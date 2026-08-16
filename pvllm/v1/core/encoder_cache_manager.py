"""The encoder output cache. R18.1.

Upstream: vllm/v1/core/encoder_cache_manager.py
Tier: A

Vision encoders are expensive and their output is reusable: the same image in two
requests produces the same embeddings, and the same image in one request produces them
once however many steps the prompt takes to prefill. This caches them, keyed by the
content hash, with a budget measured in embeddings.

Three behaviours here are worth understanding, because each is a real source of
scheduling behaviour a product will observe:

**Eviction is deferred.** Freeing a request's reference does not release the entry --
it moves to `freeable`. The embeddings stay resident until someone actually needs the
space. A request arriving with an image another request just finished with gets a hit,
which is the common case in a chat workload and would be lost by eager eviction.

**Eviction is oldest-first among unreferenced entries**, so the cache behaves like an
LRU over completed work rather than discarding whatever is convenient.

**`can_allocate` mutates.** Upstream's does too, and the name is a lie in both trees --
it evicts to make room and reports whether it succeeded. Splitting it would mean
walking the eviction candidates twice per input per step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pvllm.logger import init_logger

if TYPE_CHECKING:
    from pvllm.v1.request import Request

logger = init_logger(__name__)


class EncoderCacheManager:
    """Caches encoder outputs across requests, under a fixed budget."""

    def __init__(self, cache_size: int) -> None:
        if cache_size < 1:
            raise ValueError(
                f"encoder_cache_size must be positive, got {cache_size}: a cache that "
                f"holds nothing would make every image recompute on every step of its "
                f"own prefill"
            )
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        #: Slots held by entries nothing currently references -- reclaimable without
        #: losing work anyone is waiting on.
        self.num_freeable_slots = cache_size

        #: mm_hash -> the requests referencing it. An empty set means resident but
        #: unreferenced, which is what makes it a cache rather than a scratchpad.
        self.cached: dict[str, set[str]] = {}
        #: Unreferenced entries in eviction order, oldest first.
        self.freeable: list[str] = []
        #: request_id -> the input ids it holds.
        self.request_cached_ids: dict[str, set[int]] = {}
        #: Evicted since the last drain, for the worker to drop.
        self.freed: list[str] = []
        #: R18.1, for the metrics: how often an image was already resident.
        self.num_queries = 0
        self.num_hits = 0
        #: mm_hash -> its size in embeddings. Kept separately from `cached` because
        #: an entry's size has to survive the moment its last reference goes away.
        self._entry_size: dict[str, int] = {}

    # --- lookup --------------------------------------------------------------

    def has_cache(self, request: Request, input_id: int) -> bool:
        """Whether this item's embeddings are already resident."""
        mm_hash = request.mm_features[input_id].identifier
        return mm_hash in self.cached

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return self.request_cached_ids.get(request.request_id, set())

    # --- allocation ----------------------------------------------------------

    def can_allocate(self, request: Request, input_id: int) -> bool:
        """Whether there is room, evicting unreferenced entries to make it.

        Mutating, as upstream's is: it walks the eviction candidates and actually
        frees them. Reporting without acting would mean walking the same list twice
        for every input on every step.
        """
        num_embeds = request.mm_features[input_id].num_embeds
        if num_embeds > self.cache_size:
            # Never satisfiable. Reported rather than looped on, because the
            # scheduler would otherwise retry this input every step forever.
            return False
        if num_embeds <= self.num_free_slots:
            return True
        if num_embeds > self.num_freeable_slots:
            return False

        while self.num_free_slots < num_embeds and self.freeable:
            evicted = self.freeable.pop(0)
            references = self.cached.pop(evicted, set())
            assert not references, (
                f"encoder cache entry {evicted} was queued for eviction while "
                f"{sorted(references)} still referenced it"
            )
            self.num_free_slots += self._entry_size.pop(evicted, 0)
            # Reported to the worker so it drops the embeddings it is holding. An
            # eviction the worker never hears about is a memory leak on its side.
            self.freed.append(evicted)
        return bool(self.num_free_slots >= num_embeds)

    def allocate(self, request: Request, input_id: int) -> None:
        """Take a reference, and the space if this is the first holder."""
        feature = request.mm_features[input_id]
        mm_hash = feature.identifier
        request_id = request.request_id

        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()
            self._entry_size[mm_hash] = feature.num_embeds
            self.num_free_slots -= feature.num_embeds
            self.num_freeable_slots -= feature.num_embeds
        elif not self.cached[mm_hash]:
            # Resident but unreferenced: it was a hit, and it must come out of the
            # eviction queue before somebody reclaims it underneath its new holder.
            if mm_hash in self.freeable:
                self.freeable.remove(mm_hash)
            self.num_freeable_slots -= self._entry_size[mm_hash]

        self.cached[mm_hash].add(request_id)
        self.request_cached_ids.setdefault(request_id, set()).add(input_id)

    def record_lookup(self, hit: bool) -> None:
        """Count a cache query for the metrics. R18.1."""
        self.num_queries += 1
        self.num_hits += int(hit)

    # --- release -------------------------------------------------------------

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Drop one reference. The entry stays resident until space is needed."""
        request_id = request.request_id
        held = self.request_cached_ids.get(request_id)
        if held is not None:
            held.discard(input_id)
            if not held:
                del self.request_cached_ids[request_id]

        mm_hash = request.mm_features[input_id].identifier
        references = self.cached.get(mm_hash)
        if references is None:
            return
        references.discard(request_id)
        if not references and mm_hash not in self.freeable:
            # Guarded, because a request may reference one entry through *two* input
            # ids -- the same image twice in one message, which is an ordinary shape.
            # Queueing it twice credited its size to `num_freeable_slots` twice and
            # left a stale duplicate behind a live reference, so a later eviction
            # tripped the "referenced entry queued for eviction" assert, or under
            # `python -O` silently handed away embeddings a running request needed.
            self.freeable.append(mm_hash)
            self.num_freeable_slots += self._entry_size[mm_hash]

    def free(self, request: Request) -> None:
        """Drop every reference this request holds. Called when it finishes."""
        for input_id in list(self.get_cached_input_ids(request)):
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Hand the worker what was actually evicted, and forget it."""
        freed, self.freed = self.freed, []
        return freed

    def __repr__(self) -> str:
        return (
            f"EncoderCacheManager(size={self.cache_size}, "
            f"free={self.num_free_slots}, entries={len(self.cached)})"
        )
