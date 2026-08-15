"""KV cache block hashing and helpers.

Upstream: vllm/v1/core/kv_cache_utils.py
Tier: A

C3 binds this module exactly: prefix cache hit rate and **block hash values** must
match upstream for the same inputs. Hashing therefore has to reproduce upstream's
construction rather than merely being deterministic -- parent hash, the token tuple,
and extra keys, in that order (R6.3).

The hashing itself lands in M2 along with prefix caching. What exists now is the type
and the seam: `Request` takes a `block_hasher` callable (F8), so when hashing arrives
it plugs in without touching the request's construction path.
"""

from __future__ import annotations

from typing import NamedTuple


#: A block's identity in the prefix cache.
#:
#: A NamedTuple rather than a bare int so a hash can never be confused with a block
#: id -- the two are both ints, both flow through the KV manager, and mixing them
#: would produce a cache that silently returns the wrong blocks.
class BlockHash(NamedTuple):
    """The hash of a full block's contents, plus the tokens that produced it.

    Carrying the token ids alongside the hash lets the block pool verify a match
    rather than trusting the digest, which is what makes a hash collision a
    detectable event instead of silent data corruption. Upstream does the same.
    """

    hash_value: int
    token_ids: tuple[int, ...]


class BlockHashWithGroupId(NamedTuple):
    """A block hash scoped to one KV cache group.

    Two groups with different block sizes can produce the same digest for different
    content, so the group id is part of the cache key (R6.7).
    """

    block_hash: BlockHash
    group_id: int
