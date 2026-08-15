"""The waiting queue, under each scheduling policy.

Upstream: vllm/v1/core/sched/request_queue.py
Tier: A

R5.6. The queue's ordering *is* the admission order, so it is part of C1.

`prepend_request` matters more than it looks: a preempted request goes back to the
**front** of the waiting queue, not the back (R5.5). Sending it to the back would let
newer requests overtake it indefinitely and starve it, and the preemption trace would
stop matching upstream's.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pvllm.v1.request import Request


class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Requests waiting for admission."""

    @abstractmethod
    def add_request(self, request: Request) -> None: ...

    @abstractmethod
    def pop_request(self) -> Request: ...

    @abstractmethod
    def peek_request(self) -> Request: ...

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Put a request at the front. Used when a preempted request returns."""

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None: ...

    @abstractmethod
    def remove_request(self, request: Request) -> None: ...

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None: ...

    @abstractmethod
    def __bool__(self) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Request]: ...


class FCFSRequestQueue(deque["Request"], RequestQueue):
    """First come, first served. Arrival order is admission order."""

    def add_request(self, request: Request) -> None:
        self.append(request)

    def pop_request(self) -> Request:
        return self.popleft()

    def peek_request(self) -> Request:
        if not self:
            raise IndexError("peek_request from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self.extendleft(reversed(list(requests)))

    def remove_request(self, request: Request) -> None:
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = set(requests)
        remaining = [r for r in self if r not in to_remove]
        self.clear()
        self.extend(remaining)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return deque.__len__(self)

    def __iter__(self) -> Iterator[Request]:
        return deque.__iter__(self)


class PriorityRequestQueue(RequestQueue):
    """Ordered by priority, then arrival time, then request id (R5.6).

    A heap rather than a sorted list: admission pops one request at a time while new
    requests keep arriving, so push and pop both need to stay logarithmic.

    The tiebreak chain comes from `Request.__lt__` and is what makes the order total.
    Under a virtual clock many requests share an arrival instant, so without the
    request-id tiebreak the admission order -- and therefore C1 -- would depend on
    heap internals.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, str, Request]] = []

    @staticmethod
    def _key(request: Request) -> tuple[int, float, str, Request]:
        return (request.priority, request.arrival_time, request.request_id, request)

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, self._key(request))

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop_request from an empty queue")
        return heapq.heappop(self._heap)[3]

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek_request from an empty queue")
        return self._heap[0][3]

    def prepend_request(self, request: Request) -> None:
        """Re-insert by priority.

        A priority queue has no "front" to prepend to -- position is a function of
        the request's own key. A preempted request therefore returns to wherever its
        priority puts it, which is upstream's behaviour too.
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._heap = [entry for entry in self._heap if entry[3] is not request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = set(requests)
        self._heap = [entry for entry in self._heap if entry[3] not in to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Yields in priority order.

        Sorts a copy rather than walking the heap array, which is only partially
        ordered -- iterating the raw array would give an order that looks right for
        small queues and silently is not for large ones.
        """
        return iter(entry[3] for entry in sorted(self._heap))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    raise ValueError(f"unknown scheduling policy: {policy}")
