"""Logical paged KV block allocator and protected reusable-prefix cache."""
from dataclasses import dataclass


@dataclass
class Block:
    id: int
    owner: str
    last_used: int


class PagedKVCache:
    def __init__(self, num_blocks=64, block_size=16):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.free = list(range(num_blocks))
        self.blocks, self.by_request, self.prefixes = {}, {}, {}
        self.clock = 0

    def prefix_tokens(self, prefix_hash):
        return self.prefixes.get(prefix_hash, (0, []))[0] if prefix_hash else 0

    def allocate(self, request_id, tokens, prefix_hash=None, protect_prefix=False):
        """Allocate block pages. Caller handles preemption if eviction cannot help."""
        needed = (tokens + self.block_size - 1) // self.block_size
        while len(self.free) < needed:
            if not self._evict_one():
                return False
        ids = []
        for _ in range(needed):
            block_id = self.free.pop()
            self.clock += 1
            self.blocks[block_id] = Block(block_id, request_id, self.clock)
            ids.append(block_id)
        self.by_request.setdefault(request_id, []).extend(ids)
        if prefix_hash and protect_prefix:
            # Replacing a completed cached prefix must release its old pages;
            # otherwise repeated identical hashes would leak protected blocks.
            old = self.prefixes.get(prefix_hash)
            if old:
                for old_id in old[1]:
                    if old_id in self.blocks and all(old_id not in owner_ids for owner_ids in self.by_request.values()):
                        del self.blocks[old_id]
                        self.free.append(old_id)
            self.prefixes[prefix_hash] = (tokens, list(ids))
        return True

    def touch(self, request_id):
        self.clock += 1
        for block_id in self.by_request.get(request_id, []):
            self.blocks[block_id].last_used = self.clock

    def free_request(self, request_id, keep_prefix=False):
        ids = self.by_request.pop(request_id, [])
        protected = {block for _, blocks in self.prefixes.values() for block in blocks} if keep_prefix else set()
        for block_id in ids:
            if block_id not in protected and block_id in self.blocks:
                del self.blocks[block_id]; self.free.append(block_id)

    def evict_unused_prefixes(self, reusable_hashes):
        """Release cached prefixes only when no waiting request can reuse them."""
        for key in list(self.prefixes):
            if key not in reusable_hashes:
                _, ids = self.prefixes.pop(key)
                for block_id in ids:
                    if block_id in self.blocks:
                        del self.blocks[block_id]; self.free.append(block_id)

    def _evict_one(self):
        protected = {block for _, blocks in self.prefixes.values() for block in blocks}
        # Never evict blocks backing a reusable prefix: a waiting request may
        # still reuse them. This safety constraint takes precedence over LRU.
        candidates = [b for b in self.blocks.values() if b.id not in protected]
        if not candidates:
            return False
        victim = min(candidates, key=lambda b: b.last_used)
        self.by_request[victim.owner].remove(victim.id)
        if not self.by_request[victim.owner]:
            del self.by_request[victim.owner]
        del self.blocks[victim.id]; self.free.append(victim.id)
        return True

    def utilization(self):
        """Fraction of logical KV pages currently allocated."""
        return len(self.blocks) / self.num_blocks

    def assert_invariants(self):
        """Fail fast if page ownership or protected-prefix bookkeeping drifts."""
        assert not (set(self.blocks) & set(self.free)), "allocated page is also free"
        assert len(self.blocks) + len(self.free) == self.num_blocks, "page accounting mismatch"
        protected = {block_id for _, ids in self.prefixes.values() for block_id in ids}
        owned = {block_id for ids in self.by_request.values() for block_id in ids} | protected
        assert owned == set(self.blocks), "owner map does not match allocated pages"
        for _, ids in self.prefixes.values():
            assert set(ids) <= set(self.blocks), "prefix references a freed page"
