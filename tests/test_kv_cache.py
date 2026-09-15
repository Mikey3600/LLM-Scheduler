from scheduler.kv_cache import PagedKVCache
def test_prefix_is_reused_and_protected_from_lru():
    c=PagedKVCache(2,4); assert c.allocate('a',4,'p',True); assert c.prefix_tokens('p')==4
    assert c.allocate('b',4); c.touch('b'); assert c.allocate('c',4) # evicts b, never protected prefix a
    assert c.prefix_tokens('p')==4


def test_protected_prefix_blocks_prevent_exhaustion_eviction_until_unused():
    c = PagedKVCache(1, 4)
    assert c.allocate("prefix", 4, "shared", True)
    assert not c.allocate("other", 4)
    c.evict_unused_prefixes(set())
    assert c.allocate("other", 4)
    c.assert_invariants()
