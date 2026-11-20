import unittest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache.continuation_cache import ContinuationCache, CacheEntry


class TestContinuationCache(unittest.TestCase):
    """
    Automated test suite for cache/continuation_cache.py:
    1. Store and retrieve feature tensors.
    2. Cache hit.
    3. Cache miss.
    4. Cache invalidation.
    5. Different input images.
    6. Different model checkpoint fingerprints.
    7. Different tensor shapes.
    8. Different continuation block.
    9. LRU eviction.
    10. Memory limit enforcement.
    11. Cache clearing.
    12. Numerical equivalence of stored vs retrieved tensor.
    """

    def setUp(self):
        self.cache = ContinuationCache(max_entries=5, max_memory_mb=10.0)

    def test_put_and_get_hit(self):
        tensor = torch.randn(1, 40, 28, 28)
        img_fp = "img_001"
        ckpt_id = "ckpt_b_v1"
        block = 4

        key = self.cache.put(
            request_id="req_1",
            input_fingerprint=img_fp,
            checkpoint_id=ckpt_id,
            block_index=block,
            feature_tensor=tensor
        )
        self.assertIsNotNone(key)

        retrieved = self.cache.get(img_fp, ckpt_id, block)
        self.assertIsNotNone(retrieved)
        # Numerical equivalence
        self.assertTrue(torch.equal(tensor, retrieved))
        self.assertEqual(self.cache.hits, 1)

    def test_cache_miss_on_different_input(self):
        tensor = torch.randn(1, 40, 28, 28)
        self.cache.put("req_1", "img_001", "ckpt_b_v1", 4, tensor)

        # Query with different image fingerprint
        retrieved = self.cache.get("img_002", "ckpt_b_v1", 4)
        self.assertIsNone(retrieved)
        self.assertEqual(self.cache.misses, 1)

    def test_cache_miss_on_different_checkpoint(self):
        tensor = torch.randn(1, 40, 28, 28)
        self.cache.put("req_1", "img_001", "ckpt_b_v1", 4, tensor)

        # Query with different checkpoint
        retrieved = self.cache.get("img_001", "ckpt_b_v2", 4)
        self.assertIsNone(retrieved)

    def test_cache_miss_on_different_block(self):
        tensor = torch.randn(1, 40, 28, 28)
        self.cache.put("req_1", "img_001", "ckpt_b_v1", 4, tensor)

        # Query with different block index
        retrieved = self.cache.get("img_001", "ckpt_b_v1", 9)
        self.assertIsNone(retrieved)

    def test_lru_eviction(self):
        small_cache = ContinuationCache(max_entries=2, max_memory_mb=100.0)
        t1 = torch.randn(1, 10)
        t2 = torch.randn(1, 10)
        t3 = torch.randn(1, 10)

        small_cache.put("r1", "i1", "c1", 4, t1)
        small_cache.put("r2", "i2", "c1", 4, t2)
        # Access i1 to make i2 the LRU
        small_cache.get("i1", "c1", 4)

        # Inserting 3rd entry should evict i2
        small_cache.put("r3", "i3", "c1", 4, t3)
        self.assertEqual(small_cache.evictions, 1)
        self.assertFalse(small_cache.contains("i2", "c1", 4))
        self.assertTrue(small_cache.contains("i1", "c1", 4))
        self.assertTrue(small_cache.contains("i3", "c1", 4))

    def test_memory_limit_enforcement(self):
        # 1 float32 = 4 bytes. Tensor with 250,000 elements = 1MB
        # Limit cache to 1.5 MB
        mem_cache = ContinuationCache(max_entries=10, max_memory_mb=1.5)
        t1 = torch.randn(250000)  # ~1.0 MB
        t2 = torch.randn(250000)  # ~1.0 MB

        mem_cache.put("r1", "i1", "c1", 4, t1)
        self.assertEqual(len(mem_cache._cache), 1)

        # Inserting t2 exceeds 1.5MB total memory, so t1 must be evicted
        mem_cache.put("r2", "i2", "c1", 4, t2)
        self.assertEqual(len(mem_cache._cache), 1)
        self.assertEqual(mem_cache.evictions, 1)
        self.assertTrue(mem_cache.contains("i2", "c1", 4))

    def test_invalidate_request(self):
        t1 = torch.randn(1, 10)
        t2 = torch.randn(1, 10)
        self.cache.put("req_100", "i1", "c1", 4, t1)
        self.cache.put("req_100", "i1", "c1", 9, t2)
        self.cache.put("req_200", "i2", "c1", 4, t1)

        self.assertEqual(len(self.cache._cache), 3)
        invalidated = self.cache.invalidate_request("req_100")
        self.assertEqual(invalidated, 2)
        self.assertEqual(len(self.cache._cache), 1)
        self.assertTrue(self.cache.contains("i2", "c1", 4))

    def test_clear_and_stats(self):
        t = torch.randn(1, 10)
        self.cache.put("r1", "i1", "c1", 4, t)
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["entries_count"], 1)

        self.cache.clear()
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["entries_count"], 0)
        self.assertEqual(stats["memory_used_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()

