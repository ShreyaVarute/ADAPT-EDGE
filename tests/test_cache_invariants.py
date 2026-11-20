import unittest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache.continuation_cache import ContinuationCache


class TestCacheInvariants(unittest.TestCase):
    """
    Regression test suite enforcing strict cache invariants:
    - Incompatible tensor shapes are rejected and invalidated.
    - Incompatible tensor dtypes are rejected and invalidated.
    - Different preprocessing configurations yield different fingerprints.
    - Different inference requests do not collide.
    """

    def setUp(self):
        self.cache = ContinuationCache(max_entries=10, max_memory_mb=64.0)

    def test_shape_mismatch_rejected(self):
        tensor = torch.randn(1, 40, 28, 28)
        self.cache.put("req_1", "fp_1", "ckpt_1", 4, tensor)

        # Expecting shape (1, 80, 14, 14) but cached has (1, 40, 28, 28)
        res = self.cache.get("fp_1", "ckpt_1", 4, expected_shape=(1, 80, 14, 14))
        self.assertIsNone(res)

    def test_dtype_mismatch_rejected(self):
        tensor = torch.randn(1, 40, 28, 28, dtype=torch.float32)
        self.cache.put("req_1", "fp_1", "ckpt_1", 4, tensor)

        # Expecting float16 but cached is float32
        res = self.cache.get("fp_1", "ckpt_1", 4, expected_dtype=torch.float16)
        self.assertIsNone(res)

    def test_preprocessing_config_fingerprint_uniqueness(self):
        img = torch.randn(1, 3, 224, 224)
        config1 = {"resize": (224, 224), "norm_mean": [0.485, 0.456, 0.406]}
        config2 = {"resize": (224, 224), "norm_mean": [0.5, 0.5, 0.5]}

        fp1 = ContinuationCache.generate_input_fingerprint(img, preprocessing_config=config1)
        fp2 = ContinuationCache.generate_input_fingerprint(img, preprocessing_config=config2)

        self.assertNotEqual(fp1, fp2, "Different preprocessing configs must produce different cache fingerprints!")

    def test_different_requests_do_not_cross_leak(self):
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])

        self.cache.put("req_A", "fp_same", "ckpt_1", 4, t1)
        self.cache.put("req_B", "fp_same", "ckpt_1", 4, t2)

        out_a = self.cache.get("fp_same", "ckpt_1", 4, request_id="req_A")
        out_b = self.cache.get("fp_same", "ckpt_1", 4, request_id="req_B")

        self.assertTrue(torch.equal(out_a, t1))
        self.assertTrue(torch.equal(out_b, t2))


if __name__ == "__main__":
    unittest.main()

