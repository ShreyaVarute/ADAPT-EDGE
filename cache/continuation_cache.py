"""
Continuation Cache for ADAPT-EDGE Sequential Partial Execution.

Stores intermediate Model B feature tensors between sequential execution stages:
  - Cache entries are uniquely keyed and validated by input identity, model checkpoint hash,
    configuration, and block index.
  - Guarantees safe reuse of previously computed features across sequential probes.
  - Supports LRU eviction, maximum memory limits, explicit clearing, and device transfer.
  - Verifies tensor shape, dtype, and device compatibility before cache hit.
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import OrderedDict
import torch


class CacheEntry:
    """
    Structured entry representing a cached intermediate feature representation.
    """

    def __init__(
        self,
        key: str,
        request_id: str,
        input_fingerprint: str,
        checkpoint_id: str,
        block_index: int,
        tensor: torch.Tensor,
        device: torch.device,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.key = key
        self.request_id = request_id
        self.input_fingerprint = input_fingerprint
        self.checkpoint_id = checkpoint_id
        self.block_index = int(block_index)
        self.tensor = tensor.detach()  # Avoid storing computation graph
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = device
        self.created_at = time.time()
        self.last_accessed_at = self.created_at
        self.access_count = 0
        self.metadata = metadata or {}
        # Memory estimation in bytes
        self.memory_bytes = tensor.element_size() * tensor.nelement()

    def touch(self) -> None:
        self.last_accessed_at = time.time()
        self.access_count += 1


class ContinuationCache:
    """
    LRU continuation cache for intermediate model features.
    Ensures safe reuse across inference stages without recomputation.
    """

    def __init__(
        self,
        max_entries: int = 50,
        max_memory_mb: float = 256.0,
        offload_to_cpu: bool = False
    ):
        self.max_entries = int(max_entries)
        self.max_memory_bytes = float(max_memory_mb) * (1024.0 ** 2)
        self.offload_to_cpu = offload_to_cpu

        # OrderedDict for LRU tracking: key -> CacheEntry
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_memory_bytes: int = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0

    @staticmethod
    def generate_input_fingerprint(
        image_tensor: torch.Tensor,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deterministic sha256 fingerprint for an input tensor and its preprocessing configuration.
        Ensures different preprocessing configurations or image contents yield different fingerprints.
        """
        hasher = hashlib.sha256()
        hasher.update(str(tuple(image_tensor.shape)).encode("utf-8"))
        hasher.update(str(image_tensor.dtype).encode("utf-8"))

        if preprocessing_config:
            hasher.update(str(sorted(preprocessing_config.items())).encode("utf-8"))

        flattened = image_tensor.detach().cpu().contiguous()
        hasher.update(flattened.numpy().tobytes()[:8192])
        return hasher.hexdigest()[:16]

    @staticmethod
    def make_key(
        input_fingerprint: str,
        checkpoint_id: str,
        block_index: int,
        request_id: Optional[str] = None
    ) -> str:
        # Base key combines fingerprint, checkpoint, and block
        base = f"{input_fingerprint}_{checkpoint_id}_block{block_index}"
        if request_id:
            return f"{request_id}:{base}"
        return base

    def put(
        self,
        request_id: str,
        input_fingerprint: str,
        checkpoint_id: str,
        block_index: int,
        feature_tensor: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an intermediate feature tensor in cache.
        """
        key = self.make_key(input_fingerprint, checkpoint_id, block_index, request_id=request_id)
        orig_device = feature_tensor.device

        stored_tensor = feature_tensor.detach()
        if self.offload_to_cpu and orig_device.type != "cpu":
            stored_tensor = stored_tensor.cpu()

        entry = CacheEntry(
            key=key,
            request_id=request_id,
            input_fingerprint=input_fingerprint,
            checkpoint_id=checkpoint_id,
            block_index=block_index,
            tensor=stored_tensor,
            device=orig_device,
            metadata=metadata
        )

        # If key already exists, remove old entry and update memory
        if key in self._cache:
            self._current_memory_bytes -= self._cache[key].memory_bytes
            del self._cache[key]

        # Enforce memory and entry limits before insertion
        while (
            (len(self._cache) >= self.max_entries or 
             (self._current_memory_bytes + entry.memory_bytes > self.max_memory_bytes))
            and len(self._cache) > 0
        ):
            self._evict_lru()

        self._cache[key] = entry
        self._current_memory_bytes += entry.memory_bytes
        return key

    def get(
        self,
        input_fingerprint: str,
        checkpoint_id: str,
        block_index: int,
        request_id: Optional[str] = None,
        target_device: Optional[torch.device] = None,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached feature tensor.
        Validates input identity, checkpoint, block index, request_id, shape, and dtype.
        Restores tensor to target_device if needed.
        """
        key = self.make_key(input_fingerprint, checkpoint_id, block_index, request_id=request_id)
        if key not in self._cache:
            # If request_id was not provided, search for any entry matching (fingerprint, checkpoint, block)
            if request_id is None:
                matches = [
                    k for k, v in self._cache.items()
                    if v.input_fingerprint == input_fingerprint
                    and v.checkpoint_id == checkpoint_id
                    and v.block_index == block_index
                ]
                if matches:
                    key = matches[-1]
                else:
                    self.misses += 1
                    return None
            else:
                self.misses += 1
                return None

        entry = self._cache[key]

        # Invariant validations
        if request_id is not None and entry.request_id != request_id:
            self.invalidate(key)
            self.misses += 1
            return None

        if entry.input_fingerprint != input_fingerprint:
            self.invalidate(key)
            self.misses += 1
            return None

        if entry.checkpoint_id != checkpoint_id:
            self.invalidate(key)
            self.misses += 1
            return None

        if entry.block_index != block_index:
            self.invalidate(key)
            self.misses += 1
            return None

        if expected_shape is not None and entry.shape != tuple(expected_shape):
            self.invalidate(key)
            self.misses += 1
            return None

        if expected_dtype is not None and entry.dtype != expected_dtype:
            self.invalidate(key)
            self.misses += 1
            return None

        # Cache hit: mark LRU
        self._cache.move_to_end(key)
        entry.touch()
        self.hits += 1

        tensor = entry.tensor
        # Ensure tensor is returned on requested device
        dest_device = target_device or entry.device
        if tensor.device != dest_device:
            tensor = tensor.to(dest_device)

        return tensor

    def contains(
        self,
        input_fingerprint: str,
        checkpoint_id: str,
        block_index: int,
        request_id: Optional[str] = None
    ) -> bool:
        key = self.make_key(input_fingerprint, checkpoint_id, block_index, request_id=request_id)
        if key in self._cache:
            return True
        if request_id is None:
            return any(
                v.input_fingerprint == input_fingerprint
                and v.checkpoint_id == checkpoint_id
                and v.block_index == block_index
                for v in self._cache.values()
            )
        return False

    def invalidate(self, key: str) -> bool:
        """Explicitly invalidate a single key."""
        if key in self._cache:
            self._current_memory_bytes -= self._cache[key].memory_bytes
            del self._cache[key]
            self.invalidations += 1
            return True
        return False

    def invalidate_request(self, request_id: str) -> int:
        """Invalidate all cached blocks belonging to a completed inference request."""
        to_del = [k for k, v in self._cache.items() if v.request_id == request_id]
        for k in to_del:
            self.invalidate(k)
        return len(to_del)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry (first in OrderedDict)."""
        if self._cache:
            oldest_key, oldest_entry = self._cache.popitem(last=False)
            self._current_memory_bytes -= oldest_entry.memory_bytes
            self.evictions += 1

    def clear(self) -> None:
        self._cache.clear()
        self._current_memory_bytes = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_ratio = (self.hits / total_requests) if total_requests > 0 else 0.0
        return {
            "entries_count": len(self._cache),
            "max_entries": self.max_entries,
            "memory_used_mb": round(self._current_memory_bytes / (1024.0 ** 2), 3),
            "max_memory_mb": round(self.max_memory_bytes / (1024.0 ** 2), 3),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(hit_ratio, 4),
            "evictions": self.evictions,
            "invalidations": self.invalidations
        }

