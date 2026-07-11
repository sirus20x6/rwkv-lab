import numpy as np
import pytest
import torch

import rwkv_lab.rosa_sam as rosa_sam
from rwkv_lab.rosa_sam import (HAVE_CUDA, cuda_sam_retrieve_cf,
                               cuda_sam_workspace_bytes, sam_retrieve_cf)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_CUDA,
                    reason="Numba CUDA suffix automaton unavailable")
def test_cuda_suffix_automaton_matches_cpu_oracle():
    rng = np.random.default_rng(17)
    query = rng.integers(0, 16, (2, 31, 7), dtype=np.int32)
    key = rng.integers(0, 16, (2, 31, 7), dtype=np.int32)
    expected = sam_retrieve_cf(query, key, 16, 4)
    actual = cuda_sam_retrieve_cf(
        torch.from_numpy(query).cuda(), torch.from_numpy(key).cuda(), 16, 4)
    for expected_table, actual_table in zip(expected, actual):
        assert np.array_equal(expected_table, actual_table.cpu().numpy())
    workspace = tuple(tensor.data_ptr() for tensor in next(reversed(rosa_sam._CUDA_WORKSPACES.values())))
    cuda_sam_retrieve_cf(torch.from_numpy(query).cuda(), torch.from_numpy(key).cuda(), 16, 4)
    reused = tuple(tensor.data_ptr() for tensor in next(reversed(rosa_sam._CUDA_WORKSPACES.values())))
    assert reused == workspace
    assert cuda_sam_workspace_bytes(2, 31, 7, 16) == 2 * 7 * (2 * 31 + 5) * 19 * 4
