"""GPU regression for Q/K address indexing; skipped in CPU-only environments."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "Нужна CUDA GPU")
class GroupedReductionTests(unittest.TestCase):
    def test_grouped_and_ungrouped_against_torch(self):
        from mamba_ssm.ops.triton.mamba3.grouped_head_reduction import reduce_grouped_qk_grads_and_bias_triton

        torch.manual_seed(42)
        shape = (2, 17, 4, 8, 16)
        dq = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        dk = torch.randn_like(dq)
        for groups in (1, 2, 8):
            with self.subTest(groups=groups):
                actual = reduce_grouped_qk_grads_and_bias_triton(dq, dk, groups)
                expected = [raw.float().reshape(2, 17, 4, groups, 8 // groups, 16)
                            .sum(dim=4).to(raw.dtype) for raw in (dq, dk)]
                expected += [raw.float().sum(dim=(0, 1)).permute(1, 0, 2) for raw in (dq, dk)]
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    def test_offsets_beyond_signed_int32(self):
        from mamba_ssm.ops.triton.mamba3.grouped_head_reduction import reduce_grouped_qk_grads_and_bias_triton

        # The final row lies beyond the signed 32-bit offset range.
        # Both inputs use about 8 GiB; run separately before full SFT.
        shape = (1, 65537, 4, 64, 128)
        self.assertGreater(shape[1] * shape[2] * shape[3] * shape[4], 2**31)
        dq = torch.full(shape, 1 / 32, device="cuda", dtype=torch.bfloat16)
        dk = torch.full(shape, 1 / 16, device="cuda", dtype=torch.bfloat16)
        actual = reduce_grouped_qk_grads_and_bias_triton(dq, dk, 1)
        expected_values = (2, 4, shape[1] / 32, shape[1] / 16)
        for result, expected in zip(actual, expected_values):
            self.assertTrue(torch.all(result == expected).item())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
