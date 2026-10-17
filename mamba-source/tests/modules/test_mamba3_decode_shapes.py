"""CPU regression tests for the forward/step boundary (upstream #1014/#1022).

The real forward method is exercised with a mocked kernel step. GPU kernel
correctness and numerical equivalence require a separate CUDA test.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mamba_ssm.modules.mamba3 import Mamba3


class DecodeShapeTests(unittest.TestCase):
    def test_cached_decode_preserves_values_batch_axis_and_state_objects(self):
        for batch in (1, 2):
            with self.subTest(batch=batch):
                inputs = torch.arange(batch * 8).reshape(batch, 1, 8).float()
                expected = torch.full((batch, 8), 7.0)
                states = tuple(object() for _ in range(4))
                mixer = SimpleNamespace(
                    _get_states_from_cache=Mock(return_value=states),
                    step=Mock(return_value=(expected, *states)),
                )
                params = SimpleNamespace(seqlen_offset=3)

                actual = Mamba3.forward(mixer, inputs, inference_params=params)

                mixer._get_states_from_cache.assert_called_once_with(params, batch)
                mixer.step.assert_called_once()
                step_input, *step_states = mixer.step.call_args.args
                self.assertEqual(tuple(step_input.shape), (batch, 8))
                torch.testing.assert_close(step_input, inputs[:, 0, :])
                for before, after in zip(states, step_states):
                    self.assertIs(before, after)
                self.assertEqual(tuple(actual.shape), (batch, 1, 8))
                torch.testing.assert_close(actual[:, 0, :], expected)

    def test_cached_decode_rejects_multiple_tokens_before_calling_step(self):
        mixer = SimpleNamespace(
            _get_states_from_cache=Mock(return_value=(None,) * 4),
            step=Mock(),
        )
        with self.assertRaisesRegex(ValueError, "exactly one token"):
            Mamba3.forward(
                mixer, torch.zeros(1, 2, 8),
                inference_params=SimpleNamespace(seqlen_offset=3),
            )
        mixer.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
