"""Regressions for the shared sampler and exact-loop detector."""
import unittest
from unittest.mock import patch
import torch
from hafvillai_runtime.decoding import repetition_cycle
from mamba_ssm.utils.generation import modify_logits_for_min_p_filtering, sample


class DecodingTests(unittest.TestCase):
    def test_min_p_matches_probabilities_for_batch_and_logit_shift(self):
        logits = torch.tensor([[2.,4.,-2.,3.],[-98.,-96.,-102.,-97.]])
        probabilities = logits.softmax(-1)
        expected = probabilities >= .1*probabilities.max(-1,keepdim=True).values
        for shift in (0.,100.,-100.):
            actual = logits+shift
            modify_logits_for_min_p_filtering(actual,.1)
            self.assertTrue(torch.equal(torch.isfinite(actual),expected))

    def test_min_p_temperature_top_k_and_preserved_input(self):
        logits = torch.tensor([[2.,4.,-2.,3.],[-98.,-96.,-102.,-97.]])
        original = logits.clone()
        for k in (0,3):
            captured = []
            def multinomial(probs,num_samples):
                captured.append(probs)
                return probs.argmax(-1,keepdim=True)
            with patch.object(torch,'multinomial',multinomial):
                sample(logits,top_k=k,min_p=.1,temperature=.5)
            self.assertEqual((captured[0]>0).sum(-1).tolist(),[2,2])
        torch.testing.assert_close(logits,original)

    def test_disabled_min_p_preserves_top_p_sampling(self):
        logits = torch.tensor([[2.,4.,-2.,3.]])
        expected = (torch.tensor([4.,3.])/.7).softmax(-1)
        for k in (0,3):
            captured = []
            def multinomial(probs,num_samples):
                captured.append(probs)
                return probs.argmax(-1,keepdim=True)
            with patch.object(torch,'multinomial',multinomial):
                token=sample(logits,top_k=k,min_p=0.,top_p=.9,temperature=.7)
            row=torch.zeros_like(captured[0])
            row[0,[1,3] if k==0 else [0,1]]=expected
            torch.testing.assert_close(captured[0],row)
            self.assertEqual(token.item(),1)

    def test_cycle_requires_enough_evidence_and_preserves_prefix(self):
        self.assertIsNone(repetition_cycle(list(range(100))))
        self.assertIsNone(repetition_cycle(list(range(130))))
        prefix=list(range(100,140))
        cycle=repetition_cycle(prefix+[1,2,3,4]*30)
        self.assertEqual(cycle['period'],4)
        self.assertEqual(cycle['start'],40)


if __name__ == '__main__':
    unittest.main()
