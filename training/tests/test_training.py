"""CPU checks for training math, masking, state boundaries, and resume."""

from dataclasses import replace
from contextlib import ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import random
import re
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from training.checkpoints import load_checkpoint, save_checkpoint
from training.config import TrainConfig
from hafvillai_runtime.serialization import ChatSerializer, IGNORE_INDEX
from training.data import collate_one, dialogue_text, group_id, load_prepared, prepare, validation_groups
from hafvillai_runtime.lora import (LoRALinear, adapter_parameters, adapter_state, attach_lora,
                           load_adapter, save_adapter)
from training.model import causal_loss_sum, forward_hidden
from training.train import epoch_order, lr_multiplier, run


class Tokenizer:
    specials = {"<|begin_of_text|>": 1, "<|eot_id|>": 2,
                "<|start_header_id|>": 3, "<|end_header_id|>": 4}

    def __len__(self):
        return 512

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Нельзя автоматически добавлять второй BOS")
        result = []
        for part in re.split(r"(<\|[^>]+\|>)", text):
            result.extend([self.specials[part]] if part in self.specials else [10 + ord(c) for c in part])
        return result

    def convert_ids_to_tokens(self, index):
        return next((s for s, i in self.specials.items() if i == index), "?")


class TinyMixer(nn.Module):
    """CPU test substitute; real MIMO kernels require separate verification."""
    is_mimo = True
    chunk_size = 16

    def __init__(self, width):
        super().__init__()
        self.in_proj = nn.Linear(width, width * 2)
        self.out_proj = nn.Linear(width * 2, width)

    def forward(self, x):
        return self.out_proj(torch.tanh(self.in_proj(x)).cumsum(dim=1))


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mixer = TinyMixer(width)

    def forward(self, hidden, residual=None):
        residual = hidden if residual is None else hidden + residual
        return self.mixer(self.norm(residual)), residual


class TinyBackbone(nn.Module):
    fused_add_norm = False

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(23, 8)
        self.layers = nn.ModuleList([TinyBlock(8), TinyBlock(8)])
        self.norm_f = nn.LayerNorm(8)

    def forward(self, ids):
        hidden, residual = self.embedding(ids), None
        for block in self.layers:
            hidden, residual = block(hidden, residual)
        return self.norm_f(hidden + residual)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.lm_head = nn.Linear(8, 23, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight


class DataTests(unittest.TestCase):
    def test_content_and_assistant_mask_preserve_thinking_and_eot(self):
        tokenizer = Tokenizer()
        serializer = ChatSerializer(tokenizer)
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                    {"role": "assistant", "content": "<system_thinking>T</system_thinking> A"},
                    {"role": "user", "content": "tail"}]
        ids, labels = serializer.encode(messages)
        self.assertEqual(ids[0], serializer.bos)
        self.assertEqual(int((ids == serializer.bos).sum()), 1)
        targets = labels[labels != IGNORE_INDEX].tolist()
        self.assertEqual(targets, tokenizer.encode(messages[2]["content"]) + [serializer.eot])
        self.assertTrue(torch.equal(ids[labels != IGNORE_INDEX], labels[labels != IGNORE_INDEX]))
        prompt = serializer.prompt(messages[:-1])
        self.assertEqual(prompt[-len(serializer.headers["assistant"]):], serializer.headers["assistant"])

    def test_split_is_shared_by_dialogue_summary_and_duplicates(self):
        rows = [{"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "reply"}]}
                for i in range(10)]
        valid, groups = validation_groups(rows + [rows[0]], 42, .2)
        self.assertEqual(len(groups), 10)
        self.assertEqual(len(valid), 2)
        for row in rows:
            summary = {"messages": [{"role": "system", "content": "sum"},
                                     {"role": "user", "content": dialogue_text(row)},
                                     {"role": "assistant", "content": "summary"}]}
            self.assertEqual(group_id(row, "consciousness"), group_id(summary, "subconsciousness"))

    def test_right_padding_masks_only_tail(self):
        item = {"input_ids": torch.tensor([1, 5, 6]), "labels": torch.tensor([-100, -100, 6])}
        ids, labels = collate_one(item, 2, 16)
        self.assertEqual(ids.shape, (1, 16))
        self.assertEqual(ids[0, :3].tolist(), [1, 5, 6])
        self.assertTrue((ids[0, 3:] == 2).all())
        self.assertTrue((labels[0, 3:] == -100).all())

    def test_prepare_roundtrip_and_stale_data_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            rows = [{"messages": [{"role": "user", "content": str(i)},
                                   {"role": "assistant", "content": "long reply"}]} for i in range(10)]
            source.write_text("\n".join(json.dumps(row) for row in rows))
            config = TrainConfig("consciousness", str(source), str(root / "out"),
                                 source_dataset=str(source), prepared_dir=str(root / "prepared"))
            with patch("training.data.load_tokenizer", return_value=Tokenizer()):
                report = prepare(config, offline=True)
                manifest, train, valid = load_prepared(config)
                self.assertEqual(len(train) + len(valid), 10)
                self.assertEqual(report, manifest["report"])
                self.assertFalse({x["group"] for x in train} & {x["group"] for x in valid})
                self.assertEqual(prepare(config), report)
                with self.assertRaisesRegex(ValueError, "не обрезан"):
                    prepare(replace(config, max_seq_length=2, prepared_dir=str(root / "short")))
            source.write_text(source.read_text() + "\n{}")
            with self.assertRaisesRegex(ValueError, "изменились"):
                load_prepared(config)


class MathTests(unittest.TestCase):
    def test_mimo_autograd_wrapper_with_nonreentrant_checkpoint(self):
        # Exercise the real autograd wrapper, replacing GPU computation with differentiable Q*V.
        # This failed before saved_tensors was changed to a single read.
        module = importlib.import_module("mamba_ssm.ops.tilelang.mamba3.mamba3_mimo")
        from torch.utils.checkpoint import checkpoint

        def kernel_backward(dout, q, k, v, *args, **kwargs):
            zero = torch.zeros_like(q)
            return (dout * v, zero, dout * q, zero, zero, zero, zero, zero,
                    zero, zero, zero, zero, zero, zero, None)

        def operation(x):
            zero = torch.zeros_like(x)
            return module._Mamba3Function.apply(
                x, zero, x * 2, zero, zero, zero, zero, zero, zero, zero,
                zero, None, zero, zero, zero, 16, 2, x.dtype, False, None, False, 1e-5)

        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "angle_dt_fwd",
                                            side_effect=lambda angles, dt, **kw: (angles.clone(), None)))
            stack.enter_context(patch.object(module, "compute_dacs_segsum_triton",
                                            side_effect=lambda adt, chunk: (adt, adt, adt)))
            forward = stack.enter_context(patch.object(module, "mamba_mimo_forward",
                                            side_effect=lambda q, k, v, *a, **kw: (q * v, None, None)))
            stack.enter_context(patch.object(module, "mamba_mimo_bwd_combined", side_effect=kernel_backward))
            stack.enter_context(patch.object(module, "angle_dt_bwd",
                                            side_effect=lambda **kw: (kw["grad_out"], kw["grad_out"], None)))
            x = torch.randn(2, 3, requires_grad=True)
            output = checkpoint(operation, x, use_reentrant=False)
            torch.testing.assert_close(output, 2 * x.square())
            output.sum().backward()
            torch.testing.assert_close(x.grad, 4 * x)
            self.assertEqual(forward.call_count, 2)

    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_lora_identity_freeze_gradients_and_dtype(self):
        linear = nn.Linear(4, 5, dtype=torch.bfloat16)
        layer = LoRALinear(linear, 2, 4)
        x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        torch.testing.assert_close(layer(x), linear(x), rtol=0, atol=0)
        original = linear.weight.detach().clone()
        optimizer = torch.optim.AdamW([layer.lora_A, layer.lora_B], lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            layer(x).float().square().mean().backward()
            self.assertIsNotNone(layer.lora_A.grad)
            self.assertIsNone(linear.weight.grad)
            self.assertEqual(layer.lora_B.grad.dtype, torch.float32)
            optimizer.step()
        self.assertGreater(float(layer.lora_A.grad.abs().sum()), 0)
        self.assertEqual(layer(x).dtype, torch.bfloat16)
        torch.testing.assert_close(linear.weight, original, rtol=0, atol=0)
        self.assertIs(layer.weight, linear.weight)

    def test_chunked_loss_matches_dense_value_and_gradients(self):
        head = nn.Linear(8, 23).requires_grad_(False)
        hidden = torch.randn(2, 7, 8, requires_grad=True)
        reference = hidden.detach().clone().requires_grad_()
        labels = torch.tensor([[-100, -100, 4, 5, -100, 6, 2], [-100, 7, 8, -100, 9, -100, -100]])
        loss, count = causal_loss_sum(head, hidden, labels, chunk_tokens=2)
        dense = F.cross_entropy(head(reference[:, :-1]).reshape(-1, 23), labels[:, 1:].reshape(-1),
                                ignore_index=-100, reduction="sum")
        self.assertEqual(count, 7)
        torch.testing.assert_close(loss, dense)
        loss.backward()
        dense.backward()
        torch.testing.assert_close(hidden.grad, reference.grad)
        self.assertTrue((hidden.grad[:, -1] == 0).all())

    def test_checkpoint_frozen_embedding_matches_forward_and_backward(self):
        model = TinyModel()
        attach_lora(model, 2, 4)
        for name, p in adapter_parameters(model).items():
            if name.endswith(".lora_B"):
                nn.init.normal_(p, std=.1)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[-100, -100, 3, 4, 5]])
        expected = model.backbone(ids)
        actual = forward_hidden(model, ids, gradient_checkpointing=True)
        torch.testing.assert_close(actual, expected)
        loss, _ = causal_loss_sum(model.lm_head, actual, labels, chunk_tokens=2)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in adapter_parameters(model).items()}
        model.zero_grad(set_to_none=True)
        loss, _ = causal_loss_sum(model.lm_head, model.backbone(ids), labels, checkpoint_loss=False)
        loss.backward()
        for name, p in adapter_parameters(model).items():
            torch.testing.assert_close(p.grad, grads[name])
        self.assertIsNone(model.backbone.embedding.weight.grad)
        # Extending the sequence on the right must not alter earlier states.
        with torch.no_grad():
            padded = model.backbone(torch.cat([ids, torch.tensor([[6, 7, 8]])], dim=1))
        torch.testing.assert_close(padded[:, :5], actual)

    def test_token_normalized_accumulation_handles_unequal_lengths(self):
        head = nn.Linear(8, 23).requires_grad_(False)
        value = nn.Parameter(torch.randn(8))
        reference = value.detach().clone().requires_grad_()
        batches = [torch.tensor([[-100, 1, 2]]), torch.tensor([[-100, -100, 3, 4, 5, 6]])]
        denominator = 6
        for labels in batches:
            hidden = value.expand(1, labels.shape[1], 8)
            loss, _ = causal_loss_sum(head, hidden, labels, chunk_tokens=2)
            (loss / denominator).backward()
        total = sum(causal_loss_sum(head, reference.expand(1, labels.shape[1], 8), labels)[0]
                    for labels in batches) / denominator
        total.backward()
        torch.testing.assert_close(value.grad, reference.grad)

    def test_empty_targets_fail(self):
        with self.assertRaisesRegex(ValueError, "Нет целевых"):
            causal_loss_sum(nn.Linear(8, 23), torch.randn(1, 3, 8), torch.full((1, 3), -100))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        random.seed(19)
        torch.set_num_threads(1)

    def test_adapter_roundtrip_and_base_mismatch(self):
        model = TinyModel()
        initial_base = {k: v.clone() for k, v in model.state_dict().items()}
        attach_lora(model, 2, 4)
        for p in adapter_parameters(model).values():
            nn.init.normal_(p)
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(model, tmp, {"base": {"revision": "one"}})
            fresh = TinyModel()
            fresh.load_state_dict(initial_base)
            load_adapter(fresh, tmp, expected_base={"revision": "one"})
            for name, tensor in adapter_state(model).items():
                torch.testing.assert_close(adapter_state(fresh)[name], tensor, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "не совпадают"):
                load_adapter(TinyModel(), tmp, expected_base={"revision": "two"})
            with self.assertRaisesRegex(ValueError, "уже подключён"):
                attach_lora(fresh)

    def test_exact_resume_restores_optimizer_scheduler_rng(self):
        model = TinyModel()
        base_state = {k: v.clone() for k, v in model.state_dict().items()}
        attach_lora(model, 2, 4, dropout=.2)
        optimizer = torch.optim.AdamW(adapter_parameters(model).values(), lr=.03)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, 4, 1))
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[-100, -100, 3, 4, 5]])

        def step(m, opt, sched):
            opt.zero_grad(set_to_none=True)
            hidden = forward_hidden(m, ids)
            loss, count = causal_loss_sum(m.lm_head, hidden, labels, chunk_tokens=2)
            (loss / count).backward()
            opt.step()
            sched.step()

        manifest = {"config": {"role": "consciousness"}, "base": {"revision": "one"},
                    "tokenizer": {}, "source_fingerprint": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            step(model, optimizer, scheduler)
            saved = save_checkpoint(tmp, model, optimizer, scheduler,
                                    {"step": 1, "epoch": 0, "next_index": 1}, manifest)
            step(model, optimizer, scheduler)
            uninterrupted = adapter_state(model)
            python_value, torch_value = random.random(), torch.rand(3)
            resumed = TinyModel()
            resumed.load_state_dict(base_state)
            attach_lora(resumed, 2, 4, dropout=.2)
            opt = torch.optim.AdamW(adapter_parameters(resumed).values(), lr=.03)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, 4, 1))
            progress = load_checkpoint(saved, resumed, opt, sched, manifest)
            self.assertEqual(progress["next_index"], 1)
            step(resumed, opt, sched)
            for name, tensor in adapter_state(resumed).items():
                torch.testing.assert_close(tensor, uninterrupted[name], rtol=0, atol=0)
            self.assertEqual(random.random(), python_value)
            torch.testing.assert_close(torch.rand(3), torch_value, rtol=0, atol=0)
            self.assertEqual(sched.state_dict(), scheduler.state_dict())
            with self.assertRaisesRegex(ValueError, "другими"):
                load_checkpoint(saved, resumed, opt, sched, dict(manifest, base={"revision": "two"}))

    def test_epoch_order_reproducible_without_global_rng(self):
        state = random.getstate()
        self.assertEqual(epoch_order(100, 42, 1), epoch_order(100, 42, 1))
        self.assertNotEqual(epoch_order(100, 42, 0), epoch_order(100, 42, 1))
        self.assertEqual(state, random.getstate())

    def test_trainer_resume_after_partial_epoch_matches_uninterrupted(self):
        """Exercise the runner with a CPU substitute, including partial accumulation groups."""
        rows = []
        for size in (5, 4, 6):
            ids = torch.arange(1, size + 1, dtype=torch.int32)
            labels = ids.clone()
            labels[:2] = -100
            rows.append({"input_ids": ids, "labels": labels, "target_tokens": size - 2, "row": size})
        prepared = {"report": {"padding_token_id": 2, "tokenizer_vocab_size": 23}}
        gpu_stubs = {"is_available": True, "is_bf16_supported": True, "get_device_name": "CPU test",
                     "get_rng_state_all": [], "max_memory_allocated": 0, "max_memory_reserved": 0,
                     "is_current_stream_capturing": False}

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch("training.train.load_prepared", return_value=(prepared, rows, rows[:1])))
            stack.enter_context(patch("training.train.load_base", side_effect=lambda *a, **k: (TinyModel(), {"revision": "test"})))
            stack.enter_context(patch("training.train.assert_kernel_dtypes"))
            stack.enter_context(patch("training.train.source_fingerprint", return_value="cpu-test"))
            stack.enter_context(patch("training.train.load_tokenizer"))
            stack.enter_context(patch.object(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor))
            stack.enter_context(patch("torch.autocast", side_effect=lambda *a, **k: nullcontext()))
            for name, value in gpu_stubs.items():
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ("set_device", "manual_seed_all", "synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, name))
            config = TrainConfig("consciousness", "unused", str(Path(tmp) / "baseline"), epochs=2,
                                 gradient_accumulation=2, lora_rank=2, lora_alpha=4,
                                 lora_dropout=.2, padding_multiple=16, eval_every=2, save_every=2)
            run(config)
            interrupted = replace(config, output_dir=str(Path(tmp) / "interrupted"))

            class SimulatedInterruption(Exception):
                pass

            def stop_after_save(*args, **kwargs):
                save_checkpoint(*args, **kwargs)
                raise SimulatedInterruption

            with patch("training.train.save_checkpoint", side_effect=stop_after_save):
                with self.assertRaises(SimulatedInterruption):
                    run(interrupted)
            run(interrupted, resume="latest")
            from safetensors.torch import load_file
            baseline = load_file(str(Path(config.output_dir) / "checkpoint-000004/adapter.safetensors"))
            resumed = load_file(str(Path(interrupted.output_dir) / "checkpoint-000004/adapter.safetensors"))
            for name, tensor in baseline.items():
                torch.testing.assert_close(tensor, resumed[name], rtol=0, atol=0)
            completion = json.loads((Path(interrupted.output_dir) / "completed.json").read_text())
            self.assertEqual(completion["step"], 4)


if __name__ == "__main__":
    unittest.main()
