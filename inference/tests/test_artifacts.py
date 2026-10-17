"""Portable artifact integrity, isolation, and cloud listener configuration."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hafvillai_runtime.config import ModelSpec, SERIALIZATION
from hafvillai_runtime.files import file_sha256
from inference.artifact_store import ROLES, TOKENIZER_FILES, prepare_artifacts, verify_artifacts
from inference.api.settings import Settings
from inference.bundle import build_bundle


def fixture(root):
    base = root / "cached-base"
    base.mkdir()
    (base / "config.json").write_text('{"ssm_cfg":{"layer":"Mamba3","is_mimo":true}}\n')
    (base / "pytorch_model.bin").write_bytes(b"test base weights")
    spec = replace(ModelSpec(), model_config_sha256=file_sha256(base / "config.json"),
                   model_weights_sha256=file_sha256(base / "pytorch_model.bin"))
    runs = root / "runs"
    for role, step in zip(ROLES, (650, 242)):
        run = runs / role
        checkpoint = run / f"checkpoint-{step:06d}"
        checkpoint.mkdir(parents=True)
        (run / "best.json").write_text(json.dumps({"checkpoint": checkpoint.name}))
        metadata = {"format": "hafvillai-mamba3-lora-v1", "role": role, "step": step,
                    "base": {"model_id": spec.model_id, "revision": spec.model_revision,
                             "config_sha256": spec.model_config_sha256},
                    "tokenizer": {"id": spec.tokenizer_id, "revision": spec.tokenizer_revision,
                                  "format": SERIALIZATION}}
        (checkpoint / "adapter_config.json").write_text(json.dumps(metadata))
        (checkpoint / "adapter.safetensors").write_bytes(b"test adapter weights")
        (checkpoint / "training.pt").write_bytes(b"optimizer must stay out of the bundle")
        (run / "tokenizer").mkdir()
        for name in TOKENIZER_FILES:
            (run / "tokenizer" / name).write_text("{}")
    return runs, base, spec


class ArtifactTests(unittest.TestCase):
    def test_export_is_independent_and_excludes_optimizer_and_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, base, spec = fixture(root)
            output = root / "artifacts"
            with patch("inference.artifact_store.ModelSpec", return_value=spec):
                manifest = prepare_artifacts(runs, base, output)
                self.assertEqual(verify_artifacts(output), manifest)
                self.assertEqual(len(manifest["files"]), 12)
                self.assertFalse(list(output.rglob("training.pt")))
                self.assertNotIn(str(root), (output / "manifest.json").read_text())
                with self.assertRaises(FileExistsError):
                    prepare_artifacts(runs, base, output)
                (base / "pytorch_model.bin").write_bytes(b"changed original")
                self.assertEqual((output / "base/pytorch_model.bin").read_bytes(), b"test base weights")
                (output / "consciousness/adapter.safetensors").write_bytes(b"corrupt copy")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    verify_artifacts(output)

    def test_incompatible_metadata_and_checkpoint_traversal_leave_no_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, base, spec = fixture(root)
            metadata_path = runs / "consciousness/checkpoint-000650/adapter_config.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["tokenizer"]["revision"] = "wrong-revision"
            metadata_path.write_text(json.dumps(metadata))
            with patch("inference.artifact_store.ModelSpec", return_value=spec):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    prepare_artifacts(runs, base, root / "artifacts")
                self.assertFalse((root / "artifacts").exists())
                (runs / "consciousness/best.json").write_text('{"checkpoint":"../../outside"}')
                with self.assertRaisesRegex(ValueError, "escapes"):
                    prepare_artifacts(runs, base, root / "artifacts")

    def test_private_bundle_excludes_local_secrets_and_training_and_imports_in_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, base, spec = fixture(root)
            repository = root / "repository"
            repository.mkdir()
            # Use the real runtime sources, with synthetic weights and a private env file.
            import shutil
            project = Path(__file__).resolve().parents[2]
            for name in ("inference", "hafvillai_runtime"):
                shutil.copytree(project / name, repository / name,
                                ignore=shutil.ignore_patterns("artifacts", "dist", ".cache", "__pycache__", ".env"))
            (repository / "mamba-source").mkdir()
            (repository / "mamba-source/LICENSE").write_text("Vendor license fixture")
            (repository / ".dockerignore").write_text((project / ".dockerignore").read_text())
            (repository / "inference/.env").write_text("HAFVILLAI_API_KEY=private-test-value\n")
            (repository / "inference/secrets").mkdir()
            (repository / "inference/secrets/credentials.json").write_text('{"secret":"private"}')
            (repository / "training").mkdir()
            (repository / "training/train.py").write_text("raise AssertionError('must not be imported')\n")
            with patch("inference.artifact_store.ModelSpec", return_value=spec):
                prepare_artifacts(runs, base, repository / "inference/artifacts")
                build_bundle(repository, repository / "inference/artifacts", root / "bundle")
                verify_artifacts(root / "bundle/inference/artifacts")
            self.assertFalse((root / "bundle/inference/.env").exists())
            self.assertFalse((root / "bundle/inference/secrets").exists())
            self.assertTrue((root / "bundle/inference/.env.example").is_file())
            self.assertFalse((root / "bundle/training").exists())
            self.assertFalse((root / "bundle/hafvillai").exists())
            script = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'training', 'hafvillai', 'dataset-studio'}:
            raise AssertionError('Inference imported project-only code: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
import inference.api.engine
print('ISOLATED_RUNTIME_OK')
"""
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-I", "-c", script, str(root / "bundle")],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ISOLATED_RUNTIME_OK", result.stdout)


class DeploymentSettingsTests(unittest.TestCase):
    def test_network_binding_requires_an_api_key_for_all_entry_points(self):
        with self.assertRaisesRegex(ValueError, "API_KEY"):
            Settings(host="0.0.0.0", api_key=" ")
        self.assertEqual(Settings(host="0.0.0.0", api_key="test-key").host, "0.0.0.0")

    def test_relative_artifact_directory_uses_the_bundle_root(self):
        with patch.dict(os.environ, {"HAFVILLAI_ARTIFACTS_DIR": "inference/artifacts"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.artifacts_dir, Path(__file__).resolve().parents[1] / "artifacts")
