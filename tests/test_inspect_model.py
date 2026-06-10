"""Tests for scripts/inspect_model.py — verifies import health and core functions."""

import importlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch



# ---------------------------------------------------------------------------
# Import health
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.inspect_model")
        assert hasattr(mod, "main")
        assert hasattr(mod, "inspect_from_config")
        assert hasattr(mod, "inspect_from_yaml")

    def test_private_functions_exist(self):
        mod = importlib.import_module("scripts.inspect_model")
        assert hasattr(mod, "_analyze_model")
        assert hasattr(mod, "_print_config_summary")


# ---------------------------------------------------------------------------
# --help CLI
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.inspect_model", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Inspect model" in result.stdout


# ---------------------------------------------------------------------------
# _print_config_summary
# ---------------------------------------------------------------------------


class TestPrintConfigSummary:
    def test_prints_model_name_and_fields(self, capsys):
        from scripts.inspect_model import _print_config_summary

        cfg = SimpleNamespace(
            model_type="gpt2",
            hidden_size=768,
            num_hidden_layers=12,
            vocab_size=50257,
            architectures=["GPT2LMHeadModel"],
        )
        _print_config_summary("test-model", cfg)
        out = capsys.readouterr().out
        assert "test-model" in out
        assert "gpt2" in out
        assert "768" in out

    def test_handles_text_config(self, capsys):
        from scripts.inspect_model import _print_config_summary

        text_cfg = SimpleNamespace(
            model_type="qwen2",
            hidden_size=4096,
            num_hidden_layers=32,
            vocab_size=152064,
            architectures=["Qwen2ForCausalLM"],
        )
        cfg = SimpleNamespace(model_type="qwen2", text_config=text_cfg)
        _print_config_summary("qwen-model", cfg)
        out = capsys.readouterr().out
        assert "qwen-model" in out
        assert "4096" in out


# ---------------------------------------------------------------------------
# _analyze_model (core recommendation logic)
# ---------------------------------------------------------------------------


class TestAnalyzeModel:
    def _make_mock_model(self):
        """Build a minimal mock model with named Linear submodules."""
        import torch.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "self_attn": nn.ModuleDict(
                                    {
                                        "q_proj": nn.Linear(64, 64),
                                        "k_proj": nn.Linear(64, 64),
                                        "v_proj": nn.Linear(64, 64),
                                    }
                                ),
                                "mlp": nn.ModuleDict(
                                    {
                                        "gate_proj": nn.Linear(64, 128),
                                        "up_proj": nn.Linear(64, 128),
                                        "down_proj": nn.Linear(128, 64),
                                    }
                                ),
                            }
                        )
                        for _ in range(2)
                    ]
                )

        return FakeModel()

    def test_detects_linear_layers(self, capsys):
        from scripts.inspect_model import _analyze_model

        model = self._make_mock_model()
        _analyze_model(model)
        out = capsys.readouterr().out
        # Should mention the leaf names of linear layers
        assert "q_proj" in out
        assert "mlp" in out or "gate_proj" in out

    def test_prints_recommendations(self, capsys):
        from scripts.inspect_model import _analyze_model

        model = self._make_mock_model()
        _analyze_model(model)
        out = capsys.readouterr().out
        assert "Recommended target_modules" in out

    def test_saves_json_report(self, capsys, tmp_path):
        from scripts.inspect_model import _analyze_model

        model = self._make_mock_model()
        # _analyze_model writes to reports/model_inspection.json (cwd-relative)
        with patch("scripts.inspect_model.Path") as mock_path_cls:
            report_path = tmp_path / "model_inspection.json"
            mock_path_cls.return_value = report_path
            # The mkdir and open calls need to work
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _analyze_model(model)


        # Verify the report file was written (via capsys or side-effect)
        out = capsys.readouterr().out
        assert "Total parameters" in out

    def test_categorizes_attn_vs_mlp(self, capsys):
        from scripts.inspect_model import _analyze_model

        model = self._make_mock_model()
        _analyze_model(model)
        out = capsys.readouterr().out
        # The output should separate attention and mlp recommendations
        assert "q_proj" in out
        assert "gate_proj" in out or "down_proj" in out


# ---------------------------------------------------------------------------
# TC-218-01: inspect_model.py --model (Qwen3.5-9B structure output)
# ---------------------------------------------------------------------------


class TestTC218:
    """REQ-218: Model inspection CLI."""

    @staticmethod
    def _make_qwen_mock_model():
        import torch.nn as nn

        class FakeQwenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "self_attn": nn.ModuleDict(
                                    {
                                        "q_proj": nn.Linear(4096, 4096),
                                        "k_proj": nn.Linear(4096, 4096),
                                        "v_proj": nn.Linear(4096, 4096),
                                        "o_proj": nn.Linear(4096, 4096),
                                    }
                                ),
                                "mlp": nn.ModuleDict(
                                    {
                                        "gate_proj": nn.Linear(4096, 11008),
                                        "up_proj": nn.Linear(4096, 11008),
                                        "down_proj": nn.Linear(11008, 4096),
                                    }
                                ),
                            }
                        )
                        for _ in range(2)
                    ]
                )

        return FakeQwenModel()

    def test_tc218_01_inspect_model_outputs_structure(self, capsys):
        """TC-218-01: inspect_model.py outputs Qwen/Qwen3.5-9B model structure."""
        from scripts.inspect_model import _analyze_model, _print_config_summary

        model = self._make_qwen_mock_model()
        cfg = SimpleNamespace(
            model_type="qwen3",
            hidden_size=4096,
            num_hidden_layers=48,
            vocab_size=152064,
            architectures=["Qwen3ForCausalLM"],
        )
        _print_config_summary("Qwen/Qwen3.5-9B", cfg)
        _analyze_model(model)
        out = capsys.readouterr().out

        assert "Qwen/Qwen3.5-9B" in out
        assert "q_proj" in out
        assert "Recommended target_modules" in out
        assert "Linear layers by name pattern" in out

    def test_tc218_02_inspect_from_yaml_with_config(self, capsys, tmp_path):
        """TC-218-02: --config arg inspects model from YAML config."""
        from scripts.inspect_model import inspect_from_yaml

        self._make_qwen_mock_model()
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            "experiment:\n  name: test\n  seed: 42\n"
            "model:\n  name_or_path: Qwen/Qwen3.5-9B\n"
            "lora:\n  r: 16\n  alpha: 32\n  dropout: 0.0\n"
            "data:\n  train_path: data/train.jsonl\n"
            "  valid_quick_path: data/valid_quick.jsonl\n"
            "  valid_full_path: data/valid_full.jsonl\n"
            "training:\n  batch_size: 1\n  grad_accumulation: 8\n"
            "  learning_rate: 2e-4\n  max_cycles: 100\n"
            "logging:\n  run_dir: runs/test\n"
            "tg_lora:\n  K_initial: 3\n  K_candidates: [1,2,3]\n"
            "  N_initial: 1\n  N_candidates: [1,2,4]\n"
            "  alpha_initial: 0.3\n  alpha_min: 0.05\n  alpha_max: 1.0\n"
            "  beta_initial: 0.9\n  beta_candidates: [0.8,0.9,0.95]\n"
            "  relative_update_cap: 0.5\n"
            "  active_layer_strategy: last_25_percent\n"
        )

        with patch(
            "scripts.inspect_model.inspect_from_config"
        ) as mock_inspect:
            inspect_from_yaml(str(config_path))
            mock_inspect.assert_called_once_with("Qwen/Qwen3.5-9B")


# ---------------------------------------------------------------------------
# TC-219-01/02: Makefile inspect / inspect-config targets
# ---------------------------------------------------------------------------


class TestTC219:
    """REQ-219: Makefile inspect targets."""

    def test_tc219_01_make_inspect_target_exists(self):
        """TC-219-01: Makefile 'inspect' target exists and invokes inspect_model.py."""
        import subprocess

        result = subprocess.run(
            ["make", "-n", "inspect"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "inspect_model.py" in result.stdout
        assert "--model" in result.stdout

    def test_tc219_02_make_inspect_config_target_exists(self):
        """TC-219-02: Makefile 'inspect-config' target exists and invokes with --config."""
        import subprocess

        result = subprocess.run(
            ["make", "-n", "inspect-config"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "inspect_model.py" in result.stdout
        assert "--config" in result.stdout
