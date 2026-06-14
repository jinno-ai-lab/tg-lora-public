"""MLX backend for lm-evaluation-harness.

Usage:
    python -m lm_eval --model local-completions \
        --model_args "model=.cache/mlx_models/Qwen--Qwen3.5-9B,adapter_path=runs/mlx_qlora_baseline_500/adapters.safetensors" \
        --tasks arc_easy,hellaswag ...

Or programmatically:
    from src.eval.mlx_lm_backend import MLXLMEval
    model = MLXLMEval(model=".cache/mlx_models/Qwen--Qwen3.5-9B")
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import mlx.core as mx

# Ensure project root is on sys.path for MLX resource patches
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "4")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "32")
os.environ.setdefault("MLX_BFS_MAX_WIDTH", "4")
os.environ.setdefault("MLX_GATED_DELTA_CHUNK", "512")

from lm_eval.api.model import LM  # noqa: E402


class MLXLMEval(LM):
    """lm-eval backend backed by MLX + mlx_lm."""

    def __init__(
        self,
        model: str,
        adapter_path: str | None = None,
        batch_size: int = 1,
        max_seq_length: int = 2048,
        trust_remote_code: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        mx.random.seed(seed)

        from mlx_lm.utils import load as mlx_load
        from mlx_lm.tuner.utils import load_adapters

        print(f"[MLX] Loading model from {model}")
        self._model, self._tokenizer = mlx_load(
            model, tokenizer_config={"trust_remote_code": trust_remote_code}
        )
        self._model.freeze()

        if adapter_path:
            print(f"[MLX] Loading adapter from {adapter_path}")
            load_adapters(self._model, adapter_path)
        self._model.eval()

        self._eos_token_id = None
        if hasattr(self._tokenizer, "eos_token_id"):
            self._eos_token_id = self._tokenizer.eos_token_id
        if hasattr(self._tokenizer, "encode"):
            # Detect EOS from tokenizer
            eos = self._tokenizer.eos_token
            if eos:
                self._eos_token_id = self._tokenizer.encode(
                    eos, add_special_tokens=False
                )
                if isinstance(self._eos_token_id, list):
                    self._eos_token_id = (
                        self._eos_token_id[0] if self._eos_token_id else None
                    )

        print(f"[MLX] Model loaded. EOS token id: {self._eos_token_id}")

    # ── lm-eval required interface ──────────────────────────────

    @classmethod
    def create_from_arg_string(
        cls, arg_string: str, additional_config=None
    ) -> MLXLMEval:
        import lm_eval.utils as utils

        args = utils.simple_parse_args_string(arg_string)
        if additional_config:
            args.update({k: v for k, v in additional_config.items() if v is not None})
        return cls(**args)

    def loglikelihood(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            context, continuation = req.args
            logprob, is_greedy = self._compute_loglikelihood(context, continuation)
            results.append((logprob, is_greedy))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            (text,) = req.args
            logprob, _ = self._compute_loglikelihood("", text)
            results.append(logprob)
        return results

    def generate_until(self, requests, disable_tqdm=False):

        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", ["</s>"])
            if isinstance(until, str):
                until = [until]
            max_tokens = gen_kwargs.get("max_gen_toks", 256)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = gen_kwargs.get("do_sample", False)

            prompt_tokens = self._tokenizer.encode(context)
            if not isinstance(prompt_tokens, list):
                prompt_tokens = prompt_tokens.tolist()

            generated = self._generate(
                prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else 0.0,
                stop_strings=until,
            )
            results.append(generated)
        return results

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        tokens = self._tokenizer.encode(string)
        return tokens if isinstance(tokens, list) else tokens.tolist()

    def tok_decode(self, tokens, skip_special_tokens=True) -> list[str]:
        tokens = list(tokens)
        if not tokens or isinstance(tokens[0], (list, bytes)):
            return [
                self._tokenizer.decode(t, skip_special_tokens=skip_special_tokens)
                for t in tokens
            ]
        return [self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)]

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def eot_token_id(self):
        return self._eos_token_id

    @property
    def prefix_token_id(self):
        return self._eos_token_id

    def device(self):
        return "mlx"

    # ── Internal helpers ────────────────────────────────────────

    def _compute_loglikelihood(
        self, context: str, continuation: str
    ) -> tuple[float, bool]:
        full_text = context + continuation
        full_tokens = self._tokenizer.encode(full_text)
        if not isinstance(full_tokens, list):
            full_tokens = full_tokens.tolist()

        context_tokens = self._tokenizer.encode(context)
        if not isinstance(context_tokens, list):
            context_tokens = context_tokens.tolist()

        cont_start = len(context_tokens)
        if cont_start >= len(full_tokens):
            return 0.0, True

        # All computation stays in MLX — no numpy fallback.
        input_ids = mx.array([full_tokens])
        logits = self._model(input_ids)[0]  # (seq_len, vocab)
        mx.eval(logits)

        # Log-softmax in MLX (stable)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(log_probs)

        # Extract continuation logprobs — only scalars leave MLX
        total_logprob = 0.0
        is_greedy = True
        for i in range(cont_start, len(full_tokens)):
            token_id = full_tokens[i]
            if i > 0:
                total_logprob += float(log_probs[i - 1, token_id])
                greedy_id = int(mx.argmax(log_probs[i - 1]))
                mx.eval(greedy_id)
                if greedy_id != token_id:
                    is_greedy = False

        del logits, log_probs, input_ids
        mx.synchronize()
        gc.collect()
        if mx.get_cache_memory() > 2_000_000_000:
            mx.clear_cache()

        return total_logprob, is_greedy

    def _generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_strings: list[str] | None = None,
    ) -> str:
        tokens = mx.array(prompt_tokens)
        generated = list(prompt_tokens)

        for _ in range(max_new_tokens):
            logits = self._model(tokens[None, :])
            mx.eval(logits)

            if temperature > 0:
                mx.softmax(logits[0, -1] / temperature)
                next_token = mx.random.categorical(logits[0, -1] / temperature, axis=-1)
            else:
                next_token = mx.argmax(logits[0, -1], axis=-1)

            mx.eval(next_token)
            next_id = int(next_token)
            generated.append(next_id)

            if next_id == self._eos_token_id:
                break

            tokens = mx.array(generated)
            del logits, next_token
            mx.synchronize()
            gc.collect()

            if stop_strings:
                decoded_so_far = self._tokenizer.decode(generated[len(prompt_tokens) :])
                if any(s in decoded_so_far for s in stop_strings):
                    break

        decoded = self._tokenizer.decode(generated[len(prompt_tokens) :])
        if stop_strings:
            for s in stop_strings:
                decoded = decoded.split(s)[0]
        return decoded
