#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import traceback
from typing import Dict, Any

import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.policies import PreTrainedConfig
# Ensure PI05 custom processor steps are registered in ProcessorStepRegistry.
from lerobot.policies.pi05 import processor_pi05 as _pi05_processor  # noqa: F401
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from act_async_infer_distributed_demo.scripts.network_utils_act import (
    NetworkServer,
    log_info,
    log_error,
)


class RemoteSelectActionServerNewPI05:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.host = args.host
        self.port = args.port
        self.device = torch.device(
            args.device
            if (args.device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )

        self.shutdown_event = threading.Event()
        self.client_connected = False
        self.client_timeout = float(args.client_timeout)
        self.last_client_activity = time.time()

        self.model_lock = threading.Lock()
        self.policy = self._load_policy(args.policy_path)
        self.preprocessor, self.postprocessor = self._load_processors(args.policy_path)
        self.default_task = str(args.task)
        self.default_robot_type = str(args.robot_type)
        self._tokenizer = None
        self._tokenizer_max_length = 200
        self._tokenizer_padding = "max_length"
        self._tokenizer_truncation = True
        self._manual_tokenizer_warned = False
        self._dummy_token_warned = False
        self._language_dtype_warned = False
        self._language_shape_warned = False
        self._load_tokenizer_fallback(args.policy_path)
        self._batch_summary_logged = False
        self._image_autoscale_warned_keys: set[str] = set()

        self.network_server = NetworkServer(self.host, self.port)
        self.network_server.set_disconnect_callback(self._on_client_disconnected)
        self.network_server.register_handler("reset_policy", self._handle_reset_policy)
        self.network_server.register_handler("ping", self._handle_ping)
        self.network_server.register_handler("select_action_chunk", self._handle_select_action_chunk)

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def _load_policy(self, policy_path: str) -> PI05Policy:
        log_info(f"Loading lerobot PI05Policy from {policy_path} on {self.device} ...")
        cfg = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
        if not isinstance(cfg, PI05Config):
            raise TypeError(f"Expected PI05Config, got {type(cfg)} from {policy_path}")
        # Compatibility workaround: newer transformers PaliGemma uses a different vision attribute path.
        # For inference this flag is irrelevant, so we disable it to avoid init-time AttributeError.
        if getattr(cfg, "freeze_vision_encoder", False):
            log_info("Override config.freeze_vision_encoder=True -> False for inference compatibility.")
            cfg.freeze_vision_encoder = False
        # The exported safetensors uses metadata alias for tied embedding weights:
        # embed_tokens <- lm_head. PI05 custom loader does not resolve this alias.
        # Use strict=False to avoid aborting load, then manually sync tied weights.
        policy = PI05Policy.from_pretrained(
            policy_path,
            config=cfg,
            local_files_only=True,
            strict=False,
        )
        policy.to(self.device).eval()
        if hasattr(policy, "config") and hasattr(policy.config, "device"):
            policy.config.device = self.device.type
        self._sync_paligemma_embed_tokens_from_lm_head(policy)
        policy.reset()
        cfg = policy.config
        log_info(
            "PI05 inference config: "
            f"image_resolution={getattr(cfg, 'image_resolution', None)}, "
            f"chunk_size={getattr(cfg, 'chunk_size', None)}, "
            f"n_action_steps={getattr(cfg, 'n_action_steps', None)}"
        )
        expected_norm_map = {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}
        if hasattr(cfg, "normalization_mapping"):
            actual_norm_map = dict(getattr(cfg, "normalization_mapping"))
            log_info(f"PI05 normalization_mapping={actual_norm_map}")
            if actual_norm_map != expected_norm_map:
                log_error(
                    "Normalization mapping mismatch. "
                    f"expected={expected_norm_map}, actual={actual_norm_map}"
                )
            else:
                log_info("Normalization mapping matches training config.")
        log_info("PI05Policy loaded and reset.")
        return policy

    def _sync_paligemma_embed_tokens_from_lm_head(self, policy: PI05Policy) -> None:
        try:
            paligemma = policy.model.paligemma_with_expert.paligemma
            embed = paligemma.model.language_model.embed_tokens
            lm_head = paligemma.lm_head
            if (
                hasattr(embed, "weight")
                and hasattr(lm_head, "weight")
                and embed.weight.shape == lm_head.weight.shape
            ):
                embed.weight.data.copy_(lm_head.weight.data)
                log_info("Synced paligemma.language_model.embed_tokens.weight from paligemma.lm_head.weight.")
            else:
                log_error(
                    "Could not sync embed_tokens from lm_head due to missing attribute or shape mismatch."
                )
        except Exception as exc:
            log_error(f"Failed to sync paligemma embed_tokens from lm_head: {exc}")

    def _load_processors(
        self, policy_path: str
    ) -> tuple[PolicyProcessorPipeline | None, PolicyProcessorPipeline | None]:
        try:
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                policy_path,
                config_filename="policy_preprocessor.json",
                local_files_only=True,
            )
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                policy_path,
                config_filename="policy_postprocessor.json",
                local_files_only=True,
            )
        except Exception as exc:
            log_error(f"Failed to load PI05 preprocess/postprocess pipelines: {exc}")
            return None, None

        for step in getattr(preprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = self.device.type
            elif step.__class__.__name__ == "NormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device=self.device.type)
                expected_norm_map = {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}
                actual_norm_map = dict(getattr(step, "norm_map", {}))
                if actual_norm_map != expected_norm_map:
                    log_error(
                        "Preprocessor normalizer norm_map mismatch. "
                        f"expected={expected_norm_map}, actual={actual_norm_map}"
                    )
                else:
                    log_info("Preprocessor normalizer norm_map matches training config.")

        for step in getattr(postprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = "cpu"
            elif step.__class__.__name__ == "UnnormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device="cpu")

        log_info("Loaded PI05 policy_preprocessor.json / policy_postprocessor.json on server.")
        return preprocessor, postprocessor

    def _load_tokenizer_fallback(self, policy_path: str) -> None:
        tokenizer_name = "google/paligemma-3b-pt-224"
        cfg_path = f"{policy_path.rstrip('/')}/policy_preprocessor.json"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for step in cfg.get("steps", []):
                if step.get("registry_name") != "tokenizer_processor":
                    continue
                step_cfg = step.get("config", {})
                tokenizer_name = str(step_cfg.get("tokenizer_name", tokenizer_name))
                self._tokenizer_max_length = int(step_cfg.get("max_length", 200))
                self._tokenizer_padding = str(step_cfg.get("padding", "max_length"))
                self._tokenizer_truncation = bool(step_cfg.get("truncation", True))
                break
        except Exception as exc:
            log_error(f"Could not parse policy_preprocessor.json for tokenizer config: {exc}")

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                local_files_only=True,
            )
            log_info(
                "Loaded tokenizer fallback on server: "
                f"{tokenizer_name} (max_length={self._tokenizer_max_length})"
            )
        except Exception as exc:
            self._tokenizer = None
            log_error(f"Could not load tokenizer fallback '{tokenizer_name}': {exc}")

    def _to_torch_batch(self, batch_np: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        batch_torch: Dict[str, torch.Tensor] = {}
        for key, value in batch_np.items():
            if value is None or key == "action":
                continue
            arr = np.asarray(value)
            is_numeric_or_bool = np.issubdtype(arr.dtype, np.number) or np.issubdtype(
                arr.dtype, np.bool_
            )
            if not is_numeric_or_bool:
                log_error(f"[BAD DTYPE] skip key={key} dtype={arr.dtype}")
                continue
            is_image = "image" in key
            if is_image:
                if np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.float32) / 255.0
                    if key not in self._image_autoscale_warned_keys:
                        log_info(
                            f"Auto-scaled image key={key} from integer dtype to float32 [0,1]."
                        )
                        self._image_autoscale_warned_keys.add(key)
                elif arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                elif np.issubdtype(arr.dtype, np.floating):
                    if arr.size > 0:
                        max_val = float(np.nanmax(arr))
                        min_val = float(np.nanmin(arr))
                        if max_val > 1.5 and min_val >= 0.0:
                            arr = arr.astype(np.float32) / 255.0
                            if key not in self._image_autoscale_warned_keys:
                                log_info(
                                    f"Auto-scaled image key={key} from [0,255]-like float range "
                                    "to [0,1]."
                                )
                                self._image_autoscale_warned_keys.add(key)
                        else:
                            arr = arr.astype(np.float32, copy=False)
                else:
                    arr = arr.astype(np.float32)
            elif arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            batch_torch[key] = torch.from_numpy(arr).to(self.device)
        if len(batch_torch) == 0:
            raise ValueError("No valid numeric tensor in batch")
        return batch_torch

    def _build_observation_np_from_request(self, batch_raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
        obs_np: Dict[str, np.ndarray] = {}
        for key, value in batch_raw.items():
            if value is None or not key.startswith("observation."):
                continue
            arr = np.asarray(value)
            if arr.dtype == np.object_:
                continue
            if "image" in key:
                # Keep uint8 raw images as-is; also accept floating images.
                if np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.uint8, copy=False)
                elif np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32, copy=False)
            elif arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            obs_np[key] = arr
        return obs_np

    def _preprocess_raw_batch_on_server(
        self, batch_raw: Dict[str, Any], task: str, robot_type: str
    ) -> Dict[str, torch.Tensor]:
        obs_np = self._build_observation_np_from_request(batch_raw)
        if len(obs_np) == 0:
            raise ValueError("No observation.* fields found in request batch")

        obs_t = prepare_observation_for_inference(
            obs_np,
            self.device,
            task=task,
            robot_type=robot_type,
        )
        batch_size = 1
        if "observation.state" in obs_t and isinstance(obs_t["observation.state"], torch.Tensor):
            batch_size = int(obs_t["observation.state"].shape[0])
        # PI05 tokenizer-prep step expects iterable tasks aligned with batch size.
        obs_t["task"] = [task for _ in range(batch_size)]
        obs_t["robot_type"] = [robot_type for _ in range(batch_size)]

        if self.preprocessor is not None:
            try:
                batch_t = self.preprocessor(obs_t)
            except Exception as exc:
                log_error(f"Server preprocessor failed; fallback to raw obs tensors: {exc}")
                batch_t = obs_t
        else:
            batch_t = obs_t

        batch_torch: Dict[str, torch.Tensor] = {}
        for key, value in batch_t.items():
            if key == ACTION or value is None:
                continue
            if isinstance(value, torch.Tensor):
                batch_torch[key] = value.to(self.device)
            else:
                arr = np.asarray(value)
                is_numeric_or_bool = np.issubdtype(arr.dtype, np.number) or np.issubdtype(
                    arr.dtype, np.bool_
                )
                if not is_numeric_or_bool:
                    continue
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                batch_torch[key] = torch.from_numpy(arr).to(self.device)
        self._ensure_required_language_inputs(batch_torch, task)
        return batch_torch

    def _state_to_prompt_strings(self, state: torch.Tensor | None, task: str, batch_size: int) -> list[str]:
        cleaned_task = str(task).strip().replace("_", " ").replace("\n", " ")
        if cleaned_task == "":
            cleaned_task = "purple candy"
        if state is None:
            return [f"Task: {cleaned_task};\nAction: " for _ in range(batch_size)]

        state_np = state.detach().to(torch.float32).cpu().numpy()
        if state_np.ndim == 1:
            state_np = state_np[None, :]
        if state_np.shape[0] != batch_size:
            if state_np.shape[0] == 1:
                state_np = np.repeat(state_np, batch_size, axis=0)
            else:
                state_np = state_np[:batch_size]

        max_state_dim = 32
        if state_np.shape[1] < max_state_dim:
            pad_w = max_state_dim - state_np.shape[1]
            state_np = np.pad(state_np, ((0, 0), (0, pad_w)), mode="constant")
        elif state_np.shape[1] > max_state_dim:
            state_np = state_np[:, :max_state_dim]

        state_np = np.clip(state_np, -1.0, 1.0)
        bins = np.linspace(-1.0, 1.0, 256 + 1)[:-1]
        discretized = np.digitize(state_np, bins=bins) - 1

        prompts: list[str] = []
        for i in range(batch_size):
            state_str = " ".join(map(str, discretized[i].tolist()))
            prompts.append(f"Task: {cleaned_task}, State: {state_str};\nAction: ")
        return prompts

    def _coerce_language_inputs(self, batch_torch: Dict[str, torch.Tensor]) -> None:
        token_key = "observation.language.tokens"
        mask_key = "observation.language.attention_mask"
        if token_key not in batch_torch or mask_key not in batch_torch:
            return

        tokens = batch_torch[token_key]
        masks = batch_torch[mask_key]
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens, device=self.device)
        else:
            tokens = tokens.to(self.device)
        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks, device=self.device)
        else:
            masks = masks.to(self.device)

        changed_dtype = False
        if tokens.dtype != torch.long:
            tokens = tokens.to(dtype=torch.long)
            changed_dtype = True

        if masks.dtype != torch.bool:
            if torch.is_floating_point(masks):
                masks = masks > 0.5
            else:
                masks = masks != 0
            masks = masks.to(dtype=torch.bool)
            changed_dtype = True

        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
            changed_dtype = True
        if masks.ndim == 1:
            masks = masks.unsqueeze(0)
            changed_dtype = True

        if tokens.shape != masks.shape:
            msg = (
                "PI05 language tensor shape mismatch: "
                f"tokens={tuple(tokens.shape)} masks={tuple(masks.shape)}"
            )
            if not self._language_shape_warned:
                self._language_shape_warned = True
                log_error(msg)
            raise ValueError(msg)

        batch_torch[token_key] = tokens
        batch_torch[mask_key] = masks
        if changed_dtype and not self._language_dtype_warned:
            self._language_dtype_warned = True
            log_info(
                "Canonicalized PI05 language inputs on server: "
                "tokens->torch.long, attention_mask->torch.bool."
            )

    def _ensure_required_language_inputs(self, batch_torch: Dict[str, torch.Tensor], task: str) -> None:
        token_key = "observation.language.tokens"
        mask_key = "observation.language.attention_mask"
        if token_key in batch_torch and mask_key in batch_torch:
            self._coerce_language_inputs(batch_torch)
            return

        state = batch_torch.get("observation.state")
        batch_size = 1
        if isinstance(state, torch.Tensor) and state.ndim >= 2:
            batch_size = int(state.shape[0])
        if self._tokenizer is None:
            batch_torch[token_key] = torch.zeros(
                (batch_size, self._tokenizer_max_length), dtype=torch.long, device=self.device
            )
            dummy_mask = torch.zeros(
                (batch_size, self._tokenizer_max_length), dtype=torch.bool, device=self.device
            )
            dummy_mask[:, 0] = True
            batch_torch[mask_key] = dummy_mask
            if not self._dummy_token_warned:
                self._dummy_token_warned = True
                log_error(
                    "Tokenizer fallback unavailable; using dummy PI05 language tokens. "
                    "Install/load tokenizer cache for better behavior."
                )
            return

        prompts = self._state_to_prompt_strings(state, task, batch_size)

        encoded = self._tokenizer(
            prompts,
            max_length=self._tokenizer_max_length,
            padding=self._tokenizer_padding,
            truncation=self._tokenizer_truncation,
            return_tensors="pt",
        )
        batch_torch[token_key] = encoded["input_ids"].to(self.device)
        batch_torch[mask_key] = encoded["attention_mask"].to(self.device, dtype=torch.bool)
        self._coerce_language_inputs(batch_torch)
        if not self._manual_tokenizer_warned:
            self._manual_tokenizer_warned = True
            log_info(
                "Injected PI05 language tokens on server using tokenizer fallback "
                "(preprocessor output did not include language tensors)."
            )

    def _to_numpy_action(self, action: Any) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            arr = action.detach().cpu().numpy()
            return np.asarray(arr)
        return np.asarray(action)

    def _log_batch_summary_once(self, batch_torch: Dict[str, torch.Tensor]) -> None:
        if self._batch_summary_logged:
            return
        self._batch_summary_logged = True

        keys = sorted(batch_torch.keys())
        log_info(f"First inference batch keys ({len(keys)}): {keys}")
        for key in keys:
            value = batch_torch[key]
            if not isinstance(value, torch.Tensor):
                continue
            msg = f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}"
            if value.numel() > 0 and torch.is_floating_point(value):
                vmin = float(value.min().item())
                vmax = float(value.max().item())
                msg += f", range=[{vmin:.4f}, {vmax:.4f}]"
            log_info(msg)

            if "image" in key and value.dtype == torch.uint8:
                log_error(
                    f"{key} is uint8 raw image. Expected preprocessed float tensor from client."
                )

    def _postprocess_action_chunk(self, actions: Any) -> Any:
        if self.postprocessor is None:
            return actions
        try:
            out = self.postprocessor({ACTION: actions})
        except Exception:
            out = self.postprocessor({"action": actions})
        if isinstance(out, dict):
            if ACTION in out:
                return out[ACTION]
            if "action" in out:
                return out["action"]
        return out

    def _predict_action_chunk(
        self, batch_torch: Dict[str, torch.Tensor], n_action_steps: int | None
    ) -> np.ndarray:
        actions = self.policy.predict_action_chunk(batch_torch)
        actions = self._postprocess_action_chunk(actions)
        actions_np = np.asarray(self._to_numpy_action(actions))
        if actions_np.ndim == 3:
            actions_np = actions_np[0]
        if n_action_steps is not None:
            actions_np = actions_np[:n_action_steps]
        return actions_np

    def _validate_required_pi05_inputs(
        self, batch_torch: Dict[str, torch.Tensor], batch_raw: Dict[str, Any]
    ) -> None:
        required = (
            "observation.language.tokens",
            "observation.language.attention_mask",
        )
        missing = [k for k in required if k not in batch_torch]
        if not missing:
            return
        saw_raw_task = "task" in batch_raw
        fallback_status = (
            "server tokenizer fallback unavailable."
            if self._tokenizer is None
            else "server tokenizer fallback was available but tokens are still missing."
        )
        extra_hint = (
            " Raw key `task` was provided, but server-side PI05 preprocessing still did not "
            f"produce language tokens ({fallback_status})"
            if saw_raw_task
            else " Request must include either language tokens or enough raw observation/task fields "
            "for server-side preprocessing."
        )
        raise KeyError(
            f"Missing required PI05 inputs: {missing}. "
            f"Available keys: {sorted(batch_torch.keys())}.{extra_hint}"
        )

    def _touch_activity(self):
        self.last_client_activity = time.time()

    def _handle_ping(self, request: dict) -> dict:
        self._touch_activity()
        return {"status": "ok"}

    def _handle_reset_policy(self, request: dict) -> dict:
        self._touch_activity()
        try:
            with self.model_lock:
                self.policy.reset()
            return {"status": "ok"}
        except Exception as exc:
            log_error(f"reset_policy failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def _handle_select_action_chunk(self, request: dict) -> dict:
        self._touch_activity()
        try:
            batch = request.get("batch")
            if not isinstance(batch, dict) or len(batch) == 0:
                return {"status": "error", "message": "Invalid or empty batch"}

            task = request.get("task", batch.get("task", self.default_task))
            robot_type = request.get("robot_type", batch.get("robot_type", self.default_robot_type))
            task = str(task) if task is not None else self.default_task
            robot_type = str(robot_type) if robot_type is not None else self.default_robot_type

            n_action_steps = request.get("n_action_steps", None)
            if n_action_steps is not None:
                try:
                    n_action_steps = int(n_action_steps)
                    if n_action_steps <= 0:
                        n_action_steps = None
                except Exception:
                    n_action_steps = None

            has_language_tokens = (
                "observation.language.tokens" in batch
                and "observation.language.attention_mask" in batch
            )
            if has_language_tokens:
                batch_torch = self._to_torch_batch(batch)
            else:
                batch_torch = self._preprocess_raw_batch_on_server(batch, task, robot_type)
            self._coerce_language_inputs(batch_torch)
            self._validate_required_pi05_inputs(batch_torch, batch)
            self._log_batch_summary_once(batch_torch)
            with self.model_lock:
                with torch.no_grad():
                    actions_np = self._predict_action_chunk(
                        batch_torch, n_action_steps
                    )
            return {"status": "success", "actions": actions_np}
        except Exception as exc:
            traceback.print_exc()
            log_error(f"select_action_chunk failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def _on_client_disconnected(self):
        self.client_connected = False

    def _client_timeout_monitor(self, stop_event: threading.Event):
        while self.running and not stop_event.is_set():
            if self.client_connected:
                idle = time.time() - self.last_client_activity
                if idle > self.client_timeout:
                    log_info(
                        f"Client timeout after {idle:.1f}s, closing connection and resetting policy."
                    )
                    self.client_connected = False
                    self.network_server.close()
            time.sleep(1.0)

    def start(self):
        stop_event = threading.Event()
        while self.running:
            try:
                log_info(f"Waiting for client on {self.host}:{self.port} ...")
                if not self.network_server.start():
                    time.sleep(1.0)
                    continue

                self.client_connected = True
                self._touch_activity()

                monitor_thread = threading.Thread(
                    target=self._client_timeout_monitor,
                    args=(stop_event,),
                    daemon=True,
                )
                monitor_thread.start()

                self.network_server.handle_requests(
                    lambda: self.running and self.client_connected
                )

                stop_event.set()
                monitor_thread.join(timeout=1.0)
                stop_event.clear()

                self.network_server.close()
                with self.model_lock:
                    self.policy.reset()
                log_info("Client session closed. Policy reset.")

            except Exception as exc:
                traceback.print_exc()
                log_error(f"Server main loop error: {exc}")
                self.network_server.close()
                time.sleep(1.0)

    def stop(self):
        self.shutdown_event.set()
        self.client_connected = False
        self.network_server.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote lerobot PI05 action_chunk server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--client_timeout", type=float, default=30.0)
    parser.add_argument("--task", type=str, default="purple candy")
    parser.add_argument("--robot_type", type=str, default="w1")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    server = RemoteSelectActionServerNewPI05(args)

    def _sig_handler(sig, frame):
        log_info("Received stop signal, shutting down remote action_chunk server...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
