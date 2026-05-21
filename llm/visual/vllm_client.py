import base64
import mimetypes
import os
import re
import subprocess
import sys
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, List, Optional

from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token


def _parse_slurm_gres_idx() -> List[int]:
    """Parse GPU indices from `scontrol show job --details` GRES line.

    Looks for patterns like:  GRES=gpu:h200:1(IDX:3)  or  GRES=gpu:2(IDX:0,1)
    Returns [] if not inside a SLURM job or parsing fails.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        return []
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "job", job_id, "--details"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    import re
    # Match IDX:<digits and commas>  e.g. IDX:3  or  IDX:0,1,2
    m = re.search(r"IDX:([\d,]+)", out)
    if m:
        return [int(x) for x in m.group(1).split(",") if x.isdigit()]
    return []


def detect_gpus() -> List[int]:
    """Return GPU indices available to this process.

    Priority:
    1. CUDA_VISIBLE_DEVICES env var (set by SLURM / user on some clusters)
    2. scontrol show job --details  GRES IDX field (works when CUDA_VISIBLE_DEVICES is not set)
    3. nvidia-smi: pick GPU with most free memory (last resort / local dev)
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd not in ("NoDevFiles", "-1", ""):
        indices = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        if indices:
            return indices

    slurm_indices = _parse_slurm_gres_idx()
    if slurm_indices:
        return slurm_indices

    # Fall back to nvidia-smi: pick GPU with most free memory
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception:
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            gpus.append((int(parts[0]), int(parts[1])))

    if not gpus:
        return []

    gpus.sort(key=lambda x: x[1], reverse=True)
    return [gpus[0][0]]


def print_gpu_report(indices: List[int]) -> None:
    print("[vLLM] GPU assignment report:")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4 and int(parts[0]) in indices:
                idx, name, free, total = parts
                print(f"  GPU {idx}: {name}  {int(free):,} / {int(total):,} MiB free")
    except Exception as e:
        print(f"  (could not query nvidia-smi: {e})")
    print(f"  CUDA_VISIBLE_DEVICES will be set to: {','.join(map(str, indices))}")


class VLLMClient(LLMClientBase):
    """vLLM client that auto-starts the server if it isn't already running."""

    START_TIMEOUT = int(os.getenv("VLLM_START_TIMEOUT", "300"))
    POLL_INTERVAL = int(os.getenv("VLLM_POLL_INTERVAL", "5"))
    MIN_RESPONSE_TOKENS = 256

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, model_name: str = "vllm",
                 max_model_len: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 enable_thinking: Optional[bool] = None,
                 thinking_budget: Optional[int] = None):
        self._model = model or "meta-llama/Llama-3.1-8B-Instruct"
        self._base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self._max_model_len = max_model_len
        self._context_limit = max_model_len or 8192
        self._gpu_memory_utilization = gpu_memory_utilization
        self._enable_thinking = enable_thinking
        self._thinking_budget = thinking_budget  # cap thinking tokens independently of max_tokens
        self._vllm_process = None
        self._vllm_log_file = None
        self._started_by_me = False
        self._consecutive_timeouts = 0
        self._warned_unbounded_thinking = False
        self._warned_clamped_thinking = False

        super().__init__(model_name=model_name, model=self._model)
        self.temperature = 0.0
        self.max_tokens = 2000
        self.top_p = 1.0

        self._start_server_if_needed()

    # ------------------------------------------------------------------ #
    # Server lifecycle
    # ------------------------------------------------------------------ #

    def _health_url(self) -> str:
        parsed = urlparse(self._base_url)
        return f"{parsed.scheme}://{parsed.netloc}/health"

    def _is_server_running(self) -> bool:
        try:
            r = requests.get(self._health_url(), timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _is_port_in_use(self) -> bool:
        """Check if anything is already bound to our port (even if not yet healthy)."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", self._port())) == 0

    def _port(self) -> int:
        return urlparse(self._base_url).port or 8000

    def _lock_path(self) -> Path:
        return Path(f"/tmp/vllm_starting_{self._port()}.lock")

    def _start_server_if_needed(self):
        if self._is_server_running():
            print(f"[vLLM] Server already running at {self._base_url}")
            return

        # Port is bound but not yet healthy — another process is still loading the model.
        # Wait for it rather than launching a competing server.
        if self._is_port_in_use():
            print(f"[vLLM] Port {self._port()} is in use but server not healthy yet, waiting...")
            self._wait_for_server()
            return

        lock_path = self._lock_path()
        # Server is not running and port is free — any existing lock is stale.
        lock_path.unlink(missing_ok=True)
        self._clear_stale_lock(lock_path)
        try:
            # Exclusive create — only one worker wins this race
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            # Another worker is already starting the server — wait, but detect stale locks
            print(f"[vLLM] Another worker is starting the server, waiting...")
            self._wait_for_server(lock_path=lock_path)
            return

        # We won the race — start the server; always release the lock on exit
        try:
            self._launch_server()
        except Exception:
            lock_path.unlink(missing_ok=True)
            raise
        else:
            lock_path.unlink(missing_ok=True)

    def _clear_stale_lock(self, lock_path: Path) -> None:
        """Remove the lock file if the PID recorded inside is no longer alive."""
        if not lock_path.exists():
            return
        try:
            pid = int(lock_path.read_text().strip())
            os.kill(pid, 0)  # signal 0: check existence only, no actual signal
        except (ValueError, OSError):
            # PID is gone or unreadable — lock is stale
            print(f"[vLLM] Removing stale lock file (dead PID) at {lock_path}")
            lock_path.unlink(missing_ok=True)

    def _kill_stale_vllm_on_port(self) -> None:
        """Kill any lingering vLLM processes that held our port but are now gone."""
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{self._port()}"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            for pid_str in out.splitlines():
                pid = int(pid_str)
                print(f"[vLLM] Killing stale process on port {self._port()}: PID {pid}")
                subprocess.run(["kill", "-9", str(pid)], check=False)
        except Exception:
            pass  # lsof not available or no process found — safe to continue

    def _suggest_gpu_mem_util_from_error(self, err_text: str) -> Optional[float]:
        """Parse vLLM startup error and suggest a lower --gpu-memory-utilization.

        Example error snippet:
          Free memory on device cuda:0 (25.08/79.25 GiB) on startup is less than
          desired GPU memory utilization (0.9, 71.33 GiB).
        """
        m = re.search(r"\((\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+GiB\).*GPU memory utilization\s*\((\d+(?:\.\d+)?)", err_text)
        if not m:
            return None

        free_gib = float(m.group(1))
        total_gib = float(m.group(2))
        current_util = float(m.group(3))
        if total_gib <= 0:
            return None

        # Keep a tiny headroom margin below currently free ratio.
        safe_util = max(0.10, min(current_util - 0.05, (free_gib / total_gib) - 0.01))
        return round(safe_util, 3) if safe_util < current_util else None

    def _launch_server(self):
        self._kill_stale_vllm_on_port()

        gpu_indices = detect_gpus()
        if not gpu_indices:
            raise RuntimeError("[vLLM] No GPUs detected. Cannot start vLLM server.")

        print_gpu_report(gpu_indices)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Prepend conda env lib so subprocess gets a new enough libstdc++
        conda_lib = str(Path(sys.executable).parent.parent / "lib")
        env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

        tp = len(gpu_indices)
        log_path = Path(f"/tmp/vllm_server_{self._port()}.log")
        print(f"[vLLM] Starting server: model='{self._model}'  port={self._port()}  tensor-parallel={tp}")
        print(f"[vLLM] Server log → {log_path}")

        # Allow one adaptive retry when vLLM fails due startup free-memory/utilization mismatch.
        mem_util = self._gpu_memory_utilization
        for attempt in range(2):
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self._model,
                "--port", str(self._port()),
                "--tensor-parallel-size", str(tp),
                "--trust-remote-code",
                "--served-model-name", self._model,
                "--enable-prefix-caching",
            ]
            if self._max_model_len is not None:
                cmd += ["--max-model-len", str(self._max_model_len)]
            if mem_util is not None:
                cmd += ["--gpu-memory-utilization", str(mem_util)]

            log_file = open(log_path, "a")
            self._vllm_log_file = log_file
            self._vllm_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=env,
            )
            self._started_by_me = True

            try:
                self._wait_for_server()
                return
            except RuntimeError as e:
                out = str(e)
                if attempt == 0:
                    suggested = self._suggest_gpu_mem_util_from_error(out)
                    if suggested is not None:
                        print(
                            f"\n[vLLM] Startup failed due free-memory/utilization mismatch; "
                            f"retrying with --gpu-memory-utilization={suggested}"
                        )
                        mem_util = suggested
                        continue
                raise

    def _wait_for_server(self, lock_path: Optional[Path] = None):
        deadline = time.time() + self.START_TIMEOUT
        elapsed = 0
        print(
            f"[vLLM] Waiting up to {self.START_TIMEOUT}s for server readiness "
            f"(poll={self.POLL_INTERVAL}s)",
            flush=True,
        )
        try:
            while time.time() < deadline:
                if self._vllm_process and self._vllm_process.poll() is not None:
                    out = self._vllm_process.stdout.read() if self._vllm_process.stdout else ""
                    raise RuntimeError(f"[vLLM] Server process exited unexpectedly.\n{out}")
                # Waiting workers: lock may disappear slightly before /health is ready.
                # If port is already bound, keep waiting instead of failing fast.
                if lock_path is not None and not lock_path.exists() and not self._is_server_running():
                    if self._is_port_in_use():
                        time.sleep(self.POLL_INTERVAL)
                        continue
                    raise RuntimeError(
                        f"[vLLM] Server startup failed (lock released without server coming up). "
                        f"Check GPU availability and model path '{self._model}'."
                    )
                if self._is_server_running():
                    print(f"\n[vLLM] Server ready at {self._base_url}")
                    return
                elapsed += self.POLL_INTERVAL
                print(f"\r[vLLM] Waiting for server... {elapsed}s", end="", flush=True)
                time.sleep(self.POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n[vLLM] Interrupted — shutting down server process...")
            self.shutdown()
            raise

        raise RuntimeError(
            f"[vLLM] Server did not become ready within {self.START_TIMEOUT}s. "
            f"Check GPU availability and model path '{self._model}'."
        )

    def shutdown(self):
        if self._vllm_process and self._started_by_me:
            print("[vLLM] Shutting down server (started by this client)...")
            self._vllm_process.terminate()
            self._vllm_process.wait()
            self._vllm_process = None
            self._started_by_me = False
            self._lock_path().unlink(missing_ok=True)
        if self._vllm_log_file:
            try:
                self._vllm_log_file.close()
            except Exception:
                pass
            self._vllm_log_file = None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _effective_thinking_budget(self) -> Optional[int]:
        """Return the effective thinking budget to send to vLLM.

        Qwen's thinking budget is separate from ``max_tokens``, but the final
        answer still has to fit inside the same output cap. To avoid the model
        spending the whole generation budget inside `<think>...</think>` and
        never reaching the actual action, reserve a minimum response budget.
        """
        if not self._enable_thinking:
            return None

        requested_output = int(self.max_tokens or 1024)
        max_reasoning_budget = max(0, requested_output - self.MIN_RESPONSE_TOKENS)

        if self._thinking_budget is None:
            if not self._warned_unbounded_thinking:
                print(
                    "[vLLM] enable_thinking=True but no thinking_budget configured; "
                    "reasoning may consume the full max_tokens budget."
                )
                self._warned_unbounded_thinking = True
            return None

        effective_budget = min(int(self._thinking_budget), max_reasoning_budget)
        if effective_budget < int(self._thinking_budget) and not self._warned_clamped_thinking:
            print(
                f"[vLLM] Clamping thinking_budget from {self._thinking_budget} to {effective_budget} "
                f"to reserve at least {self.MIN_RESPONSE_TOKENS} tokens for the final answer "
                f"(max_tokens={requested_output})."
            )
            self._warned_clamped_thinking = True
        return effective_budget

    def _build_payload(self) -> dict:
        # Keep request within model context window so larger thinking budgets don't trigger 400.
        requested_output = int(self.max_tokens or 1024)
        input_budget = max(512, int(self._context_limit) - requested_output - 256)
        messages = truncate_messages_by_token(self.messages, input_budget, self._model)

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self._enable_thinking is not None:
            kwargs: dict = {"enable_thinking": self._enable_thinking}
            effective_budget = self._effective_thinking_budget()
            if effective_budget is not None:
                kwargs["thinking_budget"] = effective_budget
            payload["chat_template_kwargs"] = kwargs
        return payload

    def _encode_image_data_url(self, image_path: str) -> str:
        """Encode a local image as a data URL for OpenAI-compatible multimodal payloads."""
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _build_multimodal_messages(self, input_budget: int, image_path: str) -> list[dict[str, Any]]:
        """Convert chat history into an OpenAI-compatible multimodal message list.

        vLLM serves an OpenAI-style API, so image inputs must be embedded in the
        final user message content rather than passed as a separate argument.
        """
        messages = truncate_messages_by_token(self.messages, input_budget, self._model)
        if not messages:
            return [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": self._encode_image_data_url(image_path)}}],
            }]

        multimodal_messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            msg_copy: dict[str, Any] = dict(msg)
            if idx == len(messages) - 1 and msg_copy.get("role") == "user":
                text = msg_copy.get("content", "")
                msg_copy["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": self._encode_image_data_url(image_path)}},
                ]
            multimodal_messages.append(msg_copy)
        return multimodal_messages

    def _build_payload_for_query(self, image_path: Optional[str] = None) -> dict:
        """Build the request payload, optionally attaching an image to the latest user turn."""
        requested_output = int(self.max_tokens or 1024)
        input_budget = max(512, int(self._context_limit) - requested_output - 256)
        messages = (
            self._build_multimodal_messages(input_budget, image_path)
            if image_path else
            truncate_messages_by_token(self.messages, input_budget, self._model)
        )

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self._enable_thinking is not None:
            kwargs: dict = {"enable_thinking": self._enable_thinking}
            effective_budget = self._effective_thinking_budget()
            if effective_budget is not None:
                kwargs["thinking_budget"] = effective_budget
            payload["chat_template_kwargs"] = kwargs
        return payload

    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        self.add_message("user", prompt)
        url = self._base_url + "/v1/chat/completions"

        attempt = 0
        def _pop_last_user_message():
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()

        while True:
            try:
                payload = self._build_payload_for_query(image_path=image_path)
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": False}
                resp = requests.post(url, headers=self._headers(),
                                     json=payload, timeout=(10, 1200), stream=True)
                resp.raise_for_status()
                self._consecutive_timeouts = 0

                # Accumulate streaming SSE chunks; closing the response on timeout
                # causes vLLM to cancel the server-side generation and free KV cache.
                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                import json as _json
                import re as _re
                try:
                    for line in resp.iter_lines(chunk_size=None):
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        chunk = _json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        reasoning_parts.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                        content_parts.append(delta.get("content") or "")
                except requests.exceptions.Timeout:
                    resp.close()
                    raise
                finally:
                    resp.close()

                reasoning = "".join(reasoning_parts).strip()
                content = "".join(content_parts).strip()
                reply = f"<think>{reasoning}</think>{content}" if reasoning else content
                reply_for_history = _re.sub(r"<think>.*?</think>", "", reply, flags=_re.DOTALL).strip()
                self.add_message("assistant", reply_for_history)
                return reply
            except requests.HTTPError:
                if resp.status_code in {502, 503, 504}:
                    attempt += 1
                    time.sleep(2 ** min(attempt, 5))
                    continue
                body_preview = ""
                try:
                    body_preview = (resp.text or "")[:800]
                except Exception:
                    pass
                _pop_last_user_message()
                raise RuntimeError(
                    f"vLLM HTTP {resp.status_code} for /v1/chat/completions. "
                    f"Response body: {body_preview}"
                )
            except requests.exceptions.Timeout:
                print(f"[vLLM] Request timeout, skipping step.")
                _pop_last_user_message()
                self._consecutive_timeouts += 1
                if self._consecutive_timeouts >= 3:
                    print(f"[vLLM] {self._consecutive_timeouts} consecutive timeouts — force restarting server...")
                    self._consecutive_timeouts = 0
                    self.shutdown()
                    self._kill_stale_vllm_on_port()
                    try:
                        self._launch_server()
                    except Exception as restart_err:
                        print(f"[vLLM] Server restart failed: {restart_err}")
                return ""
            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                attempt += 1
                print(f"[vLLM] Connection error (attempt {attempt}): {e}")
                if not self._is_server_running():
                    print("[vLLM] Server appears to be down, force restarting...")
                    self.shutdown()
                    self._kill_stale_vllm_on_port()
                    try:
                        self._launch_server()
                    except Exception as restart_err:
                        print(f"[vLLM] Server restart failed: {restart_err}")
                        return ""
                if attempt < 5:
                    time.sleep(2 ** min(attempt, 5))
                    continue
                return ""
            except Exception as e:
                attempt += 1
                print(f"[vLLM] Request error (attempt {attempt}): {e}")
                if attempt < 3:
                    time.sleep(2 ** min(attempt, 5))
                    continue
                return ""
