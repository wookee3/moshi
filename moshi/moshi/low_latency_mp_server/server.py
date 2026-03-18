# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import base64
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import tarfile
import time
import typing as tp
import uuid

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from ..client_utils import log
from ..models import loaders, LMGen, LMModel, MimiModel
from ..run_inference import get_condition_tensors


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class WorkerRuntime:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    device: str | torch.device
    frame_size: int

    @classmethod
    def create(
        cls,
        *,
        hf_repo: str,
        tokenizer: str | None,
        moshi_weight: str | None,
        mimi_weight: str | None,
        lora_weight: str | None,
        config_path: str | None,
        cfg_coef: float,
        device: str,
        dtype: torch.dtype,
        fuse_lora: bool,
    ) -> "WorkerRuntime":
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo,
            moshi_weight,
            mimi_weight,
            tokenizer,
            lora_weights=lora_weight,
            config_path=config_path,
        )
        mimi = checkpoint_info.get_mimi(device=device)
        text_tokenizer = checkpoint_info.get_text_tokenizer()
        lm = checkpoint_info.get_moshi(device=device, dtype=dtype, fuse_lora=fuse_lora)
        condition_tensors = get_condition_tensors(
            checkpoint_info.model_type,
            lm,
            batch_size=1,
            cfg_coef=cfg_coef,
        )
        lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **checkpoint_info.lm_gen_config)
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)
        return cls(
            model_type=checkpoint_info.model_type,
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm_gen=lm_gen,
            device=device,
            frame_size=frame_size,
        )

    def warmup(self) -> None:
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class InferenceWorker(mp.Process):
    def __init__(
        self,
        worker_id: int,
        in_q: "mp.Queue[dict[str, tp.Any]]",
        out_q: "mp.Queue[dict[str, tp.Any]]",
        cfg: dict[str, tp.Any],
    ):
        super().__init__(name=f"moshi-worker-{worker_id}")
        self.worker_id = worker_id
        self.in_q = in_q
        self.out_q = out_q
        self.cfg = cfg

    def run(self) -> None:
        seed_all(42_424_242 + self.worker_id)
        runtime: WorkerRuntime | None = None
        all_pcm_data: np.ndarray | None = None
        skip_frames = 1
        opus_reader: sphn.OpusStreamReader | None = None
        opus_writer: sphn.OpusStreamWriter | None = None
        session_open = False

        try:
            runtime = WorkerRuntime.create(**self.cfg)
            runtime.warmup()
            self.out_q.put({"type": "ready"})
        except Exception as exc:
            self.out_q.put({"type": "fatal", "error": f"worker init failed: {exc}"})
            return

        while True:
            cmd = self.in_q.get()
            kind = cmd.get("type")

            if kind == "shutdown":
                self.out_q.put({"type": "shutdown_ack"})
                return

            if kind == "open":
                assert runtime is not None
                runtime.mimi.reset_streaming()
                runtime.lm_gen.reset_streaming()
                all_pcm_data = None
                skip_frames = 1
                opus_reader = sphn.OpusStreamReader(runtime.mimi.sample_rate)
                opus_writer = sphn.OpusStreamWriter(runtime.mimi.sample_rate)
                session_open = True
                self.out_q.put({"type": "opened"})
                continue

            if kind == "close":
                session_open = False
                all_pcm_data = None
                opus_reader = None
                opus_writer = None
                self.out_q.put({"type": "closed"})
                continue

            if kind != "audio":
                self.out_q.put({"type": "error", "error": f"unknown command: {kind}"})
                continue

            if not session_open or runtime is None or opus_reader is None or opus_writer is None:
                self.out_q.put({"type": "error", "error": "audio command without active session"})
                continue

            payload = cmd.get("payload", b"")
            t0 = time.time()
            try:
                out_messages: list[bytes] = []
                pcm = opus_reader.append_bytes(payload)
                if pcm.shape[-1] > 0:
                    all_pcm_data = pcm if all_pcm_data is None else np.concatenate((all_pcm_data, pcm))

                while all_pcm_data is not None and all_pcm_data.shape[-1] >= runtime.frame_size:
                    chunk = all_pcm_data[: runtime.frame_size]
                    all_pcm_data = all_pcm_data[runtime.frame_size:]
                    chunk_t = torch.from_numpy(chunk).to(device=runtime.device)[None, None]
                    codes = runtime.mimi.encode(chunk_t)

                    if skip_frames:
                        runtime.mimi.reset_streaming()
                        skip_frames -= 1

                    for c in range(codes.shape[-1]):
                        tokens = runtime.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        main_pcm = runtime.mimi.decode(tokens[:, 1:]).cpu()
                        opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        if len(opus_bytes) > 0:
                            out_messages.append(b"\x01" + opus_bytes)

                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            text = runtime.text_tokenizer.id_to_piece(text_token)
                            text = text.replace("▁", " ")
                            out_messages.append(b"\x02" + bytes(text, encoding="utf8"))

                self.out_q.put(
                    {
                        "type": "audio_result",
                        "messages": out_messages,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                    }
                )
            except Exception as exc:
                self.out_q.put({"type": "error", "error": f"audio processing failed: {exc}"})


@dataclass
class WorkerHandle:
    index: int
    proc: InferenceWorker
    in_q: "mp.Queue[dict[str, tp.Any]]"
    out_q: "mp.Queue[dict[str, tp.Any]]"
    device: str | None = None
    busy: bool = False
    sessions_assigned: int = 0
    failures: int = 0
    last_elapsed_ms: int | None = None
    ewma_latency_ms: float | None = None


class WorkerPool:
    def __init__(self, handles: list[WorkerHandle], rpc_timeout_s: float, ewma_alpha: float):
        self.handles = handles
        self.rpc_timeout_s = rpc_timeout_s
        self.ewma_alpha = ewma_alpha
        self._lock = asyncio.Lock()

    @staticmethod
    def create(
        worker_count: int,
        worker_cfg: dict[str, tp.Any],
        rpc_timeout_s: float,
        devices: list[str] | None = None,
        ewma_alpha: float = 0.2,
    ) -> "WorkerPool":
        handles: list[WorkerHandle] = []
        for i in range(worker_count):
            this_cfg = dict(worker_cfg)
            if devices is not None:
                this_cfg["device"] = devices[i]
            in_q: mp.Queue[dict[str, tp.Any]] = mp.Queue(maxsize=128)
            out_q: mp.Queue[dict[str, tp.Any]] = mp.Queue(maxsize=128)
            proc = InferenceWorker(i, in_q, out_q, this_cfg)
            proc.start()
            handle = WorkerHandle(
                index=i,
                proc=proc,
                in_q=in_q,
                out_q=out_q,
                device=this_cfg.get("device"),
            )
            handles.append(handle)

        # Wait for worker readiness synchronously during startup.
        for handle in handles:
            msg = handle.out_q.get(timeout=300)
            if msg.get("type") != "ready":
                raise RuntimeError(f"worker {handle.index} failed to start: {msg}")
            log("info", f"worker {handle.index} ready on {handle.device}")
        return WorkerPool(handles, rpc_timeout_s, ewma_alpha)

    async def acquire(self) -> WorkerHandle | None:
        async with self._lock:
            candidates = [h for h in self.handles if not h.busy]
            if not candidates:
                return None
            # Prefer workers with no history first, then lower EWMA latency.
            selected = min(
                candidates,
                key=lambda h: (
                    0 if h.ewma_latency_ms is None else 1,
                    h.ewma_latency_ms if h.ewma_latency_ms is not None else 0.0,
                    h.sessions_assigned,
                    h.index,
                ),
            )
            selected.busy = True
            selected.sessions_assigned += 1
            return selected
        return None

    async def release(self, handle: WorkerHandle) -> None:
        async with self._lock:
            handle.busy = False

    def rpc(self, handle: WorkerHandle, command: dict[str, tp.Any]) -> dict[str, tp.Any]:
        handle.in_q.put(command, timeout=self.rpc_timeout_s)
        return handle.out_q.get(timeout=self.rpc_timeout_s)

    def record_audio_latency(self, handle: WorkerHandle, elapsed_ms: int) -> None:
        handle.last_elapsed_ms = elapsed_ms
        if handle.ewma_latency_ms is None:
            handle.ewma_latency_ms = float(elapsed_ms)
        else:
            a = self.ewma_alpha
            handle.ewma_latency_ms = (a * float(elapsed_ms)) + ((1.0 - a) * handle.ewma_latency_ms)

    def record_failure(self, handle: WorkerHandle) -> None:
        handle.failures += 1

    def snapshot(self) -> dict[str, tp.Any]:
        workers: list[dict[str, tp.Any]] = []
        for h in self.handles:
            workers.append(
                {
                    "index": h.index,
                    "device": h.device,
                    "busy": h.busy,
                    "alive": h.proc.is_alive(),
                    "sessions_assigned": h.sessions_assigned,
                    "failures": h.failures,
                    "last_elapsed_ms": h.last_elapsed_ms,
                    "ewma_latency_ms": h.ewma_latency_ms,
                }
            )
        return {
            "rpc_timeout_s": self.rpc_timeout_s,
            "routing_ewma_alpha": self.ewma_alpha,
            "workers": workers,
        }

    async def stop(self) -> None:
        for handle in self.handles:
            if handle.proc.is_alive():
                try:
                    await asyncio.to_thread(self.rpc, handle, {"type": "shutdown"})
                except Exception:
                    pass
        for handle in self.handles:
            handle.proc.join(timeout=5)
            if handle.proc.is_alive():
                handle.proc.kill()


def resolve_static_path(static_arg: str | None) -> str | None:
    if static_arg == "none":
        return None
    if static_arg is not None:
        return static_arg
    log("info", "retrieving static content")
    dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
    dist_tgz_path = Path(dist_tgz)
    dist_dir = dist_tgz_path.parent / "dist"
    if not dist_dir.exists():
        with tarfile.open(dist_tgz_path, "r:gz") as tar:
            tar.extractall(path=dist_tgz_path.parent)
    return str(dist_dir)


async def websocket_handler(request: web.Request) -> web.StreamResponse:
    pool: WorkerPool = request.app["worker_pool"]
    handle = await pool.acquire()
    if handle is None:
        return web.json_response(
            {"error": "server busy: no available worker"},
            status=503,
        )

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    idle_timeout_s: float = request.app["idle_timeout_s"]
    max_session_duration_s: float = request.app["max_session_duration_s"]
    session_start = time.monotonic()
    last_activity = session_start

    try:
        log(
            "info",
            (
                f"session assigned worker={handle.index} device={handle.device} "
                f"ewma_ms={handle.ewma_latency_ms} assigned={handle.sessions_assigned}"
            ),
        )
        opened = await asyncio.to_thread(pool.rpc, handle, {"type": "open"})
        if opened.get("type") != "opened":
            await ws.send_bytes(b"\x05" + b"failed to open session")
            await ws.close()
            pool.record_failure(handle)
            return ws

        # Keep handshake compatible with existing clients.
        await ws.send_bytes(b"\x00")

        while True:
            now = time.monotonic()
            timeout_candidates: list[float] = []
            if idle_timeout_s > 0:
                timeout_candidates.append(max(0.05, idle_timeout_s - (now - last_activity)))
            if max_session_duration_s > 0:
                timeout_candidates.append(max(0.05, max_session_duration_s - (now - session_start)))

            timeout = min(timeout_candidates) if timeout_candidates else None
            try:
                if timeout is None:
                    message = await ws.receive()
                else:
                    message = await ws.receive(timeout=timeout)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if idle_timeout_s > 0 and (now - last_activity) >= idle_timeout_s:
                    await ws.send_bytes(b"\x05" + b"session closed: idle timeout")
                    break
                if max_session_duration_s > 0 and (now - session_start) >= max_session_duration_s:
                    await ws.send_bytes(b"\x05" + b"session closed: max session duration reached")
                    break
                continue

            if message.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                break
            if message.type != aiohttp.WSMsgType.BINARY:
                continue
            last_activity = time.monotonic()

            raw = message.data
            if not isinstance(raw, bytes) or len(raw) == 0:
                continue
            msg_kind = raw[0]
            if msg_kind != 1:
                continue

            payload = raw[1:]
            result = await asyncio.to_thread(pool.rpc, handle, {"type": "audio", "payload": payload})
            if result.get("type") == "audio_result":
                elapsed_ms = int(result.get("elapsed_ms", 0))
                pool.record_audio_latency(handle, elapsed_ms)
                for out_message in result.get("messages", []):
                    await ws.send_bytes(out_message)
            elif result.get("type") in {"error", "fatal"}:
                err = result.get("error", "worker error")
                await ws.send_bytes(b"\x05" + bytes(err, encoding="utf8"))
                pool.record_failure(handle)
                break
    finally:
        try:
            await asyncio.to_thread(pool.rpc, handle, {"type": "close"})
        except Exception:
            pass
        await pool.release(handle)
    return ws


async def metrics_handler(request: web.Request) -> web.Response:
    pool: WorkerPool = request.app["worker_pool"]
    payload = pool.snapshot()
    return web.json_response(payload)


def _rt_event(event_type: str, **kwargs: tp.Any) -> dict[str, tp.Any]:
    payload: dict[str, tp.Any] = {"type": event_type, "event_id": f"evt_{uuid.uuid4().hex}"}
    payload.update(kwargs)
    return payload


def _safe_json_loads(raw: str) -> dict[str, tp.Any] | None:
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


@dataclass
class RealtimeSessionState:
    input_audio_format: str = "audio/pcm"
    output_audio_format: str = "audio/ogg;codecs=opus"
    input_audio_opus_pages: list[bytes] | None = None
    input_pcm_residual: np.ndarray | None = None
    input_pcm_sample_rate: int = 24000
    response_counter: int = 0
    item_counter: int = 0

    def __post_init__(self) -> None:
        self.input_audio_opus_pages = []
        self._opus_writer = sphn.OpusStreamWriter(self.input_pcm_sample_rate)

    def append_input_audio_chunk(self, audio_b64: str) -> None:
        raw = base64.b64decode(audio_b64)
        # Minimal subset: treat input as PCM16 mono 24kHz to keep close to Realtime API default.
        pcm_i16 = np.frombuffer(raw, dtype=np.int16)
        pcm = pcm_i16.astype(np.float32) / 32768.0
        pages = self._opus_writer.append_pcm(pcm)
        if len(pages) > 0 and self.input_audio_opus_pages is not None:
            self.input_audio_opus_pages.append(pages)

    def drain_input_audio_pages(self) -> list[bytes]:
        pages = self.input_audio_opus_pages or []
        self.input_audio_opus_pages = []
        return pages

    def next_response_id(self) -> str:
        self.response_counter += 1
        return f"resp_{self.response_counter}_{uuid.uuid4().hex[:8]}"

    def next_item_id(self) -> str:
        self.item_counter += 1
        return f"item_{self.item_counter}_{uuid.uuid4().hex[:8]}"


async def realtime_handler(request: web.Request) -> web.StreamResponse:
    pool: WorkerPool = request.app["worker_pool"]
    handle = await pool.acquire()
    if handle is None:
        return web.json_response({"error": "server busy: no available worker"}, status=503)

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    idle_timeout_s: float = request.app["idle_timeout_s"]
    max_session_duration_s: float = request.app["max_session_duration_s"]
    session_start = time.monotonic()
    last_activity = session_start
    rt = RealtimeSessionState()

    async def send_event(event_type: str, **kwargs: tp.Any) -> None:
        await ws.send_str(json.dumps(_rt_event(event_type, **kwargs), ensure_ascii=False))

    async def process_committed_audio() -> None:
        pages = rt.drain_input_audio_pages()
        if not pages:
            return
        response_id = rt.next_response_id()
        item_id = rt.next_item_id()
        await send_event(
            "response.created",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "output": [],
                "output_modalities": ["audio"],
            },
        )
        output_index = 0
        content_index = 0
        for page in pages:
            result = await asyncio.to_thread(pool.rpc, handle, {"type": "audio", "payload": page})
            if result.get("type") != "audio_result":
                pool.record_failure(handle)
                await send_event(
                    "error",
                    error={
                        "type": "server_error",
                        "message": result.get("error", "worker error"),
                    },
                )
                return
            elapsed_ms = int(result.get("elapsed_ms", 0))
            pool.record_audio_latency(handle, elapsed_ms)
            for packed in result.get("messages", []):
                if not isinstance(packed, (bytes, bytearray)) or len(packed) == 0:
                    continue
                tag = packed[0]
                payload = bytes(packed[1:])
                if tag == 1:
                    # Keep audio as opaque encoded bytes in delta.
                    await send_event(
                        "response.output_audio.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=base64.b64encode(payload).decode("ascii"),
                    )
                elif tag == 2:
                    text_delta = payload.decode("utf8", errors="ignore")
                    await send_event(
                        "response.output_text.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=text_delta,
                    )
                    await send_event(
                        "response.output_audio_transcript.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=text_delta,
                    )
        await send_event(
            "response.output_audio.done",
            response_id=response_id,
            item_id=item_id,
            output_index=output_index,
            content_index=content_index,
        )
        await send_event(
            "response.output_text.done",
            response_id=response_id,
            item_id=item_id,
            output_index=output_index,
            content_index=content_index,
            text="",
        )
        await send_event(
            "response.output_audio_transcript.done",
            response_id=response_id,
            item_id=item_id,
            output_index=output_index,
            content_index=content_index,
            transcript="",
        )
        await send_event(
            "response.done",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "output": [],
                "output_modalities": ["audio"],
            },
        )

    try:
        log(
            "info",
            (
                f"realtime session assigned worker={handle.index} device={handle.device} "
                f"ewma_ms={handle.ewma_latency_ms} assigned={handle.sessions_assigned}"
            ),
        )
        opened = await asyncio.to_thread(pool.rpc, handle, {"type": "open"})
        if opened.get("type") != "opened":
            pool.record_failure(handle)
            await send_event("error", error={"type": "server_error", "message": "failed to open session"})
            await ws.close()
            return ws

        await send_event(
            "session.created",
            session={
                "id": f"sess_{uuid.uuid4().hex}",
                "object": "realtime.session",
                "type": "realtime",
                "model": "moshi-compatible",
                "input_audio_format": rt.input_audio_format,
                "output_audio_format": rt.output_audio_format,
                "output_modalities": ["audio"],
            },
        )

        while True:
            now = time.monotonic()
            timeout_candidates: list[float] = []
            if idle_timeout_s > 0:
                timeout_candidates.append(max(0.05, idle_timeout_s - (now - last_activity)))
            if max_session_duration_s > 0:
                timeout_candidates.append(max(0.05, max_session_duration_s - (now - session_start)))
            timeout = min(timeout_candidates) if timeout_candidates else None

            try:
                if timeout is None:
                    message = await ws.receive()
                else:
                    message = await ws.receive(timeout=timeout)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if idle_timeout_s > 0 and (now - last_activity) >= idle_timeout_s:
                    await send_event("error", error={"type": "idle_timeout", "message": "session closed: idle timeout"})
                    break
                if max_session_duration_s > 0 and (now - session_start) >= max_session_duration_s:
                    await send_event(
                        "error",
                        error={"type": "max_duration", "message": "session closed: max session duration reached"},
                    )
                    break
                continue

            if message.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                break
            if message.type != aiohttp.WSMsgType.TEXT:
                # Ignore non-text payloads for realtime endpoint.
                continue
            last_activity = time.monotonic()
            payload = _safe_json_loads(message.data)
            if payload is None:
                await send_event("error", error={"type": "invalid_request_error", "message": "invalid JSON event"})
                continue

            event_type = payload.get("type")
            if event_type == "session.update":
                session_req = payload.get("session", {}) if isinstance(payload.get("session"), dict) else {}
                in_fmt = session_req.get("input_audio_format")
                out_fmt = session_req.get("output_audio_format")
                if isinstance(in_fmt, str):
                    rt.input_audio_format = in_fmt
                if isinstance(out_fmt, str):
                    rt.output_audio_format = out_fmt
                await send_event(
                    "session.updated",
                    session={
                        "type": "realtime",
                        "input_audio_format": rt.input_audio_format,
                        "output_audio_format": rt.output_audio_format,
                        "output_modalities": ["audio"],
                    },
                )
            elif event_type == "input_audio_buffer.append":
                audio_b64 = payload.get("audio")
                if not isinstance(audio_b64, str):
                    await send_event(
                        "error",
                        error={"type": "invalid_request_error", "message": "input_audio_buffer.append missing audio"},
                    )
                    continue
                try:
                    rt.append_input_audio_chunk(audio_b64)
                except Exception as exc:
                    await send_event(
                        "error",
                        error={"type": "invalid_request_error", "message": f"bad audio payload: {exc}"},
                    )
            elif event_type == "input_audio_buffer.clear":
                rt.drain_input_audio_pages()
                await send_event("input_audio_buffer.cleared")
            elif event_type == "input_audio_buffer.commit":
                await send_event("input_audio_buffer.committed", item_id=rt.next_item_id())
                await process_committed_audio()
            elif event_type == "response.create":
                # Minimal compatibility: if there is buffered audio, treat this as trigger to run inference.
                await process_committed_audio()
            elif event_type == "response.cancel":
                # Current backend processes synchronously per chunk; no in-flight cancellation path yet.
                await send_event("response.done", response={"id": "resp_cancelled", "status": "cancelled", "output": []})
            else:
                # Gracefully ignore unsupported events with explicit error notification.
                await send_event(
                    "error",
                    error={
                        "type": "unsupported_event",
                        "message": f"event '{event_type}' is not supported in this compatibility layer",
                    },
                )
    finally:
        try:
            await asyncio.to_thread(pool.rpc, handle, {"type": "close"})
        except Exception:
            pass
        await pool.release(handle)
    return ws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--workers", default=1, type=int, help="Number of model worker processes.")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated per-worker device mapping, e.g. "
            "'cuda:0,cuda:1'. When set, workers must match device count."
        ),
    )
    parser.add_argument("--rpc-timeout-s", default=30.0, type=float, help="Worker RPC timeout.")
    parser.add_argument(
        "--routing-ewma-alpha",
        default=0.2,
        type=float,
        help="EWMA alpha for latency-based worker routing (0<alpha<=1).",
    )
    parser.add_argument(
        "--idle-timeout-s",
        default=30.0,
        type=float,
        help="Close a session if no client binary message is received for this many seconds (<=0 disables).",
    )
    parser.add_argument(
        "--max-session-duration-s",
        default=0.0,
        type=float,
        help="Hard limit for a websocket session in seconds (<=0 disables).",
    )

    parser.add_argument("--static", type=str)
    parser.add_argument("--ssl", type=str, help="Directory containing cert.pem/key.pem for HTTPS.")

    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--lora-weight", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16 rather than bfloat16.",
    )
    parser.add_argument(
        "--no_fuse_lora",
        action="store_false",
        dest="fuse_lora",
        default=True,
        help="Do not fuse LoRA layers into linear layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(42_424_242)
    if not (0.0 < args.routing_ewma_alpha <= 1.0):
        raise ValueError("--routing-ewma-alpha must be in (0, 1].")
    if args.idle_timeout_s < 0:
        raise ValueError("--idle-timeout-s must be >= 0.")
    if args.max_session_duration_s < 0:
        raise ValueError("--max-session-duration-s must be >= 0.")

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # It may already be set by the runtime.
        pass

    static_path = resolve_static_path(args.static)

    devices: list[str] | None = None
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        if not devices:
            raise ValueError("--devices was provided but no valid device entries were parsed.")
        if args.workers == 1 and len(devices) > 1:
            # User likely expects one worker per listed device.
            args.workers = len(devices)
        if args.workers != len(devices):
            raise ValueError(
                f"workers/devices mismatch: --workers={args.workers}, devices={len(devices)}. "
                "Set one worker per listed device."
            )
        log("info", f"per-worker device mapping enabled: {devices}")

    worker_cfg = {
        "hf_repo": args.hf_repo,
        "tokenizer": args.tokenizer,
        "moshi_weight": args.moshi_weight,
        "mimi_weight": args.mimi_weight,
        "lora_weight": args.lora_weight,
        "config_path": args.config_path,
        "cfg_coef": args.cfg_coef,
        "device": args.device,
        "dtype": args.dtype,
        "fuse_lora": args.fuse_lora,
    }
    worker_pool = WorkerPool.create(
        args.workers,
        worker_cfg,
        args.rpc_timeout_s,
        devices=devices,
        ewma_alpha=args.routing_ewma_alpha,
    )

    app = web.Application()
    app["worker_pool"] = worker_pool
    app["idle_timeout_s"] = args.idle_timeout_s
    app["max_session_duration_s"] = args.max_session_duration_s
    app.router.add_get("/api/chat", websocket_handler)
    app.router.add_get("/realtime", realtime_handler)
    app.router.add_get("/metrics", metrics_handler)

    if static_path is not None:
        log("info", f"serving static content from {static_path}")

        async def handle_root(_: web.Request) -> web.FileResponse:
            return web.FileResponse(os.path.join(static_path, "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    async def on_cleanup(app_: web.Application) -> None:
        pool: WorkerPool = app_["worker_pool"]
        await pool.stop()

    app.on_cleanup.append(on_cleanup)

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"low-latency mp server listening at {protocol}://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()

