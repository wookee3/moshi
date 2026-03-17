# Low-Latency Multiprocessing Python Server

This module provides a Python server prototype optimized for lower latency under modest concurrency by using:

- `aiohttp` websocket gateway for networking
- a pool of **separate model worker processes**
- one active websocket session per worker
- latency-aware worker routing with EWMA (`--routing-ewma-alpha`)

## Why this exists

The reference `moshi.server` is simple and very good for research.  
This prototype is intended for A/B comparison when you need to reduce server-side contention without moving to Rust yet.

## Key design

- Each worker process loads a full Mimi + Moshi stack once.
- Each worker handles one active session at a time.
- Gateway assigns incoming sessions to available workers.
- Message protocol remains compatible with existing client usage:
  - `0x00` handshake
  - `0x01` audio
  - `0x02` text
- Optional compatibility endpoint for OpenAI Realtime-style JSON events: `/realtime`

## Run

From repository root:

```bash
python -m moshi.low_latency_mp_server.server --host 0.0.0.0 --port 8999 --workers 2 --hf-repo kyutai/moshika-pytorch-bf16
```

Per-worker GPU pinning example:

```bash
python -m moshi.low_latency_mp_server.server --port 8999 --devices cuda:0,cuda:1 --hf-repo kyutai/moshika-pytorch-bf16
```

Timeout controls example:

```bash
python -m moshi.low_latency_mp_server.server --port 8999 --devices cuda:0,cuda:1 --idle-timeout-s 20 --max-session-duration-s 600
```

Then open:

- `http://<host>:<port>` if static files are served without SSL
- `https://<host>:<port>` when `--ssl <cert_dir>` is set

Endpoints:

- Native binary protocol: `GET /api/chat`
- Realtime-compatible JSON protocol: `GET /realtime`
- Worker stats: `GET /metrics`

## Important notes

- `--workers` means **full model replicas**. GPU memory usage scales roughly with worker count.
- `--devices` pins each worker to a specific device in order.
- When `--devices` is set, worker count must match device count (or omit `--workers` to auto-match).
- `--routing-ewma-alpha` controls how quickly routing adapts to recent worker latency.
- `--idle-timeout-s` closes sessions that stop sending audio for too long.
- `--max-session-duration-s` enforces a hard session lifetime limit.
- For large models, start with `--workers 1` and increase only if memory permits.
- This prototype focuses on latency/concurrency tradeoffs and is not a full production server.

When no worker is available, `/api/chat` returns `503` with a JSON error instead of accepting a websocket.

## Metrics endpoint

The server exposes a lightweight JSON endpoint at:

```text
/metrics
```

Example:

```bash
curl http://localhost:8999/metrics
```

Fields include worker-level runtime stats:

- `index`, `device`, `busy`, `alive`
- `sessions_assigned`, `failures`
- `last_elapsed_ms`, `ewma_latency_ms`

## Realtime compatibility subset

`/realtime` supports a minimal subset intended for practical interoperability:

- Client events:
  - `session.update`
  - `input_audio_buffer.append` (expects base64 PCM16 mono 24kHz chunks)
  - `input_audio_buffer.clear`
  - `input_audio_buffer.commit`
  - `response.create`
  - `response.cancel` (best-effort acknowledgement)
- Server events:
  - `session.created`, `session.updated`
  - `input_audio_buffer.cleared`, `input_audio_buffer.committed`
  - `response.created`, `response.done`
  - `response.output_text.delta`, `response.output_text.done`
  - `response.output_audio.delta`, `response.output_audio.done`
  - `response.output_audio_transcript.delta`, `response.output_audio_transcript.done`
  - `error`

Events outside this subset return an `error` event with `unsupported_event`.

