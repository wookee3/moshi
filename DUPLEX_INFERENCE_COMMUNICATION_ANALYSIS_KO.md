# Moshi 기반 Duplex Speech LLM 서버 구조 심층 분석 (Inference + Communication)

이 문서는 `moshi` 레포에서 **실시간 대화형(duplex) speech LLM 데모**를 만들 때 핵심이 되는 두 축:

1. **모델 인퍼런스 서버 구조**
2. **통신(WebSocket + 오디오 스트리밍) 서버 구조**

를 코드 레벨로 정리한 학습 문서다.  
목표는 "어디를 어떻게 바꿔야 지연/처리량/안정성이 개선되는지"를 빠르게 판단할 수 있도록 만드는 것이다.

---

## 1) 레포 전체에서 이 주제와 직접 관련된 구성

레포는 실질적으로 3개의 인퍼런스 스택을 제공한다.

- **PyTorch 스택 (연구/실험용)**: `moshi/`
- **Rust/Candle 스택 (프로덕션 지향)**: `rust/`
- **MLX 스택 (Mac 로컬 추론)**: `moshi_mlx/`

duplex demo 서버 관점에서 가장 중요한 축은 아래다.

- **PyTorch 서버 엔트리**: `moshi/moshi/server.py`
- **Rust 서버 엔트리**: `rust/moshi-backend/src/main.rs`, `rust/moshi-backend/src/standalone.rs`
- **Rust 실시간 스트리밍 루프**: `rust/moshi-backend/src/stream_both.rs`
- **웹 클라이언트 프로토콜/오디오 파이프라인**:
  - `client/src/protocol/encoder.ts`
  - `client/src/pages/Conversation/hooks/useSocket.ts`
  - `client/src/pages/Conversation/hooks/useUserAudio.ts`
  - `client/src/pages/Conversation/hooks/useServerAudio.ts`
  - `client/src/audio-processor.ts`

---

## 2) 큰 그림: 데이터 플로우 (마이크 -> 서버 -> 모델 -> 스피커)

### A. 입력(클라이언트 -> 서버)

1. 브라우저 마이크 캡처
2. `opus-recorder`가 Opus/Ogg 페이지로 인코딩 (`useUserAudio.ts`)
3. WebSocket binary message로 전송 (`type=0x01 audio`)

### B. 서버 인입/디코드

- PyTorch: `sphn.OpusStreamReader`
- Rust: Ogg packet reader + Opus decoder (`stream_both.rs::spawn_recv_loops`)

둘 다 PCM으로 복원 후, **Mimi 프레임 단위**로 버퍼링한다.

### C. 모델 인퍼런스

1. PCM -> Mimi `encode_step` (또는 `encode`) -> 오디오 코드 토큰
2. LM step (`LMGen.step` 또는 Rust `State.step`)으로 텍스트 토큰 + 오디오 토큰 생성
3. 생성 오디오 토큰 -> Mimi `decode_step` (또는 `decode`) -> PCM

### D. 출력(서버 -> 클라이언트)

1. PCM -> Opus 인코딩
2. WebSocket binary 송신 (`0x01 audio`, `0x02 text`, `0x04 metadata` 등)
3. 클라이언트 worker가 Opus decode
4. AudioWorklet(`audio-processor.ts`)이 버퍼링/드롭/지연 보정 후 재생

---

## 3) 핵심 개념: 지연(latency)을 결정하는 구조

Moshi 계열에서 기본 지연은 구조적으로 아래에서 생긴다.

- Mimi 프레임 크기: 약 **80ms**
  - `frame_size = sample_rate / frame_rate = 24000 / 12.5 = 1920`
  - 코드: `moshi/moshi/server.py`, `moshi/moshi/run_inference.py`, `rust/moshi-core/src/mimi.rs`
- acoustic delay: 기본 **2 토큰 스텝**
  - 코드: `rust/moshi-core/src/lm_generate_multistream.rs` (`Config::v0_1`)

즉, 모델 자체 이론 지연만 해도 대략 160ms 축이 있고, 여기에 네트워크/브라우저 버퍼/인코딩 비용이 추가된다.

---

## 4) PyTorch 서버 구조 상세 (`moshi/moshi/server.py`)

### 4.1 서버 초기화

- HF에서 체크포인트 로딩 (`loaders.CheckpointInfo.from_hf_repo`)
- `MimiModel`, `LMModel`, tokenizer 로드
- `ServerState` 생성 후 `warmup()` 수행
  - warmup 중 encode -> step -> decode를 여러 번 돌려 CUDA 경로를 예열

### 4.2 스트리밍 상태 관리

- `self.mimi.streaming_forever(1)`, `self.lm_gen.streaming_forever(1)`
- 연결마다 `reset_streaming()` 호출
- `handle_chat()`에서 `asyncio.Lock`을 잡고 처리

**중요 포인트**: 이 락 때문에 PyTorch 기본 서버는 사실상 **동시 세션 1개 직렬 처리**에 가깝다.  
데모에는 단순하지만, 멀티 유저 대응에는 병목이 된다.

### 4.3 오디오 수신/생성 루프

- 메시지 타입 첫 바이트로 구분 (`kind == 1`이면 audio)
- Opus bytes -> PCM
- 누적 버퍼가 `frame_size` 이상이면 프레임 처리
- 첫 프레임은 과거 컨텍스트 성격이라 skip 처리 (`skip_frames`)
- `mimi.encode` -> `lm_gen.step` -> `mimi.decode`
- 오디오는 `0x01`, 텍스트는 `0x02`로 전송

### 4.4 성능 관련 기법

- 내부적으로 `LMGen`은 CUDA Graph를 사용 가능 (`moshi/moshi/models/lm.py`)
- `NO_TORCH_COMPILE`, `NO_CUDA_GRAPH` 등 환경 변수로 최적화 경로 제어 가능

---

## 5) Rust 서버 구조 상세 (`rust/moshi-backend`)

Rust 쪽이 "프로덕션 지향" 설계가 더 강하다.

### 5.1 엔트리/런타임

- 엔트리: `src/main.rs`
- 서브커맨드:
  - `standalone`: 실제 서버
  - `benchmark`: 성능 측정
- Tokio multi-thread runtime
- `NoDelayAcceptor`로 TCP_NODELAY 적용 (지연 감소)

### 5.2 서버 부팅

- `standalone.rs::run`
  - TLS cert 자동 생성(없으면 self-signed)
  - `/api/chat` websocket 라우트
  - 정적 파일 서빙 (`client/dist`)
  - HTTPS 기본

### 5.3 모델 로드

- `AppStateInner::new`에서 LM + Mimi + tokenizer 로드
- 디바이스 선택:
  - CUDA/Metal 가능하면 사용
  - 옵션으로 Mimi만 CPU 분리 가능 (`use_cpu_for_mimi`)
- warm-up 수행

### 5.4 실시간 루프 (`stream_both.rs`)

핵심은 입력/인퍼런스/송신을 분리한 파이프라인이다.

1. `spawn_recv_loops`:
   - ws 수신 루프
   - Ogg/Opus -> PCM 디코드 루프
2. 모델 스레드:
   - `StreamingModel::run`
   - 내부에서 `run_with_state`, `run_with_state_mt`, `run_with_state_asr` 선택
3. sender loop:
   - StreamOut 이벤트를 받아 ws로 송신

즉, **비동기 + 스레드 분할**로 I/O와 모델 추론이 분리되어 있어 PyTorch 기본 서버보다 확장성이 높다.

### 5.5 run mode 3종

- `run_with_state`: 일반 duplex 생성
- `run_with_state_mt`: Mimi를 CPU 스레드로 분리하는 멀티스레드 경로
- `run_with_state_asr`: ASR delay 정책 포함 경로

모드 선택 기준:

- `use_cpu_for_mimi=true` -> `run_with_state_mt`
- `asr_delay_in_tokens` 설정됨 -> `run_with_state_asr`
- 그 외 -> `run_with_state`

---

## 6) 공통 프로토콜 분석 (WebSocket Message Type)

프로토콜 정의는 사실상 클라이언트/서버 코드에 내장되어 있다.

- `0x00`: handshake
- `0x01`: audio
- `0x02`: text
- `0x03`: control
- `0x04`: metadata
- `0x05`: error
- `0x06`: ping

참조:

- 클라이언트 encode/decode: `client/src/protocol/encoder.ts`
- 타입 선언: `client/src/protocol/types.ts`
- Rust enum: `rust/moshi-backend/src/stream_both.rs::MsgType`

주의할 점:

- 클라이언트 decoder에는 `0x07`(coloredtext) 분기가 있지만, 일반 encode는 `0x02 + color` 형태를 쓰는 구문이 섞여 있어 확장 시 프로토콜 정리가 필요할 수 있다.
- Rust handshake는 8바이트 payload를 포함해 보내고, 클라 decode는 현재 payload를 사실상 활용하지 않는다.

---

## 7) 모델 인퍼런스 내부: 왜 빠른가

### 7.1 PyTorch 쪽

핵심 파일: `moshi/moshi/models/lm.py`, `moshi/moshi/utils/compile.py`, `moshi/moshi/modules/transformer.py`

- `LMGen`이 step-wise 생성을 수행
- `CUDAGraphed` 래퍼로 `forward_text`, `depformer_step`를 캡처/재생
- 스트리밍 KV 캐시(`RingKVCache`)와 exec mask 기반 상태 유지
- CFG(classifier-free guidance) 분기 지원

**효율의 본질**: "프레임마다 전체 시퀀스 재연산"이 아니라, **한 step + KV cache 갱신**만 수행.

### 7.2 Rust 쪽

핵심 파일: `rust/moshi-core/src/lm_generate_multistream.rs`, `rust/moshi-core/src/kv_cache.rs`, `rust/moshi-core/src/lm.rs`

- `State::step_`가 텍스트/오디오 토큰을 시간축 지연 구조로 갱신
- repetition penalty, top-k/temp sampling 적용
- KV cache는 rotating/scattered 형태로 운영
- `.gguf` 확장자면 quantized 경로로 로딩 (`load_lm_model`)

**효율의 본질**: candle 기반 경량 추론 + 양자화 + 스트리밍 상태 머신 결합.

---

## 8) 클라이언트 오디오 파이프라인: 실제 체감 지연의 숨은 핵심

핵심 파일: `useUserAudio.ts`, `useServerAudio.ts`, `audio-processor.ts`

### 8.1 업링크(마이크 송신)

- `opus-recorder` 설정:
  - `encoderSampleRate: 24000`
  - `encoderFrameSize: 20` (ms)
  - `streamPages: true`

20ms 단위 전송은 네트워크 지연/오버헤드 트레이드오프에 영향.

### 8.2 다운링크(재생)

- worker가 Opus decode 후 Float32 PCM 전달
- AudioWorklet(`MoshiProcessor`)이 프레임 큐를 관리:
  - 시작 전 초기 버퍼 대기
  - underrun 시 partial buffer 증가
  - overrun 시 오래된 패킷 drop

즉, 클라이언트는 단순 재생이 아니라 **적응형 지연 제어기** 역할을 한다.

---

## 9) 설정 파일이 의미하는 실전 튜닝 포인트

### 9.1 Rust 설정 (`config.json`, `config-q8.json`)

- `hf_repo`: 모델 계열 선택
- `lm_model_file`, `mimi_model_file`, `text_tokenizer_file`
- `mimi_num_codebooks`: 기본 8
- `use_cpu_for_mimi`: CPU 분리 여부
- `asr_delay_in_tokens`: ASR 방식 지연 파라미터
- `addr`, `port`, `cert_dir`, `static_dir`, `log_dir`

`config-q8.json`은 LM을 gguf q8 경로로 바꿔 메모리/속도 trade-off를 잡는다.

### 9.2 PyTorch 런타임 플래그

- `--hf-repo`, `--device`, `--half`, `--cfg-coef`
- `NO_TORCH_COMPILE=1`, `NO_CUDA_GRAPH=1` 환경 변수

GPU/드라이버/파이썬 버전에 따라 torch compile 안정성이 달라질 수 있어 fallback 경로가 중요하다.

---

## 10) "어떤 부분을 어떻게 바꿀지"를 위한 우선순위 가이드

아래는 duplex demo 서버를 만들 때, ROI가 높은 순서의 변경 제안이다.

### 우선순위 1: 서버 동시성 모델 결정

- 단일 유저 데모면 PyTorch도 가능
- 멀티 유저/운영 지향이면 Rust 구조를 기본으로 가져가는 게 유리
- 특히 PyTorch `asyncio.Lock` 직렬 처리 여부를 반드시 설계적으로 재검토

### 우선순위 2: 프로토콜 명세 고정

- 메시지 타입, payload 스키마(handshake/metadata/control)를 문서화해 고정
- 클라이언트/서버 구현의 암묵적 차이(`0x07`, handshake payload)를 정리

### 우선순위 3: 지연 예산(latency budget) 분해

- 구간별 측정:
  - mic capture
  - opus encode/decode
  - ws RTT
  - mimi encode/decode
  - lm step
  - worklet buffering
- Rust는 이미 `StreamOut` 이벤트 기반 확장이 쉬워 메트릭 삽입이 편함

### 우선순위 4: 모델/디바이스 배치 최적화

- LM on GPU + Mimi on CPU 분리(`use_cpu_for_mimi`) 테스트
- q8/정밀도/탑K/온도/repetition_penalty의 품질-속도 균형점 탐색

### 우선순위 5: 클라이언트 버퍼 정책 튜닝

- 현재 AudioWorklet은 안정성 지향(드롭/증분 버퍼) 정책
- 네트워크 특성(로컬/원격)에 따라 초기 버퍼/최대 버퍼/증분 규칙을 분리 설정할 필요가 큼

---

## 11) 구현 시 바로 참고할 핵심 코드 포인트 모음

- PyTorch ws 루프: `moshi/moshi/server.py::ServerState.recv_loop`
- PyTorch 연결 진입: `moshi/moshi/server.py::handle_chat`
- PyTorch 생성기 핵심: `moshi/moshi/models/lm.py::LMGen.step`
- CUDA graph 래퍼: `moshi/moshi/utils/compile.py::CUDAGraphed`
- Rust ws 핸들러: `rust/moshi-backend/src/stream_both.rs::handle_socket`
- Rust 입력 디코드 루프: `rust/moshi-backend/src/stream_both.rs::spawn_recv_loops`
- Rust 추론 상태머신: `rust/moshi-core/src/lm_generate_multistream.rs::State`
- Rust LM 로딩(gguf/safetensors 분기): `rust/moshi-core/src/lm.rs::load_lm_model`
- 클라이언트 ws 연결: `client/src/pages/Conversation/hooks/useSocket.ts`
- 클라이언트 마이크 송신: `client/src/pages/Conversation/hooks/useUserAudio.ts`
- 클라이언트 재생 버퍼링: `client/src/audio-processor.ts`

---

## 12) 결론

이 레포에서 duplex speech LLM 데모를 위한 핵심은 아래 한 줄로 정리된다.

> **"스트리밍 상태를 유지하는 step-wise 모델 추론(Mimi + LM)과, Opus 기반 양방향 WS 파이프라인을 얼마나 안정적으로 결합하느냐"**가 성능과 체감 품질을 결정한다.

실전적으로는:

- 연구/빠른 검증은 PyTorch 경로,
- 멀티 세션/운영 지향은 Rust 경로,
- 최종 품질은 클라이언트 AudioWorklet 버퍼 정책까지 포함해서 튜닝

하는 접근이 가장 현실적이다.

