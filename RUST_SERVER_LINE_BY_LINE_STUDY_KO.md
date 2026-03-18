# Moshi Rust 서버 라인 단위 학습 노트

이 문서는 `rust` 구현을 "거의 line-by-line"로 따라가며 공부하기 위한 자료다.  
범위는 서버 동작에서 직접 중요한 파일들 위주로 잡았다.

- `rust/moshi-backend/src/main.rs`
- `rust/moshi-backend/src/standalone.rs`
- `rust/moshi-backend/src/stream_both.rs`
- `rust/moshi-core/src/lm_generate_multistream.rs`
- `rust/moshi-core/src/mimi.rs`
- `rust/moshi-core/src/kv_cache.rs`
- `rust/moshi-core/src/lm.rs` (핫패스 관련 로딩 구간)

---

## 0) 먼저 머리에 넣을 큰 구조

Rust 서버 실행 흐름은 아래 순서다.

1. `main.rs`: CLI 파싱, config 로딩, 로깅 초기화
2. `standalone.rs`: TLS/라우터/앱 상태 준비
3. `stream_both.rs::handle_socket`: websocket 연결별 스트리밍 파이프라인 시작
4. `stream_both.rs::StreamingModel::run`: Mimi 인코드 + LM step + Mimi 디코드 루프
5. `lm_generate_multistream.rs::State::step_`: 토큰 상태머신 핵심

---

## 1) `main.rs` 라인 단위 해설

### 1-1. 모듈과 CLI 스켈레톤

- `L9-L13`: 서버 구성 모듈 선언 (`audio`, `benchmark`, `standalone`, `stream_both`, `utils`)
- `L15-L29`: 최상위 인자 구조
  - `--log`, `--config`, `--silent`
  - 서브커맨드(`standalone`, `benchmark`)
- `L31-L65`: `StandaloneArgs`, `BenchmarkArgs`, `Command` 정의

핵심: 실행 모드는 명확히 `standalone`(서비스) vs `benchmark`(성능측정)로 분리됨.

### 1-2. 지연 최적화 한 줄

- `L67-L84`: `NoDelayAcceptor`
  - accept 시 `stream.set_nodelay(true)` 수행 (`L79-L81`)
  - Nagle 비활성화로 작은 패킷 즉시 전송

이건 음성 스트리밍 체감 지연에서 의미 있는 최적화 포인트다.

### 1-3. 로깅 초기화

- `L86-L110`: `tracing_init`
  - rolling file appender 구성
  - `silent`가 아니면 stdout에도 동일 레벨로 출력
  - `BuildInfo` 로그 남김

### 1-4. 실제 진입

- `L112`: Tokio 멀티스레드 런타임
- `L114`: args parse
- `L116-L153`: standalone 경로
  - config load
  - logger init
  - 필요 시 모델 다운로드 (`requires_model_download`)
  - static dir 없으면 `dist.tgz` 받아 압축 해제
  - `standalone::run` 호출
- `L154-L174`: benchmark 경로

---

## 2) `standalone.rs` 라인 단위 해설

### 2-1. 설정 타입

- `L12-L21`: `Config`
  - `cert_dir`, `static_dir`, `addr`, `port`
  - `stream_both::Config`를 flatten

- `L24-L35`: `Config::load`
  - json 읽고 deserialize
  - env var 치환 (`$HOME` 등) 적용

### 2-2. 디바이스 선택

- `L44-L55`: `device(cpu)`
  - `--cpu`면 CPU
  - 아니면 CUDA > Metal > CPU 순 fallback

### 2-3. 앱 상태 생성 (중요)

- `L57-L92`: `AppStateInner::new`
  - `L59-L61`: LM 로드 (`moshi::lm::load_streaming`)
  - `L62-L67`: Mimi 로드 (옵션으로 Mimi CPU 분리)
  - `L68-L69`: tokenizer 로드
  - `L70-L90`: warm-up
    - LM forward + depformer sample 1회
    - Mimi encode/decode step 1회
    - `device.synchronize()`

핵심: 서비스 시작 전에 커널/JIT/메모리 경로를 미리 데워 첫 응답 지연을 줄인다.

### 2-4. websocket 업그레이드 핸들러

- `L101-L110`: `stream_handler`
  - query로 session config 수신
  - `StreamingModel::new` 생성
  - `ws.on_upgrade(...)`로 소켓 처리 이관

### 2-5. 모델 파일 자동 다운로드

- `L112-L138`: `download_from_hub`
  - `hf_repo`에서 LM/Mimi/tokenizer 파일명 기준으로 다운로드
  - local config path를 실제 다운로드 경로로 덮어씀

### 2-6. 서버 런

- `L140-L171`: `run`
  - `L141-L148`: cert/key 없으면 self-signed 생성
  - `L150-L151`: rustls config 로드
  - `L152-L156`: bind address 생성
  - `L157`: app state 생성
  - `L159-L166`: router
    - `/api/chat` -> websocket
    - 나머지 fallback -> static file
  - `L168-L170`: HTTPS 서버 실행

---

## 3) `stream_both.rs` 라인 단위 해설 (핵심)

이 파일이 "통신 + 실시간 추론 orchestration"의 중심이다.

### 3-1. Config + AppState

- `L13-L27`: 스트리밍 config
  - 모델 파일/토크나이저/로그 디렉토리
  - `mimi_num_codebooks`, `use_cpu_for_mimi`, `asr_delay_in_tokens`
- `L52-L59`: 앱 상태
  - `lm_model`, `mimi_model`, `text_tokenizer`, `device`, `config`

### 3-2. 텍스트 디코딩 헬퍼

- `L61-L91`: `AppStateInner::text(...)`
  - 시작/패드/EOP 토큰 제외
  - 이전 토큰 대비 diff 방식으로 새 문자열만 뽑아냄

### 3-3. 세션 파라미터

- `L93-L121`: `SessionConfigReq`(query 입력), `SessionConfig`(실사용)
- `L136-L155`: default 채우기
  - `text_temperature`, `audio_temperature`, `topk`, `max_steps` 등
  - repetition penalty 컨텍스트/스케일 결합

### 3-4. ws 메시지 타입 정의

- `L188-L225`: `MsgType`
  - 0 handshake, 1 audio, 2 text, 3 control, 4 metadata, 5 error, 6 ping

### 3-5. 송신기 (`MsgSender`)

- `L236-L317`: 서버->클라이언트 전송 전담
  - `new`: Opus encoder + Ogg header/tags 준비
  - `send_ready`: handshake 전송
  - `send_metadata`: 세션 메타 JSON 전송
  - `send_text`: 텍스트 프레임 전송
  - `send_pcm`: PCM 큐를 960 샘플 단위로 Opus 인코딩 후 Ogg 패킷 생성/전송

960 샘플 프레임(24kHz에서 40ms)은 Opus API 제약을 반영한 실전값.

### 3-6. 추론 루프 3종

#### (A) `run_with_state_asr` (`L327-L381`)

- ASR 용 지연 정책 반영
- 초기 `asr_delay_in_tokens` 구간에서는 텍스트 입력 없이 `step_` 수행
- 이후 `state.step`으로 정상 텍스트 샘플링

#### (B) `run_with_state` (`L383-L443`)

- 일반 duplex 경로
- 입력 PCM -> Mimi `encode_step`
- 각 스텝마다 `state.step(prev_text, codes, ...)`
- `state.last_audio_tokens()` 있으면 Mimi `decode_step`으로 PCM 생성
- 텍스트/PCM을 `StreamOut` 채널로 보냄

#### (C) `run_with_state_mt` (`L445-L547`)

- Mimi CPU 분리 멀티스레드 경로
- 스레드1: PCM->encode
- 메인 스레드: LM step
- 스레드2: decode->PCM

GPU LM과 CPU Mimi를 분리해 경합을 줄이려는 설계.

### 3-7. `StreamingModel::run` 조립

- `L558-L677`:
  - metadata 생성/송신
  - sampling logits processor 구성
  - `State::new(...)` 생성
  - config 따라 실행 경로 선택
    - `use_cpu_for_mimi` -> mt
    - `asr_delay_in_tokens` -> asr
    - else 일반
  - 종료 시 transcript/토큰을 로그 파일(json + safetensors)로 저장

### 3-8. ws 수신 디코드 루프

- `spawn_recv_loops` (`L682-L758`)
  - loop1: ws binary 수신, 타입 파싱, audio payload만 duplex writer로 push
  - loop2: ogg packet reader + opus decoder, PCM 버퍼링 후 일정량 flush해 model input channel로 보냄

### 3-9. sender loop

- `L760-L778`
  - `StreamOut` 이벤트를 받아 실제 ws 프레임으로 flush
  - deadlock 회피 위해 async recv 강조 주석(`L764-L766`)

### 3-10. 최종 소켓 핸들러

- `handle_socket` (`L780-L828`)
  - websocket split
  - in/out 채널 생성
  - recv loops spawn + model thread spawn + sender loop spawn
  - `tokio::select!`로 timeout/loop 종료 감시

여기가 사실상 "session runtime supervisor"다.

---

## 4) `lm_generate_multistream.rs` 라인 단위 해설

이 파일은 텍스트+오디오 동시 생성의 상태머신 핵심이다.

### 4-1. Config

- `L13-L67`: 생성에 필요한 토큰 체계 정의
  - codebook 수, vocab 크기, acoustic delay
  - text special tokens
  - `audio_pad_token()`, `total_audio_codebooks()`

### 4-2. State 메모리 구조

- `L69-L83`
  - `audio_tokens`, `text_tokens` 큰 버퍼
  - sampling processors
  - repetition penalty, forced tokens, cfg_alpha

- `L87-L121`: `State::new`
  - 최대 step + delay만큼 버퍼 미리 할당
  - UNGENERATED 초기화

### 4-3. repetition penalty

- `L142-L183`: 최근 텍스트 컨텍스트에서 중복 토큰 penalty 적용
  - pad/eop/start 제외
  - 양/음수 logit에 따라 곱/나눗셈 분기

### 4-4. step 핵심 (`step_`)

- `L187-L289`
  - 입력 오디오 토큰을 내부 버퍼에 기록
  - 현재 step에서 모델 forward에 넣을 codebook별 입력 구성
    - delay/초기 구간/이전 step 참조 규칙
  - `model.forward_cond` 혹은 `forward_ca`
  - text sampling
  - depformer로 오디오 codebook 샘플링
  - acoustic delay 규칙에 맞춰 오디오 토큰을 과거 위치에 write-back
  - `step_idx += 1`

이 함수 하나가 duplex 시간축 정렬을 담당한다고 보면 된다.

### 4-5. 결과 조회

- `last_audio_tokens` (`L330-L342`):
  - delay를 지난 뒤에만 실제 오디오 토큰 반환
  - 아직 pad/ungenerated 상태면 `None`

---

## 5) `mimi.rs` 라인 단위 해설

### 5-1. Mimi config

- `L17-L91`: sample rate 24kHz, frame rate 12.5Hz 등 핵심 하이퍼파라미터
- `quantizer_n_q`는 codebook 수

### 5-2. 모델 구성

- `L93-L103`: encoder/decoder + transformer + up/downsample + quantizer
- `L114-L168`: 실제 모듈 조립

### 5-3. 스트리밍 API

- `encode_step` (`L192-L203`)
  - encoder -> encoder_transformer -> downsample -> quantize
  - 아직 출력이 안 나오는 스텝은 empty 반환
- `decode_step` (`L214-L222`)
  - quantizer decode -> upsample -> decoder_transformer -> decoder

### 5-4. 상태 리셋

- `reset_state` (`L224-L231`)
  - 스트리밍 상태를 세션 경계에서 초기화

---

## 6) `kv_cache.rs` 라인 단위 해설

### 6-1. 핵심 자료구조

- `ScatteredKvCache` (`L21-L51`)
- `ScatteredCacheBuilder` (`L54-L217`)
- `KvCache::Rotating` wrapper (`L219-L253`)

### 6-2. 왜 중요한가

- attention 계산 시 과거 key/value를 재사용해야 step 추론이 빠르다
- `indices_and_mask`가 각 배치의 현재 위치/마스크를 계산해 캐시에 append
- context 초과 시 회전/덮어쓰기 방식으로 메모리 bounded 유지

---

## 7) `lm.rs` (핫패스 로딩만)

전체 파일이 커서 서버 런타임 이해에 꼭 필요한 로딩 부분만 요약한다.

- `load_lm_model` (`L1009-L1031`)
  - 확장자가 `gguf`면 quantized var builder
  - 아니면 safetensors mmap
- `load_streaming` (`L1042-L1049`)
  - streaming config로 LM 로드
- `load_streaming_both_ways` (`L1051-L1058`)
  - 양방향 스트림(코드북 수 16) 설정 버전

즉, 파일 포맷 선택(q8 gguf vs bf16 safetensors)이 이 단계에서 갈린다.

---

## 8) 코드 읽기 추천 순서 (실제 공부 루프)

### 1회독 (흐름 파악)

1. `main.rs`
2. `standalone.rs`
3. `stream_both.rs`의 `handle_socket`, `spawn_recv_loops`, `sender_loop`, `StreamingModel::run`

### 2회독 (생성 알고리즘)

4. `lm_generate_multistream.rs`의 `State::new`, `step_`, `last_audio_tokens`
5. `mimi.rs`의 `encode_step`, `decode_step`

### 3회독 (성능/메모리)

6. `kv_cache.rs`
7. `lm.rs` 로딩 분기(gguf/safetensors)

---

## 9) Python 서버와 대조해 볼 때 관찰 포인트

- Rust는 websocket 수신/디코드/추론/송신이 명시적 파이프라인으로 분리되어 있음
- 세션별 supervisor(`tokio::select!`)가 timeout/loop 종료를 강하게 관리
- 종료 시 세션 summary를 파일로 남겨 디버깅/리플레이가 용이
- q8/CPU Mimi 분리 등 운영형 옵션이 코드 경로로 자연스럽게 녹아 있음

---

## 10) 다음 학습 단계 제안

이 문서로 1회독 후에는 아래 2가지를 직접 해보면 이해가 빨라진다.

1. `stream_both.rs`에서 `StreamOut::StepStart/StepPostSampling`에 타이밍 로깅 추가
2. `config.json` vs `config-q8.json`으로 p50/p95 지연 비교

둘을 해보면 "구조 이해"가 "운영 감각"으로 바로 연결된다.

