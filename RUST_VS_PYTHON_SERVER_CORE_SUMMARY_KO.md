# Duplex Speech 서버 핵심 요약 (Rust vs Python)

이 문서는 "핵심 로직만 빠르게 파악"하는 용도의 요약본이다.

---

## 1) 공통 핵심 로직 (둘 다 같은 본질)

Rust든 Python이든 실시간 duplex 음성 서버의 본질은 동일하다.

1. 클라이언트에서 오디오 수신 (WS)
2. Opus/Ogg -> PCM 디코드
3. PCM을 Mimi 프레임 단위로 버퍼링
4. `Mimi encode` -> 오디오 토큰
5. `LM step` -> 텍스트 토큰 + 오디오 토큰
6. `Mimi decode` -> PCM
7. PCM -> Opus 인코드 후 WS 송신

핵심 지연은 주로 아래에서 결정된다.

- Mimi 프레임 단위 처리(80ms)
- acoustic delay(모델 구조)
- 오디오 코덱 인/디코딩
- 네트워크 RTT
- 클라이언트 재생 버퍼링

---

## 2) Rust 서버 로직 요약

대상 파일:

- `rust/moshi-backend/src/main.rs`
- `rust/moshi-backend/src/standalone.rs`
- `rust/moshi-backend/src/stream_both.rs`
- `rust/moshi-core/src/lm_generate_multistream.rs`

핵심 설계:

- Tokio 기반 비동기 + 멀티스레드 파이프라인
- 소켓 수신/디코드/추론/송신을 분리
- session 단위 supervisor(`tokio::select!`)로 타임아웃/종료 관리
- 운영용 기능(HTTPS, 파일 로그, 모델 다운로드, q8 경로) 포함
- `TCP_NODELAY` 적용으로 지연 최적화

왜 안정적인가:

- 메모리/스레드 경합 리스크를 컴파일 단계에서 많이 제거
- 장시간 운영 시 tail latency(p95/p99) 변동성이 상대적으로 작음
- 장애 지점이 명확하고 종료/복구 루프가 보수적으로 설계됨

---

## 3) 기존 Python 서버 로직 요약

대상 파일:

- `moshi/moshi/server.py`

핵심 설계:

- aiohttp websocket + 단순 스트리밍 루프
- 모델(`mimi`, `lm_gen`) 스트리밍 상태를 연결마다 reset
- 구현이 간단해 연구/수정 속도가 빠름

제약:

- 연결 처리 구간이 사실상 직렬화되기 쉬운 구조
- 동시접속이 늘면 지연 악화 가능성이 큼
- 운영 관점 기능(세션 관리/리소스 격리/관측성)이 상대적으로 약함

---

## 4) 새 Python 서버에서 어떻게 보완했는가

대상 파일:

- `moshi/moshi/low_latency_mp_server/server.py`

적용한 보완:

- **멀티프로세스 워커 풀**
  - 워커별 모델 상주, 세션당 워커 고정
- **워커별 GPU 고정 할당**
  - `--devices cuda:0,cuda:1,...`
- **레이턴시 기반 라우팅**
  - EWMA 지표로 세션 할당 우선순위 조정
- **세션 보호 정책**
  - idle timeout
  - max session duration
- **자원 부족 시 즉시 거절**
  - 워커 없으면 `/api/chat` 503 반환
- **관측성**
  - `/metrics`로 워커 상태 확인
- **호환성 확장**
  - `/realtime` 최소 OpenAI Realtime 이벤트 subset 지원

의미:

- Python의 단점(직렬화, 운영성 부족)을 구조적으로 완화
- 하지만 런타임 특성상 Rust 수준의 장시간 결정성/안정성까지 완전히 동일하진 않음

---

## 5) Rust와 Python(보완판) 핵심 차이 요약

- **런타임 결정성**
  - Rust > Python
- **개발/실험 속도**
  - Python > Rust
- **장시간 운영 안정성**
  - Rust > Python
- **현실적 타협안**
  - Python으로 빠르게 검증 -> Rust로 핵심 경로 이관

---

## 6) 어떤 상황에서 무엇을 쓰는가

- 빠른 기능 검증/프로토타입:
  - Python 보완판(`low_latency_mp_server`)이 가장 효율적
- 상용 안정성/예측 가능한 p95,p99:
  - Rust 서버가 더 유리

즉, 지금 단계에서는 Python 보완판으로 지표/요구사항을 고정하고,  
최종 서비스 단계에서 Rust로 수렴하는 전략이 가장 현실적이다.

