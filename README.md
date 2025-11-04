# maker_challenge_suin

회의 전사·요약·툴체인 연동을 위한 내부망 우선(오프라인 친화) 도구

이 프로젝트는 외부 상용 AI 구독서비스를 사용하기 어려운 내부망 환경에서 회의 녹취를 안전하게 전사(STT)하고, 화자 분리·전문용어 보정·요약·액션아이템 추출 및 사내 툴체인(예: Atlassian, Slack 등)과 연계하는 파이프라인을 목표로 합니다. 이 README는 Gist(첫 JTBT.md)의 내용을 반영하여 요구/설계/운영 지침을 정리한 문서입니다.

## 핵심 배경(요약)
- 문제: 회의록 작성과 후속 조치 연결이 반복적이고 공수(노동력)를 많이 차지함. 외부 AI 서비스 사용이 제한된 환경에서는 자동화 도입이 어렵다.
- 목표: 내부망에서 안전하게 동작하는 자동화 파이프라인을 제공해 회의록 작업 공수를 줄이고, 툴체인 연동으로 후속 업무를 자동화한다.

## 핵심 기능
- 오디오 캡처/업로드(로컬 파일 입력)
- STT: OpenAI API(선택) 또는 로컬 Whisper 기반 전사
- 화자 분리(선택적 플러그인: pyannote/whisperx 등)
- 전문용어 사전(용어 매핑/교정)
- 요약 및 액션 아이템(액션 항목) 추출
- 툴체인 연동(Atlassian, Slack, 내부 이슈 관리 시스템)

## 보안 및 운영 원칙
- 기본 권장 모드: 로컬 모델(Whisper) 사용 — 외부 요청 최소화
- API 사용 시: `OPENAI_API_KEY`와 같은 시크릿은 환경변수 또는 안전한 시크릿 스토어에서만 사용
- 오디오/전사 파일은 필요 최소한의 보관(암호화 권장), 로그에 원문 노출 금지
- 권한/접근 제어: 파일/서비스 접근 권한을 최소화

## 설치 (로컬 개발 환경)

1. 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. 의존성 설치

```bash
pip install -r requirements.txt
# 로컬 Whisper(모델) 사용 시 별도로 PyTorch 설치가 필요합니다 (환경에 따라 CPU/GPU 선택)
# 예: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. (옵션) OpenAI API 사용 시

```bash
export OPENAI_API_KEY="sk-..."
```

## 빠른 사용 예시

Python으로 전사 수행(예제):

```python
from src import transcribe

# OpenAI API 방식
text = transcribe.transcribe_with_openai_api('data/raw/meeting1.mp3')

# 로컬 Whisper 방식
text_local = transcribe.transcribe_local_whisper('data/raw/meeting1.mp3', model_name='small')

print(text)
```

CLI(예시) — (추후 `src/cli.py` 제공 예정):

```bash
python -m src.cli run --input data/raw/meeting1.mp3 --mode local --model small
```

## 권장 폴더 구조

```
maker_challenge_suin/
├─ README.md
├─ maker_challenge_suin.code-workspace
├─ requirements.txt
├─ pyproject.toml   # optional
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ config.py
│  ├─ recorder/
│  ├─ audio/
│  ├─ stt/
│  ├─ diarization/
│  ├─ nlp/
│  ├─ connectors/
│  ├─ storage/
│  └─ orchestrator.py
├─ test/
└─ docs/
```

## 아키텍처(간단 흐름)

1. 오디오 입력(업로드 또는 녹음)
2. 전처리(샘플레이트 정규화, 노이즈 제거 옵션)
3. STT 전사(OpenAI API 또는 로컬 모델)
4. 화자 분리(옵션)
5. 전문용어 교정 및 정규화
6. 요약 및 액션 아이템 추출
7. 저장(로컬/DB) 및 툴체인에 이슈/알림 전송

## 데이터 모델(요약)
- MeetingAudio: id, path, duration, meta
- TranscriptionResult: text, segments[], language, confidence
- Segment: start, end, speaker, text
- Summary: text, bullets[]
- ActionItem: id, summary, assignee, due, source_segment

## 데이터베이스 구현 (SQLite)

이 프로젝트는 회의록 저장 및 조회를 위해 SQLite 데이터베이스를 사용합니다. 모든 회의 정보는 로컬에 안전하게 저장되며, 외부 네트워크 연결이 필요하지 않습니다.

### 데이터베이스 스키마

**meetings 테이블:**

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | INTEGER PRIMARY KEY | 자동 증가 ID |
| `name` | TEXT NOT NULL | 회의록 이름 |
| `meeting_date` | TEXT NOT NULL | 회의 날짜 (YYYY-MM-DD) |
| `audio_path` | TEXT | 오디오 파일 경로 (data/raw/meetings/) |
| `transcript_text` | TEXT | 전사된 전체 텍스트 |
| `segments_json` | TEXT | 화자별 세그먼트 JSON 배열 |
| `participants_json` | TEXT | 참석자 매핑 JSON (speaker_id → 이름) |
| `meta_json` | TEXT | 메타데이터 JSON |
| `created_at` | TEXT | 생성 시각 (자동) |

### 데이터베이스 API 사용법

**초기화:**

```python
from src import db

# 데이터베이스 초기화 (테이블 생성)
db.init_db('data/meetings.db')
```

**회의록 저장:**

```python
meeting_id = db.create_meeting(
    'data/meetings.db',
    name='2025-11-05 제품 기획 회의',
    meeting_date='2025-11-05',
    audio_path='data/raw/meetings/20251105_143022_audio.wav',
    transcript_text='[Alice] 안녕하세요.\n[Bob] 시작하겠습니다.',
    segments=[
        {"start": 0.0, "end": 1.5, "speaker": "Alice", "text": "안녕하세요."},
        {"start": 1.5, "end": 3.2, "speaker": "Bob", "text": "시작하겠습니다."}
    ],
    participants={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
    meta={"source": "ui_app", "duration": 120}
)
print(f"저장된 회의록 ID: {meeting_id}")
```

**회의록 목록 조회:**

```python
meetings = db.list_meetings('data/meetings.db')
for m in meetings:
    print(f"{m['id']}: {m['name']} ({m['meeting_date']})")
```

**특정 회의록 조회:**

```python
meeting = db.get_meeting('data/meetings.db', meeting_id)
print(f"이름: {meeting['name']}")
print(f"날짜: {meeting['meeting_date']}")
print(f"오디오: {meeting['audio_path']}")
print(f"참석자: {meeting['participants_json']}")
print(f"세그먼트 수: {len(meeting['segments_json'])}")
```

### 오디오 파일 저장 위치

전사된 회의의 오디오 파일은 `data/raw/meetings/` 폴더에 타임스탬프와 함께 저장됩니다:

```
data/raw/meetings/
├── 20251105_143022_audio.wav
├── 20251105_150315_meeting_recording.mp3
└── ...
```

### UI에서의 데이터베이스 사용

Gradio UI (`src/ui_app.py`)에서 데이터베이스를 활용하는 기능:

1. **회의록 작성 페이지**: 오디오 업로드/녹음 → 전사 → 저장
2. **회의록 목록 페이지**: 저장된 회의록 조회 및 선택
3. **회의록 열기**: 선택한 회의록의 전사 내용, 오디오 재생, 메타정보 표시
4. **참석자 매핑**: 화자 ID(SPEAKER_00 등)를 실제 이름으로 매핑

### 보안 고려사항

- 모든 데이터는 로컬 SQLite 파일에 저장됩니다 (외부 전송 없음)
- 오디오 파일은 로컬 파일시스템에만 보관됩니다
- 데이터베이스 파일 접근 권한을 적절히 설정하세요
- 민감한 회의록은 암호화된 볼륨에 저장하는 것을 권장합니다

## 구현 우선순위(권장)
1. 기본 파이프라인 스켈레톤: 로컬 Whisper 전사 + 파일 I/O
2. TerminologyManager(용어 매핑)
3. 간단한 Summarizer(rule-based)
4. Connectors(Atlassian mock)
5. 화자 분리(whisperx/pyannote 통합)
6. CLI와 문서화

## 개발 및 테스트 지침
- 네트워크/외부 API 의존 코드는 모킹(mock)하여 CI에서 안전하게 테스트
- 대용량 모델(large) 테스트는 로컬 리소스 한계로 CI에서 제외

## 운영 고려사항
- 모델 로드 시간과 메모리 요구를 명확히 문서화(서버 스펙, 권장 모델)
- 데이터 보존 정책(언제 삭제할지), 감사 로그 정책

## 향후 로드맵(예시)
- 화자 분리 정확도 향상(whisperx/pyannote)
- 요약 품질 개선(사내 LLM 또는 소형 LLM 래핑)
- 자동 이슈 분류 및 우선순위 매핑
- GUI 또는 Web 대시보드 제공

## 기여
- 이 저장소는 오픈 소스 기여를 환영합니다. 풀리퀘스트 전에 이슈를 열어 논의해 주세요.

---

이 README는 Gist(첫 JTBT.md)의 JOB-TO-BE-DONE 분석(요구/제약/해결방안)을 반영하여 작성되었습니다. 원하시면 이 문서를 바탕으로 스켈레톤 코드(예: `src/models/schemas.py`, `src/stt/base.py`, `src/orchestrator.py`)를 바로 생성해 드리겠습니다.

## 실행: Gradio UI (데모)

샘플 Gradio UI(Meeting Transcriber)를 추가했습니다. 로컬에서 실행하려면 가상환경을 활성화한 후 의존성을 설치하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

앱 실행:

```bash
./scripts/run_ui.sh
# 또는
python -m src.ui_app
```


주의: `gradio`와 일부 전사 관련 패키지(`openai-whisper`, `openai`)가 필요합니다. 로컬 Whisper를 사용하려면 PyTorch를 시스템에 맞게 별도 설치하세요.

## 로컬 Whisper 환경 구성 (macOS 권장 지침)

로컬 Whisper(`openai-whisper`)를 사용하려면 다음이 필요합니다:
- FFmpeg (오디오 디코딩/전처리)
- PyTorch (시스템/하드웨어에 맞는 빌드 — CPU / MPS(Apple Silicon) / CUDA)
- `openai-whisper` Python 패키지

아래는 macOS에서 안전하게 설치하는 권장 절차. 사용 환경(Apple Silicon vs Intel, GPU 사용 여부)에 따라 명령이 달라지므로 적절한 블록을 선택하세요.

1) Homebrew(권장) 확인 및 FFmpeg 설치

```bash
# Homebrew가 이미 설치되어 있으면 아래 커맨드로 FFmpeg 설치
brew install ffmpeg
```

2) (권장) 가상환경 생성

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

3) PyTorch 설치 (macOS)

- Apple Silicon(M1/M2) — MPS 사용(권장):

```bash
# PyTorch 설치 명령은 PyTorch 공식 사이트에서 확인하세요. 예시 (CPU/MPS):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> 참고: PyTorch의 MPS 빌드/태그는 릴리스에 따라 달라집니다. 정확한 명령은 https://pytorch.org 에서 "Get Started"를 확인하세요.

- Intel macOS 또는 CPU 전용:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4) whisper 설치 (requirements.txt에 이미 `openai-whisper`가 포함되어 있으므로 다음으로 설치)

```bash
pip install -r requirements.txt
```

5) 설치 검증

```bash
python scripts/check_whisper.py
```

이 스크립트는 `torch` 및 `whisper` 임포트를 시도하고, 사용 가능한 디바이스(CUDA/MPS/CPU)를 출력합니다. 문제가 발생하면 출력 메시지를 확인해 주세요.

문제 해결 팁:
- "No module named 'torch'" → PyTorch 설치 실패. 플랫폼(arm64 vs x86_64)에 맞는 wheel을 설치하세요.
- FFmpeg 관련 에러(파일 디코딩 실패) → `ffmpeg`가 시스템에 설치되어 있는지 확인하세요.
- 권한/가상환경 문제 → 가상환경이 활성화되어 있는지 확인하세요.

자동 설치 스크립트: `scripts/setup_local_whisper.sh` 를 추가해 두었습니다. 이 스크립트은 설치 권장 명령을 출력하고, `--auto` 옵션으로 일부 단계를 자동 실행할 수 있으나, 시스템 환경에 따라 사용 전에 명령을 검토하시기 바랍니다.


## Whisper (OpenAI) integration

이 프로젝트에 OpenAI Whisper를 추가하는 방법 예시입니다. 두 가지 방법을 지원합니다:

- OpenAI API (모델: `whisper-1`) 사용 — `OPENAI_API_KEY` 환경변수가 필요합니다.
- 로컬 Whisper 모델 사용 (`openai-whisper` 패키지, PyTorch가 필요합니다).

설치:

```bash
pip install -r requirements.txt
# 로컬 Whisper를 GPU로 사용하려면 PyTorch를 별도로 설치하세요 (https://pytorch.org 참조)
```

간단 사용 예 (OpenAI API):

```python
from src import transcribe

text = transcribe.transcribe_with_openai_api('audio.mp3')
print(text)
```

로컬 Whisper 사용 예:

```python
from src import transcribe

text = transcribe.transcribe_local_whisper('audio.mp3', model_name='small')
print(text)
```

참고: OpenAI API를 쓰려면 `OPENAI_API_KEY` 환경변수를 설정하세요.
