"""Gradio UI for Meeting Transcriber (sample implementation based on MD spec).

UI features (sample):
- Upload audio (.mp3/.wav)
- Select model (local-small/local-medium/openai-whisper-1)
- Select mode (local/api)
- Transcribe button -> outputs transcript, summary, action items

Notes:
- This is a lightweight demo UI. The actual transcription uses `src.transcribe` functions.
- If local Whisper or OpenAI API are not configured, the UI will show error messages returned by the transcribe functions.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from src import transcribe
from src import db as meeting_db


def _summarize_text(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # naive summary: first paragraph or first max_chars
    parts = text.split("\n\n")
    return parts[0][:max_chars] + ("..." if len(parts[0]) > max_chars else "")


def _extract_action_items(text: str) -> List[Dict]:
    # very naive rule-based extractor: look for lines that start with verbs or contain keywords
    if not text:
        return []
    items = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    idx = 1
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in ("action", "todo", "follow up", "follow-up", "assign", "할당", "해야")) or ln.endswith(".") and len(ln.split()) < 12:
            items.append({"id": idx, "summary": ln})
            idx += 1
        # also pick short lines starting with verbs (naive)
        elif len(ln.split()) < 6 and ln.split()[0].istitle():
            items.append({"id": idx, "summary": ln})
            idx += 1
    return items


def run_transcription(
    file_obj,
    model_select: str,
    mode_select: str,
    use_api_summary: bool,
    api_key: str = "",
) -> Tuple[str, str, List[Dict], List[Dict], str]:
    """Handler called by Gradio when user clicks Transcribe.

    file_obj: uploaded file object from Gradio (has .name attribute or is path)
    model_select: one of ['local-small','local-medium','openai-whisper-1']
    mode_select: 'local' or 'api'
    use_api_summary: whether to use OpenAI chat API for summarization
    api_key: OpenAI API key (if use_api_summary is True)
    """
    if not file_obj:
        return "", "", [], [], ""

    # Gradio File returns a tempfile with attribute `name` pointing to path.
    audio_path = getattr(file_obj, "name", None) or (file_obj if isinstance(file_obj, str) else None)
    if not audio_path or not os.path.exists(audio_path):
        return "Error: uploaded audio file not found.", "", [], [], ""

    try:
        if mode_select == "api":
            text = transcribe.transcribe_with_openai_api(audio_path)
            # Use API summary if requested
            if use_api_summary:
                try:
                    summary = transcribe.summarize_with_chat_api(text, api_key=api_key or None)
                except Exception as e:
                    summary = f"API 요약 오류: {e}\n\n" + _summarize_text(text)
            else:
                summary = _summarize_text(text)
            actions = _extract_action_items(text)
            # API 모드에선 세그먼트가 없을 수 있으니 전체를 하나의 세그먼트로 저장
            api_segments = [{"start": 0.0, "end": 0.0, "speaker": "SPEAKER_00", "text": text}]
            return text, summary, actions, api_segments, audio_path
        else:
            # model_select like 'local-small' -> model_name 'small'
            # model_select like 'local-small' -> model_name 'small'
            model_name = model_select.split("-")[-1]
            # Prefer pyannote-based diarization if available; fall back to heuristic implementation
            try:
                segments = transcribe.transcribe_local_whisper_with_pyannote(audio_path, whisper_model_name=model_name)
            except Exception:
                segments = transcribe.transcribe_local_whisper_with_speakers(audio_path, model_name=model_name)

            # format speaker separated transcript into a readable string
            parts = []
            for seg in segments:
                speaker = seg.get("speaker")
                text_seg = seg.get("text", "").strip()
                parts.append(f"[{speaker}] {text_seg}")
            full_text = "\n".join(parts)

            # Use API summary if requested
            if use_api_summary:
                try:
                    summary = transcribe.summarize_with_chat_api(full_text, api_key=api_key or None)
                except Exception as e:
                    summary = f"API 요약 오류: {e}\n\n" + _summarize_text(full_text)
            else:
                summary = _summarize_text(full_text)
            actions = _extract_action_items(full_text)
            return full_text, summary, actions, segments, audio_path
    except Exception as exc:
        return f"Error during transcription: {exc}", "", [], [], ""


def create_app():
    # import gradio lazily so module import won't fail when gradio isn't installed
    import gradio as gr

    # DB 준비
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw", "meetings")
    DB_PATH = os.path.join(DATA_DIR, "meetings.db")
    os.makedirs(RAW_DIR, exist_ok=True)
    meeting_db.init_db(DB_PATH)

    with gr.Blocks(title="Meeting Transcriber") as demo:
        gr.Markdown("# Meeting Transcriber")

        # 페이지 전환용 상태
        page_state = gr.State(value="list")  # 'list' | 'transcribe'

        # 공용 상태: 마지막 결과 보관
        last_audio_path = gr.State(value="")
        last_segments = gr.State(value=[])
        last_transcript_text = gr.State(value="")

        # ----------------------
        # 회의록 목록 페이지
        # ----------------------
        with gr.Group(visible=True) as page_list:
            gr.Markdown("## 회의록 목록")
            with gr.Row():
                new_btn = gr.Button("➕ 새로 만들기", variant="primary")
                refresh_btn = gr.Button("🔄 새로고침")

            meetings_dropdown = gr.Dropdown(label="저장된 회의록", choices=[], value=None, interactive=True)
            open_btn = gr.Button("열기")

            # 미리보기 영역
            with gr.Row():
                with gr.Column(scale=1):
                    preview_transcript = gr.Textbox(label="회의록 내용 (미리보기)", lines=14, interactive=False)
                with gr.Column(scale=1):
                    preview_audio = gr.Audio(label="오디오 재생", interactive=False)
                    preview_meta = gr.JSON(label="메타정보", value={})

        # ----------------------
        # 회의록 작성(기존 Transcribe) 페이지
        # ----------------------
        with gr.Group(visible=False) as page_transcribe:
            gr.Markdown("## 회의록 작성")
            with gr.Row():
                back_to_list_btn = gr.Button("← 목록으로")
                save_btn = gr.Button("💾 저장", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Input methods**: upload an audio file or record with your microphone (browser).")
                    audio_upload = gr.File(label="Upload audio", file_types=[".mp3", ".wav"], interactive=True)
                    mic_record = gr.Audio(sources=["microphone"], type="filepath", label="Record from microphone")
                    model_select = gr.Dropdown(choices=["local-small", "local-medium", "openai-whisper-1"], value="local-small", label="Model")
                    mode_select = gr.Radio(choices=["local", "api"], value="local", label="Mode")

                    # Summary options
                    gr.Markdown("**Summary Options**")
                    use_api_summary = gr.Checkbox(label="Use API-based Summary (GitHub Copilot/OpenAI)", value=False)
                    api_key_input = gr.Textbox(label="OpenAI API Key (optional, uses env if empty)", type="password", value="")

                    transcribe_btn = gr.Button("Transcribe")
                with gr.Column(scale=1):
                    transcript = gr.Textbox(label="Transcript", lines=12, interactive=False)
                    summary = gr.Textbox(label="Summary", lines=6, interactive=False)
                    action_items = gr.JSON(label="Action Items")

            # 저장 패널: 전사 후 자동 표시. 단일 '저장' 버튼으로 저장 수행.
            with gr.Group(visible=False) as save_panel:
                gr.Markdown("### 저장 정보")
                save_name = gr.Textbox(label="회의록 이름", placeholder="예: 2025-11-03 제품 기획 회의")
                save_date = gr.Textbox(label="회의록 날짜 (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
                gr.Markdown("참석자 매핑을 편집하세요 (SPEAKER_ID → 이름). 비워두면 원래 라벨을 사용합니다.")
                participants_df = gr.Dataframe(headers=["speaker_id", "name"], datatype=["str", "str"], row_count=(0, "dynamic"), col_count=(2, "fixed"), label="참석자 매핑")
                save_feedback = gr.Markdown(visible=False)

        # 헬퍼: 목록 로드
        def _load_meetings():
            items = meeting_db.list_meetings(DB_PATH)
            # 표시용 텍스트와 실제 value(id)를 분리하기 어렵다면 choices에 튜플 대신 문자열 사용
            display = [f"{it['id']}: {it['name']} ({it['meeting_date']})" for it in items]
            values = [str(it['id']) for it in items]
            # Gradio Dropdown은 (label, value) 페어를 직접 지원하지 않으므로 label을 value로 통일하고 내부적으로 id 파싱
            return display

        def _extract_id_from_choice(choice: Optional[str]) -> Optional[int]:
            if not choice:
                return None
            try:
                return int(choice.split(":", 1)[0])
            except Exception:
                return None

        # 목록 새로고침
        def refresh_list():
            return gr.update(choices=_load_meetings(), value=None)

        refresh_btn.click(fn=refresh_list, outputs=[meetings_dropdown])

        # 새로 만들기 -> 작성 페이지로 전환
        def go_to_transcribe():
            return gr.update(visible=False), gr.update(visible=True), "transcribe"

        new_btn.click(fn=go_to_transcribe, outputs=[page_list, page_transcribe, page_state])

        # 목록으로 돌아가기
        def go_to_list():
            return gr.update(visible=True), gr.update(visible=False), "list", refresh_list()

        back_to_list_btn.click(fn=go_to_list, outputs=[page_list, page_transcribe, page_state, meetings_dropdown])

        # 업로드/녹음 선택 후 전사 실행
        def _choose_and_run(uploaded, recorded, model, mode, use_api_sum, api_key):
            audio_path = None
            if recorded:
                audio_path = recorded
            elif uploaded:
                if isinstance(uploaded, dict) and 'name' in uploaded:
                    audio_path = uploaded['name']
                else:
                    audio_path = getattr(uploaded, 'name', None) or uploaded

            if not audio_path:
                return "", "", [], [], "", "", gr.update(visible=False), gr.update(value=[])

            t, s, a, segs, apath = run_transcription(type('F', (), {'name': audio_path}), model, mode, use_api_sum, api_key)
            # full_text, summary, actions, segments, audio_path
            # 참가자 기본 행 구성: 고유 speaker_id 목록
            uniq = []
            seen = set()
            for seg in (segs or []):
                sid = seg.get("speaker")
                if sid and sid not in seen:
                    seen.add(sid)
                    uniq.append([sid, ""])  # 이름은 비워둠
            return t, s, a, segs, apath, t, gr.update(visible=True), gr.update(value=uniq)

        transcribe_btn.click(
            fn=_choose_and_run,
            inputs=[audio_upload, mic_record, model_select, mode_select, use_api_summary, api_key_input],
            outputs=[transcript, summary, action_items, last_segments, last_audio_path, last_transcript_text, save_panel, participants_df],
        )

        # 저장 실행
        def _parse_participants_from_df(rows: List[List[str]]) -> Dict[str, str]:
            mapping: Dict[str, str] = {}
            for row in (rows or []):
                if not row:
                    continue
                sid = (row[0] or "").strip()
                nm = (row[1] or "").strip() if len(row) > 1 else ""
                if sid:
                    mapping[sid] = nm
            return mapping

        def do_save(name: str, date_text: str, participants_rows: List[List[str]], segs: List[Dict], apath: str, full_text: str):
            if not name:
                name = f"회의록_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                datetime.strptime(date_text, "%Y-%m-%d")
            except Exception:
                date_text = datetime.now().strftime("%Y-%m-%d")

            # 참가자 매핑 파싱 및 스피커 이름 치환
            pmap = _parse_participants_from_df(participants_rows)
            # 세그먼트 복사 및 레이블 치환
            resolved_segments: List[Dict] = []
            for seg in (segs or []):
                new_seg = dict(seg)
                label = new_seg.get("speaker")
                if label in pmap and pmap[label]:
                    new_seg["speaker"] = pmap[label]
                resolved_segments.append(new_seg)

            # 텍스트도 치환
            resolved_text = "\n".join([f"[{seg.get('speaker')}] {seg.get('text','').strip()}" for seg in resolved_segments]) if resolved_segments else full_text

            # 오디오 복사
            saved_audio_path = None
            if apath and os.path.exists(apath):
                base = os.path.basename(apath)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = os.path.join(RAW_DIR, f"{timestamp}_{base}")
                try:
                    shutil.copy2(apath, dest)
                    saved_audio_path = dest
                except Exception:
                    saved_audio_path = apath  # 복사 실패 시 원본 경로 보관

            # DB 저장
            meeting_id = meeting_db.create_meeting(
                DB_PATH,
                name=name,
                meeting_date=date_text,
                audio_path=saved_audio_path,
                transcript_text=resolved_text,
                segments=resolved_segments or (segs or []),
                participants=pmap,
                meta={"source": "ui_app"},
            )

            # 저장 후: 목록 페이지로 전환 + 목록 갱신
            list_choices = _load_meetings()
            success_msg = f"저장 완료 (ID: {meeting_id})"
            return (
                gr.update(visible=True),   # show list page
                gr.update(visible=False),  # hide transcribe page
                "list",
                gr.update(choices=list_choices, value=None),
                success_msg,
                gr.update(visible=False),  # hide save panel
            )

        # 단일 저장 버튼: 입력값으로 즉시 저장 수행
        save_btn.click(
            fn=do_save,
            inputs=[save_name, save_date, participants_df, last_segments, last_audio_path, last_transcript_text],
            outputs=[page_list, page_transcribe, page_state, meetings_dropdown, save_feedback, save_panel],
        )

        # 회의록 열기
        def open_meeting(choice: Optional[str]):
            mid = _extract_id_from_choice(choice)
            if not mid:
                return "", None, {}
            data = meeting_db.get_meeting(DB_PATH, mid)
            if not data:
                return "", None, {}
            return data.get("transcript_text") or "", data.get("audio_path"), {
                "name": data.get("name"),
                "meeting_date": data.get("meeting_date"),
                "participants": data.get("participants_json"),
                "created_at": data.get("created_at"),
            }

        open_btn.click(fn=open_meeting, inputs=[meetings_dropdown], outputs=[preview_transcript, preview_audio, preview_meta])

        # 초기 목록 로드
        demo.load(fn=lambda: gr.update(choices=_load_meetings()), outputs=[meetings_dropdown])

    return demo


if __name__ == "__main__":
    app = create_app()
    # default: localhost; set share=True if you want an external link (beware security)
    app.launch()
