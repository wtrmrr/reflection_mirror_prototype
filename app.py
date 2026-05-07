import os
import json
import uuid
import asyncio
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from hume import AsyncHumeClient
from hume.expression_measurement.stream import (
    Config,
    StreamLanguage,
)

# --------------------------------------------------
# App setup
# --------------------------------------------------

st.set_page_config(
    page_title="Reflection Mirror Prototype",
    page_icon="🪞",
    layout="wide",
)

st.title("🪞 Reflection Mirror Prototype")
st.caption(
    "Prototype for expression-informed reflection using Hume Expression Measurement."
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_api_key() -> Optional[str]:
    return os.getenv("HUME_API_KEY") or st.secrets.get("HUME_API_KEY", None)

def ensure_session_state():
    if "saved_runs" not in st.session_state:
        st.session_state.saved_runs = []

def safe_get(obj: Any, path: List[str], default=None):
    """
    Safely access nested attributes or dict keys.
    """
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur

def object_to_jsonable(obj: Any):
    """
    Convert SDK objects to something st.json can display.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [object_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: object_to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {
            k: object_to_jsonable(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    return str(obj)

def top_emotions_from_prediction(prediction: Any, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Pull top emotions from a single prediction object.
    Expects prediction.emotions to be a list with .name and .score.
    """
    emotions = safe_get(prediction, ["emotions"], []) or []
    normalized = []
    for emotion in emotions:
        name = safe_get(emotion, ["name"], "unknown")
        score = safe_get(emotion, ["score"], 0.0)
        normalized.append({"label": name, "score": float(score)})
    normalized.sort(key=lambda x: x["score"], reverse=True)
    return normalized[:top_n]

def parse_language_predictions(result: Any) -> List[Dict[str, Any]]:
    """
    Parse result.language.predictions from Hume stream send_text or send_file.
    """
    preds = safe_get(result, ["language", "predictions"], []) or []
    rows = []
    for p in preds:
        rows.append(
            {
                "modality": "language",
                "text": safe_get(p, ["text"], ""),
                "time_begin": safe_get(p, ["time", "begin"], None),
                "time_end": safe_get(p, ["time", "end"], None),
                "top_emotions": top_emotions_from_prediction(p),
            }
        )
    return rows

def parse_prosody_predictions(result: Any) -> List[Dict[str, Any]]:
    """
    Parse result.prosody.predictions from send_file.
    """
    preds = safe_get(result, ["prosody", "predictions"], []) or []
    rows = []
    for p in preds:
        rows.append(
            {
                "modality": "prosody",
                "text": safe_get(p, ["text"], ""),
                "time_begin": safe_get(p, ["time", "begin"], None),
                "time_end": safe_get(p, ["time", "end"], None),
                "top_emotions": top_emotions_from_prediction(p),
            }
        )
    return rows

def flatten_top_signals(parsed_rows: List[Dict[str, Any]], top_n: int = 8) -> List[Dict[str, Any]]:
    """
    Aggregate emotion scores across parsed rows into a single top-signals view.
    """
    score_map: Dict[str, List[float]] = {}
    for row in parsed_rows:
        for item in row.get("top_emotions", []):
            score_map.setdefault(item["label"], []).append(item["score"])

    merged = []
    for label, scores in score_map.items():
        merged.append(
            {
                "label": label,
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "count": len(scores),
            }
        )

    merged.sort(key=lambda x: (x["avg_score"], x["max_score"]), reverse=True)
    return merged[:top_n]

def build_reflection_text(top_signals: List[Dict[str, Any]]) -> str:
    """
    Lightweight reflection language.
    """
    if not top_signals:
        return (
            "No strong observer-facing signal was extracted from this sample. "
            "Try a longer input or a different modality."
        )

    labels = [x["label"] for x in top_signals[:3]]
    joined = ", ".join(labels[:-1]) + (f", and {labels[-1]}" if len(labels) > 1 else labels[0])

    return (
        f"This sample may come across with signals related to {joined}. "
        "Treat this as a possible interpretation of expression, not a direct readout of internal emotion."
    )

def signals_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for row in rows:
        top = row.get("top_emotions", [])
        best_label = top[0]["label"] if top else None
        best_score = top[0]["score"] if top else None
        flat.append(
            {
                "modality": row.get("modality"),
                "text": row.get("text"),
                "time_begin": row.get("time_begin"),
                "time_end": row.get("time_end"),
                "top_label": best_label,
                "top_score": best_score,
            }
        )
    return pd.DataFrame(flat)

async def analyze_text_with_hume(api_key: str, text: str) -> Dict[str, Any]:
    """
    Stream text to Hume Expression Measurement.
    """
    client = AsyncHumeClient(api_key=api_key)

    async with client.expression_measurement.stream.connect() as socket:
        result = await socket.send_text(
            text=text,
            config=Config(
                language=StreamLanguage(granularity="sentence")
            ),
        )

    parsed_language = parse_language_predictions(result)
    top_signals = flatten_top_signals(parsed_language)

    return {
        "kind": "text",
        "timestamp": datetime.utcnow().isoformat(),
        "input_preview": text[:200],
        "raw_result": object_to_jsonable(result),
        "parsed_rows": parsed_language,
        "top_signals": top_signals,
        "reflection": build_reflection_text(top_signals),
    }

async def analyze_file_with_hume(
    api_key: str,
    file_path: str,
    include_language: bool = True,
    include_prosody: bool = True,
    include_burst: bool = False,
    include_face: bool = False,
) -> Dict[str, Any]:
    client = AsyncHumeClient(api_key=api_key)

    config = Config(
        language=StreamLanguage(granularity="sentence") if include_language else None,
        prosody={} if include_prosody else None,
        burst={} if include_burst else None,
        face={} if include_face else None,
    )

    async with client.expression_measurement.stream.connect() as socket:
        result = await socket.send_file(
            file_path,
            config=config,
        )

    parsed_rows = []
    parsed_rows.extend(parse_language_predictions(result))
    parsed_rows.extend(parse_prosody_predictions(result))
    parsed_rows.extend(parse_burst_predictions(result))
    parsed_rows.extend(parse_face_predictions(result))

    top_signals = flatten_top_signals(parsed_rows)

    return {
        "kind": "media",
        "timestamp": datetime.utcnow().isoformat(),
        "input_preview": os.path.basename(file_path),
        "raw_result": object_to_jsonable(result),
        "parsed_rows": parsed_rows,
        "top_signals": top_signals,
        "reflection": build_reflection_text(top_signals),
    }

def save_run(run: Dict[str, Any]):
    run_copy = dict(run)
    run_copy["id"] = str(uuid.uuid4())[:8]
    st.session_state.saved_runs.insert(0, run_copy)

def render_run_summary(run: Dict[str, Any], title: str = "Result"):
    st.subheader(title)

    top_signals = run.get("top_signals", [])
    if top_signals:
        cols = st.columns(min(4, len(top_signals)))
        for i, sig in enumerate(top_signals[:4]):
            cols[i].metric(
                label=sig["label"],
                value=f"{sig['avg_score']:.3f}",
                help=f"Seen in {sig['count']} segment(s)",
            )
    else:
        st.info("No top signals available yet.")

    st.markdown("**Possible read**")
    st.write(run.get("reflection", ""))

    parsed_rows = run.get("parsed_rows", [])
    if parsed_rows:
        df = signals_to_dataframe(parsed_rows)
        st.markdown("**Segments**")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No parsed segment rows available.")

    with st.expander("Raw JSON"):
        st.json(run.get("raw_result", {}))

def parse_burst_predictions(result: Any) -> List[Dict[str, Any]]:
    preds = safe_get(result, ["burst", "predictions"], []) or []
    rows = []
    for p in preds:
        rows.append(
            {
                "modality": "burst",
                "text": safe_get(p, ["text"], ""),
                "time_begin": safe_get(p, ["time", "begin"], None),
                "time_end": safe_get(p, ["time", "end"], None),
                "top_emotions": top_emotions_from_prediction(p),
            }
        )
    return rows


def parse_face_predictions(result: Any) -> List[Dict[str, Any]]:
    preds = safe_get(result, ["face", "predictions"], []) or []
    rows = []
    for p in preds:
        rows.append(
            {
                "modality": "face",
                "text": safe_get(p, ["description"], "") or safe_get(p, ["text"], ""),
                "time_begin": safe_get(p, ["time", "begin"], None),
                "time_end": safe_get(p, ["time", "end"], None),
                "top_emotions": top_emotions_from_prediction(p),
                "action_units": object_to_jsonable(safe_get(p, ["action_units"], [])),
            }
        )
    return rows

def detect_media_kind(suffix: str) -> str:
    if suffix in [".mp3", ".wav", ".m4a"]:
        return "audio"
    if suffix in [".mp4", ".mov"]:
        return "video"
    if suffix in [".jpg", ".jpeg", ".png"]:
        return "image"
    return "unknown"


# --------------------------------------------------
# Main app
# --------------------------------------------------

ensure_session_state()
api_key = get_api_key()

if not api_key:
    st.error("Missing HUME_API_KEY. Add it as an environment variable or Streamlit secret.")
    st.stop()

st.info(
    "Use careful wording: this prototype reflects possible observer-facing interpretation of expression, not verified internal emotion."
)

tab_text, tab_media, tab_compare, tab_saved = st.tabs(
    ["Text reflection", "Audio/video reflection", "Compare", "Saved runs"]
)

# -------------------------
# Text tab
# -------------------------
with tab_text:
    st.markdown("### Paste text")
    text_input = st.text_area(
        "Enter text to analyze",
        height=180,
        placeholder="Example: I understand your concerns, but we need to move quickly.",
    )

    col1, col2 = st.columns([1, 4])
    analyze_text_btn = col1.button("Analyze text", type="primary")
    clear_text_btn = col2.button("Clear text")

    if clear_text_btn:
        st.rerun()

    if analyze_text_btn:
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing text with Hume..."):
                try:
                    run = asyncio.run(analyze_text_with_hume(api_key, text_input))
                    save_run(run)
                    render_run_summary(run, "Text reflection result")
                except Exception as e:
                    st.exception(e)

# -------------------------
# Media tab
# -------------------------
with tab_media:
    st.markdown("### Upload audio, video, or image")
    uploaded = st.file_uploader(
        "Choose a file",
        type=["mp3", "wav", "m4a", "mp4", "mov", "jpg", "jpeg", "png"],
        help="Streaming supports short audio/video payloads and images up to 3000 x 3000 pixels.",
    )

    colA, colB, colC, colD = st.columns(4)
    include_language = colA.checkbox("Analyze language", value=True)
    include_prosody = colB.checkbox("Analyze prosody", value=True)
    include_burst = colC.checkbox("Analyze vocal burst", value=False)
    include_face = colD.checkbox("Analyze facial expression", value=False)

    if uploaded is not None:
        suffix = "." + uploaded.name.split(".")[-1].lower()

        if suffix in [".mp3", ".wav", ".m4a"]:
            st.audio(uploaded)
            media_kind = "audio"
        elif suffix in [".mp4", ".mov"]:
            st.video(uploaded)
            media_kind = "video"
        elif suffix in [".jpg", ".jpeg", ".png"]:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
            media_kind = "image"
        else:
            media_kind = "unknown"

        if media_kind == "image" and (include_language or include_prosody or include_burst):
            st.warning(
                "For images, facial expression is the relevant model. Other selected models may return no useful output."
            )

        if st.button("Analyze media", type="primary"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    temp_path = tmp.name

                with st.spinner("Sending local file to Hume..."):
                    run = asyncio.run(
                        analyze_file_with_hume(
                            api_key=api_key,
                            file_path=temp_path,
                            include_language=include_language,
                            include_prosody=include_prosody,
                            include_burst=include_burst,
                            include_face=include_face,
                        )
                    )

                save_run(run)
                render_run_summary(run, "Media reflection result")

            except Exception as e:
                st.exception(e)

# -------------------------
# Compare tab
# -------------------------
with tab_compare:
    st.markdown("### Compare saved runs")

    saved_runs = st.session_state.saved_runs
    if len(saved_runs) < 2:
        st.info("Analyze at least two samples to compare them.")
    else:
        options = {
            f"{r['id']} | {r['kind']} | {r['timestamp']} | {r['input_preview'][:40]}": r
            for r in saved_runs
        }

        left_key = st.selectbox("First run", list(options.keys()), key="compare_left")
        right_key = st.selectbox("Second run", list(options.keys()), key="compare_right")

        left_run = options[left_key]
        right_run = options[right_key]

        c1, c2 = st.columns(2)

        with c1:
            render_run_summary(left_run, "First run")

        with c2:
            render_run_summary(right_run, "Second run")

        left_labels = {x["label"]: x for x in left_run.get("top_signals", [])}
        right_labels = {x["label"]: x for x in right_run.get("top_signals", [])}
        all_labels = sorted(set(left_labels.keys()) | set(right_labels.keys()))

        comparison_rows = []
        for label in all_labels:
            comparison_rows.append(
                {
                    "label": label,
                    "left_avg_score": left_labels.get(label, {}).get("avg_score"),
                    "right_avg_score": right_labels.get(label, {}).get("avg_score"),
                }
            )

        st.markdown("### Signal comparison")
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

# -------------------------
# Saved runs tab
# -------------------------
with tab_saved:
    st.markdown("### Session history")

    saved_runs = st.session_state.saved_runs
    if not saved_runs:
        st.info("No runs saved yet.")
    else:
        for run in saved_runs:
            with st.expander(
                f"{run['id']} | {run['kind']} | {run['timestamp']} | {run['input_preview'][:60]}"
            ):
                render_run_summary(run, "Saved result")

                json_str = json.dumps(run["raw_result"], indent=2, default=str)
                st.download_button(
                    label=f"Download JSON for {run['id']}",
                    data=json_str,
                    file_name=f"reflection_run_{run['id']}.json",
                    mime="application/json",
                )