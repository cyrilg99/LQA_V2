import os
import io
import tempfile
from typing import List, Tuple
import pandas as pd
import streamlit as st
import torch

# --- Dependencies Check ---
try:
    from bertalign import Bertalign
    from comet import download_model, load_from_checkpoint
    # NOTE: laser_encoders removed - not actually used in the code
except ImportError as e:
    st.error(f"Missing dependency: {e}. Check your requirements.txt.")
    st.stop()

# ---- UI config ----
st.set_page_config(page_title="LQA — Bertalign + COMET", layout="wide")
st.title("LQA — Alignement Strict • Évaluation (COMET / XCOMET)")

# =========================
# Utils: I/O
# =========================
def read_uploaded_text(file) -> str:
    name = (file.name or "").lower()
    data = file.getvalue()
    if name.endswith((".txt", ".md", ".csv", ".tsv")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        import docx
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        os.unlink(tmp_path)
        return "\n".join([p.text for p in doc.paragraphs])
    if name.endswith(".pdf"):
        import pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        acc = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                acc.append(page.extract_text() or "")
        os.unlink(tmp_path)
        return "\n".join(acc)
    return data.decode("utf-8", errors="ignore")

# =========================
# Alignment: Strict Bertalign
# =========================
def align_texts_strict(src_text: str, tgt_text: str) -> List[Tuple[str, str]]:
    try:
        # Bertalign handles sentence splitting internally
        aligner = Bertalign(src_text, tgt_text)
        aligner.align_sents()
        
        pairs = []
        if hasattr(aligner, "result") and aligner.result:
            for bead in aligner.result:
                src_line = aligner._get_line(bead[0], aligner.src_sents)
                tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
                pairs.append((src_line.strip(), tgt_line.strip()))
            return pairs
        else:
            raise ValueError("Bertalign produced no results.")
    except Exception as e:
        st.error(f"Bertalign Alignment Error: {e}")
        raise e

# =========================
# COMET Scoring
# =========================
@st.cache_resource(show_spinner=True)
def load_comet_model(model_id: str):
    ckpt = download_model(model_id)
    return load_from_checkpoint(ckpt)

def score_with_comet(model, data: list):
    gpu_count = 1 if torch.cuda.is_available() else 0
    # Use small batch size for Streamlit Cloud RAM limits
    out = model.predict(data, batch_size=4, gpus=gpu_count)
    return out.scores

# =========================
# Main UI Logic
# =========================
if "aligned_df" not in st.session_state:
    st.session_state.aligned_df = None

col_src, col_mt, col_ref = st.columns(3)
with col_src:
    up_src = st.file_uploader("Source File", key="src")
with col_mt:
    up_mt = st.file_uploader("MT Output", key="mt")
with col_ref:
    up_ref = st.file_uploader("Reference (Optional)", key="ref")

if up_src and up_mt:
    src_txt = read_uploaded_text(up_src)
    mt_txt = read_uploaded_text(up_mt)
    ref_txt = read_uploaded_text(up_ref) if up_ref else None

    metrics = ["COMETKiwi (QE)", "XCOMET-XL (Reference-based)"]
    choice = st.selectbox("Select Metric", options=metrics)
    needs_ref = "XCOMET" in choice
    model_id = "Unbabel/wmt22-cometkiwi-da" if not needs_ref else "Unbabel/XCOMET-XL"

    if st.button("Run LQA Analysis", type="primary"):
        if needs_ref and not ref_txt:
            st.error("Reference file is required for XCOMET.")
        else:
            with st.spinner("Aligning with Bertalign..."):
                pairs = align_texts_strict(src_txt, mt_txt)
                df = pd.DataFrame(pairs, columns=["source", "translation"])
            
            if needs_ref:
                with st.spinner("Aligning Reference..."):
                    ref_pairs = align_texts_strict(src_txt, ref_txt)
                    ref_map = {p[0]: p[1] for p in ref_pairs}
                    df["reference"] = df["source"].map(ref_map).fillna("")

            with st.spinner("Scoring with COMET..."):
                model = load_comet_model(model_id)
                eval_data = [{"src": r.source, "mt": r.translation, "ref": getattr(r, "reference", "")} for r in df.itertuples()]
                df["score"] = score_with_comet(model, eval_data)
                st.session_state.aligned_df = df

    if st.session_state.aligned_df is not None:
        df = st.session_state.aligned_df
        st.metric("System Average Score", f"{df['score'].mean():.4f}")
        st.dataframe(df.style.background_gradient(subset=['score'], cmap="RdYlGn"), use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "lqa_results.csv")
