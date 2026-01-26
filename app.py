import os
import io
import math
import tempfile
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

# Safety check for imports to guide the user
try:
    import torch
    from comet import download_model, load_from_checkpoint
    from bertalign import Bertalign
except ImportError as e:
    st.error(f"Erreur de dépendance : {e}. Assurez-vous d'avoir un fichier requirements.txt.")
    st.stop()

# ---- UI config ----
st.set_page_config(page_title="LQA — Bertalign + COMET", layout="wide")
st.title("LQA — Alignement Strict • Évaluation (COMET / XCOMET)")

# =========================
# Utils: I/O
# =========================
def read_uploaded_text(file) -> str:
    """Read text from uploaded file (.txt, .md, .csv, .docx, .pdf)."""
    name = (file.name or "").lower()
    data = file.getvalue()
    
    if name.endswith((".txt", ".md", ".csv", ".tsv")):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")

    if name.endswith(".docx"):
        try:
            import docx
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            doc = docx.Document(tmp_path)
            os.unlink(tmp_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            return ""

    if name.endswith(".pdf"):
        try:
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
        except Exception:
            return ""

    return data.decode("utf-8", errors="ignore")

# =========================
# Alignment: Strict Bertalign
# =========================
def align_texts_strict(src_text: str, tgt_text: str) -> List[Tuple[str, str]]:
    """Strictly uses Bertalign; fails if alignment cannot be performed."""
    try:
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
            raise ValueError("Bertalign n'a produit aucun résultat.")
    except Exception as e:
        st.error(f"Erreur Bertalign : {e}")
        raise e

# =========================
# COMET Scoring
# =========================
@st.cache_resource(show_spinner=True)
def load_comet_model(model_id: str):
    ckpt = download_model(model_id)
    return load_from_checkpoint(ckpt)

def score_with_comet(model, data: List[dict], batch_size: int = 8):
    gpu_count = 1 if torch.cuda.is_available() else 0
    out = model.predict(data, batch_size=batch_size, gpus=gpu_count)
    return out.scores

# =========================
# UI Main Logic
# =========================
if "aligned_df" not in st.session_state:
    st.session_state.aligned_df = None

col_src, col_mt, col_ref = st.columns(3)

with col_src:
    up_src = st.file_uploader("Source", key="src")
with col_mt:
    up_mt = st.file_uploader("Traduction", key="mt")
with col_ref:
    up_ref = st.file_uploader("Référence (XCOMET)", key="ref")

if up_src and up_mt:
    src_txt = read_uploaded_text(up_src)
    mt_txt = read_uploaded_text(up_mt)
    ref_txt = read_uploaded_text(up_ref) if up_ref else None

    metrics = ["COMETKiwi (Sans réf)", "XCOMET-XL (Avec réf)"]
    choice = st.selectbox("Métrique", options=metrics)
    needs_ref = "XCOMET" in choice
    model_id = "Unbabel/wmt22-cometkiwi-da" if not needs_ref else "Unbabel/XCOMET-XL"

    if st.button("Lancer l'analyse", type="primary"):
        if needs_ref and not ref_txt:
            st.error("Référence manquante.")
        else:
            # 1. Align MT
            with st.spinner("Alignement MT..."):
                pairs = align_texts_strict(src_txt, mt_txt)
                df = pd.DataFrame(pairs, columns=["source", "traduction"])
            
            # 2. Align Ref if needed
            if needs_ref:
                with st.spinner("Alignement Référence..."):
                    ref_pairs = align_texts_strict(src_txt, ref_txt)
                    ref_map = {p[0]: p[1] for p in ref_pairs}
                    df["reference"] = df["source"].map(ref_map).fillna("")

            # 3. Score
            with st.spinner("Évaluation..."):
                model = load_comet_model(model_id)
                eval_data = [{"src": r.source, "mt": r.traduction, "ref": r.get("reference", "")} for r in df.itertuples()]
                df["score"] = score_with_comet(model, eval_data)
                st.session_state.aligned_df = df

    if st.session_state.aligned_df is not None:
        df = st.session_state.aligned_df
        st.metric("Score Global", f"{df['score'].mean():.3f}")
        
        # Highlight low scores
        st.dataframe(df.style.background_gradient(subset=['score'], cmap="RdYlGn"), use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger CSV", data=csv, file_name="lqa_results.csv")
