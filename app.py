import os
import io
import math
import tempfile
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
import torch

# ---- UI config ----
st.set_page_config(page_title="LQA — Alignement + Évaluation (COMET/XCOMET)", layout="wide")
st.title("LQA — Upload • Alignement • Évaluation (COMET / XCOMET)")

# =========================
# Utils: I/O & preprocessing
# =========================
def read_uploaded_text(file) -> str:
    """Read text from uploaded file (.txt, .md, .csv as plain text; .docx optional; .pdf optional)."""
    name = (file.name or "").lower()
    data = file.getvalue()
    for ext in (".txt", ".md", ".csv", ".tsv"):
        if name.endswith(ext):
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
        except Exception as e:
            st.warning(f"Impossible de lire DOCX ({e}).")
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
        except Exception as e:
            st.warning(f"Impossible de lire PDF ({e}).")
            return ""

    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")

# =========================
# Alignment: Bertalign (Strict)
# =========================
def align_texts_strict_bertalign(src_text: str, tgt_text: str) -> List[Tuple[str, str]]:
    """
    Strictly uses Bertalign for alignment. 
    If it fails or is not installed, it raises an exception.
    """
    try:
        from bertalign import Bertalign
        
        # Initialize Bertalign
        # It handles sentence splitting internally or uses the text provided.
        aligner = Bertalign(src_text, tgt_text)
        aligner.align_sents()
        
        pairs = []
        if hasattr(aligner, "result") and aligner.result:
            for bead in aligner.result:
                # Reconstruct the aligned lines based on indices
                src_line = aligner._get_line(bead[0], aligner.src_sents)
                tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
                pairs.append((src_line.strip(), tgt_line.strip()))
            return pairs
        else:
            raise ValueError("Bertalign n'a produit aucun résultat.")
            
    except ImportError:
        st.error("❌ La bibliothèque 'bertalign' n'est pas installée. Exécutez : pip install bertalign")
        raise
    except Exception as e:
        st.error(f"❌ Erreur lors de l'alignement avec Bertalign : {str(e)}")
        raise

# =========================
# COMET / XCOMET scoring
# =========================
@st.cache_resource(show_spinner=True)
def load_comet_model(model_id: str):
    from comet import download_model, load_from_checkpoint
    ckpt = download_model(model_id)
    model = load_from_checkpoint(ckpt)
    return model

def score_with_comet(model, data: List[dict], batch_size: int = 8, gpus: int = 0):
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return out.scores, getattr(out, "system_score", None), getattr(out, "metadata", {})

# =========================
# UI — Main Logic
# =========================
if "aligned_df" not in st.session_state:
    st.session_state.aligned_df = None

col_src, col_mt, col_ref = st.columns([1, 1, 1])

with col_src:
    st.subheader("Source")
    up_src = st.file_uploader("Fichier source", key="src_file")

with col_mt:
    st.subheader("Traduction")
    up_mt = st.file_uploader("Fichier traduction", key="mt_file")

with col_ref:
    st.subheader("Référence")
    up_ref = st.file_uploader("Référence (optionnelle)", key="ref_file")

if up_src and up_mt:
    # 1. READ TEXTS
    src_text = read_uploaded_text(up_src)
    mt_text  = read_uploaded_text(up_mt)
    ref_text = read_uploaded_text(up_ref) if up_ref else None

    st.markdown("---")
    
    # 2. METRIC SELECTION
    metrics = ["COMETKiwi (QE, sans référence)", "XCOMET-XL (avec référence)"]
    choice = st.selectbox("Choisir la métrique d'évaluation", options=metrics)
    needs_ref = "XCOMET" in choice
    model_id = "Unbabel/wmt22-cometkiwi-da" if not needs_ref else "Unbabel/XCOMET-XL"

    # 3. ACTION
    if st.button("Lancer Bertalign + Évaluation", type="primary"):
        if needs_ref and not ref_text:
            st.error("⚠️ XCOMET nécessite un fichier de référence.")
        else:
            try:
                # Step 1: Strict Bertalign
                with st.spinner("Alignement Bertalign (Source <> Traduction)..."):
                    pairs = align_texts_strict_bertalign(src_text, mt_text)
                    df = pd.DataFrame(pairs, columns=["source", "traduction"])

                # Step 2: Reference Alignment for XCOMET
                if needs_ref and ref_text:
                    with st.spinner("Alignement Bertalign (Source <> Référence)..."):
                        ref_pairs = align_texts_strict_bertalign(src_text, ref_text)
                        ref_map = {p[0]: p[1] for p in ref_pairs}
                        df["reference"] = df["source"].map(ref_map).fillna("")

                # Step 3: Scoring
                with st.spinner(f"Chargement de {model_id} et évaluation..."):
                    gpu_count = 1 if torch.cuda.is_available() else 0
                    model = load_comet_model(model_id)
                    
                    eval_data = []
                    for _, row in df.iterrows():
                        item = {"src": row["source"], "mt": row["traduction"]}
                        if needs_ref:
                            item["ref"] = row.get("reference", "")
                        eval_data.append(item)

                    scores, sys_score, meta = score_with_comet(model, eval_data, gpus=gpu_count)
                    df["score"] = scores
                    st.session_state.aligned_df = df
                    st.success("Analyse terminée avec succès.")

            except Exception:
                st.error("L'analyse a été interrompue en raison d'une erreur d'alignement.")

    # 4. RESULTS DISPLAY
    if st.session_state.aligned_df is not None:
        df = st.session_state.aligned_df
        
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Score Global (COMET)", f"{df['score'].mean():.3f}")
        
        # LQA Visual Thresholds
        low_threshold = 0.45
        
        def style_rows(row):
            if row['score'] < low_threshold:
                return ['background-color: #ffe6e6'] * len(row)
            return [''] * len(row)

        st.subheader("Analyse détaillée des segments")
        st.dataframe(df.style.apply(style_rows, axis=1), use_container_width=True)

        # Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les résultats (CSV)", data=csv, file_name="lqa_bertalign_results.csv", mime="text/csv")
else:
    st.info("Téléversez les fichiers requis pour activer Bertalign.")
