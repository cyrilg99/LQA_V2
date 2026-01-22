import os
import io
import math
import tempfile
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

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
    # Fast-path text types
    for ext in (".txt", ".md", ".csv", ".tsv"):
        if name.endswith(ext):
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return data.decode("latin-1", errors="ignore")

    # Optional: DOCX
    if name.endswith(".docx"):
        try:
            import docx  # python-docx
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            doc = docx.Document(tmp_path)
            os.unlink(tmp_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.warning(f"Impossible de lire DOCX ({e}).")
            return ""

    # Optional: PDF
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

    # Fallback: try UTF-8
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


@st.cache_resource(show_spinner=False)
def get_sentence_splitter():
    """Load SaT sentence splitter (wtpsplit)."""
    try:
        from wtpsplit import SaT  # SaT — Segment Any Text
        sat = SaT("sat-3l-sm")
        return ("sat", sat)
    except Exception as e:
        # Lightweight regex as last resort
        return ("regex", None)


def split_sentences(text: str) -> List[str]:
    """Split into sentence-like segments using SaT if available, else regex."""
    mode, splitter = get_sentence_splitter()
    text = (text or "").strip()
    if not text:
        return []
    if mode == "sat":
        try:
            sents = splitter.split(text)
            # strip & drop empty
            return [s.strip() for s in sents if s and s.strip()]
        except Exception:
            pass
    # Regex fallback (approx.)
    parts = re.split(r'(?<=[\.\!\?\u2047\u203C\u203D\u3002\uff01\uff1f])\s+', text)
    return [p.strip() for p in parts if p and p.strip()]


# =========================
# Alignment: bertalign + fallbacks
# =========================
def try_run_bertalign(src_text: str, tgt_text: str) -> Optional[List[Tuple[str, str]]]:
    """Use bertalign if available (API compatible with bfsujason/bertalign)."""
    try:
        from bertalign import Bertalign  # pip install bertalign
        aligner = Bertalign(src_text, tgt_text)
        aligner.align_sents()
        pairs = []
        # aligner.result holds "beads"; _get_line reconstructs groups
        for bead in getattr(aligner, "result", []):
            src_line = aligner._get_line(bead[0], aligner.src_sents)  # type: ignore[attr-defined]
            tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)  # type: ignore[attr-defined]
            pairs.append((src_line.strip(), tgt_line.strip()))
        return pairs or None
    except Exception:
        return None


def align_sentences_dp(src_sents: List[str], tgt_sents: List[str]) -> List[Tuple[str, str]]:
    """Deterministic DP fallback inspired by ton script existant (longueur/penalty)."""
    ns, nt = len(src_sents), len(tgt_sents)
    INF = 10**9
    moves = [(1, 1), (1, 2), (2, 1), (1, 0), (0, 1)]
    dp = [[INF] * (nt + 1) for _ in range(ns + 1)]
    back = [[None] * (nt + 1) for _ in range(ns + 1)]
    dp[0][0] = 0
    for i in range(ns + 1):
        for j in range(nt + 1):
            if dp[i][j] >= INF:
                continue
            for di, dj in moves:
                ni, nj = i + di, j + dj
                if ni > ns or nj > nt:
                    continue
                s_len = sum(len(src_sents[k]) for k in range(i, ni))
                t_len = sum(len(tgt_sents[k]) for k in range(j, nj))
                cost = 0.0 if (s_len == 0 and t_len == 0) else (math.log(max(1, s_len)) - math.log(max(1, t_len))) ** 2
                penalty = 0.5 if (di + dj > 2) else 0.0
                newc = dp[i][j] + cost + penalty
                if newc < dp[ni][nj]:
                    dp[ni][nj] = newc
                    back[ni][nj] = (i, j)
    i, j = ns, nt
    pairs = []
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            s = " ".join(src_sents[:i]) if i else ""
            t = " ".join(tgt_sents[:j]) if j else ""
            pairs.insert(0, (s, t))
            break
        pi, pj = prev
        s_grp = " ".join(src_sents[pi:i])
        t_grp = " ".join(tgt_sents[pj:j])
        pairs.insert(0, (s_grp, t_grp))
        i, j = pi, pj
    return [(s.strip(), t.strip()) for s, t in pairs if s.strip() or t.strip()]


def align_texts(src_text: str, tgt_text: str) -> List[Tuple[str, str]]:
    """Try bertalign first; if not, split + DP fallback."""
    pairs = try_run_bertalign(src_text, tgt_text)
    if pairs is not None:
        return pairs
    src_sents = split_sentences(src_text)
    tgt_sents = split_sentences(tgt_text)
    return align_sentences_dp(src_sents, tgt_sents)


# =========================
# COMET / XCOMET scoring
# =========================
@st.cache_resource(show_spinner=True)
def load_comet_model(model_id: str):
    """Download + load a COMET model (Hugging Face login/licence may be required)."""
    from comet import download_model, load_from_checkpoint
    ckpt = download_model(model_id)   # returns .../checkpoints/model.ckpt
    model = load_from_checkpoint(ckpt)
    return model

def score_with_comet(model, data: List[dict], batch_size: int = 8, gpus: int = 0):
    """Run model.predict on list[{'src', 'mt', ('ref')?}] and return segment scores."""
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    # For COMET/XCOMET, .scores is segment-level array; system_score also available
    scores = out.scores
    meta = getattr(out, "metadata", {})
    return scores, getattr(out, "system_score", None), meta


# =========================
# UI — Uploads
# =========================
col_src, col_mt, col_ref = st.columns([1, 1, 1])

with col_src:
    st.subheader("Source")
    up_src = st.file_uploader("Téléverser le fichier **source**", key="src_file")

with col_mt:
    st.subheader("Traduction")
    up_mt = st.file_uploader("Téléverser la **traduction** (MT ou humaine)", key="mt_file")

with col_ref:
    st.subheader("Référence (optionnelle)")
    up_ref = st.file_uploader("Téléverser la **référence** (pour XCOMET)", key="ref_file")

if up_src and up_mt:
    src_text = read_uploaded_text(up_src)
    mt_text  = read_uploaded_text(up_mt)
    ref_text = read_uploaded_text(up_ref) if up_ref else None

    with st.expander("Aperçu (source / traduction / référence)", expanded=False):
        st.text_area("Source", src_text[:5000], height=150)
        st.text_area("Traduction", mt_text[:5000], height=150)
        if ref_text is not None:
            st.text_area("Référence", ref_text[:5000], height=150)

    st.markdown("---")
    st.subheader("Alignement")

    if st.button("Aligner maintenant", type="primary"):
        with st.spinner("Segmentation + alignement en cours…"):
            pairs = align_texts(src_text, mt_text)
            df = pd.DataFrame(pairs, columns=["source", "traduction"])
            st.success(f"{len(df)} paires alignées.")

        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("---")
        st.subheader("Évaluation")

        # Choix du modèle
        metrics = ["COMETKiwi (QE, sans référence)", "XCOMET-XL (avec référence)"]
        choice = st.selectbox("Choisir la métrique", options=metrics)

        # Déterminer modèle et données
        model_id = None
        needs_ref = False
        if choice.startswith("COMETKiwi"):
            model_id = "Unbabel/wmt22-cometkiwi-da"   # QE: src + mt
            needs_ref = False
        else:
            model_id = "Unbabel/XCOMET-XL"           # Explainable: src + mt + ref
            needs_ref = True

        if needs_ref and (ref_text is None):
            st.warning("XCOMET requiert une **référence**. Téléverse une référence ou choisis COMETKiwi.")
        else:
            # Construire les items data à partir du tableau aligné
            data = []
            for _, row in df.iterrows():
                item = {"src": row["source"], "mt": row["traduction"]}
                if needs_ref:
                    item["ref"] = row["source"] if ref_text is None else row["source"]  # default: source as anchor
                data.append(item)

            # Charger modèle + prédire
            with st.spinner(f"Téléchargement/chargement du modèle ({model_id})…"):
                model = load_comet_model(model_id)

            with st.spinner("Évaluation des segments…"):
                # gpus=0 => CPU friendly
                scores, sys_score, meta = score_with_comet(model, data, batch_size=8, gpus=0)

            # Ajouter au tableau
            score_col = "score_qe" if not needs_ref else "score_xcomet"
            df[score_col] = scores

            st.success("Évaluation terminée.")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Score système (moyenne)", f"{(sys_score or df[score_col].mean()):.3f}")
            with c2:
                st.caption("Distribution des scores")
                st.bar_chart(df[score_col])

            st.dataframe(df.head(50), use_container_width=True)

            # Export
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                "Télécharger les paires alignées + scores (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="aligned_scored.csv",
                mime="text/csv",
            )
else:
    st.info("Téléverse au minimum **source** et **traduction** pour commencer.")

