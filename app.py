import os
import io
import tempfile
from typing import List, Tuple
import pandas as pd
import streamlit as st
import torch
import gc

# === HUGGING FACE AUTHENTICATION ===
if "HUGGINGFACE_TOKEN" in st.secrets:
    from huggingface_hub import login
    try:
        login(token=st.secrets["HUGGINGFACE_TOKEN"])
    except Exception as e:
        st.warning(f"⚠️ HF authentication issue: {e}")

# --- Dependencies Check ---
try:
    from bertalign import Bertalign
    from comet import download_model, load_from_checkpoint
except ImportError as e:
    st.error(f"Missing dependency: {e}. Check your requirements.txt.")
    st.stop()

# ---- UI config ----
st.set_page_config(page_title="LQA — Bertalign + COMET", layout="wide")
st.title("LQA — Alignement Strict • Évaluation (COMET / XCOMET)")

# =========================
# Memory Management
# =========================
def clear_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        aligner = Bertalign(src_text, tgt_text)
        aligner.align_sents()
        
        pairs = []
        if hasattr(aligner, "result") and aligner.result:
            for bead in aligner.result:
                src_line = aligner._get_line(bead[0], aligner.src_sents)
                tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
                if src_line.strip() and tgt_line.strip():
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
    """Load COMET model with memory management"""
    try:
        clear_memory()
        st.info(f"Loading model: {model_id}")
        ckpt = download_model(model_id)
        model = load_from_checkpoint(ckpt)
        st.success(f"✅ Model loaded: {model_id}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model '{model_id}'")
        st.error(str(e))
        
        if "memory" in str(e).lower() or "oom" in str(e).lower():
            st.error("💾 OUT OF MEMORY ERROR")
            st.info("Try one of these solutions:")
            st.markdown("""
            1. **Use XCOMET-lite** (278M params - much lighter!)
            2. **Process fewer segments** (split your file)
            3. **Use a smaller model** (COMET-22 instead of XCOMET-XL)
            4. **Upgrade Streamlit Cloud** (paid tier has more RAM)
            """)
        elif "401" in str(e) or "gated" in str(e).lower():
            st.error("🔒 Authentication required for this model")
            st.info("Add your HF token to Streamlit Secrets")
        raise

def score_with_comet(model, data: list, batch_size: int = 1):
    """Score with COMET using minimal memory"""
    try:
        clear_memory()
        gpu_count = 1 if torch.cuda.is_available() else 0
        
        # Use batch_size=1 for maximum memory safety
        out = model.predict(data, batch_size=batch_size, gpus=gpu_count)
        return out.scores
    except Exception as e:
        if "memory" in str(e).lower() or "oom" in str(e).lower():
            st.error("💾 Out of memory during scoring!")
            st.info("The model loaded but ran out of memory during prediction.")
            st.markdown("""
            **Solutions:**
            - Reduce file size (process fewer segments)
            - Use XCOMET-lite (much more memory efficient)
            - Upgrade to Streamlit Cloud paid tier
            """)
        raise

# =========================
# Main UI Logic
# =========================
if "aligned_df" not in st.session_state:
    st.session_state.aligned_df = None

# Sidebar with model info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown("""
    ### Memory Requirements (approx)
    
    **Lightweight (< 1GB RAM):**
    - ✅ XCOMET-lite (278M params)
    - ✅ COMET-22 (560M params)
    
    **Medium (1-2GB RAM):**
    - ⚠️ COMETKiwi (560M params)
    
    **Heavy (2-4GB+ RAM):**
    - ❌ XCOMET-XL (10.7B params)
    - ❌ XCOMET-XXL (10.7B params)
    
    **Free Streamlit Cloud: 1GB RAM**
    
    Use lightweight models for best results!
    """)

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

    # Model selection - lightweight models first
    metrics = [
        "XCOMET-lite ⚡ (Fastest, 278M params)",
        "COMET-22 (Medium, 560M params)",
        "COMETKiwi (QE - No Reference, 560M params)",
        "XCOMET-XL ⚠️ (Heavy, may crash on free tier)"
    ]
    choice = st.selectbox(
        "Select Metric",
        options=metrics,
        help="XCOMET-lite recommended for Streamlit Cloud free tier"
    )
    
    # Map to model IDs
    if "XCOMET-lite" in choice:
        model_id = "Unbabel/XCOMET-lite"
        needs_ref = True
    elif "COMET-22" in choice:
        model_id = "Unbabel/wmt22-comet-da"
        needs_ref = True
    elif "COMETKiwi" in choice:
        model_id = "Unbabel/wmt22-cometkiwi-da"
        needs_ref = False
    else:  # XCOMET-XL
        model_id = "Unbabel/XCOMET-XL"
        needs_ref = True
        st.warning("⚠️ XCOMET-XL is very large and may crash on free tier. Consider XCOMET-lite instead.")

    # Show warning for large files
    if src_txt:
        src_lines = len(src_txt.split('\n'))
        if src_lines > 100:
            st.warning(f"⚠️ Large file detected ({src_lines} lines). Consider splitting into smaller chunks to avoid memory issues.")

    if st.button("Run LQA Analysis", type="primary"):
        if needs_ref and not ref_txt:
            st.error("Reference file is required for this model.")
        else:
            try:
                # Clear memory before starting
                clear_memory()
                
                with st.spinner("Aligning with Bertalign..."):
                    pairs = align_texts_strict(src_txt, mt_txt)
                    st.info(f"Aligned {len(pairs)} segment pairs")
                    df = pd.DataFrame(pairs, columns=["source", "translation"])
                
                if needs_ref:
                    with st.spinner("Aligning Reference..."):
                        ref_pairs = align_texts_strict(src_txt, ref_txt)
                        ref_map = {p[0]: p[1] for p in ref_pairs}
                        df["reference"] = df["source"].map(ref_map).fillna("")

                with st.spinner(f"Loading model... (this may take 2-3 minutes on first run)"):
                    model = load_comet_model(model_id)
                
                with st.spinner(f"Scoring {len(df)} segments..."):
                    eval_data = [
                        {
                            "src": r.source,
                            "mt": r.translation,
                            "ref": getattr(r, "reference", "")
                        }
                        for r in df.itertuples()
                    ]
                    
                    # Use batch_size=1 for maximum safety
                    df["score"] = score_with_comet(model, eval_data, batch_size=1)
                    st.session_state.aligned_df = df
                    
                    # Clear memory after scoring
                    clear_memory()
                    
                    st.success("✅ Analysis complete!")
                    
            except MemoryError as e:
                st.error("💾 OUT OF MEMORY!")
                st.error("Your file is too large for the available memory.")
                st.info("Try: 1) Use XCOMET-lite, 2) Split file into smaller chunks, 3) Upgrade Streamlit tier")
            except Exception as e:
                st.error("❌ Analysis failed")
                st.error(str(e))
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())

    if st.session_state.aligned_df is not None:
        df = st.session_state.aligned_df
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Score", f"{df['score'].mean():.4f}")
        with col2:
            st.metric("Segments", len(df))
        with col3:
            st.metric("Std Dev", f"{df['score'].std():.4f}")
        
        # Results table
        st.dataframe(
            df.style.background_gradient(subset=['score'], cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
            height=400
        )
        
        # Download
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False),
                "lqa_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            # Excel download
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            st.download_button(
                "⬇️ Download Excel",
                excel_buffer.getvalue(),
                "lqa_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
