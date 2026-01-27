import os
import io
import tempfile
from typing import List, Tuple
import pandas as pd
import streamlit as st
import torch
import gc

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

# Display info banner
st.info("💡 **Using public COMET models** - No authentication required!")

# =========================
# Memory Management
# =========================
def clear_memory():
    """Force garbage collection"""
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
        st.info(f"📥 Downloading model: {model_id}")
        st.info("⏳ First run may take 2-3 minutes...")
        ckpt = download_model(model_id)
        model = load_from_checkpoint(ckpt)
        st.success(f"✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model '{model_id}'")
        st.error(str(e))
        raise

def score_with_comet(model, data: list):
    """Score with COMET using minimal memory"""
    try:
        clear_memory()
        gpu_count = 1 if torch.cuda.is_available() else 0
        
        # Use batch_size=1 for maximum safety on free tier
        with st.spinner(f"Scoring {len(data)} segments (this may take a few minutes)..."):
            out = model.predict(data, batch_size=1, gpus=gpu_count)
        return out.scores
    except Exception as e:
        st.error("Failed during scoring")
        st.error(str(e))
        raise

# =========================
# Main UI Logic
# =========================
if "aligned_df" not in st.session_state:
    st.session_state.aligned_df = None

# Sidebar with model info
with st.sidebar:
    st.header("ℹ️ Available Models")
    st.markdown("""
    ### Public Models (No Auth)
    
    All models below are **publicly available** 
    and work on Streamlit Cloud free tier:
    
    - ✅ **COMET-22** (~1GB RAM)
    - ✅ **eTranslation-COMET** (~1GB RAM)  
    - ✅ **COMET-20** (~800MB RAM)
    
    All require reference translation.
    
    ---
    
    ### Need QE (No Reference)?
    
    COMETKiwi requires authentication.
    
    To use it:
    1. Get HF token
    2. Request access to model
    3. Add token to Streamlit Secrets
    
    [See documentation →](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)
    """)

col_src, col_mt, col_ref = st.columns(3)
with col_src:
    up_src = st.file_uploader("📄 Source File", key="src")
with col_mt:
    up_mt = st.file_uploader("🔄 MT Output", key="mt")
with col_ref:
    up_ref = st.file_uploader("✅ Reference", key="ref", 
                              help="Reference translation (required)")

if up_src and up_mt:
    src_txt = read_uploaded_text(up_src)
    mt_txt = read_uploaded_text(up_mt)
    ref_txt = read_uploaded_text(up_ref) if up_ref else None

    # ONLY PUBLIC MODELS - NO AUTHENTICATION NEEDED
    st.subheader("Select Quality Metric")
    
    metrics = [
        "COMET-22 (wmt22-comet-da) - Recommended",
        "eTranslation-COMET - For EU Languages",
        "COMET-20 (wmt20-comet-da) - Lighter/Faster"
    ]
    
    choice = st.selectbox(
        "Model",
        options=metrics,
        help="All models require reference translation"
    )
    
    # Map to model IDs - ALL PUBLIC
    if "wmt22-comet-da" in choice:
        model_id = "Unbabel/wmt22-comet-da"
    elif "eTranslation" in choice:
        model_id = "Unbabel/eTranslation-COMET"
    else:  # wmt20
        model_id = "Unbabel/wmt20-comet-da"

    # Show file info
    if src_txt:
        src_lines = len([l for l in src_txt.split('\n') if l.strip()])
        st.caption(f"Source: {src_lines} lines")
        if src_lines > 200:
            st.warning(f"⚠️ Large file ({src_lines} lines). Consider splitting to avoid memory issues.")

    if st.button("🚀 Run LQA Analysis", type="primary"):
        if not ref_txt:
            st.error("❌ Reference file is required for all public models.")
            st.info("💡 Upload a reference translation to continue.")
        else:
            try:
                # Clear memory before starting
                clear_memory()
                
                # Alignment
                with st.spinner("🔗 Aligning source and MT with Bertalign..."):
                    pairs = align_texts_strict(src_txt, mt_txt)
                    st.success(f"✅ Aligned {len(pairs)} segment pairs")
                    df = pd.DataFrame(pairs, columns=["source", "translation"])
                
                # Reference alignment
                with st.spinner("🔗 Aligning reference..."):
                    ref_pairs = align_texts_strict(src_txt, ref_txt)
                    ref_map = {p[0]: p[1] for p in ref_pairs}
                    df["reference"] = df["source"].map(ref_map).fillna("")
                    
                    missing_refs = (df["reference"] == "").sum()
                    if missing_refs > 0:
                        st.warning(f"⚠️ {missing_refs} segments missing reference")

                # Load model
                model = load_comet_model(model_id)
                
                # Score
                eval_data = [
                    {
                        "src": row.source,
                        "mt": row.translation,
                        "ref": row.reference
                    }
                    for row in df.itertuples()
                ]
                
                df["score"] = score_with_comet(model, eval_data)
                st.session_state.aligned_df = df
                
                # Clear memory after scoring
                clear_memory()
                
                st.balloons()
                st.success("🎉 Analysis complete!")
                    
            except MemoryError:
                st.error("💾 OUT OF MEMORY!")
                st.error("Try: 1) Split file into smaller chunks, 2) Use COMET-20, 3) Upgrade Streamlit tier")
            except Exception as e:
                st.error("❌ Analysis failed")
                st.error(str(e))
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())

    # Display results
    if st.session_state.aligned_df is not None:
        df = st.session_state.aligned_df
        
        st.markdown("---")
        st.header("📊 Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Score", f"{df['score'].mean():.4f}")
        with col2:
            st.metric("Median", f"{df['score'].median():.4f}")
        with col3:
            st.metric("Std Dev", f"{df['score'].std():.4f}")
        with col4:
            st.metric("Segments", len(df))
        
        # Distribution
        with st.expander("📈 Score Distribution"):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['score'], bins=20, edgecolor='black')
            ax.set_xlabel('COMET Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Score Distribution')
            st.pyplot(fig)
        
        # Results table
        st.subheader("Segment-level Results")
        st.dataframe(
            df.style.background_gradient(subset=['score'], cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
            height=400
        )
        
        # Download buttons
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                "lqa_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            st.download_button(
                "⬇️ Download Excel",
                excel_buffer.getvalue(),
                "lqa_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

else:
    st.info("👆 Upload source, MT, and reference files to begin analysis")
