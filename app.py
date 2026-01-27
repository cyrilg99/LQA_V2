import os
import io
import tempfile
from typing import List, Tuple, Optional
import pandas as pd
import streamlit as st
import torch

# --- Dependencies Check & LASER Setup ---
try:
    from bertalign import Bertalign
    from comet import download_model, load_from_checkpoint
    from laser_encoders import LaserEncoderPipeline
except ImportError as e:
    st.error(f"Missing dependency: {e}. Check your requirements.txt.")
    st.stop()

# ---- UI config ----
st.set_page_config(
    page_title="LQA – Bertalign + COMET",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better UI
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("LQA – Alignement Strict • Évaluation (COMET / XCOMET)")

# =========================
# Utils: I/O
# =========================
def read_uploaded_text(file) -> str:
    """
    Read text from various file formats.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If file format is not supported
    """
    if file is None:
        return ""
    
    name = (file.name or "").lower()
    data = file.getvalue()
    
    try:
        if name.endswith((".txt", ".md")):
            return data.decode("utf-8", errors="ignore")
        
        elif name.endswith((".csv", ".tsv")):
            # Handle CSV/TSV - assume single column or concatenate
            df = pd.read_csv(io.BytesIO(data), sep='\t' if name.endswith('.tsv') else ',')
            return "\n".join(df.iloc[:, 0].astype(str).tolist())
        
        elif name.endswith(".docx"):
            import docx
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                doc = docx.Document(tmp_path)
                return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            finally:
                os.unlink(tmp_path)
        
        elif name.endswith(".pdf"):
            import pdfplumber
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                acc = []
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            acc.append(text)
                return "\n".join(acc)
            finally:
                os.unlink(tmp_path)
        
        else:
            # Try UTF-8 decode as fallback
            return data.decode("utf-8", errors="ignore")
    
    except Exception as e:
        st.error(f"Error reading file {name}: {str(e)}")
        raise ValueError(f"Could not read file {name}") from e


# =========================
# Alignment: Strict Bertalign
# =========================
def align_texts_strict(src_text: str, tgt_text: str) -> List[Tuple[str, str]]:
    """
    Align source and target texts using Bertalign.
    
    Args:
        src_text: Source text
        tgt_text: Target text
        
    Returns:
        List of (source, target) aligned pairs
        
    Raises:
        ValueError: If alignment fails
    """
    if not src_text or not tgt_text:
        raise ValueError("Source and target texts cannot be empty")
    
    try:
        # Bertalign handles sentence splitting internally
        aligner = Bertalign(src_text, tgt_text)
        aligner.align_sents()
        
        pairs = []
        if hasattr(aligner, "result") and aligner.result:
            for bead in aligner.result:
                # Get aligned segments
                src_line = aligner._get_line(bead[0], aligner.src_sents)
                tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
                
                # Only add non-empty pairs
                if src_line.strip() and tgt_line.strip():
                    pairs.append((src_line.strip(), tgt_line.strip()))
            
            if not pairs:
                raise ValueError("Alignment produced no valid pairs")
            return pairs
        else:
            raise ValueError("Bertalign produced no results")
    
    except Exception as e:
        st.error(f"Bertalign Alignment Error: {e}")
        raise


# =========================
# COMET Scoring
# =========================
@st.cache_resource(show_spinner=True)
def load_comet_model(model_id: str):
    """Load and cache COMET model."""
    try:
        ckpt = download_model(model_id)
        return load_from_checkpoint(ckpt)
    except Exception as e:
        st.error(f"Error loading COMET model: {e}")
        raise


def score_with_comet(model, data: list, batch_size: int = 4) -> list:
    """
    Score translations using COMET.
    
    Args:
        model: Loaded COMET model
        data: List of dicts with 'src', 'mt', and optionally 'ref'
        batch_size: Batch size for prediction
        
    Returns:
        List of scores
    """
    if not data:
        return []
    
    try:
        gpu_count = 1 if torch.cuda.is_available() else 0
        out = model.predict(data, batch_size=batch_size, gpus=gpu_count)
        return out.scores
    except Exception as e:
        st.error(f"Error during COMET scoring: {e}")
        raise


# =========================
# Analysis Functions
# =========================
def analyze_scores(df: pd.DataFrame) -> dict:
    """Generate detailed statistics from scores."""
    scores = df['score']
    return {
        'mean': scores.mean(),
        'median': scores.median(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'q25': scores.quantile(0.25),
        'q75': scores.quantile(0.75),
    }


def categorize_quality(score: float) -> str:
    """Categorize translation quality based on score."""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    elif score >= 0.2:
        return "Poor"
    else:
        return "Very Poor"


# =========================
# Main UI Logic
# =========================

# Initialize session state
if "aligned_df" not in st.session_state:
    st.session_state.aligned_df = None
if "analysis_stats" not in st.session_state:
    st.session_state.analysis_stats = None

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    batch_size = st.slider("Batch Size", min_value=1, max_value=16, value=4, 
                          help="Lower values use less memory")
    show_details = st.checkbox("Show detailed statistics", value=True)
    color_threshold = st.slider("Quality threshold for highlighting", 
                               min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool aligns source and target texts using **Bertalign** 
    and evaluates translation quality with **COMET** metrics.
    
    - **COMETKiwi**: Quality estimation without reference
    - **XCOMET-XL**: Reference-based evaluation
    """)

# Main interface
col_src, col_mt, col_ref = st.columns(3)

with col_src:
    up_src = st.file_uploader(
        "📄 Source File",
        key="src",
        help="Upload source text (.txt, .pdf, .docx, .csv, .tsv)"
    )
    if up_src:
        st.success(f"Loaded: {up_src.name}")

with col_mt:
    up_mt = st.file_uploader(
        "🔄 MT Output",
        key="mt",
        help="Upload machine translation output"
    )
    if up_mt:
        st.success(f"Loaded: {up_mt.name}")

with col_ref:
    up_ref = st.file_uploader(
        "✅ Reference (Optional)",
        key="ref",
        help="Upload reference translation (required for XCOMET)"
    )
    if up_ref:
        st.success(f"Loaded: {up_ref.name}")

# Metric selection and analysis
if up_src and up_mt:
    st.markdown("---")
    
    col_metric, col_action = st.columns([2, 1])
    
    with col_metric:
        metrics = ["COMETKiwi (QE)", "XCOMET-XL (Reference-based)"]
        choice = st.selectbox("📊 Select Metric", options=metrics)
        needs_ref = "XCOMET" in choice
        model_id = "Unbabel/wmt22-cometkiwi-da" if not needs_ref else "Unbabel/XCOMET-XL"
        
        if needs_ref and not up_ref:
            st.warning("⚠️ Reference file required for XCOMET")
    
    with col_action:
        st.write("")  # Spacing
        st.write("")  # Spacing
        run_analysis = st.button("🚀 Run LQA Analysis", type="primary", use_container_width=True)
    
    # Run analysis
    if run_analysis:
        if needs_ref and not up_ref:
            st.error("❌ Please upload a reference file for XCOMET evaluation")
        else:
            try:
                # Read files
                with st.spinner("📖 Reading files..."):
                    src_txt = read_uploaded_text(up_src)
                    mt_txt = read_uploaded_text(up_mt)
                    ref_txt = read_uploaded_text(up_ref) if up_ref else None
                    
                    # Validate
                    if not src_txt.strip():
                        st.error("Source file appears to be empty")
                        st.stop()
                    if not mt_txt.strip():
                        st.error("MT file appears to be empty")
                        st.stop()
                
                # Align source and MT
                with st.spinner("🔗 Aligning source and MT with Bertalign..."):
                    pairs = align_texts_strict(src_txt, mt_txt)
                    df = pd.DataFrame(pairs, columns=["source", "translation"])
                    st.info(f"Aligned {len(pairs)} segment pairs")
                
                # Align reference if needed
                if needs_ref:
                    with st.spinner("🔗 Aligning reference..."):
                        ref_pairs = align_texts_strict(src_txt, ref_txt)
                        ref_map = {p[0]: p[1] for p in ref_pairs}
                        df["reference"] = df["source"].map(ref_map).fillna("")
                        
                        # Warn about missing references
                        missing_refs = df["reference"].isna().sum() + (df["reference"] == "").sum()
                        if missing_refs > 0:
                            st.warning(f"⚠️ {missing_refs} segments missing reference translations")
                
                # Score with COMET
                with st.spinner(f"🧮 Scoring with {choice}..."):
                    model = load_comet_model(model_id)
                    eval_data = [
                        {
                            "src": row.source,
                            "mt": row.translation,
                            "ref": getattr(row, "reference", "")
                        }
                        for row in df.itertuples()
                    ]
                    df["score"] = score_with_comet(model, eval_data, batch_size=batch_size)
                    df["quality"] = df["score"].apply(categorize_quality)
                    
                    # Store results
                    st.session_state.aligned_df = df
                    st.session_state.analysis_stats = analyze_scores(df)
                
                st.success("✅ Analysis complete!")
            
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

# Display results
if st.session_state.aligned_df is not None:
    df = st.session_state.aligned_df
    stats = st.session_state.analysis_stats
    
    st.markdown("---")
    st.header("📊 Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Score", f"{stats['mean']:.4f}")
    with col2:
        st.metric("Median Score", f"{stats['median']:.4f}")
    with col3:
        st.metric("Std Deviation", f"{stats['std']:.4f}")
    with col4:
        st.metric("Segments", len(df))
    
    # Detailed statistics
    if show_details:
        with st.expander("📈 Detailed Statistics"):
            detail_cols = st.columns(5)
            with detail_cols[0]:
                st.metric("Min", f"{stats['min']:.4f}")
            with detail_cols[1]:
                st.metric("25th %ile", f"{stats['q25']:.4f}")
            with detail_cols[2]:
                st.metric("Median", f"{stats['median']:.4f}")
            with detail_cols[3]:
                st.metric("75th %ile", f"{stats['q75']:.4f}")
            with detail_cols[4]:
                st.metric("Max", f"{stats['max']:.4f}")
            
            # Quality distribution
            st.subheader("Quality Distribution")
            quality_dist = df['quality'].value_counts().sort_index()
            st.bar_chart(quality_dist)
    
    # Results table
    st.subheader("Segment-level Results")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        quality_filter = st.multiselect(
            "Filter by quality",
            options=["Excellent", "Good", "Fair", "Poor", "Very Poor"],
            default=[]
        )
    with filter_col2:
        score_range = st.slider(
            "Filter by score range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01
        )
    
    # Apply filters
    filtered_df = df.copy()
    if quality_filter:
        filtered_df = filtered_df[filtered_df['quality'].isin(quality_filter)]
    filtered_df = filtered_df[
        (filtered_df['score'] >= score_range[0]) & 
        (filtered_df['score'] <= score_range[1])
    ]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} segments")
    
    # Display table with formatting
    st.dataframe(
        filtered_df.style.background_gradient(
            subset=['score'],
            cmap="RdYlGn",
            vmin=0,
            vmax=1
        ),
        use_container_width=True,
        height=400
    )
    
    # Download options
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Results (CSV)",
            data=csv_data,
            file_name="lqa_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        # Create Excel file with formatting
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, index=False, sheet_name='Statistics')
        
        st.download_button(
            label="⬇️ Download Full Results (Excel)",
            data=excel_buffer.getvalue(),
            file_name="lqa_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Low-quality segments highlight
    low_quality = df[df['score'] < color_threshold]
    if len(low_quality) > 0:
        st.markdown("---")
        with st.expander(f"⚠️ {len(low_quality)} segments below quality threshold ({color_threshold})"):
            st.dataframe(
                low_quality.style.background_gradient(
                    subset=['score'],
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1
                ),
                use_container_width=True
            )

else:
    st.info("👆 Upload source and MT files to begin analysis")
