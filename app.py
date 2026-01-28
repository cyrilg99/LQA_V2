import os
import io
import tempfile
import pandas as pd
import streamlit as st
import torch
import gc

# --- Dependencies Check ---
try:
    from comet import download_model, load_from_checkpoint
except ImportError as e:
    st.error(f"Missing dependency: {e}. Check your requirements.txt.")
    st.stop()

# ---- UI config ----
st.set_page_config(page_title="LQA — COMET Scoring", layout="wide")
st.title("LQA — COMET Quality Evaluation")

st.info("📊 **Upload pre-aligned Excel files** with Source and MT columns")

# =========================
# Memory Management
# =========================
def clear_memory():
    """Force garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# Excel Processing
# =========================
def read_excel_file(file) -> pd.DataFrame:
    """
    Read Excel file with source and MT columns.
    Expected format:
    - Column 1: Source segments
    - Column 2: MT (translation) segments
    - Optional Column 3: Reference segments
    """
    try:
        # Read Excel file
        df = pd.read_excel(file, engine='openpyxl')
        
        # Validate columns
        if len(df.columns) < 2:
            raise ValueError("Excel file must have at least 2 columns (Source, MT)")
        
        # Rename columns for consistency
        if len(df.columns) == 2:
            df.columns = ['source', 'translation']
        elif len(df.columns) >= 3:
            df.columns = ['source', 'translation', 'reference'] + list(df.columns[3:])
        
        # Convert to string and clean
        df['source'] = df['source'].astype(str).str.strip()
        df['translation'] = df['translation'].astype(str).str.strip()
        if 'reference' in df.columns:
            df['reference'] = df['reference'].astype(str).str.strip()
        
        # Remove empty rows
        df = df[(df['source'] != '') & (df['source'] != 'nan')]
        df = df[(df['translation'] != '') & (df['translation'] != 'nan')]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        raise

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
        with st.spinner(f"Scoring {len(data)} segments..."):
            out = model.predict(data, batch_size=1, gpus=gpu_count)
        return out.scores
    except Exception as e:
        st.error("Failed during scoring")
        st.error(str(e))
        raise

# =========================
# Main UI Logic
# =========================
if "results_df" not in st.session_state:
    st.session_state.results_df = None

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    ### Excel File Format
    
    Your Excel file should have:
    
    **Required columns:**
    1. **Column A**: Source segments
    2. **Column B**: MT/Translation segments
    
    **Optional:**
    3. **Column C**: Reference segments
    
    **Example:**
    | Source | Translation | Reference |
    |--------|-------------|-----------|
    | Hello world | Bonjour monde | Bonjour le monde |
    | How are you? | Comment allez-vous? | Comment vas-tu? |
    
    ---
    
    ### Available Models
    
    **Lightweight (Best for Free Tier):**
    - ✅ COMET-22 (~1GB RAM)
    - ✅ COMET-20 (~800MB RAM)
    
    **Reference Required:**
    All public models require a reference 
    translation in Column C.
    
    ---
    
    ### Memory Usage
    
    **Free Streamlit Cloud: 1GB RAM**
    
    Estimated capacity:
    - Small files (<100 segments): ✅
    - Medium (100-500): ✅
    - Large (>500): ⚠️ May need splitting
    """)

# File uploader
st.subheader("📁 Upload Excel File")

uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=['xlsx', 'xls'],
    help="Excel file with Source (A), Translation (B), and optionally Reference (C) columns"
)

if uploaded_file is not None:
    try:
        # Read and display Excel file
        with st.spinner("Reading Excel file..."):
            df = read_excel_file(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} segments from Excel file")
        
        # Show preview
        with st.expander("📄 Preview Data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Check for reference column
        has_reference = 'reference' in df.columns and not df['reference'].isna().all()
        
        if has_reference:
            st.info("✅ Reference column detected - can use reference-based models")
        else:
            st.warning("⚠️ No reference column - only QE models available (require authentication)")
        
        # Model selection
        st.subheader("📊 Select Quality Metric")
        
        if has_reference:
            metrics = [
                "COMET-22 (wmt22-comet-da) - Recommended",
                "eTranslation-COMET - For EU Languages",
                "COMET-20 (wmt20-comet-da) - Lighter/Faster"
            ]
        else:
            metrics = [
                "⚠️ No reference detected - Add Column C for reference-based models"
            ]
            st.error("Please add a reference column (Column C) to use public models")
        
        choice = st.selectbox(
            "Model",
            options=metrics,
            disabled=not has_reference,
            help="All public models require reference translation"
        )
        
        # Map to model IDs
        if has_reference:
            if "wmt22-comet-da" in choice:
                model_id = "Unbabel/wmt22-comet-da"
            elif "eTranslation" in choice:
                model_id = "Unbabel/eTranslation-COMET"
            else:  # wmt20
                model_id = "Unbabel/wmt20-comet-da"
        
        # Warning for large files
        if len(df) > 500:
            st.warning(f"⚠️ Large file ({len(df)} segments). Consider splitting to avoid memory issues.")
        
        # Run analysis button
        if st.button("🚀 Run COMET Analysis", type="primary", disabled=not has_reference):
            try:
                clear_memory()
                
                # Prepare data for COMET
                with st.spinner("Preparing data..."):
                    eval_data = []
                    for idx, row in df.iterrows():
                        data_point = {
                            "src": str(row['source']),
                            "mt": str(row['translation']),
                            "ref": str(row.get('reference', ''))
                        }
                        eval_data.append(data_point)
                
                st.info(f"📝 Prepared {len(eval_data)} segments for evaluation")
                
                # Load model
                model = load_comet_model(model_id)
                
                # Score
                scores = score_with_comet(model, eval_data)
                
                # Add scores to dataframe
                df['score'] = scores
                
                # Add quality category
                def categorize_quality(score):
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
                
                df['quality'] = df['score'].apply(categorize_quality)
                
                # Store results
                st.session_state.results_df = df
                
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
    
    except Exception as e:
        st.error(f"Failed to read Excel file: {str(e)}")
        st.info("Make sure your Excel file has at least 2 columns: Source and Translation")

# Display results
if st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    st.markdown("---")
    st.header("📊 Results")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Score", f"{df['score'].mean():.4f}")
    with col2:
        st.metric("Median", f"{df['score'].median():.4f}")
    with col3:
        st.metric("Std Dev", f"{df['score'].std():.4f}")
    with col4:
        st.metric("Segments", len(df))
    
    # Quality distribution
    with st.expander("📈 Quality Distribution"):
        quality_counts = df['quality'].value_counts()
        st.bar_chart(quality_counts)
        
        # Show percentages
        st.subheader("Quality Breakdown")
        for quality in ["Excellent", "Good", "Fair", "Poor", "Very Poor"]:
            if quality in quality_counts.index:
                count = quality_counts[quality]
                percentage = (count / len(df)) * 100
                st.write(f"**{quality}**: {count} segments ({percentage:.1f}%)")
    
    # Filters
    st.subheader("🔍 Filter Results")
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        quality_filter = st.multiselect(
            "Filter by quality",
            options=["Excellent", "Good", "Fair", "Poor", "Very Poor"],
            default=[]
        )
    
    with col_f2:
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
    
    # Results table
    st.subheader("Segment-level Results")
    
    # Determine which columns to display
    display_cols = ['source', 'translation']
    if 'reference' in filtered_df.columns:
        display_cols.append('reference')
    display_cols.extend(['score', 'quality'])
    
    st.dataframe(
        filtered_df[display_cols].style.background_gradient(
            subset=['score'], 
            cmap="RdYlGn", 
            vmin=0, 
            vmax=1
        ),
        use_container_width=True,
        height=400
    )
    
    # Download section
    st.markdown("---")
    st.subheader("⬇️ Download Results")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # CSV download
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Download CSV",
            csv_data,
            "comet_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        # Excel download with formatting
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            
            # Add statistics sheet
            stats_data = {
                'Metric': ['Average', 'Median', 'Std Dev', 'Min', 'Max', 'Count'],
                'Value': [
                    df['score'].mean(),
                    df['score'].median(),
                    df['score'].std(),
                    df['score'].min(),
                    df['score'].max(),
                    len(df)
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, index=False, sheet_name='Statistics')
        
        st.download_button(
            "📊 Download Excel (with stats)",
            excel_buffer.getvalue(),
            "comet_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

else:
    if uploaded_file is None:
        st.info("👆 Upload an Excel file to begin analysis")
