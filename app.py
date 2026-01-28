import os
import io
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

st.info("📊 **Memory-Optimized Excel Processing** - Handles large files by chunking")

# =========================
# Memory Management
# =========================
def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# Excel Processing
# =========================
def read_excel_file(file) -> pd.DataFrame:
    """Read and validate Excel file"""
    try:
        df = pd.read_excel(file, engine='openpyxl')
        
        if len(df.columns) < 2:
            raise ValueError("Excel file must have at least 2 columns (Source, MT)")
        
        # Rename columns
        if len(df.columns) == 2:
            df.columns = ['source', 'translation']
        elif len(df.columns) >= 3:
            df.columns = ['source', 'translation', 'reference'] + list(df.columns[3:])
        
        # Clean data
        df['source'] = df['source'].astype(str).str.strip()
        df['translation'] = df['translation'].astype(str).str.strip()
        if 'reference' in df.columns:
            df['reference'] = df['reference'].astype(str).str.strip()
        
        # Remove empty rows
        df = df[(df['source'] != '') & (df['source'] != 'nan')]
        df = df[(df['translation'] != '') & (df['translation'] != 'nan')]
        
        df = df.reset_index(drop=True)
        return df
        
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        raise

# =========================
# COMET Scoring with Chunking
# =========================
@st.cache_resource(show_spinner=True)
def load_comet_model(model_id: str):
    """Load COMET model"""
    try:
        clear_memory()
        ckpt = download_model(model_id)
        model = load_from_checkpoint(ckpt)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise

def score_with_comet_chunked(model, data: list, chunk_size: int = 10):
    """
    Score in small chunks to avoid OOM errors.
    Process 10 segments at a time with memory cleanup between chunks.
    """
    all_scores = []
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            status_text.text(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} segments)...")
            
            # Clear memory before each chunk
            clear_memory()
            
            # Score this chunk
            out = model.predict(chunk, batch_size=1, gpus=0)  # Force CPU, batch=1
            all_scores.extend(out.scores)
            
            # Update progress
            progress = min((i + chunk_size) / len(data), 1.0)
            progress_bar.progress(progress)
            
            # Clean up
            clear_memory()
        
        progress_bar.progress(1.0)
        status_text.text("✅ Scoring complete!")
        
        return all_scores
        
    except Exception as e:
        st.error(f"Error during scoring: {e}")
        raise
    finally:
        clear_memory()

# =========================
# Main UI
# =========================
if "results_df" not in st.session_state:
    st.session_state.results_df = None

with st.sidebar:
    st.header("⚙️ Settings")
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Number of segments to process at once. Lower = safer but slower."
    )
    
    st.markdown("---")
    st.header("📋 Instructions")
    st.markdown("""
    ### Excel Format
    
    **Columns:**
    - A: Source segments
    - B: Translation segments
    - C: Reference (required)
    
    ### Memory Tips
    
    **Free tier limits:**
    - Max ~500 segments recommended
    - Use chunk size 10-20 for safety
    - Larger chunks = faster but riskier
    
    **If app crashes:**
    - Reduce chunk size to 5
    - Split Excel file
    - Process fewer segments
    """)

st.subheader("📁 Upload Excel File")

uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=['xlsx', 'xls'],
    help="Excel with Source (A), Translation (B), Reference (C)"
)

if uploaded_file is not None:
    try:
        with st.spinner("Reading Excel file..."):
            df = read_excel_file(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} segments")
        
        # Show preview
        with st.expander("📄 Preview (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Check reference
        has_reference = 'reference' in df.columns and not df['reference'].isna().all()
        
        if not has_reference:
            st.error("❌ No reference column detected. Add Column C with reference translations.")
            st.stop()
        
        # File size warning
        if len(df) > 500:
            st.warning(f"⚠️ Large file ({len(df)} segments). Consider splitting.")
            st.info(f"Estimated processing time: {len(df) * 2 // 60} minutes")
        
        # Model selection
        st.subheader("📊 Select Model")
        
        model_choice = st.radio(
            "Choose model",
            [
                "COMET-22 (Best quality, ~1GB RAM)",
                "COMET-20 (Lighter, ~800MB RAM)"
            ],
            help="COMET-20 recommended for files >200 segments"
        )
        
        if "COMET-22" in model_choice:
            model_id = "Unbabel/wmt22-comet-da"
        else:
            model_id = "Unbabel/wmt20-comet-da"
        
        st.caption(f"Selected: {model_id}")
        st.caption(f"Chunk size: {chunk_size} segments per batch")
        
        # Analysis button
        if st.button("🚀 Run COMET Analysis", type="primary"):
            try:
                clear_memory()
                
                # Prepare data
                with st.spinner("Preparing data..."):
                    eval_data = []
                    for idx, row in df.iterrows():
                        eval_data.append({
                            "src": str(row['source']),
                            "mt": str(row['translation']),
                            "ref": str(row['reference'])
                        })
                
                st.info(f"📝 Processing {len(eval_data)} segments in chunks of {chunk_size}")
                
                # Load model
                with st.spinner("Loading COMET model (may take 2-3 min first time)..."):
                    model = load_comet_model(model_id)
                    st.success("✅ Model loaded!")
                
                # Score with chunking
                st.subheader("Scoring Progress")
                scores = score_with_comet_chunked(model, eval_data, chunk_size=chunk_size)
                
                # Add results
                df['score'] = scores
                
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
                st.session_state.results_df = df
                
                clear_memory()
                st.balloons()
                st.success("🎉 Analysis complete!")
                
            except MemoryError:
                st.error("💾 OUT OF MEMORY!")
                st.error("Solutions:")
                st.markdown("""
                1. **Reduce chunk size** to 5
                2. **Split your Excel file** into smaller parts
                3. **Use COMET-20** (lighter model)
                4. **Upgrade Streamlit** to paid tier
                """)
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                with st.expander("Error details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Failed to read Excel: {str(e)}")

# Display results
if st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    st.markdown("---")
    st.header("📊 Results")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{df['score'].mean():.4f}")
    with col2:
        st.metric("Median", f"{df['score'].median():.4f}")
    with col3:
        st.metric("Std Dev", f"{df['score'].std():.4f}")
    with col4:
        st.metric("Segments", len(df))
    
    # Quality breakdown
    with st.expander("📈 Quality Distribution"):
        quality_counts = df['quality'].value_counts()
        st.bar_chart(quality_counts)
    
    # Results table
    st.subheader("Results Table")
    
    display_cols = ['source', 'translation', 'reference', 'score', 'quality']
    st.dataframe(
        df[display_cols].style.background_gradient(
            subset=['score'], 
            cmap="RdYlGn", 
            vmin=0, 
            vmax=1
        ),
        use_container_width=True,
        height=400
    )
    
    # Downloads
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Download CSV",
            csv_data,
            "comet_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        st.download_button(
            "📊 Download Excel",
            excel_buffer.getvalue(),
            "comet_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

else:
    if uploaded_file is None:
        st.info("👆 Upload an Excel file to begin")
