# ===========================
# Block 1: Module Docstring and Imports (Part 1)
# ===========================
"""
Advanced Semantic Keyword Clustering Application
"""

import os
import time
import json
import logging
import warnings
import tempfile
import hashlib
import gc
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
# ===========================
# Block 2: Optional Library Detection
# ===========================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    OpenAI = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
# ===========================
# Block 3: Configuration and Constants
# ===========================
MAX_KEYWORDS = 25000
OPENAI_TIMEOUT = 60.0
OPENAI_MAX_RETRIES = 3
MAX_MEMORY_WARNING = 800  # MB
BATCH_SIZE = 100
MIN_CLUSTER_SIZE = 2
MAX_CLUSTERS = 50
# ===========================
# Block 4: Logging Configuration
# ===========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
# ===========================
# Block 5: load_and_validate_csv Function
# ===========================
def load_and_validate_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and validate CSV file with enhanced error handling.
    Expects the CSV to have at least a 'keyword' column.
    """
    try:
        df = pd.read_csv(uploaded_file)
        if 'keyword' not in df.columns:
            st.error("Uploaded CSV must contain a 'keyword' column.")
            return None
        df = df[['keyword']].dropna().drop_duplicates().reset_index(drop=True)
        if df.shape[0] < 10:
            st.error("Please upload at least 10 unique keywords.")
            return None
        if df.shape[0] > MAX_KEYWORDS:
            st.error(f"Please limit the number of keywords to {MAX_KEYWORDS}.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None
# ===========================
# Block 6: compute_embeddings Function
# ===========================
def compute_embeddings(
    keywords: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Compute embeddings for a list of keywords using a SentenceTransformer model.
    If OpenAI is available and selected, switch accordingly.
    """
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(keywords, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.error(f"Embedding computation error: {str(e)}")
        return np.zeros((len(keywords), 384))
# ===========================
# Block 7: perform_clustering Function
# ===========================
def perform_clustering(
    embeddings: np.ndarray,
    num_clusters: int,
    linkage_method: str = "ward"
) -> np.ndarray:
    """
    Perform agglomerative clustering on embeddings.
    """
    try:
        clustering_model = AgglomerativeClustering(
            n_clusters=num_clusters,
            affinity="euclidean",
            linkage=linkage_method
        )
        labels = clustering_model.fit_predict(embeddings)
        return labels
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        return np.zeros(embeddings.shape[0], dtype=int)
# ===========================
# Block 8: calculate_cluster_coherence Function
# ===========================
def calculate_cluster_coherence(embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate coherence score for each cluster with robust error handling.
    """
    # If embeddings or labels are None, return an empty dictionary
    if embeddings is None or cluster_labels is None:
        return {}

    # If lengths do not match, log a warning and return empty
    if len(embeddings) != len(cluster_labels):
        logger.warning("Embeddings and labels length mismatch in coherence calculation")
        return {}

    try:
        unique_labels = np.unique(cluster_labels)
        coherence_scores: Dict[int, float] = {}

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]

            if len(cluster_indices) == 0:
                coherence_scores[label] = 0.0
                continue

            cluster_embs = embeddings[cluster_indices]

            # If only one element, consider perfect coherence
            if len(cluster_embs) < 2:
                coherence_scores[label] = 1.0
                continue

            try:
                # Calculate similarity matrix within the cluster
                similarities = cosine_similarity(cluster_embs)
            except Exception as e:
                logger.error(f"Error calculating similarity for cluster {label}: {str(e)}")
                coherence_scores[label] = 0.0
                continue

            # If the matrix is empty, assign default moderate coherence
            if similarities.size == 0:
                coherence_scores[label] = 0.5  # Default moderate coherence
                continue

            # Get upper triangle (excluding diagonal)
            n = similarities.shape[0]
            if n > 1:
                # Create mask for upper triangle excluding diagonal
                mask = np.triu(np.ones((n, n)), k=1).astype(bool)
                upper_triangle = similarities[mask]

                if len(upper_triangle) > 0:
                    coherence = np.mean(upper_triangle)
                    # Ensure coherence is in valid range
                    coherence = max(0.0, min(1.0, coherence))
                else:
                    coherence = 0.0
            else:
                coherence = 1.0

            coherence_scores[label] = coherence

        return coherence_scores
    except Exception as e:
        logger.error(f"Exception in calculate_cluster_coherence: {str(e)}")
        return {}
# ===========================
# Block 9: extract_representative_keywords Function
# ===========================
def extract_representative_keywords(
    keywords: List[str],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    num_representatives: int = 3
) -> Dict[int, List[str]]:
    """
    Extract representative keywords for each cluster by finding those closest to the cluster centroid.
    """
    representatives: Dict[int, List[str]] = {}
    try:
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            if len(cluster_indices) == 0:
                representatives[label] = []
                continue

            cluster_embs = embeddings[cluster_indices]
            centroid = np.mean(cluster_embs, axis=0)
            similarities = cosine_similarity(cluster_embs, centroid.reshape(1, -1)).flatten()
            top_indices = similarities.argsort()[::-1][:num_representatives]
            rep_keywords = [keywords[cluster_indices[i]] for i in top_indices]
            representatives[label] = rep_keywords
        return representatives
    except Exception as e:
        logger.error(f"Error extracting representatives: {str(e)}")
        return {}
# ===========================
# Block 10: build_dendrogram Function
# ===========================
def build_dendrogram(embeddings: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Build linkage matrix and labels for dendrogram plotting.
    """
    try:
        Z = linkage(embeddings, method='ward')
        labels = [str(i) for i in range(embeddings.shape[0])]
        return Z, labels
    except Exception as e:
        logger.error(f"Dendrogram building error: {str(e)}")
        return np.array([]), []
# ===========================
# Block 11: get_system_status Function
# ===========================
def get_system_status() -> Dict[str, Any]:
    """
    Get current system status, including CPU, memory, and disk usage.
    """
    try:
        status: Dict[str, Any] = {}
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            status = {
                "cpu_percent": cpu_percent,
                "available_gb": mem.available / (1024**3),
                "total_gb": mem.total / (1024**3),
                "used_percent": mem.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_used_percent": disk.percent
            }
        else:
            status = {"cpu_percent": 0.0, "available_gb": 0.0, "total_gb": 0.0, "used_percent": 0.0,
                      "disk_total_gb": 0.0, "disk_used_percent": 0.0}
        return status
    except Exception as e:
        logger.error(f"System status error: {str(e)}")
        return {"cpu_percent": 0.0, "available_gb": 0.0, "total_gb": 0.0, "used_percent": 0.0,
                "disk_total_gb": 0.0, "disk_used_percent": 0.0}
# ===========================
# Block 12: show_welcome_screen Function
# ===========================
def show_welcome_screen():
    """Show enhanced welcome screen with comprehensive information"""
    try:
        # Main header with styling
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f1f1f; font-size: 3rem; margin-bottom: 0.5rem;">
                🔍 Semantic Keyword Clustering
            </h1>
            <h3 style="color: #666; font-weight: 300; margin-bottom: 2rem;">
                Advanced AI-Powered SEO Keyword Analysis Platform
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h4>🚀 Fast Processing</h4>
                <p>Analyze thousands of keywords in seconds.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h4>🤖 AI-Driven Insights</h4>
                <p>Leverage state-of-the-art embeddings and clustering.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h4>📊 Enterprise Features</h4>
                <ul>
                    <li>Process up to 25K keywords</li>
                    <li>Multiple export formats</li>
                    <li>Interactive dashboards</li>
                    <li>Quality scoring & metrics</li>
                    <li>Performance visualizations</li>
                    <li>Business value analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Application info footer
        st.write("---")
        st.write("**Application Version:** 1.0.0")
        st.write("**Build Date:** 2025-01-01")
        st.write("**Python Version:** " + ".".join(map(str, __import__('sys').version_info[:3])))

        try:
            st.write(f"**Pandas:** {pd.__version__}")
            st.write(f"**NumPy:** {np.__version__}")
            st.write(f"**Scikit-learn:** {__import__('sklearn').__version__}")
        except Exception:
            pass

    except Exception as e:
        st.error(f"Error displaying welcome screen: {str(e)}")
# ===========================
# Block 13: show_processing_screen Function
# ===========================
def show_processing_screen(config: Dict[str, Any]):
    """Show processing options and handle keyword embedding & clustering"""
    try:
        st.header("Processing Configuration")

        # Sidebar: clustering parameters
        num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=MAX_CLUSTERS, value=5)
        linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])

        # Progress bar placeholder
        progress_bar = st.progress(0)

        # State: have we started processing?
        if 'processing_started' not in st.session_state:
            st.session_state.processing_started = False

        if st.button("🚀 Start Processing", use_container_width=True):
            st.session_state.processing_started = True

        if st.session_state.processing_started:
            # Step 1: Load keywords from config
            keywords = config.get("keywords", [])
            st.write(f"Loaded {len(keywords)} keywords.")

            # Step 2: Compute embeddings
            progress_bar.progress(10, text="Computing embeddings...")
            embeddings = compute_embeddings(keywords)
            progress_bar.progress(30, text="Embeddings computed.")

            # Step 3: Perform clustering
            progress_bar.progress(40, text="Performing clustering...")
            labels = perform_clustering(embeddings, num_clusters, linkage_method)
            progress_bar.progress(60, text="Clustering completed.")

            # Step 4: Calculate coherence
            progress_bar.progress(70, text="Calculating cluster coherence...")
            coherence_scores = calculate_cluster_coherence(embeddings, labels)
            progress_bar.progress(80, text="Coherence calculated.")

            # Step 5: Extract representatives
            progress_bar.progress(85, text="Extracting representative keywords...")
            representatives = extract_representative_keywords(keywords, embeddings, labels)
            progress_bar.progress(90, text="Representatives extracted.")

            # Step 6: Build dendrogram
            progress_bar.progress(95, text="Building dendrogram data...")
            Z, dendro_labels = build_dendrogram(embeddings)
            progress_bar.progress(100, text="Processing complete.")

            # Store results in session state
            st.session_state.keywords = keywords
            st.session_state.embeddings = embeddings
            st.session_state.labels = labels
            st.session_state.coherence_scores = coherence_scores
            st.session_state.representatives = representatives
            st.session_state.dendrogram = (Z, dendro_labels)

            st.success("Processing completed successfully!")

    except Exception as e:
        st.error(f"Processing screen error: {str(e)}")
# ===========================
# Block 14: show_results_screen Function
# ===========================
def show_results_screen():
    """Show interactive results: clusters, coherence, representatives, and visualizations"""
    try:
        if 'labels' not in st.session_state:
            st.warning("Please run processing first.")
            return

        keywords = st.session_state.keywords
        embeddings = st.session_state.embeddings
        labels = st.session_state.labels
        coherence_scores = st.session_state.coherence_scores
        representatives = st.session_state.representatives
        Z, dendro_labels = st.session_state.dendrogram

        st.header("Cluster Results")

        # Summary statistics
        unique_labels = np.unique(labels)
        cluster_sizes = [np.sum(labels == lbl) for lbl in unique_labels]
        avg_coherence = np.mean(list(coherence_scores.values())) if coherence_scores else 0.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clusters", len(unique_labels))
        with col2:
            st.metric("Average Coherence", f"{avg_coherence:.2f}")
        with col3:
            st.metric("Keywords Processed", len(keywords))

        st.write("### Cluster Details")
        for lbl in unique_labels:
            st.subheader(f"Cluster {lbl} (Size: {cluster_sizes[lbl]})")
            st.write(f"Coherence Score: {coherence_scores.get(lbl, 0.0):.2f}")
            st.write("Representative Keywords:")
            reps = representatives.get(lbl, [])
            for kw in reps:
                st.write(f"- {kw}")
            st.write("---")

        # Dendrogram visualization
        st.write("### Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, labels=[keywords[int(i)] for i in dendro_labels], leaf_rotation=90)
        st.pyplot(fig)

        # 2D PCA scatter plot
        st.write("### 2D PCA Scatter Plot")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        scatter = ax2.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        legend1 = ax2.legend(*scatter.legend_elements(), title="Clusters")
        ax2.add_artist(legend1)
        ax2.set_xlabel("PCA Component 1")
        ax2.set_ylabel("PCA Component 2")
        st.pyplot(fig2)

        # Export options
        st.write("### Export Results")
        df_results = pd.DataFrame({
            "keyword": keywords,
            "cluster": labels,
            "coherence": [coherence_scores.get(lbl, 0.0) for lbl in labels]
        })
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="cluster_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Results screen error: {str(e)}")
# ===========================
# Block 15: main Function and __main__ Guard
# ===========================
def main():
    """Main Streamlit application routing and state management."""
    try:
        st.set_page_config(
            page_title="Semantic Keyword Clustering",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ("Welcome", "Processing", "Results")
        )

        # Handle file upload on Welcome page
        if page == "Welcome":
            show_welcome_screen()
            uploaded_file = st.file_uploader("Upload CSV of Keywords", type=["csv"], accept_multiple_files=False)
            if uploaded_file is not None:
                df_keywords = load_and_validate_csv(uploaded_file)
                if df_keywords is not None:
                    st.session_state.keywords = df_keywords['keyword'].tolist()
                    st.success("File uploaded and validated.")
        elif page == "Processing":
            if 'keywords' not in st.session_state:
                st.warning("Please upload and validate a CSV of keywords on the Welcome page first.")
            else:
                config = {"keywords": st.session_state.keywords}
                show_processing_screen(config)
        elif page == "Results":
            show_results_screen()

        # Display system status in the footer
        with st.expander("System Status", expanded=False):
            status = get_system_status()
            st.write(f"CPU Usage: {status.get('cpu_percent', 0.0):.1f}%")
            st.write(f"Memory Usage: {status.get('used_percent', 0.0):.1f}%")
            st.write(f"Disk Usage: {status.get('disk_used_percent', 0.0):.1f}%")

    except Exception as e:
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
