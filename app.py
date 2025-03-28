import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from io import StringIO

# For OpenAI, import with error handling
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

# Try to import advanced libraries
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False

try:
    import spacy
    try:
        # Load English model if your keywords are in English
        nlp = spacy.load("en_core_web_sm")
        spacy_available = True
    except:
        spacy_available = False
except ImportError:
    spacy_available = False

try:
    from textblob import TextBlob
    textblob_available = True
except ImportError:
    textblob_available = False

try:
    import hdbscan
    hdbscan_available = True
except ImportError:
    hdbscan_available = False

# Download NLTK resources at startup
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass  # Continue even if downloads fail

################################################################
#          COST CALCULATION AND SUPPORT FUNCTIONS
################################################################

def calculate_api_cost(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    """
    Calculates the estimated cost of using the OpenAI API based on the number of keywords.
    """
    # Updated prices (March 2025) - Adjust if new pricing is available
    EMBEDDING_COST_PER_1K = 0.02  # text-embedding-3-small per 1K tokens
    
    # GPT-3.5-Turbo costs
    GPT35_INPUT_COST_PER_1K = 0.0005
    GPT35_OUTPUT_COST_PER_1K = 0.0015
    
    # GPT-4 costs
    GPT4_INPUT_COST_PER_1K = 0.03
    GPT4_OUTPUT_COST_PER_1K = 0.06
    
    results = {
        "embedding_cost": 0,
        "naming_cost": 0,
        "total_cost": 0,
        "processed_keywords": 0
    }
    
    # 1. Embedding cost (limited to 5000 keywords)
    keywords_for_embeddings = min(num_keywords, 5000)
    results["processed_keywords"] = keywords_for_embeddings
    
    # Estimate ~2 tokens per keyword
    estimated_tokens = keywords_for_embeddings * 2
    results["embedding_cost"] = (estimated_tokens / 1000) * EMBEDDING_COST_PER_1K
    
    # 2. Naming cost
    avg_tokens_per_cluster = 200   # prompt + representative keywords
    avg_output_tokens_per_cluster = 80  # output tokens (name + description)
    
    estimated_input_tokens = num_clusters * avg_tokens_per_cluster
    estimated_output_tokens = num_clusters * avg_output_tokens_per_cluster
    
    if selected_model == "gpt-3.5-turbo":
        input_cost = (estimated_input_tokens / 1000) * GPT35_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT35_OUTPUT_COST_PER_1K
    else:  # GPT-4
        input_cost = (estimated_input_tokens / 1000) * GPT4_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT4_OUTPUT_COST_PER_1K
    
    results["naming_cost"] = input_cost + output_cost
    results["total_cost"] = results["embedding_cost"] + results["naming_cost"]
    
    return results

def add_cost_calculator():
    st.sidebar.markdown("---")
    with st.sidebar.expander("💰 API Cost Calculator", expanded=False):
        st.markdown("""
        ### API Cost Calculator
        
        Calculate approximate OpenAI costs for your keywords.
        """)
        
        calc_num_keywords = st.number_input(
            "Number of keywords to process", 
            min_value=100, 
            max_value=100000, 
            value=1000,
            step=500
        )
        calc_num_clusters = st.number_input(
            "Approximate number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            step=1
        )
        calc_model = st.radio(
            "Model for naming clusters",
            options=["gpt-3.5-turbo", "gpt-4"],
            index=0,
            horizontal=True
        )
        
        if st.button("Calculate Estimated Cost", use_container_width=True):
            cost_results = calculate_api_cost(calc_num_keywords, calc_model, calc_num_clusters)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Keywords processed with OpenAI", 
                    f"{cost_results['processed_keywords']:,}",
                    help="OpenAI processes up to 5,000 keywords; the rest are handled via similarity propagation."
                )
                st.metric(
                    "Embeddings cost", 
                    f"${cost_results['embedding_cost']:.4f}",
                    help="Cost for text-embedding-3-small"
                )
            with col2:
                st.metric(
                    "Cluster naming cost", 
                    f"${cost_results['naming_cost']:.4f}",
                    help=f"Cost using {calc_model} for naming/describing clusters"
                )
                st.metric(
                    "TOTAL COST", 
                    f"${cost_results['total_cost']:.4f}",
                    help="Approximate total cost"
                )
            
            st.info("""
            **Note:** This is only an estimate. Actual costs may vary 
            based on the length of keywords and complexity of the clusters.
            Using Sentence Transformers instead of OpenAI embeddings is $0.
            """)

def show_csv_cost_estimate(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    if num_keywords > 0:
        cost_results = calculate_api_cost(num_keywords, selected_model, num_clusters)
        
        with st.sidebar.expander("💰 Estimated Cost (Current CSV)", expanded=True):
            st.markdown(f"### Estimated Cost for {num_keywords:,} Keywords")
            
            st.markdown(f"""
            - **Keywords processed with OpenAI**: {cost_results['processed_keywords']:,}
            - **Embeddings cost**: ${cost_results['embedding_cost']:.4f}
            - **Cluster naming cost**: ${cost_results['naming_cost']:.4f}
            - **TOTAL COST**: ${cost_results['total_cost']:.4f}
            """)
            
            if cost_results['processed_keywords'] < num_keywords:
                st.info(f"""
                {cost_results['processed_keywords']:,} keywords will be processed directly with OpenAI.
                The remaining {num_keywords - cost_results['processed_keywords']:,} will use
                similarity propagation.
                """)
            
            st.markdown("""
            **Cost Savings**: If you prefer not to use OpenAI, you can 
            use SentenceTransformers at no cost with decent results.
            """)

################################################################
#          SEMANTIC PREPROCESSING
################################################################

def enhanced_preprocessing(text, use_lemmatization=True):
    """Enhanced preprocessing using spaCy or TextBlob if available."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Option 1: spaCy
        if spacy_available:
            doc = nlp(text.lower())
            entities = [ent.text for ent in doc.ents]
            tokens = []
            for token in doc:
                if not token.is_stop and token.is_alpha and len(token.text) > 1:
                    tokens.append(token.lemma_)
            
            # Bigrams
            bigrams = []
            for i in range(len(doc) - 1):
                if (not doc[i].is_stop and not doc[i+1].is_stop 
                    and doc[i].is_alpha and doc[i+1].is_alpha):
                    bigrams.append(f"{doc[i].lemma_}_{doc[i+1].lemma_}")
            
            processed_parts = tokens + bigrams + entities
            return " ".join(processed_parts)
        
        # Option 2: TextBlob
        elif textblob_available:
            from textblob import TextBlob
            blob = TextBlob(text.lower())
            noun_phrases = list(blob.noun_phrases)
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
            
            words = [w for w in blob.words if len(w) > 1 and w.lower() not in stop_words]
            
            if use_lemmatization:
                lemmatizer = WordNetLemmatizer()
                lemmas = [lemmatizer.lemmatize(w) for w in words]
                processed_parts = lemmas + noun_phrases
            else:
                processed_parts = words + noun_phrases
            
            return " ".join(processed_parts)
        
        # Fallback: standard NLTK
        else:
            return preprocess_text(text, use_lemmatization)
    
    except Exception:
        return text.lower() if isinstance(text, str) else ""

def preprocess_text(text, use_lemmatization=True):
    """Basic NLTK-based preprocessing."""
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
        
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        if use_lemmatization:
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            except:
                pass
        
        return " ".join(tokens)
    except Exception:
        return text.lower() if isinstance(text, str) else ""

def preprocess_keywords(keywords, use_advanced=True):
    processed_keywords = []
    progress_bar = st.progress(0)
    total = len(keywords)
    
    if use_advanced:
        if spacy_available:
            st.success("Using advanced preprocessing with spaCy")
        elif textblob_available:
            st.success("Using alternative preprocessing with TextBlob")
        else:
            st.info("Using standard preprocessing with NLTK")
    else:
        st.info("Using standard preprocessing with NLTK (advanced preprocessing disabled)")
    
    for i, keyword in enumerate(keywords):
        if use_advanced and (spacy_available or textblob_available):
            processed_keywords.append(enhanced_preprocessing(keyword))
        else:
            processed_keywords.append(preprocess_text(keyword))
        
        if i % 100 == 0:
            progress_bar.progress(min(i / total, 1.0))
    
    progress_bar.progress(1.0)
    return processed_keywords

################################################################
#          EMBEDDING GENERATION
################################################################

def generate_embeddings(df, openai_available, openai_api_key=None):
    """
    Generate embeddings using OpenAI if possible, else SentenceTransformers, or fallback TF-IDF.
    """
    st.info("Generating embeddings for keywords...")
    
    # Attempt OpenAI embeddings
    if openai_available and openai_api_key:
        try:
            st.info("Using OpenAI embeddings (high semantic precision)")
            os.environ["OPENAI_API_KEY"] = openai_api_key
            client = OpenAI()
            keywords = df['keyword_processed'].fillna('').tolist()
            all_embeddings = []
            
            if len(keywords) > 5000:
                st.warning(f"Limiting to 5000 representative keywords out of {len(keywords)} total.")
                step = max(1, len(keywords) // 5000)
                sample_indices = list(range(0, len(keywords), step))[:5000]
                sample_keywords = [keywords[i] for i in sample_indices]
                
                progress_bar = st.progress(0)
                st.info("Requesting embeddings from OpenAI (this may take a few minutes)...")
                
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=sample_keywords
                )
                progress_bar.progress(0.5)
                
                sample_embeddings = np.array([item.embedding for item in response.data])
                
                st.info("Propagating embeddings to remaining keywords via similarity...")
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(keywords)
                
                all_embeddings = np.zeros((len(keywords), len(sample_embeddings[0])))
                for i, idx in enumerate(sample_indices):
                    all_embeddings[idx] = sample_embeddings[i]
                
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(3, len(sample_indices)))
                nn.fit(tfidf_matrix[sample_indices])
                
                remaining_indices = [i for i in range(len(keywords)) if i not in sample_indices]
                
                for i, idx in enumerate(remaining_indices):
                    distances, neighbors = nn.kneighbors(tfidf_matrix[idx:idx+1])
                    weights = 1.0 / (1.0 + distances[0])
                    weights = weights / weights.sum()
                    
                    weighted_embedding = np.zeros_like(sample_embeddings[0])
                    for j, weight in zip(neighbors[0], weights):
                        similar_idx = sample_indices[j]
                        weighted_embedding += weight * all_embeddings[similar_idx]
                    
                    all_embeddings[idx] = weighted_embedding
                    
                    if i % 100 == 0:
                        prog_val = 0.5 + min(0.5, (i / len(remaining_indices)) * 0.5)
                        progress_bar.progress(prog_val)
                
                progress_bar.progress(1.0)
            
            else:
                progress_bar = st.progress(0)
                st.info(f"Requesting embeddings for all {len(keywords)} keywords from OpenAI...")
                batch_size = 1000
                for i in range(0, len(keywords), batch_size):
                    batch_end = min(i + batch_size, len(keywords))
                    batch = keywords[i:batch_end]
                    
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    progress_bar.progress(min(1.0, batch_end / len(keywords)))
                
                progress_bar.progress(1.0)
            
            embeddings = np.array(all_embeddings) if isinstance(all_embeddings, list) else all_embeddings
            st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (OpenAI)")
            return embeddings
                
        except Exception as e:
            st.error(f"Error generating embeddings with OpenAI: {str(e)}")
            st.info("Falling back to SentenceTransformers...")
    
    # SentenceTransformers fallback
    if sentence_transformers_available:
        try:
            st.success("Using SentenceTransformer (free fallback)")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            progress_bar = st.progress(0)
            keywords = df['keyword_processed'].fillna('').tolist()
            batch_size = 512
            all_embeddings = []
            
            for i in range(0, len(keywords), batch_size):
                batch = keywords[i:i+batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                progress_bar.progress(min(1.0, (i + batch_size) / len(keywords)))
            
            progress_bar.progress(1.0)
            embeddings = np.array(all_embeddings)
            st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (SentenceTransformers)")
            return embeddings
        except Exception as e:
            st.error(f"Error with SentenceTransformers: {str(e)}")
    
    # TF-IDF last resort
    st.warning("Using TF-IDF as a last resort (less semantic precision).")
    return generate_tfidf_embeddings(df['keyword_processed'].fillna(''))

def generate_tfidf_embeddings(texts, min_df=1, max_df=0.95):
    st.info("Generating TF-IDF vectors for keywords...")
    progress_bar = st.progress(0)
    try:
        vectorizer = TfidfVectorizer(
            max_features=300,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        clean_texts = [t if isinstance(t, str) else " " for t in texts]
        
        progress_bar.progress(0.3)
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
        progress_bar.progress(0.8)
        
        embeddings = tfidf_matrix.toarray()
        progress_bar.progress(1.0)
        
        st.success(f"✅ Generated {embeddings.shape[1]} TF-IDF vectors")
        return embeddings
    except Exception as e:
        st.error(f"Error generating TF-IDF embeddings: {str(e)}")
        st.warning("Generating random vectors as a last resort.")
        random_embeddings = np.random.rand(len(texts), 100)
        return random_embeddings

################################################################
#          CLUSTERING ALGORITHMS
################################################################

def improved_clustering(embeddings, num_clusters=None, min_cluster_size=5):
    """Applies advanced clustering (HDBSCAN + fallback to hierarchical or K-Means)."""
    st.info("Applying advanced clustering algorithms...")
    
    # Auto-determine if num_clusters not provided
    if num_clusters is None:
        try:
            from sklearn.metrics import silhouette_score
            st.info("Determining optimal number of clusters (silhouette)...")
            sil_scores = []
            max_clusters = min(30, len(embeddings) // 5)
            range_n_clusters = range(2, max(3, max_clusters))
            progress_bar = st.progress(0)
            
            from sklearn.cluster import KMeans
            for i, n_clusters in enumerate(range_n_clusters):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=2)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                if len(set(cluster_labels)) > 1:
                    try:
                        if len(embeddings) > 5000:
                            sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
                            sample_score = silhouette_score(embeddings[sample_indices], cluster_labels[sample_indices])
                        else:
                            sample_score = silhouette_score(embeddings, cluster_labels)
                        sil_scores.append(sample_score)
                    except:
                        sil_scores.append(0)
                else:
                    sil_scores.append(0)
                
                progress_bar.progress((i + 1) / len(range_n_clusters))
            
            if sil_scores:
                best_num_clusters = list(range_n_clusters)[np.argmax(sil_scores)]
                st.success(f"Optimal number of clusters: {best_num_clusters}")
                num_clusters = best_num_clusters
            else:
                st.warning("Could not determine the optimal number of clusters. Using default.")
        except Exception as e:
            st.error(f"Error determining optimal number of clusters: {str(e)}")
    
    # HDBSCAN
    if hdbscan_available:
        try:
            st.info("Applying HDBSCAN for natural cluster detection...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            unique_clusters = np.unique(cluster_labels)
            non_noise_clusters = [c for c in unique_clusters if c != -1]
            
            if (len(non_noise_clusters) > 1 
                and len(non_noise_clusters) <= num_clusters * 2):
                st.success(f"HDBSCAN found {len(non_noise_clusters)} natural clusters")
                
                # Reassign noise points to nearest cluster
                if -1 in unique_clusters:
                    noise_indices = np.where(cluster_labels == -1)[0]
                    for idx in noise_indices:
                        min_dist = float('inf')
                        nearest_cluster = non_noise_clusters[0]
                        for cluster in non_noise_clusters:
                            cluster_points = embeddings[cluster_labels == cluster]
                            if len(cluster_points) > 0:
                                centroid = np.mean(cluster_points, axis=0)
                                dist = np.linalg.norm(embeddings[idx] - centroid)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_cluster = cluster
                        cluster_labels[idx] = nearest_cluster
                
                # Remap cluster IDs starting from 1
                old_to_new = {old_id: new_id + 1 for new_id, old_id in enumerate(np.unique(cluster_labels))}
                cluster_labels = np.array([old_to_new[label] for label in cluster_labels])
                return cluster_labels
        except Exception as e:
            st.warning(f"Error with HDBSCAN: {str(e)}. Using hierarchical clustering next.")
    
    # Hierarchical fallback
    try:
        st.info("Applying agglomerative hierarchical clustering...")
        methods = ['ward', 'complete', 'average']
        best_method = 'ward'
        
        if len(embeddings) < 5000:
            coherence_scores = []
            
            for method in methods:
                try:
                    Z = linkage(embeddings, method=method)
                    labels = fcluster(Z, t=num_clusters, criterion="maxclust")
                    # Quick measure of coherence: centroid approach
                    coherence = 0
                    for cid in np.unique(labels):
                        cluster_vectors = embeddings[labels == cid]
                        if len(cluster_vectors) > 1:
                            centroid = np.mean(cluster_vectors, axis=0)
                            dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
                            coherence += np.mean(1 / (1 + dists))
                    coherence_scores.append(coherence / len(np.unique(labels)))
                except:
                    coherence_scores.append(0)
            
            if coherence_scores:
                best_method = methods[np.argmax(coherence_scores)]
                st.success(f"Optimal linkage method: {best_method}")
        
        Z = linkage(embeddings, method=best_method)
        labels = fcluster(Z, t=num_clusters, criterion="maxclust")
        return labels
        
    except Exception as e:
        st.error(f"Hierarchical clustering error: {str(e)}")
        st.warning("Using K-Means as a final fallback.")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings) + 1

def refine_clusters(df, embeddings, original_cluster_column='cluster_id'):
    """Refines clusters by identifying and correcting poor assignments."""
    st.info("Refining clusters to improve coherence...")
    df['original_cluster'] = df[original_cluster_column]
    
    outliers = []
    for cluster_id in df[original_cluster_column].unique():
        cluster_indices = df[df[original_cluster_column] == cluster_id].index.tolist()
        if len(cluster_indices) <= 3:
            continue
        cluster_embs = np.array([embeddings[i] for i in cluster_indices])
        centroid = np.mean(cluster_embs, axis=0)
        distances = [np.linalg.norm(embeddings[i] - centroid) for i in cluster_indices]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist == 0:
            continue
        normalized_distances = [(d - mean_dist) / std_dist for d in distances]
        for i, norm_dist in enumerate(normalized_distances):
            if norm_dist > 2.0:
                outliers.append((cluster_indices[i], cluster_id, norm_dist))
    
    # Reassign outliers
    reassigned = 0
    for idx, original_cluster, _ in outliers:
        keyword_embedding = embeddings[idx]
        min_distance = float('inf')
        best_cluster = original_cluster
        for c_id in df[original_cluster_column].unique():
            if c_id == original_cluster:
                continue
            indices = df[df[original_cluster_column] == c_id].index.tolist()
            cluster_embeddings = np.array([embeddings[i] for i in indices])
            centroid = np.mean(cluster_embeddings, axis=0)
            distance = np.linalg.norm(keyword_embedding - centroid)
            if distance < min_distance:
                min_distance = distance
                best_cluster = c_id
        
        if best_cluster != original_cluster:
            df.loc[idx, original_cluster_column] = best_cluster
            reassigned += 1
    
    # Merge highly similar clusters
    similar_pairs = []
    clusters = df[original_cluster_column].unique()
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            idx1 = df[df[original_cluster_column] == c1].index.tolist()
            idx2 = df[df[original_cluster_column] == c2].index.tolist()
            if len(idx1) < 3 or len(idx2) < 3:
                continue
            emb1 = np.array([embeddings[i] for i in idx1])
            emb2 = np.array([embeddings[i] for i in idx2])
            centroid1 = np.mean(emb1, axis=0)
            centroid2 = np.mean(emb2, axis=0)
            sim = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1)*np.linalg.norm(centroid2))
            if sim > 0.8:
                similar_pairs.append((c1, c2, sim))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    clusters_merged = 0
    processed_clusters = set()
    for (c1, c2, _) in similar_pairs:
        if c1 in processed_clusters or c2 in processed_clusters:
            continue
        keep_id = min(c1, c2)
        remove_id = max(c1, c2)
        df.loc[df[original_cluster_column] == remove_id, original_cluster_column] = keep_id
        processed_clusters.add(remove_id)
        clusters_merged += 1
        if clusters_merged >= len(clusters) // 4:
            break
    
    st.success(f"Refinement done: {reassigned} keywords reassigned, {clusters_merged} clusters merged.")
    return df

################################################################
#          GENERATE CLUSTER NAMES (WITH CUSTOM PROMPT)
################################################################

def generate_cluster_names(
    clusters_with_representatives, 
    client, 
    model="gpt-3.5-turbo",
    custom_prompt=None
):
    """
    Generates SEO-oriented cluster names & descriptions using an English prompt.
    The user can provide a custom prompt; if not, a default English prompt is used.
    Returns a dict {cluster_id: (cluster_name, cluster_description)}.
    """
    if not clusters_with_representatives:
        return {}

    results = {}
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Generating SEO-friendly cluster names/descriptions...")

    # Default prompt if none provided
    if not custom_prompt:
        custom_prompt = (
            "You are an expert in SEO and content marketing. Below you'll see several clusters "
            "with a list of representative keywords. Your task is to assign each cluster a short, "
            "clear name (3-6 words) and write a concise SEO meta description (1 or 2 sentences), "
            "briefly explaining the topic and likely search intent. Respond in JSON.\n\n"
        )

    # Build the actual prompt
    naming_prompt = custom_prompt.strip() + "\n\n"
    naming_prompt += (
        "Return the answer ONLY as a JSON object named 'clusters', containing elements with the fields:\n"
        "  {\n"
        "    \"cluster_id\": <number>,\n"
        "    \"cluster_name\": \"Short SEO name\",\n"
        "    \"cluster_description\": \"Brief SEO description\"\n"
        "  }\n\n"
        "Here are the clusters:\n"
    )
    
    for cluster_id, keywords in clusters_with_representatives.items():
        sample_kws = keywords[:15]
        naming_prompt += f"- Cluster {cluster_id}: {', '.join(sample_kws)}\n"
    
    naming_prompt += "\nPlease respond with ONLY the JSON array 'clusters'. Nothing else."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": naming_prompt}],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        # Try to parse the JSON
        json_data = None
        
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            # Attempt to locate JSON in the text
            match = re.search(r'(\{.*\"clusters\".*\})', content, re.DOTALL)
            if match:
                possible_json = match.group(1)
                possible_json = possible_json.replace("'", '"')
                possible_json = re.sub(r',\s*}', '}', possible_json)
                possible_json = re.sub(r',\s*\]', ']', possible_json)
                try:
                    json_data = json.loads(possible_json)
                except:
                    pass

        if not json_data or "clusters" not in json_data:
            st.warning("Could not parse JSON from GPT response. Showing raw text:")
            st.text_area("GPT Raw Response", content, height=300)
            # Fallback: assign generic names
            for c_id in clusters_with_representatives.keys():
                results[c_id] = (f"Cluster {c_id}", f"Generic description for cluster {c_id}")
            return results

        # Extract cluster data
        cluster_array = json_data["clusters"]
        for item in cluster_array:
            c_id = item.get("cluster_id")
            c_name = item.get("cluster_name", f"Cluster {c_id}")
            c_desc = item.get("cluster_description", "No SEO description provided")
            if c_id is not None:
                results[c_id] = (c_name, c_desc)

    except Exception as e:
        st.error(f"Error generating names with OpenAI: {str(e)}")
        # Fallback
        for c_id in clusters_with_representatives.keys():
            results[c_id] = (f"Cluster {c_id}", f"Fallback description for cluster {c_id}")

    progress_bar.progress(1.0)
    progress_text.text("✅ SEO cluster naming done.")
    return results

################################################################
#          EVALUATION FUNCTIONS
################################################################

def evaluate_cluster_quality(df, embeddings, cluster_column='cluster_id'):
    """Evaluates cluster quality with multiple metrics and plots."""
    st.subheader("Advanced Cluster Quality Evaluation")
    
    metrics = {
        'silhouette': [],
        'density': [],
        'separation': [],
        'coherence': []
    }
    
    centroids = {}
    for cid in df[cluster_column].unique():
        indices = df[df[cluster_column] == cid].index.tolist()
        centroids[cid] = np.mean(np.array([embeddings[i] for i in indices]), axis=0)
    
    cluster_progress = st.progress(0)
    unique_clusters = df[cluster_column].unique()
    
    for i, cluster_id in enumerate(unique_clusters):
        indices = df[df[cluster_column] == cluster_id].index.tolist()
        cluster_vectors = np.array([embeddings[i] for i in indices])
        centroid = centroids[cluster_id]
        
        # 1. Density
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        density = 1 / (1 + np.mean(distances)) if distances else 0
        metrics['density'].append((cluster_id, density))
        
        # 2. Coherence (average cosine similarity)
        coherence = calculate_cluster_coherence(cluster_vectors)
        metrics['coherence'].append((cluster_id, coherence))
        
        # 3. Separation
        min_sep = float('inf')
        for other_id, other_centroid in centroids.items():
            if other_id != cluster_id:
                dist = np.linalg.norm(centroid - other_centroid)
                if dist < min_sep:
                    min_sep = dist
        if min_sep != float('inf'):
            metrics['separation'].append((cluster_id, min_sep))
        
        cluster_progress.progress((i + 1) / len(unique_clusters))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Coherence vs. Size")
        coherence_data = pd.DataFrame(metrics['coherence'], columns=['cluster_id', 'score'])
        size_df = df.groupby(cluster_column)['keyword'].count().reset_index()
        coherence_data = coherence_data.merge(size_df, on='cluster_id')
        coherence_data = coherence_data.merge(
            df.drop_duplicates(cluster_column)[['cluster_id', 'cluster_name']],
            on='cluster_id'
        )
        fig = px.scatter(
            coherence_data,
            x='score',
            y='keyword',
            color='score',
            size='keyword',
            hover_data=['cluster_name'],
            labels={'score': 'Semantic Coherence', 'keyword': 'Cluster Size'},
            title='Coherence vs. Cluster Size',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Separation vs. Density")
        if metrics['separation']:
            separation_data = pd.DataFrame(metrics['separation'], columns=['cluster_id', 'separation'])
            density_data = pd.DataFrame(metrics['density'], columns=['cluster_id', 'density'])
            combined_data = separation_data.merge(density_data, on='cluster_id')
            combined_data = combined_data.merge(
                df.drop_duplicates(cluster_column)[['cluster_id', 'cluster_name']],
                on='cluster_id'
            )
            fig2 = px.scatter(
                combined_data,
                x='separation',
                y='density',
                color='density',
                hover_data=['cluster_name'],
                labels={'separation': 'Inter-cluster Separation', 'density': 'Cluster Density'},
                title='Separation vs. Density',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Cluster Diagnostics")
    coherence_threshold = np.percentile([x[1] for x in metrics['coherence']], 25)
    problematic = [x[0] for x in metrics['coherence'] if x[1] < coherence_threshold]
    
    for cid, cval in metrics['coherence']:
        df.loc[df[cluster_column] == cid, 'cluster_coherence'] = cval
    
    if problematic:
        st.warning(f"Clusters with low semantic coherence: {problematic}")
        st.info("""
        Recommendations to improve:
        - Increase the number of clusters
        - Review the keywords in these clusters
        - Use higher-quality embeddings
        - Consider splitting them manually
        """)
    else:
        st.success("All clusters have good semantic coherence.")
    
    return df

def calculate_cluster_coherence(cluster_embeddings):
    """Calculate average cosine similarity to the centroid for coherence."""
    if len(cluster_embeddings) <= 1:
        return 1.0
    try:
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities = []
        for emb in cluster_embeddings:
            norm_emb = np.linalg.norm(emb)
            norm_centroid = np.linalg.norm(centroid)
            if norm_emb > 0 and norm_centroid > 0:
                sim = np.dot(emb, centroid) / (norm_emb * norm_centroid)
                similarities.append(sim)
            else:
                similarities.append(0.0)
        return np.mean(similarities) if similarities else 0.0
    except Exception as e:
        st.warning(f"Error calculating coherence: {str(e)}")
        return 0.5

################################################################
#          AI-BASED CLUSTER EVALUATION (Optional)
################################################################

def evaluate_and_refine_clusters(df, client, model="gpt-3.5-turbo"):
    """Uses OpenAI to evaluate cluster quality and suggest improvements (sample-based)."""
    st.subheader("AI-Powered Cluster Quality Evaluation")
    unique_clusters = df['cluster_id'].unique()
    if len(unique_clusters) == 0:
        st.warning("No clusters found to evaluate.")
        return {}
    
    # Sample some clusters
    sample_clusters = np.random.choice(unique_clusters, size=min(5, len(unique_clusters)), replace=False)
    eval_progress = st.progress(0)
    eval_results = {}
    
    for i, cid in enumerate(sample_clusters):
        cluster_keywords = df[df['cluster_id'] == cid]['keyword'].tolist()
        cluster_name = df[df['cluster_id'] == cid]['cluster_name'].iloc[0]
        sample_keywords = cluster_keywords[:50]
        
        prompt = f"""Evaluate the semantic coherence of the following keyword cluster and its name:

Cluster name: {cluster_name}
Sample keywords: {', '.join(sample_keywords)}

Return a JSON with:
- coherence_score (1-10)
- outlier_keywords (list of strings)
- needs_splitting (boolean)
- improvement_suggestions (list of strings)

Example:
{{
  "coherence_score": 8,
  "outlier_keywords": ["keyword1", "keyword2"],
  "needs_splitting": false,
  "improvement_suggestions": ["Suggestion 1", "Suggestion 2"]
}}
"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            
            content = response.choices[0].message.content
            try:
                evaluation_data = json.loads(content)
            except:
                # If JSON parse fails, store raw
                evaluation_data = {"error": "Could not parse JSON", "raw_text": content}
            
            eval_results[cid] = {
                "name": cluster_name,
                "evaluation": evaluation_data
            }
            
            with st.expander(f"Cluster {cid}: {cluster_name}", expanded=False):
                st.write(evaluation_data)
            
        except Exception as e:
            st.error(f"Error evaluating cluster {cid}: {str(e)}")
        
        eval_progress.progress((i + 1) / len(sample_clusters))
    
    # Overall summary
    if eval_results:
        st.subheader("Overall Cluster Quality Assessment")
        coherence_scores = []
        for data in eval_results.values():
            score = data["evaluation"].get("coherence_score", 0)
            if isinstance(score, (int, float)):
                coherence_scores.append(score)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        needs_splitting = sum(1 for d in eval_results.values() if d["evaluation"].get("needs_splitting", False))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Coherence", f"{avg_coherence:.1f}/10")
        with col2:
            st.metric("Clusters Needing Split", f"{needs_splitting}/{len(eval_results)}")
        
        if avg_coherence < 6:
            st.warning("⚠️ Overall coherence is low. Consider increasing the number of clusters.")
        elif avg_coherence > 8:
            st.success("✅ Overall coherence is quite good.")
        else:
            st.info("ℹ️ Moderate coherence. Consider refining based on individual suggestions.")
    
    return eval_results

################################################################
#          MAIN CLUSTERING PIPELINE
################################################################

def run_clustering(
    uploaded_file, 
    openai_api_key, 
    num_clusters, 
    pca_variance, 
    max_pca_components, 
    min_df, 
    max_df, 
    gpt_model,
    user_prompt
):
    """Executes the full clustering pipeline and returns the final DataFrame."""
    if uploaded_file is None:
        st.warning("Please upload a CSV file with keywords.")
        return False, None
    
    st.info("Starting advanced semantic clustering pipeline...")
    
    client = None
    if openai_api_key and openai_available:
        try:
            if openai_api_key.strip() == "":
                st.info("No valid OpenAI API Key provided. Will use alternative methods.")
            else:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                client = OpenAI()
                # Simple connectivity check
                try:
                    _ = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=5
                    )
                    st.success("✅ Connected to OpenAI successfully.")
                except Exception as e:
                    st.error(f"Error checking OpenAI connection: {str(e)}")
                    client = None
        except Exception as e:
            st.error(f"Error configuring OpenAI client: {str(e)}")
            client = None
    elif not openai_available:
        st.warning("OpenAI library is not installed. No OpenAI functionality available.")
    else:
        st.info("No OpenAI API Key provided. Will use free alternatives.")
    
    try:
        # Load CSV
        try:
            df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
            num_keywords = len(df)
            st.success(f"✅ Loaded {num_keywords} keywords from CSV.")
            show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            st.info("Trying alternative format (sep=None, engine='python')...")
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                df = pd.read_csv(StringIO(content), sep=None, engine='python', header=None)
                df.columns = ["keyword"]
                num_keywords = len(df)
                st.success(f"✅ Loaded {num_keywords} keywords from CSV (alternative).")
                show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
            except Exception as e2:
                st.error(f"Could not read CSV: {str(e2)}")
                return False, None
        
        # Preprocessing
        st.subheader("Keyword Preprocessing")
        st.info("Preprocessing keywords with advanced NLP...")
        use_advanced = spacy_available
        if use_advanced:
            st.success("Using advanced spaCy analysis")
        else:
            st.info("Using standard NLTK/TextBlob analysis")
        
        keywords_processed = preprocess_keywords(df["keyword"].tolist(), use_advanced=use_advanced)
        df['keyword_processed'] = keywords_processed
        st.success("✅ Preprocessing complete.")
        
        # Embeddings
        st.subheader("Generating Semantic Vectors (Embeddings)")
        keyword_embeddings = generate_embeddings(df, openai_available, openai_api_key)
        
        # Dimensionality reduction (PCA)
        if keyword_embeddings.shape[1] > max_pca_components:
            st.subheader("Dimensionality Reduction (PCA)")
            try:
                pca_progress = st.progress(0)
                pca_text = st.empty()
                pca_text.text("Analyzing PCA explained variance...")
                
                pca = PCA()
                pca.fit(keyword_embeddings)
                cum_var = np.cumsum(pca.explained_variance_ratio_)
                pca_progress.progress(0.3)
                
                target_var = pca_variance / 100.0
                n_components = np.argmax(cum_var >= target_var) + 1
                if n_components == 1 and len(cum_var) > 1:
                    n_components = min(max_pca_components, len(cum_var))
                
                pca_text.text(f"Components for {pca_variance}% variance: {n_components}")
                pca_progress.progress(0.6)
                
                max_components = min(n_components, max_pca_components)
                pca = PCA(n_components=max_components)
                keyword_embeddings_reduced = pca.fit_transform(keyword_embeddings)
                pca_progress.progress(1.0)
                pca_text.text(f"✅ PCA applied: {max_components} dimensions (covering ~{pca_variance}% variance)")
            except Exception as e:
                st.error(f"Error applying PCA: {str(e)}")
                st.info("Proceeding without PCA.")
                keyword_embeddings_reduced = keyword_embeddings
        else:
            keyword_embeddings_reduced = keyword_embeddings
            st.info(f"No PCA needed (dimension is {keyword_embeddings.shape[1]}).")
        
        # Clustering
        st.subheader("Advanced Semantic Clustering")
        try:
            cluster_labels = improved_clustering(keyword_embeddings_reduced, num_clusters=num_clusters)
            df["cluster_id"] = cluster_labels
            st.success(f"✅ {len(df['cluster_id'].unique())} clusters created.")
        except Exception as e:
            st.error(f"Error in advanced clustering: {str(e)}")
            st.info("Trying basic K-Means fallback...")
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                df["cluster_id"] = kmeans.fit_predict(keyword_embeddings_reduced) + 1
                st.success("✅ K-Means fallback succeeded.")
            except Exception as e2:
                st.error(f"Error in K-Means fallback: {str(e2)}")
                st.warning("Assigning random clusters as a final resort.")
                df["cluster_id"] = np.random.randint(1, num_clusters + 1, size=len(df))
        
        # Refinement
        st.subheader("Cluster Refinement")
        df = refine_clusters(df, keyword_embeddings_reduced)
        final_clusters = len(df['cluster_id'].unique())
        st.success(f"✅ Refinement complete: {final_clusters} final clusters.")
        
        # Representative keywords
        st.subheader("Representative Keywords")
        rep_progress = st.progress(0)
        rep_text = st.empty()
        rep_text.text("Finding representative keywords...")
        clusters_with_representatives = {}
        
        try:
            unique_cluster_ids = df['cluster_id'].unique()
            for i, cnum in enumerate(unique_cluster_ids):
                csize = len(df[df['cluster_id'] == cnum])
                n_rep = min(20, csize)
                indices = df[df['cluster_id'] == cnum].index.tolist()
                c_embs = np.array([keyword_embeddings_reduced[idx] for idx in indices])
                centroid = np.mean(c_embs, axis=0)
                distances = [np.linalg.norm(keyword_embeddings_reduced[idx] - centroid) for idx in indices]
                sorted_indices = np.argsort(distances)[:n_rep]
                rep_indices = [indices[idx] for idx in sorted_indices]
                rep_kws = df.loc[rep_indices, 'keyword'].tolist()
                clusters_with_representatives[cnum] = rep_kws
                rep_progress.progress((i+1) / len(unique_cluster_ids))
            
            rep_progress.progress(1.0)
            rep_text.text(f"✅ Representative keywords identified for {len(clusters_with_representatives)} clusters.")
        except Exception as e:
            st.error(f"Error finding representative keywords: {str(e)}")
            for cnum in df['cluster_id'].unique():
                cluster_kws = df[df['cluster_id'] == cnum]['keyword'].tolist()
                clusters_with_representatives[cnum] = cluster_kws[:min(20, len(cluster_kws))]
            st.warning("Using basic fallback for representatives.")
        
        # Generate cluster names & descriptions
        if client:
            st.subheader("Generating Cluster Names & Descriptions (SEO-focused)")
            try:
                cluster_names = generate_cluster_names(
                    clusters_with_representatives, 
                    client, 
                    model=gpt_model,
                    custom_prompt=user_prompt  # <--- we pass the user-defined prompt here
                )
            except Exception as e:
                st.error(f"Error generating cluster names: {str(e)}")
                cluster_names = {k: (f"Cluster {k}", f"Keywords group {k}") 
                                 for k in df['cluster_id'].unique()}
        else:
            st.warning("No OpenAI client available. Using generic cluster names.")
            cluster_names = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
        
        # Apply cluster names & descriptions
        df['cluster_name'] = ''
        df['cluster_description'] = ''
        df['representative'] = False
        for cnum, (name, desc) in cluster_names.items():
            df.loc[df['cluster_id'] == cnum, 'cluster_name'] = name
            df.loc[df['cluster_id'] == cnum, 'cluster_description'] = desc
            for kw in clusters_with_representatives.get(cnum, []):
                match_idx = df[(df['cluster_id'] == cnum) & (df['keyword'] == kw)].index
                if not match_idx.empty:
                    df.loc[match_idx, 'representative'] = True
        
        # Evaluate cluster quality
        df = evaluate_cluster_quality(df, keyword_embeddings_reduced)
        
        # AI-powered cluster evaluation (optional)
        if client:
            try:
                eval_results = evaluate_and_refine_clusters(df, client, model=gpt_model)
                st.session_state.cluster_evaluation = eval_results
            except Exception as e:
                st.error(f"Error during AI-driven evaluation: {str(e)}")
        
        return True, df
    
    except Exception as e:
        st.error(f"Error in the clustering pipeline: {str(e)}")
        return False, None
    
    return True, None

################################################################
#          MAIN STREAMLIT APP
################################################################

st.set_page_config(
    page_title="Advanced Semantic Keyword Clustering",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #fffbcc;
        padding: 0.2rem 0.5rem;
        border-radius: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Advanced Semantic Keyword Clustering</div>", unsafe_allow_html=True)
st.markdown("""
This application clusters semantically similar keywords using advanced NLP and clustering methods. 
Upload a CSV of keywords, configure parameters, and get high semantic correlation clusters.
""")

with st.expander("Semantic Libraries Status", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        if openai_available:
            st.success("✅ OpenAI available")
        else:
            st.warning("⚠️ OpenAI not installed")
        
        if sentence_transformers_available:
            st.success("✅ SentenceTransformers available")
        else:
            st.warning("⚠️ SentenceTransformers not installed\n```\npip install sentence-transformers\n```")
    with col2:
        if spacy_available:
            st.success("✅ SpaCy available")
        else:
            st.warning("⚠️ SpaCy not available")
        
        if hdbscan_available:
            st.success("✅ HDBSCAN available")
        else:
            st.warning("⚠️ HDBSCAN not available")
    with col3:
        st.info("Install them for more advanced features")

# Session state
if 'process_complete' not in st.session_state:
    st.session_state.process_complete = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

st.sidebar.markdown("<div class='sub-header'>Configuration</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your keywords CSV", type=['csv'])

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key (optional)",
    type="password",
    help="Enter your OpenAI API Key for high-quality embeddings. If omitted, free SentenceTransformers or TF-IDF will be used."
)

if openai_available:
    if openai_api_key:
        st.sidebar.success("✅ OpenAI key provided - will use OpenAI for embeddings")
    else:
        if sentence_transformers_available:
            st.sidebar.info("No OpenAI key - will use SentenceTransformers (free)")
        else:
            st.sidebar.warning("No OpenAI key, no SentenceTransformers - fallback to TF-IDF (less precision)")
else:
    if sentence_transformers_available:
        st.sidebar.info("OpenAI not installed - using SentenceTransformers")
    else:
        st.sidebar.error("No advanced embedding method - TF-IDF fallback")

st.sidebar.markdown("<div class='sub-header'>Parameters</div>", unsafe_allow_html=True)

with st.sidebar.expander("ℹ️ Parameters Guide", expanded=False):
    st.markdown("""
    ### Clustering Parameters
    **Number of clusters**: how many groups to form.
    **PCA explained variance**: how much information to keep (in %) during dimensionality reduction.
    **Max PCA components**: upper bound on PCA dimensions.
    **Minimum term frequency**: ignore extremely rare words.
    **Maximum term frequency (%)**: ignore extremely common words.
    """)
    st.info("If unsure, keep defaults. For large datasets, consider more clusters or lower PCA variance.")

num_clusters = st.sidebar.slider("Number of clusters", 2, 50, 10)
pca_variance = st.sidebar.slider("PCA explained variance (%)", 50, 99, 95)
max_pca_components = st.sidebar.slider("Max PCA components", 10, 300, 100)

st.sidebar.markdown("<div class='sub-header'>Advanced options</div>", unsafe_allow_html=True)
min_df = st.sidebar.slider("Minimum term frequency", 1, 10, 1)
max_df = st.sidebar.slider("Maximum term frequency (%)", 50, 100, 95)
gpt_model = st.sidebar.selectbox("Model for naming clusters", ["gpt-3.5-turbo", "gpt-4"], index=0)

# ------------------------------------
# Language & Custom Prompt section
# ------------------------------------
st.sidebar.markdown("### Language & Custom Prompt for SEO Naming")
language_option = st.sidebar.selectbox(
    "Target Language (for reference only)",
    ("English", "Spanish", "French", "German", "Portuguese"),
    index=0
)

default_prompt = (
    "You are an expert in SEO and content marketing. Below you'll see several clusters "
    "with a list of representative keywords. Your task is to assign each cluster a short, "
    "clear name (3-6 words) and write a concise SEO meta description (1 or 2 sentences) "
    "briefly explaining the topic and likely search intent. Respond only in JSON."
)
user_prompt = st.sidebar.text_area(
    "Custom Prompt for Cluster Naming",
    value=default_prompt,
    height=200
)

add_cost_calculator()

# Trigger process
if uploaded_file is not None and not st.session_state.process_complete:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Advanced Semantic Clustering", type="primary", use_container_width=True):
            success, results = run_clustering(
                uploaded_file,
                openai_api_key,
                num_clusters,
                pca_variance,
                max_pca_components,
                min_df,
                max_df,
                gpt_model,
                user_prompt
            )
            if success and results is not None:
                st.session_state.df_results = results
                st.session_state.process_complete = True
                st.markdown("<div class='success-box'>✅ Semantic clustering completed successfully!</div>", unsafe_allow_html=True)

# If process is complete, show results
if st.session_state.process_complete and st.session_state.df_results is not None:
    st.markdown("<div class='main-header'>Clustering Results</div>", unsafe_allow_html=True)
    df = st.session_state.df_results
    
    with st.expander("Visualizations", expanded=True):
        st.subheader("Cluster Distribution")
        cluster_sizes = df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
        cluster_sizes['label'] = cluster_sizes.apply(lambda x: f"{x['cluster_name']} (ID: {x['cluster_id']})", axis=1)
        
        fig = px.bar(
            cluster_sizes,
            x='label',
            y='count',
            color='count',
            labels={'count': 'Number of Keywords', 'label': 'Cluster'},
            title='Size of Each Cluster',
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Semantic Coherence of Clusters")
        coherence_data = df.groupby(['cluster_id', 'cluster_name'])['cluster_coherence'].mean().reset_index()
        coherence_data['label'] = coherence_data.apply(lambda x: f"{x['cluster_name']} (ID: {x['cluster_id']})", axis=1)
        
        fig2 = px.bar(
            coherence_data,
            x='label',
            y='cluster_coherence',
            color='cluster_coherence',
            labels={'cluster_coherence': 'Coherence', 'label': 'Cluster'},
            title='Semantic Coherence by Cluster',
            color_continuous_scale=px.colors.sequential.Greens
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Explore Clusters", expanded=True):
        cluster_options = [
            f"{row['cluster_name']} (ID: {row['cluster_id']})"
            for _, row in df.drop_duplicates(['cluster_id', 'cluster_name'])[['cluster_id', 'cluster_name']].iterrows()
        ]
        selected_cluster = st.selectbox("Select a cluster to explore:", cluster_options)
        
        if selected_cluster:
            cid = int(selected_cluster.split("ID: ")[1].split(")")[0])
            cluster_df = df[df['cluster_id'] == cid].copy()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {cluster_df['cluster_name'].iloc[0]}")
                st.markdown(f"**Description:** {cluster_df['cluster_description'].iloc[0]}")
                st.markdown(f"**Total keywords:** {len(cluster_df)}")
            with col2:
                st.markdown(f"**Semantic coherence:** {cluster_df['cluster_coherence'].iloc[0]:.3f}")
                st.markdown("**Representative keywords:**")
                reps = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                if reps:
                    st.markdown("<ul>" + "".join([f"<li>{kw}</li>" for kw in reps[:10]]) + "</ul>", unsafe_allow_html=True)
            
            st.markdown("### All keywords in this cluster")
            st.dataframe(cluster_df[['keyword']], use_container_width=True)

    with st.expander("Download Results"):
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_data,
            file_name="semantic_clustered_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.subheader("Clusters Summary")
        summary_df = df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
        summary_df.columns = ['ID', 'Name', 'Description', 'Number of Keywords']
        
        coherence_df = df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id')
        summary_df.drop('cluster_id', axis=1, inplace=True)
        summary_df.rename(columns={'cluster_coherence': 'Coherence'}, inplace=True)
        
        def get_rep_keywords(cid):
            reps = df[(df['cluster_id'] == cid) & (df['representative'] == True)]['keyword'].tolist()
            return ', '.join(reps[:5])
        
        summary_df['Representative Keywords'] = summary_df['ID'].apply(get_rep_keywords)
        st.dataframe(summary_df, use_container_width=True)
        
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Clusters Summary",
            data=csv_summary,
            file_name="semantic_clusters_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

if st.session_state.process_complete:
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state.process_complete = False
        st.session_state.df_results = None
        st.experimental_rerun()

with st.expander("More Information about Advanced Semantic Clustering"):
    st.markdown("""
    ### How does it work?
    1. **Linguistic Preprocessing** (spaCy/TextBlob/NLTK).
    2. **Embeddings** (OpenAI if key provided, else SentenceTransformers, else TF-IDF).
    3. **Dimensionality Reduction** (PCA, optional).
    4. **Clustering** (HDBSCAN, hierarchical, or fallback to K-Means).
    5. **Refinement** (outlier detection, merging similar clusters).
    6. **Evaluation** (coherence, density, separation).
    
    ### Tips
    - For best quality, use an OpenAI API Key for embeddings (limited to 5,000 direct embeddings).
    - For more than 5,000 keywords, the rest are assigned via similarity propagation.
    - Increase the number of clusters for more granularity.
    - Pay attention to clusters with low coherence—try splitting or refining them.
    """)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Developed for advanced semantic keyword clustering | Version 2.2 with custom prompts
    </div>
    """, 
    unsafe_allow_html=True
)
