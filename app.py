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

# Try to import advanced libraries with error handling
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")  # You can also use es_core_news_sm for Spanish
        spacy_available = True
    except:
        # If the model is not downloaded
        spacy_available = False
except ImportError:
    spacy_available = False

# Try to import TextBlob as an alternative to spaCy
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

# Download NLTK resources at start to avoid later problems
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    pass  # Continue even if download fails

# Function to calculate the estimated API cost
def calculate_api_cost(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    """
    Calculates the estimated cost of using the OpenAI API based on the number of keywords
    
    Args:
        num_keywords: Total number of keywords
        selected_model: Model for naming clusters (gpt-3.5-turbo or gpt-4)
        num_clusters: Estimated number of clusters
    
    Returns:
        dict: Breakdown of costs by component and total cost
    """
    # Updated prices (March 2025)
    EMBEDDING_COST_PER_1K = 0.02  # text-embedding-3-small per 1K tokens
    
    # GPT-3.5-Turbo costs
    GPT35_INPUT_COST_PER_1K = 0.0005
    GPT35_OUTPUT_COST_PER_1K = 0.0015
    
    # GPT-4 costs
    GPT4_INPUT_COST_PER_1K = 0.03
    GPT4_OUTPUT_COST_PER_1K = 0.06
    
    # Results
    results = {
        "embedding_cost": 0,
        "naming_cost": 0,
        "total_cost": 0,
        "processed_keywords": 0
    }
    
    # 1. Embedding cost - limited to 5000 keywords
    keywords_for_embeddings = min(num_keywords, 5000)
    results["processed_keywords"] = keywords_for_embeddings
    
    # We estimate an average of 2 tokens per keyword (some will have 1, others more)
    estimated_tokens = keywords_for_embeddings * 2
    results["embedding_cost"] = (estimated_tokens / 1000) * EMBEDDING_COST_PER_1K
    
    # 2. Cost of naming clusters
    # We estimate tokens for naming clusters (depends on the number of clusters)
    # The prompt for analysis + representative keywords (approx. 15 per cluster) + Response
    avg_tokens_per_cluster = 200  # Tokens per cluster for input (including keywords)
    avg_output_tokens_per_cluster = 80  # Output tokens per cluster (name + description in JSON)
    
    estimated_input_tokens = num_clusters * avg_tokens_per_cluster
    estimated_output_tokens = num_clusters * avg_output_tokens_per_cluster
    
    if selected_model == "gpt-3.5-turbo":
        input_cost = (estimated_input_tokens / 1000) * GPT35_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT35_OUTPUT_COST_PER_1K
    else:  # gpt-4
        input_cost = (estimated_input_tokens / 1000) * GPT4_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT4_OUTPUT_COST_PER_1K
    
    results["naming_cost"] = input_cost + output_cost
    
    # 3. Total cost
    results["total_cost"] = results["embedding_cost"] + results["naming_cost"]
    
    return results

# Cost calculator widget for Streamlit
def add_cost_calculator():
    st.sidebar.markdown("---")
    with st.sidebar.expander("💰 API Cost Calculator", expanded=False):
        st.markdown("""
        ### API Cost Calculator
        
        Calculate the approximate cost of processing your keywords with OpenAI.
        """)
        
        # Number of keywords input
        calc_num_keywords = st.number_input(
            "Number of keywords to process", 
            min_value=100, 
            max_value=100000, 
            value=1000,
            step=500
        )
        
        # Number of clusters input
        calc_num_clusters = st.number_input(
            "Approximate number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            step=1
        )
        
        # Model selection
        calc_model = st.radio(
            "Model for naming clusters",
            options=["gpt-3.5-turbo", "gpt-4"],
            index=0,
            horizontal=True
        )
        
        # Calculate button
        if st.button("Calculate Estimated Cost", use_container_width=True):
            cost_results = calculate_api_cost(calc_num_keywords, calc_model, calc_num_clusters)
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Keywords processed with OpenAI", 
                    f"{cost_results['processed_keywords']:,}",
                    help="OpenAI processes up to 5,000 keywords, the rest are propagated through similarity"
                )
                
                st.metric(
                    "Embeddings cost", 
                    f"${cost_results['embedding_cost']:.4f}",
                    help="Cost of generating semantic vectors with text-embedding-3-small"
                )
            
            with col2:
                st.metric(
                    "Cluster naming cost", 
                    f"${cost_results['naming_cost']:.4f}",
                    help=f"Cost of generating names and descriptions using {calc_model}"
                )
                
                st.metric(
                    "TOTAL COST", 
                    f"${cost_results['total_cost']:.4f}",
                    help="Estimated total API cost (does not include computational resources)"
                )
            
            st.info("""
            **Note:** This is an approximate estimate. The actual cost may vary 
            depending on the length of the keywords and the complexity of the clusters.
            Using Sentence Transformers as an alternative, the cost is $0.
            """)

# Function to display estimated cost of the loaded CSV
def show_csv_cost_estimate(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    if num_keywords > 0:
        cost_results = calculate_api_cost(num_keywords, selected_model, num_clusters)
        
        with st.sidebar.expander("💰 Estimated Cost (Current CSV)", expanded=True):
            st.markdown(f"### Estimated Cost for {num_keywords:,} Keywords")
            
            # Show breakdown
            st.markdown(f"""
            - **Keywords processed with OpenAI**: {cost_results['processed_keywords']:,}
            - **Embeddings cost**: ${cost_results['embedding_cost']:.4f}
            - **Cluster naming cost**: ${cost_results['naming_cost']:.4f}
            - **TOTAL COST**: ${cost_results['total_cost']:.4f}
            """)
            
            if cost_results['processed_keywords'] < num_keywords:
                st.info(f"""
                {cost_results['processed_keywords']:,} keywords will be processed directly with OpenAI.
                The remaining {num_keywords - cost_results['processed_keywords']:,} will be processed
                using semantic similarity propagation.
                """)
            
            st.markdown("""
            **Cost Savings**: If you prefer not to use OpenAI, you can 
            use SentenceTransformers as a free alternative with 
            good results.
            """)

# IMPROVEMENT 4: Enhanced Semantic Preprocessing
def enhanced_preprocessing(text, use_lemmatization=True):
    """Enhanced preprocessing with entity treatment and n-grams"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Option 1: Use spaCy for more advanced linguistic analysis
        if spacy_available:
            doc = nlp(text.lower())
            
            # Preserve complete named entities
            entities = [ent.text for ent in doc.ents]
            
            # Extract relevant tokens (not stopwords)
            tokens = []
            for token in doc:
                if not token.is_stop and token.is_alpha and len(token.text) > 1:
                    tokens.append(token.lemma_)
            
            # Extract relevant bigrams
            bigrams = []
            for i in range(len(doc) - 1):
                if (not doc[i].is_stop and not doc[i+1].is_stop and 
                    doc[i].is_alpha and doc[i+1].is_alpha):
                    bigrams.append(f"{doc[i].lemma_}_{doc[i+1].lemma_}")
            
            # Combine everything preserving entities
            processed_parts = tokens + bigrams + entities
            return " ".join(processed_parts)
        
        # Option 2: Use TextBlob as a simpler alternative
        elif textblob_available:
            blob = TextBlob(text.lower())
            # Extract nouns and noun phrases (good for keywords)
            noun_phrases = list(blob.noun_phrases)
            # Tokens without stopwords
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'in', 'on', 'to', 'for'}
            
            words = [word for word in blob.words if len(word) > 1 and word.lower() not in stop_words]
            
            # Lemmatization if required
            if use_lemmatization:
                lemmas = [word.lemmatize() for word in words]
                processed_parts = lemmas + noun_phrases
            else:
                processed_parts = words + noun_phrases
            
            return " ".join(processed_parts)
        
        # Option 3: Fallback to the original method with NLTK
        else:
            return preprocess_text(text, use_lemmatization)
    except Exception as e:
        return text.lower() if isinstance(text, str) else ""

# Original preprocessing function as fallback
def preprocess_text(text, use_lemmatization=True):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Load stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Basic fallback if stopwords can't be loaded
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'in', 'on', 'to', 'for'}
        
        # Filter stopwords
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        # Lemmatization (optional)
        if use_lemmatization:
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            except Exception as e:
                pass  # Continue without lemmatization if it fails
        
        return " ".join(tokens)
    except Exception as e:
        # Error handling to ensure something is always returned
        return text.lower() if isinstance(text, str) else ""

# Function to preprocess keywords
def preprocess_keywords(keywords, use_advanced=True):
    processed_keywords = []
    
    progress_bar = st.progress(0)
    total = len(keywords)
    
    # Report which preprocessing method is being used
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
        
        # Update progress bar every 100 items
        if i % 100 == 0:
            progress_bar.progress(min(i / total, 1.0))
    
    progress_bar.progress(1.0)
    return processed_keywords

# IMPROVEMENT 1: Enhanced embeddings with OpenAI priority and 5000 keywords limit
def generate_embeddings(df, openai_available, openai_api_key=None):
    st.info("Generating embeddings for keywords...")
    
    # Option 1: Use OpenAI if available and API key is provided
    if openai_available and openai_api_key:
        try:
            st.info("Using OpenAI embeddings (high semantic precision)")
            # Configure OpenAI
            os.environ["OPENAI_API_KEY"] = openai_api_key
            client = OpenAI()
            
            # Process in batches to minimize costs
            keywords = df['keyword_processed'].fillna('').tolist()
            all_embeddings = []
            
            # Increased to 5000 keywords (instead of 1000)
            if len(keywords) > 5000:
                st.warning(f"Limiting to 5000 representative keywords out of {len(keywords)} total")
                # Select keywords strategically (not just the first ones)
                step = max(1, len(keywords) // 5000)
                sample_indices = list(range(0, len(keywords), step))[:5000]
                sample_keywords = [keywords[i] for i in sample_indices]
                
                progress_bar = st.progress(0)
                st.info("Processing embeddings with OpenAI (this may take a few minutes)...")
                
                # Create embeddings for sample
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=sample_keywords
                )
                progress_bar.progress(0.5)
                
                # Extract embeddings
                sample_embeddings = np.array([item.embedding for item in response.data])
                
                # Propagate embeddings to the rest by TF-IDF similarity
                st.info("Propagating embeddings to the rest of keywords...")
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(keywords)
                
                all_embeddings = np.zeros((len(keywords), len(sample_embeddings[0])))
                # Assign embeddings to the sample
                for i, idx in enumerate(sample_indices):
                    all_embeddings[idx] = sample_embeddings[i]
                
                # Process remaining keywords using a more sophisticated approach with top-3 nearest neighbors
                remaining_indices = [i for i in range(len(keywords)) if i not in sample_indices]

                # Use top-3 similar embeddings instead of just the most similar one
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(3, len(sample_indices)))
                nn.fit(tfidf_matrix[sample_indices])

                for i, idx in enumerate(remaining_indices):
                    # Get the 3 most similar keywords
                    distances, neighbors = nn.kneighbors(tfidf_matrix[idx:idx+1])
                    
                    # Calculate weighted embedding based on similarity
                    weights = 1.0 / (1.0 + distances[0])
                    weights = weights / weights.sum()  # Normalize weights
                    
                    # Apply embedding as weighted average of the 3 most similar ones
                    weighted_embedding = np.zeros_like(sample_embeddings[0])
                    for j, weight in zip(neighbors[0], weights):
                        similar_idx = sample_indices[j]
                        weighted_embedding += weight * all_embeddings[similar_idx]
                    
                    all_embeddings[idx] = weighted_embedding
                    
                    # Update progress for the second half
                    if i % 100 == 0:
                        progress_bar.progress(0.5 + min(0.5, (i / len(remaining_indices) * 0.5)))
                
                progress_bar.progress(1.0)
            else:
                # If less than 5000, process all
                progress_bar = st.progress(0)
                st.info(f"Processing embeddings for all {len(keywords)} keywords with OpenAI...")
                
                # Process in batches of 1000 to avoid API limits
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
            st.success(f"✅ Generated embeddings of {embeddings.shape[1]} dimensions using OpenAI")
            return embeddings
                
        except Exception as e:
            st.error(f"Error generating embeddings with OpenAI: {str(e)}")
            st.info("Trying with Sentence Transformers as an alternative...")

    # Option 2: Use Sentence Transformers as fallback (free)
    if sentence_transformers_available:
        try:
            st.success("Using SentenceTransformer as fallback (free)")
            
            # Use a multilingual model if your keywords are in multiple languages
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            progress_bar = st.progress(0)
            keywords = df['keyword_processed'].fillna('').tolist()
            
            # Process in batches to avoid memory issues
            batch_size = 512
            all_embeddings = []
            
            for i in range(0, len(keywords), batch_size):
                batch = keywords[i:i+batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                progress_bar.progress(min(1.0, (i + batch_size) / len(keywords)))
                
            progress_bar.progress(1.0)
            embeddings = np.array(all_embeddings)
            st.success(f"✅ Generated embeddings of {embeddings.shape[1]} dimensions using SentenceTransformer")
            return embeddings
        except Exception as e:
            st.error(f"Error with SentenceTransformer: {str(e)}")
    
    # Option 3: Fallback to TF-IDF (less precise) as a last resort
    st.warning("Using TF-IDF as a last resort (less semantically precise)")
    return generate_tfidf_embeddings(df['keyword_processed'].fillna(''))

# Original TF-IDF function as fallback
def generate_tfidf_embeddings(texts, min_df=1, max_df=0.95):
    st.info("Generating TF-IDF vectors for keywords...")
    progress_bar = st.progress(0)
    
    try:
        # Create a vectorizer with configurable parameters
        vectorizer = TfidfVectorizer(
            max_features=300,  # Limit features to prevent memory issues
            min_df=min_df,     # Ignore terms that appear in fewer than N documents
            max_df=max_df,     # Ignore terms that appear in more than N% of documents
            stop_words='english'
        )
        
        # Ensure there are no null values
        clean_texts = [t if isinstance(t, str) and t else " " for t in texts]
        
        # Generate TF-IDF matrix
        progress_bar.progress(0.3)
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
        progress_bar.progress(0.8)
        
        # Convert to dense array
        embeddings = tfidf_matrix.toarray()
        progress_bar.progress(1.0)
        
        st.success(f"✅ Generated {embeddings.shape[1]} TF-IDF vectors")
        return embeddings
    except Exception as e:
        st.error(f"Error generating TF-IDF embeddings: {str(e)}")
        # Last resort: generate random vectors
        st.warning("Generating random vectors as a last resort")
        random_embeddings = np.random.rand(len(texts), 100)
        return random_embeddings
# IMPROVEMENT 2: Enhanced clustering algorithm
def improved_clustering(embeddings, num_clusters=None, min_cluster_size=5):
    st.info("Applying advanced clustering algorithms...")
    
    # Automatically determine the optimal number of clusters if not specified
    if num_clusters is None:
        try:
            from sklearn.metrics import silhouette_score
            
            st.info("Finding optimal number of clusters...")
            sil_scores = []
            max_clusters = min(30, len(embeddings) // 5)
            range_n_clusters = range(2, max(3, max_clusters))
            
            progress_bar = st.progress(0)
            
            # Calculate silhouette score for different numbers of clusters
            for i, n_clusters in enumerate(range_n_clusters):
                # Use K-Means for testing as it's faster
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=2)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette (if there are enough samples)
                if len(set(cluster_labels)) > 1:
                    try:
                        # Use a sample to calculate silhouette if there is a lot of data
                        if len(embeddings) > 5000:
                            sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
                            sample_score = silhouette_score(
                                embeddings[sample_indices], 
                                cluster_labels[sample_indices]
                            )
                        else:
                            sample_score = silhouette_score(embeddings, cluster_labels)
                        sil_scores.append(sample_score)
                    except:
                        sil_scores.append(0)
                else:
                    sil_scores.append(0)
                    
                progress_bar.progress((i + 1) / len(range_n_clusters))
                    
            # Select the number of clusters with the best score
            if sil_scores:
                best_num_clusters = range_n_clusters[np.argmax(sil_scores)]
                st.success(f"Optimal number of clusters determined: {best_num_clusters}")
                num_clusters = best_num_clusters
            else:
                st.warning("Could not determine the optimal number of clusters. Using default value.")
        except Exception as e:
            st.error(f"Error determining optimal number of clusters: {str(e)}")
    
    # Try HDBSCAN if available (better for irregular shaped clusters)
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
            
            # Check if HDBSCAN found a reasonable structure
            # Limit the maximum number of clusters to a reasonable value
            unique_clusters = np.unique(cluster_labels)
            non_noise_clusters = [c for c in unique_clusters if c != -1]
            
            if len(non_noise_clusters) > 1 and len(non_noise_clusters) <= num_clusters * 2:
                st.success(f"HDBSCAN identified {len(non_noise_clusters)} natural clusters")
                
                # Reassign cluster -1 (noise) to the nearest cluster
                if -1 in unique_clusters:
                    noise_indices = np.where(cluster_labels == -1)[0]
                    for idx in noise_indices:
                        # Find the nearest centroid
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
                
                # Reassign IDs to start from 1
                old_to_new = {old_id: new_id + 1 for new_id, old_id in enumerate(np.unique(cluster_labels))}
                cluster_labels = np.array([old_to_new[label] for label in cluster_labels])
                
                return cluster_labels
        except Exception as e:
            st.warning(f"Error with HDBSCAN: {str(e)}. Using hierarchical clustering.")
            
    # Fallback to hierarchical clustering
    try:
        st.info("Applying agglomerative hierarchical clustering...")
        # Try different linkage methods to find the best one
        methods = ['ward', 'complete', 'average']
        best_method = 'ward'  # Default value
        
        # If the dataset is not too large, try different methods
        if len(embeddings) < 5000:
            coherence_scores = []
            
            for method in methods:
                try:
                    Z = linkage(embeddings, method=method)
                    labels = fcluster(Z, t=num_clusters, criterion="maxclust")
                    
                    # Calculate average coherence
                    coherence = 0
                    for cluster_id in np.unique(labels):
                        cluster_vectors = embeddings[labels == cluster_id]
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
        
        # Apply clustering with the best method
        Z = linkage(embeddings, method=best_method)
        labels = fcluster(Z, t=num_clusters, criterion="maxclust")
        
        return labels
        
    except Exception as e:
        st.error(f"Error in hierarchical clustering: {str(e)}")
        
        # Last resort: K-Means
        st.warning("Using K-Means as an alternative")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings) + 1  # +1 to start from 1

# IMPROVEMENT 3: Post-Clustering Refinement
def refine_clusters(df, embeddings, original_cluster_column='cluster_id'):
    """Refines clusters by identifying and correcting poor assignments"""
    st.info("Refining clusters to improve semantic coherence...")
    
    # Save original assignments
    df['original_cluster'] = df[original_cluster_column]
    
    # 1. Identify semantic outliers in each cluster
    outliers = []
    for cluster_id in df[original_cluster_column].unique():
        # Get indices of this cluster
        cluster_indices = df[df[original_cluster_column] == cluster_id].index.tolist()
        
        if len(cluster_indices) <= 3:  # Very small clusters, don't refine
            continue
            
        # Calculate cluster centroid
        cluster_embeddings = np.array([embeddings[i] for i in cluster_indices])
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate distances to centroid
        distances = [np.linalg.norm(embeddings[i] - centroid) for i in cluster_indices]
        
        # Normalize distances for this cluster
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist == 0:
            continue
            
        normalized_distances = [(d - mean_dist) / std_dist for d in distances]
        
        # Identify outliers (keywords very far from the centroid)
        for i, norm_dist in enumerate(normalized_distances):
            if norm_dist > 2.0:  # More than 2 standard deviations
                outliers.append((cluster_indices[i], cluster_id, norm_dist))
    
    # 2. Reassign outliers to more appropriate clusters
    reassigned = 0
    for idx, original_cluster, _ in outliers:
        keyword_embedding = embeddings[idx]
        
        # Find closest cluster (excluding the original)
        min_distance = float('inf')
        best_cluster = original_cluster
        
        for cluster_id in df[original_cluster_column].unique():
            if cluster_id == original_cluster:
                continue
                
            # Get indices of this cluster
            cluster_indices = df[df[original_cluster_column] == cluster_id].index.tolist()
            
            # Calculate centroid
            cluster_embeddings = np.array([embeddings[i] for i in cluster_indices])
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate distance
            distance = np.linalg.norm(keyword_embedding - centroid)
            
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_id
        
        # Reassign if we found a better cluster
        if best_cluster != original_cluster:
            df.loc[idx, original_cluster_column] = best_cluster
            reassigned += 1
            
    # 3. Combine too similar clusters
    similar_pairs = []
    clusters = df[original_cluster_column].unique()
    
    for i, cluster1 in enumerate(clusters):
        for cluster2 in clusters[i+1:]:
            # Calculate centroids
            indices1 = df[df[original_cluster_column] == cluster1].index.tolist()
            indices2 = df[df[original_cluster_column] == cluster2].index.tolist()
            
            if len(indices1) < 3 or len(indices2) < 3:
                continue  # Ignore very small clusters
                
            centroid1 = np.mean(np.array([embeddings[i] for i in indices1]), axis=0)
            centroid2 = np.mean(np.array([embeddings[i] for i in indices2]), axis=0)
            
            # Calculate cosine similarity
            similarity = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
            
            if similarity > 0.8:  # High threshold for merging
                similar_pairs.append((cluster1, cluster2, similarity))
    
    # Sort by similarity to combine the most similar first
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Combine clusters (keeping the lowest ID)
    clusters_merged = 0
    processed_clusters = set()
    
    for cluster1, cluster2, _ in similar_pairs:
        if cluster1 in processed_clusters or cluster2 in processed_clusters:
            continue  # Avoid combining already processed clusters
            
        # Choose the lowest ID to keep
        keep_id = min(cluster1, cluster2)
        remove_id = max(cluster1, cluster2)
        
        # Reassign keywords from the cluster to be eliminated
        df.loc[df[original_cluster_column] == remove_id, original_cluster_column] = keep_id
        
        processed_clusters.add(remove_id)
        clusters_merged += 1
        
        # Limit number of merges
        if clusters_merged >= len(clusters) // 4:  # Maximum 25% of merges
            break
    
    st.success(f"Refinement completed: {reassigned} keywords reassigned, {clusters_merged} clusters merged.")
    return df

# Function to generate cluster names with OpenAI
def generate_cluster_names(clusters_with_representatives, client, model="gpt-3.5-turbo"):
    if not clusters_with_representatives:
        return {}

    results = {}
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Generating names and descriptions for clusters...")

    # Process clusters in smaller batches to manage costs
    batch_size = 5
    total_batches = (len(clusters_with_representatives) + batch_size - 1) // batch_size
    
    for batch_idx, batch_start in enumerate(range(0, len(clusters_with_representatives), batch_size)):
        batch_end = min(batch_start + batch_size, len(clusters_with_representatives))
        batch_clusters = list(clusters_with_representatives.items())[batch_start:batch_end]

        try:
            # Prompt analyze clusters
            analysis_prompt = """I'll provide representative keywords for several SEO clusters. For each cluster:
1. Identify the main intent/purpose (informational, transactional, navigational)
2. Determine precise industry/vertical these keywords belong to
3. Identify specific entity types (brands, product categories, services)
4. Find meaningful relationships and hierarchies between terms
5. Determine the most distinctive feature of this keyword group

Be exceptionally detailed and focus on semantic intent rather than surface-level word similarities.

"""

            # Track the cluster order to match the response
            cluster_order = []
            for cluster_id, keywords in batch_clusters:
                cluster_order.append(cluster_id)
                analysis_prompt += f"Cluster {cluster_id} representative keywords: {', '.join(keywords[:20])}\n\n"

            analysis_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500
            )

            analysis_text = analysis_response.choices[0].message.content.strip()

            # Prompt naming clusters
            naming_prompt = f"""Based on the semantic analysis of these keyword clusters, provide:

1. A HIGHLY SPECIFIC name (3-5 words) that clearly identifies the:
   - Primary user intent
   - Specific industry/vertical 
   - Distinctive feature of this cluster

2. A detailed description (2 sentences) explaining:
   - The clear semantic relationship between keywords
   - How these keywords differ from other similar clusters
   - The likely search intent behind these terms

Prioritize precision and clarity over generality. Names should be immediately useful for SEO and content planning.

Format your response as a JSON array with cluster_id, cluster_name, and description.
"""

            # Define a function schema for structured cluster naming
            function_definition = {
                "name": "classify_keyword_cluster",
                "description": "Classifies a group of keywords into a specific semantic cluster",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cluster_id": {
                                        "type": "integer",
                                        "description": "The ID of the cluster"
                                    },
                                    "cluster_name": {
                                        "type": "string",
                                        "description": "A specific, descriptive name (3-5 words) for the cluster"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "A detailed description explaining the semantic relationship"
                                    },
                                    "primary_intent": {
                                        "type": "string",
                                        "description": "The dominant user intent: informational, navigational, or transactional",
                                        "enum": ["informational", "navigational", "transactional", "mixed"]
                                    },
                                    "distinctive_features": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of 2-3 distinctive semantic features of this cluster"
                                    }
                                },
                                "required": ["cluster_id", "cluster_name", "description"]
                            }
                        }
                    },
                    "required": ["clusters"]
                }
            }

            # Set up function call parameters
            completion_params = {
                "model": model,
                "messages": [{"role": "user", "content": naming_prompt}],
                "temperature": 0.2,
                "max_tokens": 800
            }

            # Use different approach based on model capabilities
            if "gpt-3.5-turbo" in model or "gpt-4" in model:
                # Modern models support function calling
                completion_params["tools"] = [{"type": "function", "function": function_definition}]
                completion_params["tool_choice"] = {"type": "function", "function": {"name": "classify_keyword_cluster"}}
            else:
                # Fallback for older models
                completion_params["response_format"] = {"type": "json_object"}

            # Make the API call
            naming_response = client.chat.completions.create(**completion_params)
            
            # Extract the response based on whether it was a function call or direct response
            if hasattr(naming_response.choices[0].message, 'tool_calls') and naming_response.choices[0].message.tool_calls:
                # Extract from function call
                function_call = naming_response.choices[0].message.tool_calls[0]
                naming_text = function_call.function.arguments
            else:
                # Extract from direct message content
                naming_text = naming_response.choices[0].message.content.strip()

# Process the JSON response
try:
    import re
    import json
    
    # First try direct parsing
    try:
        data = json.loads(naming_text)
    except json.JSONDecodeError:
        # Advanced text cleaning before attempting to extract JSON
        # Remove any text before the first '{' and after the last '}'
        json_start = naming_text.find('{')
        json_end = naming_text.rfind('}')
        array_start = naming_text.find('[')
        array_end = naming_text.rfind(']')
        
        # Determine if it's more likely to be an object or an array
        if (json_start != -1 and json_end != -1) and (array_start == -1 or json_start < array_start):
            clean_text = naming_text[json_start:json_end+1]
        elif array_start != -1 and array_end != -1:
            clean_text = naming_text[array_start:array_end+1]
        else:
            clean_text = naming_text
        
        # Replace single quotes with double quotes
        clean_text = clean_text.replace("'", '"')
        
        # Try to fix keys without quotes (a common error)
        clean_text = re.sub(r'([{,])\s*(\w+):', r'\1 "\2":', clean_text)
        
        # Remove invalid characters between strings
        clean_text = re.sub(r'"\s*:\s*"', '":"', clean_text)
        clean_text = re.sub(r'"\s*,\s*"', '","', clean_text)
        
        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError:
            # If it still fails, try to detect and fix more common errors
            # Remove commas after the last element of objects or arrays
            clean_text = re.sub(r',\s*}', '}', clean_text)
            clean_text = re.sub(r',\s*\]', ']', clean_text)
            
            try:
                data = json.loads(clean_text)
            except json.JSONDecodeError:
                # Last resort: handle parsing failure
                data = {}  # or some default value
except Exception as e:
    # Handle any other unexpected errors
    st.error(f"Unexpected error during JSON parsing: {str(e)}")
    data = {}  # or some default value
            
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        # If it's still failing, look for any JSON-like structure
        # First try with JSON objects
        json_match = re.search(r'({.*})', naming_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # Then try with JSON arrays
                array_match = re.search(r'(\[.*\])', naming_text, re.DOTALL)
                if array_match:
                    try:
                        array_data = json.loads(array_match.group(1))
                        data = {"clusters": array_data}
                    except json.JSONDecodeError:
                        # Last resort: manual parsing of the structure
                        st.warning("JSON parsing failed. Performing manual extraction...")
                        # Instead of raising an error, create a default structure
                        data = {"clusters": []}
                else:
                    # Instead of raising an error, create a default structure
                    data = {"clusters": []}
        else:
            raise ValueError("Could not extract valid JSON from response")

    # Handle different JSON structures that might be returned
    clusters_data = None
    if "clusters" in data:
        clusters_data = data["clusters"]
    elif isinstance(data, list):
        clusters_data = data
    else:
        # Try to find any list in the response
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                clusters_data = value
                break

    if clusters_data:
        # Match clusters by position if IDs don't match
        if len(clusters_data) == len(cluster_order):
            for i, cluster_info in enumerate(clusters_data):
                actual_id = cluster_order[i]
                results[actual_id] = (
                    cluster_info.get("cluster_name", f"Cluster {actual_id}"),
                    cluster_info.get("description", "No description provided.")
                )
except Exception as e:
    st.warning(f"Error analyzing JSON response for batch {batch_idx+1}/{total_batches}: {str(e)}")
    # Fallback for this batch: try to manually extract the information
    try:
        # Fallback more manual parsing for both models
        for i, cluster_id in enumerate(cluster_order):
            # Look for "Cluster {cluster_id}" or similar pattern in the text
            cluster_section = re.search(f"Cluster {cluster_id}[:\s]*(.*?)(?:Cluster \d|$)", 
                                       naming_text, re.DOTALL | re.IGNORECASE)
            
            if cluster_section:
                section_text = cluster_section.group(1).strip()
                # Try to extract name and description
                name_match = re.search(r"name[:\s]*(.*?)(?:description|\n|$)", 
                                     section_text, re.DOTALL | re.IGNORECASE)
                desc_match = re.search(r"description[:\s]*(.*?)(?:\n\n|$)", 
                                      section_text, re.DOTALL | re.IGNORECASE)
                
                name = name_match.group(1).strip() if name_match else f"Cluster {cluster_id}"
                desc = desc_match.group(1).strip() if desc_match else f"Keywords group {cluster_id}"
                
                results[cluster_id] = (name, desc)
            else:
                results[cluster_id] = (f"Cluster {cluster_id}", f"Keywords group {cluster_id}")
    except Exception:
        # Ultimate fallback if all parsing fails
        for cluster_id in cluster_order:
            results[cluster_id] = (f"Cluster {cluster_id}", f"Keywords group {cluster_id}")

    # Progress update inside the batch loop
    progress_bar.progress((batch_idx + 1) / total_batches)
    time.sleep(1)  # Avoid rate limits
except Exception as e:
    st.warning(f"Error generating names for batch {batch_idx+1}/{total_batches}: {str(e)}")
    # Provide default names
    for cluster_id, _ in batch_clusters:
        results[cluster_id] = (f"Cluster {cluster_id}", "Error generating description")

# Ensure all clusters have names
for cluster_id in clusters_with_representatives.keys():
    if cluster_id not in results:
        results[cluster_id] = (f"Cluster {cluster_id}", f"Keywords group {cluster_id}")

progress_bar.progress(1.0)
progress_text.text(f"✅ Generated names and descriptions for {len(results)} clusters")

return results
# IMPROVEMENT 5: Advanced cluster evaluation
def evaluate_cluster_quality(df, embeddings, cluster_column='cluster_id'):
    """Evaluates cluster quality using multiple metrics"""
    st.subheader("Advanced Cluster Quality Evaluation")
    
    metrics = {
        'silhouette': [],
        'density': [],
        'separation': [],
        'coherence': []
    }
    
    # Calculate centroids for all clusters
    centroids = {}
    for cluster_id in df[cluster_column].unique():
        indices = df[df[cluster_column] == cluster_id].index.tolist()
        centroids[cluster_id] = np.mean(np.array([embeddings[i] for i in indices]), axis=0)
    
    # Evaluate each cluster
    cluster_progress = st.progress(0)
    for i, cluster_id in enumerate(df[cluster_column].unique()):
        indices = df[df[cluster_column] == cluster_id].index.tolist()
        cluster_vectors = np.array([embeddings[i] for i in indices])
        centroid = centroids[cluster_id]
        
        # 1. Density (average distance to center)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        density = 1 / (1 + np.mean(distances)) if distances else 0
        metrics['density'].append((cluster_id, density))
        
        # 2. Coherence (average cosine similarity between vectors)
        coherence = calculate_cluster_coherence(cluster_vectors)
        metrics['coherence'].append((cluster_id, coherence))
        
        # 3. Separation (minimum distance to another centroid)
        min_separation = float('inf')
        for other_id, other_centroid in centroids.items():
            if other_id != cluster_id:
                separation = np.linalg.norm(centroid - other_centroid)
                min_separation = min(min_separation, separation)
        
        if min_separation != float('inf'):
            metrics['separation'].append((cluster_id, min_separation))
            
        cluster_progress.progress((i + 1) / len(df[cluster_column].unique()))
    
    # Visualize metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Coherence vs size chart
        coherence_data = pd.DataFrame(metrics['coherence'], columns=['cluster_id', 'score'])
        coherence_data = coherence_data.merge(
            df.groupby(cluster_column)['keyword'].count().reset_index(),
            left_on='cluster_id', right_on=cluster_column
        )
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
            labels={
                'score': 'Semantic Coherence', 
                'keyword': 'Cluster Size'
            },
            title='Relationship between Coherence and Size',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Separation vs density chart
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
                labels={
                    'separation': 'Separation Between Clusters',
                    'density': 'Cluster Density'
                },
                title='Separation vs Density',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Identify problematic clusters
    st.subheader("Cluster Diagnostics")
    
    # Calculate thresholds
    coherence_threshold = np.percentile([x[1] for x in metrics['coherence']], 25)
    problematic = [x[0] for x in metrics['coherence'] if x[1] < coherence_threshold]
    
    # Add coherence to original dataframe
    for cluster_id, coherence in metrics['coherence']:
        df.loc[df[cluster_column] == cluster_id, 'cluster_coherence'] = coherence
    
    if problematic:
        st.warning(f"Clusters with low semantic coherence: {problematic}")
        st.info("""
        Recommendations to improve:
        - Consider increasing the number of clusters
        - Review the keywords in these specific clusters
        - Try using higher quality embeddings
        - Consider manually splitting these clusters
        """)
    else:
        st.success("All clusters have good semantic coherence")
        
    return df

# Basic function to calculate coherence
def calculate_cluster_coherence(cluster_embeddings):
    """Calculate semantic coherence of a cluster based on embedding similarity"""
    if len(cluster_embeddings) <= 1:
        return 1.0  # Perfect coherence for single element

    try:
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Calculate average cosine similarity to centroid
        similarities = []
        for emb in cluster_embeddings:
            # Avoid division by zero
            norm_emb = np.linalg.norm(emb)
            norm_centroid = np.linalg.norm(centroid)
            if norm_emb > 0 and norm_centroid > 0:
                similarity = np.dot(emb, centroid) / (norm_emb * norm_centroid)
                similarities.append(similarity)
            else:
                similarities.append(0.0)

        return np.mean(similarities) if similarities else 0.0
    except Exception as e:
        st.warning(f"Error calculating coherence: {str(e)}")
        return 0.5  # Default value in case of error

def evaluate_and_refine_clusters(df, client, model="gpt-3.5-turbo"):
    """Uses OpenAI to evaluate cluster quality and suggest improvements"""
    st.subheader("AI-Powered Cluster Quality Evaluation")
    
    # Select a sample of clusters for evaluation (up to 5)
    sample_clusters = np.random.choice(df['cluster_id'].unique(), 
                                      size=min(5, len(df['cluster_id'].unique())), 
                                      replace=False)
    
    eval_progress = st.progress(0)
    eval_results = {}
    
    for i, cluster_id in enumerate(sample_clusters):
        # Get keywords from this cluster
        cluster_keywords = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
        cluster_name = df[df['cluster_id'] == cluster_id]['cluster_name'].iloc[0]
        
        # Limit to 50 keywords for evaluation
        sample_keywords = cluster_keywords[:50]
        
        prompt = f"""Evaluate this keyword cluster for semantic coherence and provide recommendations.

Cluster name: {cluster_name}
Sample keywords: {', '.join(sample_keywords)}

Answer these questions:
1. Coherence score (1-10): How semantically coherent is this cluster?
2. Outliers: Identify any keywords that don't seem to belong.
3. Sub-clusters: Should this cluster be split further? If so, suggest how.
4. Improvement: How could we improve this cluster's definition?

Provide your analysis in a structured JSON format with the following fields:
- coherence_score: numeric score from 1-10
- outlier_keywords: array of keywords that should be removed
- needs_splitting: boolean indicating if this cluster should be split
- suggested_subclusters: array of objects with 'name' and 'keywords' fields (if needs_splitting is true)
- improvement_suggestions: array of specific action items to improve this cluster
"""
        
        # Define function schema for structured response
        function_definition = {
            "name": "evaluate_keyword_cluster",
            "description": "Evaluates the semantic coherence of a keyword cluster",
            "parameters": {
                "type": "object",
                "properties": {
                    "coherence_score": {
                        "type": "number",
                        "description": "Semantic coherence score from 1-10"
                    },
                    "outlier_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords that don't belong in this cluster"
                    },
                    "needs_splitting": {
                        "type": "boolean",
                        "description": "Whether this cluster should be split into subclusters"
                    },
                    "suggested_subclusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "description": "Suggested subclusters if needs_splitting is true"
                    },
                    "improvement_suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific action items to improve this cluster"
                    }
                },
                "required": ["coherence_score", "outlier_keywords", "needs_splitting", "improvement_suggestions"]
            }
        }
        
        try:
            # Make API call with function calling
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "function", "function": function_definition}],
                tool_choice={"type": "function", "function": {"name": "evaluate_keyword_cluster"}},
                temperature=0.2
            )
            
            # Process the response
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                # Extract from function call
                tool_call = response.choices[0].message.tool_calls[0]
                evaluation_text = tool_call.function.arguments
                evaluation_data = json.loads(evaluation_text)
            else:
                # Extract from direct message content
                evaluation_text = response.choices[0].message.content
                # Try to parse JSON from text
                try:
                    evaluation_data = json.loads(evaluation_text)
                except:
                    evaluation_data = {"error": "Could not parse JSON", "raw_text": evaluation_text}
            
            # Store results
            eval_results[cluster_id] = {
                "name": cluster_name,
                "evaluation": evaluation_data
            }
            
            # Display evaluation
            with st.expander(f"Cluster {cluster_id}: {cluster_name} (Score: {evaluation_data.get('coherence_score', 'N/A')}/10)", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Cluster Quality")
                    st.metric("Coherence Score", f"{evaluation_data.get('coherence_score', 'N/A')}/10")
                    st.write("**Needs Splitting:**", "Yes" if evaluation_data.get('needs_splitting', False) else "No")
                    
                    if evaluation_data.get('outlier_keywords'):
                        st.write("**Outlier Keywords:**")
                        st.write(", ".join(evaluation_data['outlier_keywords'][:10]))
                        if len(evaluation_data['outlier_keywords']) > 10:
                            st.write(f"...and {len(evaluation_data['outlier_keywords']) - 10} more")
                
                with col2:
                    st.subheader("Improvement Suggestions")
                    for suggestion in evaluation_data.get('improvement_suggestions', []):
                        st.write(f"- {suggestion}")
                
                if evaluation_data.get('needs_splitting') and evaluation_data.get('suggested_subclusters'):
                    st.subheader("Suggested Subclusters")
                    for i, subcluster in enumerate(evaluation_data['suggested_subclusters']):
                        st.write(f"**{i+1}. {subcluster.get('name', 'Unnamed')}**")
                        st.write(", ".join(subcluster.get('keywords', [])[:10]))
                        if len(subcluster.get('keywords', [])) > 10:
                            st.write(f"...and {len(subcluster.get('keywords', [])) - 10} more")
            
        except Exception as e:
            st.error(f"Error evaluating cluster {cluster_id}: {str(e)}")
        
    # Update progress
    eval_progress.progress((i + 1) / len(sample_clusters))
    
    # Final recommendations based on overall evaluation
    if eval_results:
        st.subheader("Overall Cluster Quality Assessment")
        
        # Calculate average coherence
        coherence_scores = [eval_data["evaluation"].get("coherence_score", 0) 
                           for eval_data in eval_results.values() 
                           if isinstance(eval_data["evaluation"].get("coherence_score", 0), (int, float))]
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        
        # Count clusters needing splitting
        needs_splitting = sum(1 for eval_data in eval_results.values() 
                             if eval_data["evaluation"].get("needs_splitting", False))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Coherence", f"{avg_coherence:.1f}/10")
        with col2:
            st.metric("Clusters Needing Split", f"{needs_splitting}/{len(eval_results)}")
        
        # Overall recommendations
        if avg_coherence < 6:
            st.warning("⚠️ Overall cluster coherence is low. Consider increasing the number of clusters.")
        elif avg_coherence > 8:
            st.success("✅ Clusters show good coherence. Current configuration is working well.")
        else:
            st.info("ℹ️ Cluster coherence is moderate. Consider refinements based on individual evaluations.")
    
    return eval_results

# Main function to run improved clustering
def run_clustering(uploaded_file, openai_api_key, num_clusters, pca_variance, max_pca_components, min_df, max_df, gpt_model):
    """Executes the complete clustering process and returns the results"""
    if uploaded_file is None:
        st.warning("Please upload a CSV file with keywords.")
        return False, None
    
    st.info("Starting advanced semantic clustering process...")
    
    # Configure OpenAI client if API key is provided
    client = None
    if openai_api_key and openai_available:
        try:
            # Verify that the API key is not empty
            if openai_api_key.strip() == "":
                st.info("No valid OpenAI API Key provided. Clusters will have generic names.")
            else:
                # Set the API key as an environment variable
                os.environ["OPENAI_API_KEY"] = openai_api_key
                
                # Create the client without additional parameters
                client = OpenAI()
                
                # Verify the connection with a simple request
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=5
                    )
                    st.success("✅ Connection with OpenAI established successfully")
                except Exception as e:
                    st.error(f"Error verifying connection with OpenAI: {str(e)}")
                    st.error("Possible cause: Invalid API Key or connection problems")
                    client = None
        except Exception as e:
            st.error(f"Error configuring OpenAI client: {str(e)}")
            st.info("Continuing without OpenAI functionality")
            client = None
    elif not openai_available:
        st.warning("OpenAI library is not available. Continuing without OpenAI functionality.")
    elif not openai_api_key or openai_api_key.strip() == "":
        st.info("No OpenAI API Key provided. SentenceTransformers will be used as a free alternative.")
    
    try:
        # Load and process the CSV
        try:
            df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
            num_keywords = len(df)
            st.success(f"✅ Loaded {num_keywords} keywords from the CSV file")
            
            # Show cost estimate based on the loaded CSV
            show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
            
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            st.info("Trying alternative format...")
            # Try other formats/separators
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                df = pd.read_csv(StringIO(content), sep=None, engine='python', header=None)
                df.columns = ["keyword"]
                num_keywords = len(df)
                st.success(f"✅ Loaded {num_keywords} keywords from the CSV file (alternative format)")
                
                # Show cost estimate based on the loaded CSV
                show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
                
            except Exception as e2:
                st.error(f"Could not read the CSV file: {str(e2)}")
                return False, None
                
        # Preprocess keywords
        st.subheader("Keyword Preprocessing")
        st.info("Preprocessing keywords with improved semantic analysis...")
        
        # IMPROVEMENT 4: Use improved semantic preprocessing
        use_advanced = spacy_available
        if use_advanced:
            st.success("Using advanced preprocessing with linguistic analysis")
        else:
            st.info("Using standard preprocessing (SpaCy not available)")
            
        keywords_processed = preprocess_keywords(df["keyword"].tolist(), use_advanced=use_advanced)
        df['keyword_processed'] = keywords_processed
        st.success("✅ Keywords preprocessed correctly")
        
        # Generate improved embeddings
        st.subheader("Generating Semantic Vectors")
        
        # IMPROVEMENT 1: Use high-quality embeddings
        keyword_embeddings = generate_embeddings(df, openai_available, openai_api_key)
        
        # Apply PCA if embeddings are high-dimensional
        if keyword_embeddings.shape[1] > max_pca_components:
            st.subheader("Dimensionality Reduction (PCA)")
            
            try:
                pca_progress = st.progress(0)
                pca_text = st.empty()
                pca_text.text("Analyzing explained variance...")
                
                # Determine optimal number of components experimentally
                pca = PCA()
                pca.fit(keyword_embeddings)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                pca_progress.progress(0.3)
                
                # Find the number of components that explain the desired variance
                target_variance = pca_variance / 100.0
                n_components = np.argmax(cumulative_variance >= target_variance) + 1
                # If there aren't enough components for the desired variance, use the maximum
                if n_components == 1 and len(cumulative_variance) > 1:
                    n_components = min(max_pca_components, len(cumulative_variance))
                    
                pca_text.text(f"Components for {pca_variance}% variance: {n_components}")
                pca_progress.progress(0.6)
                
                # Use that number (with a reasonable cap)
                max_components = min(n_components, max_pca_components)
                pca = PCA(n_components=max_components)
                keyword_embeddings_reduced = pca.fit_transform(keyword_embeddings)
                
                pca_progress.progress(1.0)
                pca_text.text(f"✅ PCA applied: {max_components} dimensions ({pca_variance}% of explained variance)")
            except Exception as e:
                st.error(f"Error applying PCA: {str(e)}")
                st.info("Continuing without dimensionality reduction")
                # In case of error, keep the original embeddings
                keyword_embeddings_reduced = keyword_embeddings
        else:
            # PCA not needed if dimensionality is already appropriate
            keyword_embeddings_reduced = keyword_embeddings
            st.info(f"Appropriate embedding dimensionality ({keyword_embeddings.shape[1]}). PCA not required.")
            
        # Apply improved clustering
        st.subheader("Advanced Semantic Clustering")
        
        # IMPROVEMENT 2: Use improved clustering algorithm
        try:
            cluster_labels = improved_clustering(keyword_embeddings_reduced, num_clusters=num_clusters)
            df["cluster_id"] = cluster_labels
            st.success(f"✅ Keywords grouped into {len(df['cluster_id'].unique())} semantic clusters")
        except Exception as e:
            st.error(f"Error in advanced clustering: {str(e)}")
            st.info("Trying alternative clustering...")
            
            # Fallback: Assign clusters in a more basic way
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                df["cluster_id"] = kmeans.fit_predict(keyword_embeddings_reduced) + 1
                st.success("✅ Clustering completed using K-Means as an alternative")
            except Exception as e2:
                st.error(f"Error in alternative clustering: {str(e2)}")
                # Last resort: assign random clusters
                df["cluster_id"] = np.random.randint(1, num_clusters + 1, size=len(df))
                st.warning("⚠️ Random clusters have been assigned as a last resort")
        
        # IMPROVEMENT 3: Refine clusters
        st.subheader("Cluster Refinement")
        df = refine_clusters(df, keyword_embeddings_reduced)
        num_clusters_after_refinement = len(df['cluster_id'].unique())
        st.success(f"✅ Refinement completed: {num_clusters_after_refinement} final clusters")
        
        # Identify representative keywords for each cluster
        st.subheader("Cluster Analysis")
        
        rep_progress = st.progress(0)
        rep_text = st.empty()
        rep_text.text("Identifying representative keywords...")
        
        clusters_with_representatives = {}
        try:
            for i, cluster_num in enumerate(df['cluster_id'].unique()):
                cluster_size = len(df[df['cluster_id'] == cluster_num])
                n_representatives = min(20, cluster_size)
                
                # Get indices of keywords in this cluster
                indices = df[df['cluster_id'] == cluster_num].index.tolist()
                
                # Calculate centroid of the cluster
                cluster_embeddings = np.array([keyword_embeddings_reduced[idx] for idx in indices])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distance to centroid for each keyword
                distances = [np.linalg.norm(keyword_embeddings_reduced[i] - centroid) for i in indices]
                
                # Get indices of keywords closest to centroid
                sorted_indices = np.argsort(distances)[:n_representatives]
                representative_indices = [indices[i] for i in sorted_indices]
                representative_keywords = df.iloc[representative_indices]['keyword'].tolist()
                
                clusters_with_representatives[cluster_num] = representative_keywords
                
                # Update progress
                rep_progress.progress((i+1) / len(df['cluster_id'].unique()))
            
            rep_progress.progress(1.0)
            rep_text.text(f"✅ Identified representative keywords for {len(clusters_with_representatives)} clusters")
        except Exception as e:
            st.error(f"Error identifying representative keywords: {str(e)}")
            # Fallback: take the first N keywords of each cluster
            for cluster_num in df['cluster_id'].unique():
                cluster_keywords = df[df['cluster_id'] == cluster_num]['keyword'].tolist()
                clusters_with_representatives[cluster_num] = cluster_keywords[:min(20, len(cluster_keywords))]
            st.warning("Basic representative keywords have been selected as an alternative")
            
        # Generate names for clusters if OpenAI is available
        if client:
            st.subheader("Generating Cluster Names")
            try:
                cluster_names = generate_cluster_names(
                    clusters_with_representatives, 
                    client,
                    model=gpt_model
                )
            except Exception as e:
                st.error(f"Error generating cluster names: {str(e)}")
                cluster_names = {k: (f"Cluster {k}", f"Keyword group {k}") for k in df['cluster_id'].unique()}
        else:
            st.warning("Cannot generate cluster names without OpenAI API Key")
            cluster_names = {k: (f"Cluster {k}", f"Keyword group {k}") for k in df['cluster_id'].unique()}
        
        # Apply results to DataFrame
        df['cluster_name'] = ''
        df['cluster_description'] = ''
        df['representative'] = False
        
        for cluster_num, (name, description) in cluster_names.items():
            df.loc[df['cluster_id'] == cluster_num, 'cluster_name'] = name
            df.loc[df['cluster_id'] == cluster_num, 'cluster_description'] = description
            
            # Mark representative keywords
            for keyword in clusters_with_representatives.get(cluster_num, []):
                matching_indices = df[(df['cluster_id'] == cluster_num) & (df['keyword'] == keyword)].index
                if not matching_indices.empty:
                    df.loc[matching_indices, 'representative'] = True
        
        # IMPROVEMENT 5: Advanced cluster quality evaluation
        df = evaluate_cluster_quality(df, keyword_embeddings_reduced)
        
        # IMPROVEMENT 6: AI-powered cluster evaluation and refinement
        if client:
            try:
                # Evaluate clusters with OpenAI
                eval_results = evaluate_and_refine_clusters(df, client, model=gpt_model)
                # Store evaluation results in the session state for later use
                st.session_state.cluster_evaluation = eval_results
            except Exception as e:
                st.error(f"Error during cluster evaluation: {str(e)}")
                st.info("Continuing without evaluation")
        
        # Return the results
        return True, df
    
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return False, None
        
    # Final return as a safeguard
    return True, df
#############################
# MAIN APPLICATION
#############################

# Page configuration
st.set_page_config(
    page_title="Advanced Semantic Keyword Clustering",
    page_icon="🔍",
    layout="wide"
)

# CSS styles to improve appearance
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

# Title and description
st.markdown("<div class='main-header'>Advanced Semantic Keyword Clustering</div>", unsafe_allow_html=True)
st.markdown("""
This application allows you to group semantically similar keywords using advanced NLP and clustering techniques.
Upload your CSV file with keywords and configure the parameters to obtain high semantic correlation clusters.
""")

# Display advanced libraries status
with st.expander("Semantic Libraries Status", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        if openai_available:
            st.success("✅ OpenAI available (with API Key)")
        else:
            st.warning("⚠️ OpenAI not available")
            
        if sentence_transformers_available:
            st.success("✅ SentenceTransformers available (free)")
        else:
            st.warning("⚠️ SentenceTransformers not available")
            st.markdown("""
            To install:
            ```
            pip install sentence-transformers
            ```
            """)
    
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
        # Installation information
        st.info("""
        For more functionality:
        ```
        pip install sentence-transformers spacy hdbscan
        python -m spacy download en_core_web_sm
        ```
        """)

# Session initialization
if 'process_complete' not in st.session_state:
    st.session_state.process_complete = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

# Sidebar for configuration
st.sidebar.markdown("<div class='sub-header'>Configuration</div>", unsafe_allow_html=True)

# 1. Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your keywords CSV file", type=['csv'])

# 2. OpenAI API Key (message adjusted to indicate preference)
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key (recommended)",
    type="password", 
    help="Provide your OpenAI API Key for high-quality embeddings (up to 5000 keywords). If not provided, SentenceTransformers will be used as a free alternative."
)

# Display semantic processing status
if openai_available:
    if openai_api_key:
        st.sidebar.success("✅ API Key provided - OpenAI will be used for high-precision embeddings")
    else:
        if sentence_transformers_available:
            st.sidebar.info("ℹ️ No API Key - SentenceTransformers will be used as a free alternative")
        else:
            st.sidebar.warning("⚠️ No API Key and no SentenceTransformers - TF-IDF will be used (reduced precision)")
else:
    if sentence_transformers_available:
        st.sidebar.info("ℹ️ OpenAI not available - SentenceTransformers will be used as a free alternative")
    else:
        st.sidebar.error("❌ Advanced methods not available - TF-IDF will be used (reduced precision)")

# 3. Clustering parameters
st.sidebar.markdown("<div class='sub-header'>Parameters</div>", unsafe_allow_html=True)

# Parameters explanation panel - Placed before sliders
with st.sidebar.expander("ℹ️ Parameters Guide", expanded=False):
    st.markdown("""
    ### Clustering Parameters Guide
    
    Here you'll find explanations about each parameter and how to adjust it for better results:
    
    #### Number of clusters
    **What is it?** The number of groups into which your keywords will be divided.
    
    **How to use it:** 
    - **↑ Increase** if you need a more detailed and specific division by topics.
    - **↓ Decrease** if you prefer more general and broader groups.
    
    **Result:**
    - **High values** (15-30): Many small and very specific groups.
    - **Low values** (5-10): Fewer but broader thematic groups.
    - **Ideal:** Generally between 8-15 for 1000 keywords. Increase proportionally with the number of keywords.
    
    ---
    
    #### PCA explained variance (%)
    **What is it?** Determines how much original information is preserved when simplifying the data. Think of this as the "level of detail" that is maintained.
    
    **How to use it:**
    - **↑ Increase** for greater precision and to preserve more semantic nuances.
    - **↓ Decrease** to speed up processing with large datasets.
    
    **Result:**
    - **High values** (95-99%): Greater semantic precision but slower.
    - **Low values** (80-90%): Faster processing but some nuances may be lost.
    - **Ideal:** 90-95% offers a good balance between precision and speed.
    
    ---
    
    #### Maximum PCA components
    **What is it?** Limits the maximum complexity of the analysis model. Similar to setting a limit to avoid excessive complexity.
    
    **How to use it:**
    - **↑ Increase** for large datasets or with high thematic diversity.
    - **↓ Decrease** for smaller datasets or those centered on a single topic.
    
    **Result:**
    - **High values** (100-200): Captures more complex relationships between words.
    - **Low values** (30-75): More efficient but may oversimplify.
    - **Ideal:** Between 75-100 for most cases.
    
    ---
    
    #### Minimum term frequency
    **What is it?** Ignores words that appear in very few keywords. Helps filter out rare words or typographical errors.
    
    **How to use it:**
    - **↑ Increase** to eliminate uncommon terms and possible noise.
    - **↓ Decrease** to include infrequent terms that might be important.
    
    **Result:**
    - **High values** (3-5): Eliminates more rare terms, "cleaner" clustering.
    - **Low values** (1-2): Preserves uncommon terms, may retain more noise.
    - **Ideal:** 1-2 for small datasets, 2-3 for large datasets (+5000 keywords).
    
    ---
    
    #### Maximum term frequency (%)
    **What is it?** Ignores words that appear in a high percentage of keywords. Similar to eliminating "wildcard" words that are everywhere.
    
    **How to use it:**
    - **↑ Increase** to include more common terms.
    - **↓ Decrease** to filter out too generic words.
    
    **Result:**
    - **High values** (90-100%): Includes almost all terms, even very common ones.
    - **Low values** (70-85%): Focus on more distinctive words, ignoring generic ones.
    - **Ideal:** 85-95% works well for most datasets.
    """)
    
    st.info("""
    **Tip:** If you're not sure, keep the default values. The application is optimized to work well with these parameters in most cases.
    
    For large datasets (+5000 keywords), consider slightly increasing the number of clusters and reducing the PCA explained variance to maintain reasonable processing times.
    """)

# Sliders for parameters with improved descriptions
num_clusters = st.sidebar.slider(
    "Number of clusters", 
    min_value=2, 
    max_value=50, 
    value=10, 
    help="Number of groups into which your keywords will be divided. More clusters = more specific groups."
)

pca_variance = st.sidebar.slider(
    "PCA explained variance (%)", 
    min_value=50, 
    max_value=99, 
    value=95, 
    help="Percentage of information that is preserved. Higher value = greater precision but slower."
)

max_pca_components = st.sidebar.slider(
    "Maximum PCA components", 
    min_value=10, 
    max_value=300, 
    value=100, 
    help="Complexity limit of the model. Higher value = captures more complex relationships."
)

# 4. Advanced options
st.sidebar.markdown("<div class='sub-header'>Advanced options</div>", unsafe_allow_html=True)

min_df = st.sidebar.slider(
    "Minimum term frequency", 
    min_value=1, 
    max_value=10, 
    value=1, 
    help="Ignores infrequent terms. Higher value = eliminates more rare words."
)

max_df = st.sidebar.slider(
    "Maximum term frequency (%)", 
    min_value=50, 
    max_value=100, 
    value=95, 
    help="Ignores too common terms. Lower value = eliminates more generic words."
)

gpt_model = st.sidebar.selectbox(
    "Model for naming clusters", 
    ["gpt-3.5-turbo", "gpt-4"], 
    index=0,
    help="GPT-4 provides more accurate names but is more expensive and slower."
)

# Cost calculator widget for Streamlit
add_cost_calculator()

# Button to run clustering
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
                gpt_model
            )
            if success:
                st.session_state.df_results = results
                st.session_state.process_complete = True
                st.markdown("<div class='success-box'>✅ Semantic clustering completed successfully!</div>", unsafe_allow_html=True)

# Show results if process is complete
if st.session_state.process_complete and st.session_state.df_results is not None:
    st.markdown("<div class='main-header'>Clustering Results</div>", unsafe_allow_html=True)
    
    df = st.session_state.df_results
    
    # Tab to show visualizations
    with st.expander("Visualizations", expanded=True):
        # Bar chart with cluster sizes
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
        
        # Cluster coherence chart
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

    # Tab to explore clusters
    with st.expander("Explore Clusters", expanded=True):
        # Cluster selector
        cluster_options = [f"{row['cluster_name']} (ID: {row['cluster_id']})" for _, row in 
                          df.drop_duplicates(['cluster_id', 'cluster_name'])[['cluster_id', 'cluster_name']].iterrows()]
        selected_cluster = st.selectbox("Select a cluster to explore:", cluster_options)
        
        if selected_cluster:
            # Get selected cluster ID
            cluster_id = int(selected_cluster.split("ID: ")[1].split(")")[0])
            
            # Filter data for selected cluster
            cluster_df = df[df['cluster_id'] == cluster_id].copy()
            
            # Show cluster information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {cluster_df['cluster_name'].iloc[0]}")
                st.markdown(f"**Description:** {cluster_df['cluster_description'].iloc[0]}")
                st.markdown(f"**Total keywords:** {len(cluster_df)}")
            
            with col2:
                st.markdown(f"**Semantic coherence:** {cluster_df['cluster_coherence'].iloc[0]:.3f}")
                st.markdown("**Representative keywords:**")
                rep_keywords = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                if rep_keywords:
                    st.markdown("<ul>" + "".join([f"<li>{kw}</li>" for kw in rep_keywords[:10]]) + "</ul>", unsafe_allow_html=True)
            
            # Show all keywords in the cluster
            st.markdown("### All keywords")
            st.dataframe(cluster_df[['keyword']], use_container_width=True)

    # Tab to download results
    with st.expander("Download Results"):
        # Option to download complete CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV with all results",
            data=csv,
            file_name="semantic_clustered_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Option to download summary
        st.subheader("Clusters Summary")
        summary_df = df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
        summary_df.columns = ['ID', 'Name', 'Description', 'Number of Keywords']
        
        # Add coherence
        coherence_df = df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id')
        summary_df.drop('cluster_id', axis=1, inplace=True)
        summary_df.rename(columns={'cluster_coherence': 'Coherence'}, inplace=True)
        
        # Add representative keywords
        def get_rep_keywords(cluster_id):
            reps = df[(df['cluster_id'] == cluster_id) & (df['representative'] == True)]['keyword'].tolist()
            return ', '.join(reps[:5])
        
        summary_df['Representative Keywords'] = summary_df['ID'].apply(get_rep_keywords)
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Download summary
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Download clusters summary",
            data=csv_summary,
            file_name="semantic_clusters_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

# Reset button
if st.session_state.process_complete:
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state.process_complete = False
        st.session_state.df_results = None
        st.rerun()

# Additional information
with st.expander("Information about Advanced Semantic Clustering"):
    st.markdown("""
    ### How does this advanced semantic clustering work?
    
    1. **Linguistic Preprocessing**: Keywords are analyzed using advanced NLP to extract named entities, relevant bigrams, and significant tokens.
    
    2. **High-Quality Embeddings**: Latest generation embedding models are used:
       - OpenAI Embeddings (up to 5000 keywords) if API key is provided
       - Sentence Transformers (free) as an alternative or fallback
       - TF-IDF as a last resort
    
    3. **Intelligent Dimensionality Reduction**: Optimized PCA to preserve the most important semantic relationships.
    
    4. **Advanced Clustering**: Algorithms that automatically discover the optimal structure:
       - HDBSCAN to detect natural clusters
       - Optimized agglomerative hierarchical clustering
       - Automatic determination of the number of clusters
    
    5. **Post-Clustering Refinement**: Identifies and corrects problematic assignments:
       - Detection of semantic outliers
       - Fusion of very similar clusters
       - Reassignment of misclassified keywords
    
    6. **Multi-Metric Evaluation**: Rigorous analysis of cluster quality:
       - Internal semantic coherence
       - Density and compactness
       - Separation between clusters
       - Diagnosis of problematic clusters
    
    ### Tips for better results
    
    - **Keyword quality**: Clustering works better when keywords are related to the same domain or industry.
    
    - **Preprocessing**: Make sure your keywords don't contain spelling errors or strange characters.
    
    - **OpenAI API Key**: Provide an API Key for higher quality embeddings, although SentenceTransformers offers good results at no cost.
    
    - **Number of clusters**: Consider using automatic determination of the optimal number of clusters.
    
    - **Iterative evaluation**: Examine clusters with low coherence and consider adjusting parameters or dividing them.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Developed for advanced semantic keyword clustering | Version 2.1 with OpenAI/SentenceTransformers hybrid
    </div>
    """, 
    unsafe_allow_html=True
)
