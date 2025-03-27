# BLOCK 1 - START 
# Imports and Setup

import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from functools import partial
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
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Setup streamlit caching
st.cache_data.clear()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# For OpenAI, import with error handling
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False
    logger.warning("OpenAI package not available. Will use alternatives.")

# Try to import advanced libraries with error handling
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    logger.warning("SentenceTransformer not available. Will use alternatives.")

try:
    import spacy
    try:
        # Try loading spaCy without subprocess
        nlp = spacy.load("en_core_web_sm")
        spacy_available = True
    except OSError:
        # Model not found, but no subprocess to download it
        logger.warning("spaCy model not found, continuing without it")
        spacy_available = False
    except Exception as e:
        spacy_available = False
        logger.warning(f"spaCy error: {e}")
except ImportError:
    spacy_available = False
    logger.warning("spaCy not available. Will use alternatives.")

# Try to import TextBlob as alternative to spaCy
try:
    from textblob import TextBlob
    textblob_available = True
except ImportError:
    textblob_available = False
    logger.warning("TextBlob not available. Will use alternatives.")

try:
    import hdbscan
    hdbscan_available = True
except ImportError:
    hdbscan_available = False
    logger.warning("HDBSCAN not available. Will use hierarchical clustering.")

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    kmeans_available = True
except ImportError:
    kmeans_available = False
    logger.warning("scikit-learn clustering not available.")

# Define default stopwords to avoid NLTK downloads
DEFAULT_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'in', 
    'on', 'to', 'for', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should',
    'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'their', 'his',
    'her', 'its', 'ours', 'yours', 'theirs', 'myself', 'yourself', 'himself', 'herself',
    'itself', 'ourselves', 'yourselves', 'themselves'
}

# Create NLTK data directory if it doesn't exist - safer than downloading
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir)
    except Exception as e:
        logger.warning(f"Could not create NLTK data directory: {e}")

# BLOCK 1 - END

# BLOCK 2 - START 
# Core Classes - KeywordProcessor

class KeywordProcessor:
    """Main class for keyword processing with caching built-in"""
    
    def __init__(self, use_openai=False, openai_api_key=None, num_clusters=10, min_cluster_size=5):
        self.use_openai = use_openai and openai_available
        self.openai_api_key = openai_api_key
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.client = None
        self.sentence_model = None
        
        # Initialize OpenAI client if available
        if self.use_openai and self.openai_api_key:
            try:
                os.environ["OPENAI_API_KEY"] = self.openai_api_key
                self.client = OpenAI()
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.use_openai = False
        
        # Initialize SentenceTransformer if available
        if sentence_transformers_available:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("SentenceTransformer model loaded")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def preprocess_keywords_cached(self, keywords, use_advanced=True):
        """Cached version of keyword preprocessing"""
        return self._preprocess_keywords(keywords, use_advanced)
    
    def _preprocess_keywords(self, keywords, use_advanced=True):
        """Internal method for keyword preprocessing"""
        processed_keywords = []
        
        progress_bar = st.progress(0)
        total = len(keywords)
        batch_size = max(1, min(500, total // 10))  # Dynamic batch size
        
        # Perform preprocessing in batches
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch = keywords[i:batch_end]
            
            # Process the batch
            batch_processed = []
            for keyword in batch:
                if use_advanced and (spacy_available or textblob_available):
                    batch_processed.append(self._enhanced_preprocessing(keyword))
                else:
                    batch_processed.append(self._preprocess_text(keyword))
            
            processed_keywords.extend(batch_processed)
            progress_bar.progress(min(batch_end / total, 1.0))
        
        progress_bar.progress(1.0)
        return processed_keywords
    
    def _enhanced_preprocessing(self, text):
        """Enhanced preprocessing with linguistic analysis - without relying on NLTK downloads"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Option 1: Use spaCy for advanced linguistic analysis
            if spacy_available:
                doc = nlp(text.lower())
                
                # Preserve named entities
                entities = [ent.text for ent in doc.ents]
                
                # Extract relevant tokens (not stopwords)
                tokens = []
                for token in doc:
                    if not token.is_stop and token.is_alpha and len(token.text) > 1:
                        tokens.append(token.lemma_)
                
                # Extract meaningful bigrams
                bigrams = []
                for i in range(len(doc) - 1):
                    if (not doc[i].is_stop and not doc[i+1].is_stop and 
                        doc[i].is_alpha and doc[i+1].is_alpha):
                        bigrams.append(f"{doc[i].lemma_}_{doc[i+1].lemma_}")
                
                # Combine everything preserving entities
                processed_parts = tokens + bigrams + entities
                return " ".join(processed_parts)
            
            # Option 2: Use TextBlob as simpler alternative
            elif textblob_available:
                blob = TextBlob(text.lower())
                # Extract noun phrases (good for keywords)
                noun_phrases = list(blob.noun_phrases)
                
                # Get tokens without stopwords
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    # Use default stopwords
                    stop_words = DEFAULT_STOPWORDS
                
                words = [word for word in blob.words if len(word) > 1 and word.lower() not in stop_words]
                
                # Lemmatization
                lemmas = [word.lemmatize() for word in words]
                processed_parts = lemmas + noun_phrases
                
                return " ".join(processed_parts)
            
            # Option 3: Fallback to basic method
            else:
                return self._preprocess_text(text)
        except Exception as e:
            logger.warning(f"Enhanced preprocessing failed: {e}")
            return text.lower() if isinstance(text, str) else ""
    
    def _preprocess_text(self, text):
        """Basic text preprocessing without relying on NLTK downloads"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Simple tokenization without NLTK
            tokens = text.split()
            
            # Filter out short words and stopwords
            try:
                # Try to use NLTK stopwords if available without downloading
                stop_words = set(stopwords.words('english'))
            except:
                # Use default stopwords if NLTK resources aren't available
                stop_words = DEFAULT_STOPWORDS
            
            # Filter tokens
            tokens = [t for t in tokens if len(t) > 1 and t.isalpha() and t not in stop_words]
            
            # Basic lemmatization (if available)
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            except Exception as e:
                logger.warning(f"Lemmatization failed: {e}")
            
            return " ".join(tokens)
        except Exception as e:
            logger.warning(f"Basic preprocessing failed: {e}")
            return text.lower() if isinstance(text, str) else ""
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_embeddings_cached(self, processed_keywords, use_openai=True):
        """Cached version of embedding generation"""
        return self._generate_embeddings(processed_keywords, use_openai)
    
    def _generate_embeddings(self, processed_keywords, use_openai=True):
        """Generate embeddings using available methods"""
        st.info("Generating embeddings for keywords...")
        
        # Option 1: Use OpenAI if available and requested
        if use_openai and self.use_openai and self.client:
            try:
                return self._generate_openai_embeddings(processed_keywords)
            except Exception as e:
                logger.error(f"OpenAI embeddings failed: {e}")
                st.warning("OpenAI embeddings failed. Trying SentenceTransformers...")
        
        # Option 2: Use SentenceTransformers as fallback
        if sentence_transformers_available and self.sentence_model:
            try:
                return self._generate_sentence_transformer_embeddings(processed_keywords)
            except Exception as e:
                logger.error(f"SentenceTransformer embeddings failed: {e}")
                st.warning("SentenceTransformer embeddings failed. Using TF-IDF...")
        
        # Option 3: Fallback to TF-IDF
        return self._generate_tfidf_embeddings(processed_keywords)
    
    def _generate_openai_embeddings(self, processed_keywords):
        """Generate embeddings using OpenAI API with smart sampling"""
        st.info("Using OpenAI embeddings (high semantic precision)")
        
        # Process keywords in batches to optimize costs
        keywords = [k if isinstance(k, str) and k.strip() else " " for k in processed_keywords]
        all_embeddings = []
        
        # Limit to 5000 keywords with smart sampling
        if len(keywords) > 5000:
            st.warning(f"Limiting to 5000 representative keywords from {len(keywords)} total")
            
            # Step 1: Create quick TF-IDF representation for initial clustering
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(keywords)
            
            # Step 2: Quick clustering to find representative samples
            mini_kmeans = MiniBatchKMeans(
                n_clusters=min(100, len(keywords)//100), 
                batch_size=1024,
                random_state=42
            )
            initial_clusters = mini_kmeans.fit_predict(tfidf_matrix)
            
            # Step 3: Sample evenly from each cluster
            sample_indices = []
            for cluster in range(mini_kmeans.n_clusters):
                cluster_indices = np.where(initial_clusters == cluster)[0]
                if len(cluster_indices) > 0:
                    # Sample proportionally to cluster size
                    n_samples = max(1, int(5000 * (len(cluster_indices) / len(keywords))))
                    if len(cluster_indices) <= n_samples:
                        sample_indices.extend(cluster_indices)
                    else:
                        sample_indices.extend(np.random.choice(
                            cluster_indices, n_samples, replace=False))
            
            # Ensure we have exactly 5000 samples
            if len(sample_indices) > 5000:
                sample_indices = np.random.choice(sample_indices, 5000, replace=False)
            elif len(sample_indices) < 5000:
                remaining = np.setdiff1d(np.arange(len(keywords)), sample_indices)
                if len(remaining) > 0:
                    additional = np.random.choice(
                        remaining, 
                        min(5000-len(sample_indices), len(remaining)), 
                        replace=False
                    )
                    sample_indices = np.concatenate([sample_indices, additional])
            
            sample_keywords = [keywords[i] for i in sample_indices]
            
            # Generate embeddings for sample
            progress_bar = st.progress(0)
            st.info("Processing embeddings with OpenAI (this will take a few minutes)...")
            
            # Process in batches of 1000
            batch_size = 1000
            sample_embeddings = []
            
            for i in range(0, len(sample_keywords), batch_size):
                batch_end = min(i + batch_size, len(sample_keywords))
                batch = sample_keywords[i:batch_end]
                
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                sample_embeddings.extend(batch_embeddings)
                progress_bar.progress(min(0.5, batch_end / len(sample_keywords)))
            
            # Convert to numpy array
            sample_embeddings = np.array(sample_embeddings)
            
            # Propagate embeddings to all keywords using nearest neighbors
            st.info("Propagating embeddings to remaining keywords...")
            
            # Create TF-IDF for all keywords for similarity comparison
            full_vectorizer = TfidfVectorizer(max_features=300)
            full_tfidf_matrix = full_vectorizer.fit_transform(keywords)
            
            # Initialize embeddings array for all keywords
            all_embeddings = np.zeros((len(keywords), sample_embeddings.shape[1]))
            
            # Assign sample embeddings
            for i, idx in enumerate(sample_indices):
                all_embeddings[idx] = sample_embeddings[i]
            
            # Find similar keywords for the rest
            remaining_indices = [i for i in range(len(keywords)) if i not in sample_indices]
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(remaining_indices), batch_size):
                batch_end = min(i + batch_size, len(remaining_indices))
                batch_indices = remaining_indices[i:batch_end]
                
                for idx in batch_indices:
                    # Find most similar keyword from samples
                    similarities = cosine_similarity(
                        full_tfidf_matrix[idx:idx+1],
                        full_tfidf_matrix[sample_indices]
                    )[0]
                    most_similar_idx = sample_indices[np.argmax(similarities)]
                    all_embeddings[idx] = all_embeddings[most_similar_idx]
                
                # Update progress
                progress_percent = 0.5 + 0.5 * (batch_end / len(remaining_indices))
                progress_bar.progress(min(1.0, progress_percent))
            
            progress_bar.progress(1.0)
            
        else:
            # If fewer than 5000, process all keywords
            progress_bar = st.progress(0)
            st.info(f"Processing embeddings for all {len(keywords)} keywords with OpenAI...")
            
            # Process in batches of 1000
            batch_size = 1000
            for i in range(0, len(keywords), batch_size):
                batch_end = min(i + batch_size, len(keywords))
                batch = keywords[i:batch_end]
                
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                progress_bar.progress(min(1.0, batch_end / len(keywords)))
            
            # Convert to numpy array
            all_embeddings = np.array(all_embeddings) if isinstance(all_embeddings, list) else all_embeddings
        
        st.success(f"✅ Generated embeddings of {all_embeddings.shape[1]} dimensions using OpenAI")
        return all_embeddings
    
    def _generate_sentence_transformer_embeddings(self, processed_keywords):
        """Generate embeddings using SentenceTransformer"""
        st.success("Using SentenceTransformer (free alternative)")
        
        progress_bar = st.progress(0)
        keywords = [k if isinstance(k, str) and k.strip() else " " for k in processed_keywords]
        
        # Process in batches to manage memory
        batch_size = 256  # Smaller batch size for better progress reporting
        all_embeddings = []
        
        for i in range(0, len(keywords), batch_size):
            batch_end = min(i + batch_size, len(keywords))
            batch = keywords[i:batch_end]
            
            batch_embeddings = self.sentence_model.encode(
                batch, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            all_embeddings.extend(batch_embeddings)
            progress_bar.progress(min(1.0, batch_end / len(keywords)))
        
        progress_bar.progress(1.0)
        embeddings = np.array(all_embeddings)
        
        st.success(f"✅ Generated embeddings of {embeddings.shape[1]} dimensions using SentenceTransformer")
        return embeddings
    
    def _generate_tfidf_embeddings(self, processed_keywords, min_df=1, max_df=0.95):
        """Generate TF-IDF embeddings as last resort"""
        st.warning("Using TF-IDF as last resort (less semantically precise)")
        
        progress_bar = st.progress(0)
        
        try:
            # Create vectorizer with configurable parameters
            vectorizer = TfidfVectorizer(
                max_features=300,  # Limit features to prevent memory issues
                min_df=min_df,     # Ignore terms in fewer than N documents
                max_df=max_df,     # Ignore terms in more than N% of documents
                stop_words='english'
            )
            
            # Ensure no null values
            clean_texts = [t if isinstance(t, str) and t.strip() else " " for t in processed_keywords]
            
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
            logger.error(f"Error generating TF-IDF embeddings: {e}")
            # Last resort: random vectors
            st.warning("Generating random vectors as absolute last resort")
            random_embeddings = np.random.rand(len(processed_keywords), 100)
            return random_embeddings

# BLOCK 2 - END

# BLOCK 3 - START 
# Core Classes - ClusteringEngine

class ClusteringEngine:
    """Handle clustering operations with advanced algorithms and fallbacks"""
    
    def __init__(self, num_clusters=10, min_cluster_size=5):
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def cluster_cached(self, embeddings, num_clusters=None, min_cluster_size=None):
        """Cached version of clustering"""
        num_clusters = num_clusters or self.num_clusters
        min_cluster_size = min_cluster_size or self.min_cluster_size
        return self._perform_clustering(embeddings, num_clusters, min_cluster_size)
    
    def _perform_clustering(self, embeddings, num_clusters=None, min_cluster_size=None):
        """Main clustering method with fallbacks"""
        num_clusters = num_clusters or self.num_clusters
        min_cluster_size = min_cluster_size or self.min_cluster_size
        
        st.info("Applying advanced clustering algorithms...")
        
        # Determine optimal number of clusters if not specified
        if num_clusters is None or num_clusters <= 0:
            num_clusters = self._find_optimal_clusters(embeddings)
        
        # Try HDBSCAN for natural clustering
        if hdbscan_available and len(embeddings) < 50000:  # HDBSCAN struggles with very large datasets
            try:
                st.info("Applying HDBSCAN for natural cluster detection...")
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=1,
                    cluster_selection_epsilon=0.5,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True  # Enable prediction for outliers
                )
                
                cluster_labels = clusterer.fit_predict(embeddings)
                
                # Check if HDBSCAN found a reasonable structure
                unique_clusters = np.unique(cluster_labels)
                non_noise_clusters = [c for c in unique_clusters if c != -1]
                
                if len(non_noise_clusters) > 1 and len(non_noise_clusters) <= num_clusters * 2:
                    st.success(f"HDBSCAN identified {len(non_noise_clusters)} natural clusters")
                    
                    # Reassign noise points to nearest cluster
                    if -1 in unique_clusters:
                        noise_indices = np.where(cluster_labels == -1)[0]
                        if len(noise_indices) > 0:
                            # Use approximate_predict for efficient reassignment
                            new_labels, _ = hdbscan.approximate_predict(
                                clusterer, 
                                embeddings[noise_indices]
                            )
                            
                            # Update labels
                            for i, idx in enumerate(noise_indices):
                                if new_labels[i] != -1:  # If prediction successful
                                    cluster_labels[idx] = new_labels[i]
                                else:
                                    # Find nearest centroid manually
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
                    
                    # Remap IDs to start from 1
                    mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(np.unique(cluster_labels))}
                    cluster_labels = np.array([mapping[label] for label in cluster_labels])
                    
                    return cluster_labels
                else:
                    st.info("HDBSCAN clustering produced too many or too few clusters. Trying hierarchical clustering...")
            except Exception as e:
                logger.warning(f"HDBSCAN failed: {e}")
                st.warning("HDBSCAN clustering failed. Trying hierarchical clustering...")
        
        # Try hierarchical clustering
        try:
            st.info("Applying hierarchical agglomerative clustering...")
            
            # For large datasets, use sample to determine best method
            methods = ['ward', 'complete', 'average']
            best_method = 'ward'  # Default value
            
            # Only test methods if dataset is not too large
            if len(embeddings) < 5000:
                coherence_scores = []
                
                for method in methods:
                    try:
                        # Use sample for very large datasets
                        sample_size = min(5000, len(embeddings))
                        if len(embeddings) > sample_size:
                            sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
                            sample_embeddings = embeddings[sample_indices]
                        else:
                            sample_embeddings = embeddings
                        
                        Z = linkage(sample_embeddings, method=method)
                        labels = fcluster(Z, t=num_clusters, criterion="maxclust")
                        
                        # Calculate average coherence
                        coherence = 0
                        for cluster_id in np.unique(labels):
                            cluster_vectors = sample_embeddings[labels == cluster_id]
                            if len(cluster_vectors) > 1:
                                centroid = np.mean(cluster_vectors, axis=0)
                                dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
                                coherence += np.mean(1 / (1 + dists))
                        
                        coherence_scores.append(coherence / len(np.unique(labels)))
                    except Exception as e:
                        logger.warning(f"Method {method} failed: {e}")
                        coherence_scores.append(0)
                
                if coherence_scores:
                    best_method = methods[np.argmax(coherence_scores)]
                    st.success(f"Best linkage method: {best_method}")
            
            # Apply clustering with the best method
            # For very large datasets, use sample as prototypes
            if len(embeddings) > 10000:
                # Sample prototypes for faster hierarchical clustering
                sample_size = 10000
                sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
                sample_embeddings = embeddings[sample_indices]
                
                # Cluster the prototypes
                Z = linkage(sample_embeddings, method=best_method)
                prototype_labels = fcluster(Z, t=num_clusters, criterion="maxclust")
                
                # Assign each point to nearest prototype cluster
                labels = np.zeros(len(embeddings), dtype=int)
                
                # First assign the sampled points
                for i, idx in enumerate(sample_indices):
                    labels[idx] = prototype_labels[i]
                
                # Then assign the rest based on nearest centroid
                centroids = {}
                for cluster_id in np.unique(prototype_labels):
                    cluster_points = sample_embeddings[prototype_labels == cluster_id]
                    centroids[cluster_id] = np.mean(cluster_points, axis=0)
                
                # Assign remaining points in batches
                remaining_indices = np.setdiff1d(np.arange(len(embeddings)), sample_indices)
                batch_size = 1000
                
                for i in range(0, len(remaining_indices), batch_size):
                    batch_end = min(i + batch_size, len(remaining_indices))
                    batch_indices = remaining_indices[i:batch_end]
                    
                    for idx in batch_indices:
                        min_dist = float('inf')
                        best_cluster = 1
                        
                        for cluster_id, centroid in centroids.items():
                            dist = np.linalg.norm(embeddings[idx] - centroid)
                            if dist < min_dist:
                                min_dist = dist
                                best_cluster = cluster_id
                        
                        labels[idx] = best_cluster
            else:
                # Direct hierarchical clustering for smaller datasets
                Z = linkage(embeddings, method=best_method)
                labels = fcluster(Z, t=num_clusters, criterion="maxclust")
            
            return labels
            
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            st.warning("Hierarchical clustering failed. Using K-Means as fallback.")
        
        # Fallback to K-Means
        try:
            if kmeans_available:
                # Use MiniBatchKMeans for very large datasets
                if len(embeddings) > 10000:
                    st.info("Using MiniBatchKMeans for large dataset...")
                    kmeans = MiniBatchKMeans(
                        n_clusters=num_clusters, 
                        random_state=42, 
                        batch_size=1024,
                        n_init=3
                    )
                else:
                    st.info("Using KMeans...")
                    kmeans = KMeans(
                        n_clusters=num_clusters, 
                        random_state=42, 
                        n_init=10
                    )
                
                return kmeans.fit_predict(embeddings) + 1  # +1 to start from 1
            else:
                raise ImportError("KMeans not available")
        except Exception as e:
            logger.error(f"K-Means clustering failed: {e}")
            st.error("All clustering methods failed. Using random assignment as last resort.")
            
            # Last resort: random assignment
            return np.random.randint(1, num_clusters + 1, size=len(embeddings))
    
    def _find_optimal_clusters(self, embeddings):
        """Find optimal number of clusters using silhouette score"""
        try:
            from sklearn.metrics import silhouette_score
            
            st.info("Finding optimal number of clusters...")
            sil_scores = []
            max_clusters = min(30, len(embeddings) // 5)
            range_n_clusters = range(2, max(3, max_clusters))
            
            progress_bar = st.progress(0)
            
            # Use sample for large datasets
            sample_size = min(5000, len(embeddings))
            if len(embeddings) > sample_size:
                sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
                sample_embeddings = embeddings[sample_indices]
            else:
                sample_embeddings = embeddings
            
            # Test different numbers of clusters
            for i, n_clusters in enumerate(range_n_clusters):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=2)
                cluster_labels = kmeans.fit_predict(sample_embeddings)
                
                if len(set(cluster_labels)) > 1:
                    try:
                        score = silhouette_score(sample_embeddings, cluster_labels)
                        sil_scores.append(score)
                    except:
                        sil_scores.append(0)
                else:
                    sil_scores.append(0)
                
                progress_bar.progress((i + 1) / len(range_n_clusters))
            
            # Select number of clusters with best score
            if sil_scores:
                best_num_clusters = range_n_clusters[np.argmax(sil_scores)]
                st.success(f"Optimal number of clusters determined: {best_num_clusters}")
                return best_num_clusters
            
        except Exception as e:
            logger.warning(f"Optimal cluster determination failed: {e}")
        
        # Default fallback
        return max(2, min(self.num_clusters, len(embeddings) // 5))
    
    def refine_clusters(self, df, embeddings, original_cluster_column='cluster_id'):
        """Refine clusters by identifying and correcting poor assignments"""
        st.info("Refining clusters to improve semantic coherence...")
        
        # Save original assignments
        df['original_cluster'] = df[original_cluster_column]
        
        # Process in batches to reduce memory usage
        batch_size = min(5000, len(df))
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end].copy()
            batch_indices = batch_df.index.tolist()
            
            # 1. Identify semantic outliers in each cluster
            outliers = []
            for cluster_id in batch_df[original_cluster_column].unique():
                # Get indices of this cluster in the batch
                cluster_indices = batch_df[batch_df[original_cluster_column] == cluster_id].index.tolist()
                cluster_positions = [batch_indices.index(i) for i in cluster_indices]
                
                if len(cluster_positions) <= 3:  # Clusters too small, don't refine
                    continue
                
                # Calculate cluster centroid
                cluster_embeddings = np.array([embeddings[batch_indices.index(i)] for i in cluster_indices])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distances to centroid
                distances = [np.linalg.norm(embeddings[batch_indices.index(i)] - centroid) for i in cluster_indices]
                
                # Normalize distances for this cluster
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                if std_dist == 0:
                    continue
                
                normalized_distances = [(d - mean_dist) / std_dist for d in distances]
                
                # Identify outliers (keywords very far from centroid)
                for i, norm_dist in enumerate(normalized_distances):
                    if norm_dist > 2.0:  # More than 2 standard deviations
                        outliers.append((cluster_indices[i], cluster_id, norm_dist))
            
            # 2. Reassign outliers to more appropriate clusters
            reassigned = 0
            for idx, original_cluster, _ in outliers:
                batch_pos = batch_indices.index(idx)
                keyword_embedding = embeddings[batch_pos]
                
                # Find closest cluster (excluding the original)
                min_distance = float('inf')
                best_cluster = original_cluster
                
                for cluster_id in batch_df[original_cluster_column].unique():
                    if cluster_id == original_cluster:
                        continue
                    
                    # Get indices of this cluster
                    cluster_indices = batch_df[batch_df[original_cluster_column] == cluster_id].index.tolist()
                    cluster_positions = [batch_indices.index(i) for i in cluster_indices]
                    
                    if not cluster_positions:
                        continue
                    
                    # Calculate centroid
                    cluster_embeddings = np.array([embeddings[p] for p in cluster_positions])
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Calculate distance
                    distance = np.linalg.norm(keyword_embedding - centroid)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id
                
                # Reassign if found a better cluster
                if best_cluster != original_cluster:
                    df.loc[idx, original_cluster_column] = best_cluster
                    reassigned += 1
        
        # 3. Combine very similar clusters
        similar_pairs = []
        clusters = df[original_cluster_column].unique()
        
        # Calculate all cluster centroids
        centroids = {}
        for cluster_id in clusters:
            indices = df[df[original_cluster_column] == cluster_id].index.tolist()
            if len(indices) >= 3:  # Skip tiny clusters
                centroids[cluster_id] = np.mean(np.array([embeddings[df.index.get_loc(i)] for i in indices]), axis=0)
        
        # Find similar pairs
        for i, cluster1 in enumerate(centroids.keys()):
            for cluster2 in list(centroids.keys())[i+1:]:
                # Calculate cosine similarity
                similarity = np.dot(centroids[cluster1], centroids[cluster2]) / (
                    np.linalg.norm(centroids[cluster1]) * np.linalg.norm(centroids[cluster2]))
                
                if similarity > 0.8:  # High threshold for merging
                    similar_pairs.append((cluster1, cluster2, similarity))
        
        # Sort by similarity to merge most similar first
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Combine clusters (keeping the lower ID)
        clusters_merged = 0
        processed_clusters = set()
        
        for cluster1, cluster2, _ in similar_pairs:
            if cluster1 in processed_clusters or cluster2 in processed_clusters:
                continue  # Skip already processed clusters
                
            # Keep the ID with more members
            size1 = len(df[df[original_cluster_column] == cluster1])
            size2 = len(df[df[original_cluster_column] == cluster2])
            
            if size1 >= size2:
                keep_id, remove_id = cluster1, cluster2
            else:
                keep_id, remove_id = cluster2, cluster1
            
            # Reassign keywords from the cluster to remove
            df.loc[df[original_cluster_column] == remove_id, original_cluster_column] = keep_id
            
            processed_clusters.add(remove_id)
            clusters_merged += 1
            
            # Limit number of merges
            if clusters_merged >= len(clusters) // 4:  # Maximum 25% of merges
                break
        
        st.success(f"Refinement completed: {reassigned} keywords reassigned, {clusters_merged} clusters merged.")
        return df

# BLOCK 3 - END

# BLOCK 4 - START 
# Cost Calculation and Post-Processing Utilities

class ClusterAnalyzer:
    """Analyze and evaluate cluster quality and name clusters"""
    
    def __init__(self, gpt_model="gpt-3.5-turbo"):
        self.gpt_model = gpt_model
    
    def evaluate_cluster_quality(self, df, embeddings, cluster_column='cluster_id'):
        """Evaluate clusters using multiple metrics"""
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
        
        # For large datasets, process in batches to avoid memory issues
        all_cluster_ids = list(df[cluster_column].unique())
        for i, cluster_id in enumerate(all_cluster_ids):
            indices = df[df[cluster_column] == cluster_id].index.tolist()
            
            # Use sampling for very large clusters
            if len(indices) > 1000:
                sample_indices = np.random.choice(indices, 1000, replace=False)
            else:
                sample_indices = indices
                
            cluster_vectors = np.array([embeddings[i] for i in sample_indices])
            centroid = centroids[cluster_id]
            
            # 1. Density (average distance to center)
            distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
            density = 1 / (1 + np.mean(distances)) if distances else 0
            metrics['density'].append((cluster_id, density))
            
            # 2. Coherence (average cosine similarity between vectors)
            coherence = self._calculate_cluster_coherence(cluster_vectors)
            metrics['coherence'].append((cluster_id, coherence))
            
            # 3. Separation (minimum distance to another centroid)
            min_separation = float('inf')
            for other_id, other_centroid in centroids.items():
                if other_id != cluster_id:
                    separation = np.linalg.norm(centroid - other_centroid)
                    min_separation = min(min_separation, separation)
            
            if min_separation != float('inf'):
                metrics['separation'].append((cluster_id, min_separation))
                
            cluster_progress.progress((i + 1) / len(all_cluster_ids))
        
        # Add coherence to the original dataframe
        for cluster_id, coherence in metrics['coherence']:
            df.loc[df[cluster_column] == cluster_id, 'cluster_coherence'] = coherence
        
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
                title='Coherence vs Cluster Size',
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
                        'separation': 'Cluster Separation',
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
        
        if problematic:
            st.warning(f"Clusters with low semantic coherence: {problematic}")
            st.info("""
            Recommendations for improvement:
            - Consider increasing the number of clusters
            - Review keywords in these specific clusters
            - Try using higher quality embeddings
            - Consider manually splitting these clusters
            """)
        else:
            st.success("All clusters have good semantic coherence")
            
        return df
    
    def _calculate_cluster_coherence(self, cluster_embeddings):
        """Calculate semantic coherence based on embedding similarity"""
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
            logger.warning(f"Error calculating coherence: {e}")
            return 0.5  # Default value in case of error
    
    def find_representative_keywords(self, df, embeddings, cluster_column='cluster_id'):
        """Find representative keywords for each cluster"""
        st.subheader("Cluster Representatives Analysis")
        
        rep_progress = st.progress(0)
        rep_text = st.empty()
        rep_text.text("Identifying representative keywords...")
        
        clusters_with_representatives = {}
        
        try:
            # Process in batches for large datasets
            for i, cluster_num in enumerate(df[cluster_column].unique()):
                # Get keywords in this cluster
                cluster_df = df[df[cluster_column] == cluster_num]
                cluster_size = len(cluster_df)
                n_representatives = min(20, cluster_size)
                
                # Get indices of keywords in this cluster
                indices = cluster_df.index.tolist()
                
                # Calculate centroid
                cluster_embeddings = np.array([embeddings[df.index.get_loc(i)] for i in indices])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # For large clusters, use MiniBatchKMeans to find diverse representatives
                if cluster_size > 100:
                    # Use k-means to find diverse representatives
                    n_subclusters = min(n_representatives, cluster_size // 5)
                    kmeans = MiniBatchKMeans(n_clusters=n_subclusters, random_state=42)
                    subcluster_labels = kmeans.fit_predict(cluster_embeddings)
                    
                    # Take closest to each subcluster centroid
                    representative_indices = []
                    for j in range(n_subclusters):
                        subcluster_points = cluster_embeddings[subcluster_labels == j]
                        if len(subcluster_points) == 0:
                            continue
                            
                        subcluster_centroid = np.mean(subcluster_points, axis=0)
                        subcl_indices = [indices[k] for k in range(len(indices)) 
                                        if subcluster_labels[k] == j]
                        subcl_embeddings = [cluster_embeddings[k] for k in range(len(cluster_embeddings))
                                          if subcluster_labels[k] == j]
                        
                        # Find closest to subcluster centroid
                        distances = [np.linalg.norm(emb - subcluster_centroid) for emb in subcl_embeddings]
                        if distances:
                            closest_idx = np.argmin(distances)
                            representative_indices.append(subcl_indices[closest_idx])
                    
                    # If we need more, add closest to overall centroid
                    if len(representative_indices) < n_representatives:
                        remaining = n_representatives - len(representative_indices)
                        used_indices = set(representative_indices)
                        
                        distances = [(i, np.linalg.norm(emb - centroid)) 
                                    for i, emb in zip(indices, cluster_embeddings)
                                    if i not in used_indices]
                        
                        distances.sort(key=lambda x: x[1])
                        representative_indices.extend([idx for idx, _ in distances[:remaining]])
                else:
                    # For small clusters, just use distance to centroid
                    distances = [(i, np.linalg.norm(embeddings[df.index.get_loc(i)] - centroid)) 
                                for i in indices]
                    distances.sort(key=lambda x: x[1])
                    representative_indices = [idx for idx, _ in distances[:n_representatives]]
                
                # Get the actual keywords
                representative_keywords = df.loc[representative_indices]['keyword'].tolist()
                clusters_with_representatives[cluster_num] = representative_keywords
                
                # Update progress
                rep_progress.progress((i+1) / len(df[cluster_column].unique()))
            
            rep_progress.progress(1.0)
            rep_text.text(f"✅ Identified representative keywords for {len(clusters_with_representatives)} clusters")
            
        except Exception as e:
            logger.error(f"Error identifying representative keywords: {e}")
            # Fallback: take the first N keywords of each cluster
            for cluster_num in df[cluster_column].unique():
                cluster_keywords = df[df[cluster_column] == cluster_num]['keyword'].tolist()
                clusters_with_representatives[cluster_num] = cluster_keywords[:min(20, len(cluster_keywords))]
            st.warning("Using basic representative keywords as fallback")
        
        return clusters_with_representatives
    
    def generate_cluster_names(self, clusters_with_representatives, client, model=None):
        """Generate names for clusters using OpenAI"""
        model = model or self.gpt_model
        
        if not clusters_with_representatives:
            return {}

        results = {}
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text("Generating names and descriptions for clusters...")

        # Process clusters in smaller batches to manage costs
        batch_size = 5  # Process 5 clusters at a time
        total_batches = (len(clusters_with_representatives) + batch_size - 1) // batch_size
        
        for batch_idx, batch_start in enumerate(range(0, len(clusters_with_representatives), batch_size)):
            batch_end = min(batch_start + batch_size, len(clusters_with_representatives))
            batch_clusters = list(clusters_with_representatives.items())[batch_start:batch_end]

            try:
                # Improved analysis prompt for better understanding of clusters
                analysis_prompt = """I'll provide representative keywords for several semantic clusters. For each numbered cluster, analyze the keywords to identify:
1. The main topic or theme
2. User intent or purpose 
3. Any patterns in language or terms
4. What makes this group distinct from others

Be insightful in your analysis.

"""
                # Track cluster order
                cluster_order = []
                for cluster_id, keywords in batch_clusters:
                    cluster_order.append(cluster_id)
                    analysis_prompt += f"Cluster {cluster_id} keywords: {', '.join(keywords[:20])}\n\n"

                # Get analysis of the clusters
                analysis_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=500
                )

                analysis_text = analysis_response.choices[0].message.content.strip()

                # Improved naming prompt with clear structure requirements
                naming_prompt = f"""Based on the following analysis of keyword clusters, provide an accurate name and description for each.

Analysis:
{analysis_text}

For each cluster, provide:
1. A specific, descriptive name (3-5 words) that clearly identifies the theme
2. A concise description (1-2 sentences) 

Format your response as valid JSON with this exact structure:
{{
  "clusters": [
    {{"cluster_id": {cluster_order[0]}, "cluster_name": "Name for cluster {cluster_order[0]}", "description": "Description for cluster {cluster_order[0]}"}},
    {{"cluster_id": {cluster_order[1] if len(cluster_order) > 1 else cluster_order[0]}, "cluster_name": "Name for cluster {cluster_order[1] if len(cluster_order) > 1 else cluster_order[0]}", "description": "Description for cluster {cluster_order[1] if len(cluster_order) > 1 else cluster_order[0]}"}},
    ...
  ]
}}

Ensure the JSON is properly formatted, with no trailing commas and properly closed brackets.
"""

                # Configure parameters based on model capabilities
                completion_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": naming_prompt}],
                    "temperature": 0.2,
                    "max_tokens": 600
                }
                
                # Use response_format for compatible models
                if "gpt" in model.lower():
                    completion_params["response_format"] = {"type": "json_object"}
                    
                naming_response = client.chat.completions.create(**completion_params)
                naming_text = naming_response.choices[0].message.content.strip()

                # Parse the JSON response with robust error handling
                try:
                    # First attempt: Direct JSON parsing
                    import json
                    data = json.loads(naming_text)
                    
                    # Extract clusters data
                    if "clusters" in data and isinstance(data["clusters"], list):
                        clusters_data = data["clusters"]
                        
                        # Process each cluster in the response
                        for cluster_info in clusters_data:
                            if "cluster_id" in cluster_info and "cluster_name" in cluster_info:
                                cluster_id = cluster_info["cluster_id"]
                                results[cluster_id] = (
                                    cluster_info.get("cluster_name", f"Cluster {cluster_id}"),
                                    cluster_info.get("description", "No description provided.")
                                )
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}")
                    
                    # Fallback: Extract information using regex
                    try:
                        import re
                        for cluster_id in cluster_order:
                            # Look for patterns like "cluster_id": 1, "cluster_name": "Name"
                            pattern = rf'"cluster_id"\s*:\s*{cluster_id}\s*,\s*"cluster_name"\s*:\s*"([^"]+)"'
                            name_match = re.search(pattern, naming_text)
                            
                            # Look for description
                            desc_pattern = rf'"cluster_id"\s*:\s*{cluster_id}[^}}]+?"description"\s*:\s*"([^"]+)"'
                            desc_match = re.search(desc_pattern, naming_text)
                            
                            name = name_match.group(1) if name_match else f"Cluster {cluster_id}"
                            desc = desc_match.group(1) if desc_match else f"Group of keywords {cluster_id}"
                            
                            results[cluster_id] = (name, desc)
                    except Exception:
                        # Ultimate fallback for this batch
                        for cluster_id in cluster_order:
                            results[cluster_id] = (f"Cluster {cluster_id}", f"Group of keywords {cluster_id}")

            except Exception as e:
                logger.error(f"Error generating names for batch {batch_idx+1}/{total_batches}: {e}")
                # Provide default names
                for cluster_id, _ in batch_clusters:
                    results[cluster_id] = (f"Cluster {cluster_id}", f"Group of keywords {cluster_id}")
            
            # Update progress
            progress_bar.progress((batch_idx + 1) / total_batches)
            time.sleep(0.5)  # Small delay to avoid rate limits
        
        # Ensure all clusters have names
        for cluster_id in clusters_with_representatives:
            if cluster_id not in results:
                results[cluster_id] = (f"Cluster {cluster_id}", f"Group of keywords {cluster_id}")
        
        progress_bar.progress(1.0)
        progress_text.text(f"✅ Generated names and descriptions for {len(results)} clusters")
        
        return results

def calculate_api_cost(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    """Calculate estimated OpenAI API cost"""
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
    
    # Estimate average of 2 tokens per keyword
    estimated_tokens = keywords_for_embeddings * 2
    results["embedding_cost"] = (estimated_tokens / 1000) * EMBEDDING_COST_PER_1K
    
    # 2. Cluster naming cost
    # Estimate tokens for naming clusters
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

# BLOCK 4 - END

# BLOCK 5 - START 
# Streamlit UI Functions

def add_cost_calculator():
    """Add cost calculator widget to sidebar"""
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
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Keywords processed with OpenAI", 
                    f"{cost_results['processed_keywords']:,}",
                    help="OpenAI processes up to 5,000 keywords, the rest are processed using similarity propagation"
                )
                
                st.metric(
                    "Embedding cost", 
                    f"${cost_results['embedding_cost']:.4f}",
                    help="Cost of generating semantic vectors with text-embedding-3-small"
                )
            
            with col2:
                st.metric(
                    "Cost for naming clusters", 
                    f"${cost_results['naming_cost']:.4f}",
                    help=f"Cost of generating names and descriptions using {calc_model}"
                )
                
                st.metric(
                    "TOTAL COST", 
                    f"${cost_results['total_cost']:.4f}",
                    help="Total estimated API cost (does not include computational resources)"
                )
            
            st.info("""
            **Note:** This is an approximate estimate. The actual cost may vary 
            depending on keyword length and cluster complexity.
            Using SentenceTransformers as an alternative, the cost is $0.
            """)

def show_csv_cost_estimate(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    """Show estimated cost of processing the loaded CSV"""
    if num_keywords > 0:
        cost_results = calculate_api_cost(num_keywords, selected_model, num_clusters)
        
        with st.sidebar.expander("💰 Estimated Cost (Current CSV)", expanded=True):
            st.markdown(f"### Estimated Cost for {num_keywords:,} Keywords")
            
            # Show breakdown
            st.markdown(f"""
            - **Keywords processed with OpenAI**: {cost_results['processed_keywords']:,}
            - **Embedding cost**: ${cost_results['embedding_cost']:.4f}
            - **Cost for naming clusters**: ${cost_results['naming_cost']:.4f}
            - **TOTAL COST**: ${cost_results['total_cost']:.4f}
            """)
            
            if cost_results['processed_keywords'] < num_keywords:
                st.info(f"""
                {cost_results['processed_keywords']:,} keywords are directly processed with OpenAI.
                The remaining {num_keywords - cost_results['processed_keywords']:,} are processed 
                using semantic similarity propagation.
                """)
            
            st.markdown("""
            **Cost savings**: If you prefer not to use OpenAI, you can 
            use SentenceTransformers as a free alternative with 
            good results.
            """)

def create_sample_data():
    """Create sample data for demonstration"""
    return pd.DataFrame({
        "keyword": [
            "seo tools", "keyword research tools", "backlink checker", "rank tracker software",
            "google keyword planner alternative", "ahrefs vs semrush", "free seo tools",
            "technical seo guide", "how to improve website ranking", "google algorithm updates",
            "meta description optimization", "how to do keyword research", "best on-page seo practices",
            "how to build backlinks", "local seo tips", "mobile seo best practices",
            "voice search optimization", "amazon product ranking", "youtube seo tips",
            "wordpress seo plugin", "ecommerce seo checklist", "content marketing strategy",
            "long tail keywords", "semantic search", "google search console tutorial",
            "seo audit checklist", "website speed optimization", "seo for beginners",
            "link building strategies", "seo competitor analysis", "schema markup guide",
            "what is domain authority", "google penalties recovery", "international seo",
            "keyword difficulty", "site structure for seo", "featured snippets optimization",
            "core web vitals", "what is ctr in seo", "canonical tags guide", "robots.txt examples"
        ]
    })

def display_results(df, embeddings=None):
    """Display results of clustering in Streamlit"""
    st.markdown("<div class='main-header'>Clustering Results</div>", unsafe_allow_html=True)
    
    # Show visualizations
    with st.expander("Visualizations", expanded=True):
        # Cluster size distribution
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
        
        # Cluster coherence 
        if 'cluster_coherence' in df.columns:
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
    
    # Cluster explorer
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
                if 'cluster_coherence' in cluster_df.columns:
                    st.markdown(f"**Semantic coherence:** {cluster_df['cluster_coherence'].iloc[0]:.3f}")
                
                st.markdown("**Representative keywords:**")
                rep_keywords = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                if rep_keywords:
                    st.markdown("<ul>" + "".join([f"<li>{kw}</li>" for kw in rep_keywords[:10]]) + "</ul>", unsafe_allow_html=True)
            
            # Show all keywords in the cluster
            st.markdown("### All keywords")
            st.dataframe(cluster_df[['keyword']], use_container_width=True)
    
    # Download options
    with st.expander("Download Results"):
        # Option to download full CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV with all results",
            data=csv,
            file_name="semantic_clustered_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Option to download summary
        st.subheader("Cluster Summary")
        summary_df = df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
        summary_df.columns = ['ID', 'Name', 'Description', 'Number of Keywords']
        
        # Add coherence if available
        if 'cluster_coherence' in df.columns:
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
            label="Download cluster summary",
            data=csv_summary,
            file_name="semantic_clusters_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

# BLOCK 5 - END

# BLOCK 6 - START 
#  Main Application

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Advanced Semantic Keyword Clustering",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add CSS styles
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
    Group semantically similar keywords using advanced NLP and clustering techniques.
    Upload your CSV file with keywords and configure parameters to get high-quality semantic clusters.
    """)

    # Show library status
    with st.expander("Semantic library status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if openai_available:
                st.success("✅ OpenAI available (requires API Key)")
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
                st.success("✅ spaCy available")
            else:
                st.warning("⚠️ spaCy not available")
                
            if hdbscan_available:
                st.success("✅ HDBSCAN available")
            else:
                st.warning("⚠️ HDBSCAN not available")
                
        with col3:
            # Installation info
            st.info("""
            For more features:
            ```
            pip install sentence-transformers spacy hdbscan
            python -m spacy download en_core_web_sm
            ```
            """)

    # Initialize session state
    if 'process_complete' not in st.session_state:
        st.session_state.process_complete = False
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None

    # Sidebar configuration
    st.sidebar.markdown("<div class='sub-header'>Configuration</div>", unsafe_allow_html=True)

    # 1. Upload CSV or use sample data
    uploaded_file = st.sidebar.file_uploader("Upload your keywords CSV", type=['csv'])
    
    # Sample data option
    use_sample_data = False
    if uploaded_file is None:
        use_sample_data = st.sidebar.checkbox("Use sample data for demonstration")
        if use_sample_data:
            st.sidebar.success("✅ Using sample SEO keywords for demonstration")

    # 2. OpenAI API Key
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key (recommended)",
        type="password", 
        help="Provide your OpenAI API Key for high-quality embeddings (up to 5000 keywords). If not provided, SentenceTransformers will be used as a free alternative."
    )

    # Show semantic processing status
    if openai_available:
        if openai_api_key:
            st.sidebar.success("✅ API Key provided - Using OpenAI for high precision embeddings")
        else:
            if sentence_transformers_available:
                st.sidebar.info("ℹ️ No API Key - Using SentenceTransformers as free alternative")
            else:
                st.sidebar.warning("⚠️ No API Key or SentenceTransformers - Using TF-IDF (reduced precision)")
    else:
        if sentence_transformers_available:
            st.sidebar.info("ℹ️ OpenAI not available - Using SentenceTransformers as free alternative")
        else:
            st.sidebar.error("❌ Advanced methods not available - Using TF-IDF (reduced precision)")

    # 3. Clustering parameters
    st.sidebar.markdown("<div class='sub-header'>Parameters</div>", unsafe_allow_html=True)

    # Parameters explanation panel
    with st.sidebar.expander("ℹ️ Parameter Guide", expanded=False):
        st.markdown("""
        ### Clustering Parameter Guide
        
        #### Number of clusters
        **What is it?** The number of groups your keywords will be divided into.
        
        **How to use it:** 
        - **↑ Increase** for more detailed and specific topic division.
        - **↓ Decrease** for broader groups.
        
        **Result:**
        - **High values** (15-30): Many small, specific groups.
        - **Low values** (5-10): Fewer but broader groups.
        - **Ideal:** Generally 8-15 for 1000 keywords. Increase proportionally with keyword count.
        
        ---
        
        #### Explained PCA variance (%)
        **What is it?** Determines how much original information is preserved when simplifying the data.
        
        **How to use it:**
        - **↑ Increase** for greater precision and preserving semantic nuances.
        - **↓ Decrease** to speed up processing with large datasets.
        
        **Result:**
        - **High values** (95-99%): Greater semantic precision but slower.
        - **Low values** (80-90%): Faster processing but may lose some nuances.
        - **Ideal:** 90-95% offers a good balance between precision and speed.
        
        ---
        
        #### Maximum PCA components
        **What is it?** Limits the maximum complexity of the analysis model.
        
        **How to use it:**
        - **↑ Increase** for large datasets with high thematic diversity.
        - **↓ Decrease** for smaller datasets or those centered on a single theme.
        
        **Result:**
        - **High values** (100-200): Captures more complex relationships between words.
        - **Low values** (30-75): More efficient but may oversimplify.
        - **Ideal:** Between 75-100 for most cases.
        
        ---
        
        #### Minimum term frequency
        **What is it?** Ignores words that appear in very few keywords. Helps filter rare words or typos.
        
        **How to use it:**
        - **↑ Increase** to eliminate uncommon terms and possible noise.
        - **↓ Decrease** to include less frequent terms that might be important.
        
        **Result:**
        - **High values** (3-5): Eliminates more rare terms, "cleaner" clustering.
        - **Low values** (1-2): Preserves uncommon terms, may retain more noise.
        - **Ideal:** 1-2 for small datasets, 2-3 for large datasets (+5000 keywords).
        
        ---
        
        #### Maximum term frequency (%)
        **What is it?** Ignores words that appear in a high percentage of keywords.
        
        **How to use it:**
        - **↑ Increase** to include more common terms.
        - **↓ Decrease** to filter out overly generic words.
        
        **Result:**
        - **High values** (90-100%): Includes almost all terms, even very common ones.
        - **Low values** (70-85%): Focus on more distinctive words, ignoring generic ones.
        - **Ideal:** 85-95% works well for most datasets.
        """)
        
        st.info("""
        **Tip:** If unsure, keep the default values. The application is optimized to work well with these parameters in most cases.
        
        For large datasets (+5000 keywords), consider slightly increasing the number of clusters and reducing the explained PCA variance to maintain reasonable processing times.
        """)

    # Parameter sliders with improved descriptions
    num_clusters = st.sidebar.slider(
        "Number of clusters", 
        min_value=2, 
        max_value=50, 
        value=10, 
        help="Number of groups your keywords will be divided into. More clusters = more specific groups."
    )

    pca_variance = st.sidebar.slider(
        "Explained PCA variance (%)", 
        min_value=50, 
        max_value=99, 
        value=90,  # Reduced from 95 to 90 for better performance
        help="Percentage of information preserved. Higher value = greater precision but slower."
    )

    max_pca_components = st.sidebar.slider(
        "Maximum PCA components", 
        min_value=10, 
        max_value=300, 
        value=100, 
        help="Maximum complexity limit. Higher value = captures more complex relationships."
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
        help="Ignores overly common terms. Lower value = eliminates more generic words."
    )

    gpt_model = st.sidebar.selectbox(
        "Model for naming clusters", 
        ["gpt-3.5-turbo", "gpt-4"], 
        index=0,
        help="GPT-4 provides more precise names but is more expensive and slower."
    )

    # Add cost calculator to sidebar
    add_cost_calculator()

    # Data processing section
    if uploaded_file is not None or use_sample_data:
        # Show estimated cost based on CSV
        if not st.session_state.process_complete:
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    num_keywords = len(df)
                    st.success(f"✅ Loaded {num_keywords} keywords from CSV")
                    show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    st.info("Trying alternative format...")
                    try:
                        content = uploaded_file.getvalue().decode('utf-8')
                        df = pd.read_csv(StringIO(content), sep=None, engine='python')
                        if len(df.columns) == 1:
                            df.columns = ["keyword"]
                        else:
                            df = df.iloc[:, 0].to_frame()
                            df.columns = ["keyword"]
                        num_keywords = len(df)
                        st.success(f"✅ Loaded {num_keywords} keywords from CSV (alternative format)")
                        show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
                    except Exception as e2:
                        st.error(f"Could not read CSV: {e2}")
                        uploaded_file = None
            else:  # Sample data
                df = create_sample_data()
                num_keywords = len(df)
                st.success(f"✅ Using {num_keywords} sample keywords for demonstration")
                show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)

        # Button to run clustering
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if not st.session_state.process_complete:
                if st.button("Start Advanced Semantic Clustering", type="primary", use_container_width=True):
                    # Define input data
                    if uploaded_file is not None:
                        try:
                            if 'df' not in locals():
                                df = pd.read_csv(uploaded_file, header=None)
                                df.columns = ["keyword"]
                            if len(df.columns) != 1:
                                df = df.iloc[:, 0].to_frame()
                                df.columns = ["keyword"]
                        except Exception as e:
                            st.error(f"Error reading CSV: {e}")
                            return
                    else:  # use_sample_data must be True
                        df = create_sample_data()
                    
                    # Run the complete clustering pipeline
                    with st.spinner("Processing keywords..."):
                        try:
                            # Initialize processor
                            processor = KeywordProcessor(
                                use_openai=(openai_api_key is not None and openai_api_key.strip() != ""),
                                openai_api_key=openai_api_key,
                                num_clusters=num_clusters
                            )
                            
                            # Step 1: Preprocess keywords
                            st.subheader("Keyword Preprocessing")
                            processed_keywords = processor.preprocess_keywords_cached(df["keyword"].tolist())
                            df['keyword_processed'] = processed_keywords
                            st.success("✅ Keywords preprocessed successfully")
                            
                            # Step 2: Generate embeddings
                            st.subheader("Semantic Vector Generation")
                            embeddings = processor.generate_embeddings_cached(
                                df['keyword_processed'].tolist(),
                                use_openai=(openai_api_key is not None and openai_api_key.strip() != "")
                            )
                            st.success(f"✅ Generated {embeddings.shape[1]}-dimensional semantic vectors")
                            
                            # Step 3: Apply PCA if embeddings are high-dimensional
                            if embeddings.shape[1] > max_pca_components:
                                st.subheader("Dimensionality Reduction (PCA)")
                                
                                try:
                                    pca_progress = st.progress(0)
                                    pca_text = st.empty()
                                    pca_text.text("Analyzing explained variance...")
                                    
                                    # Find optimal number of components
                                    pca = PCA()
                                    pca.fit(embeddings)
                                    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                                    pca_progress.progress(0.3)
                                    
                                    # Find components for target variance
                                    target_variance = pca_variance / 100.0
                                    n_components = np.argmax(cumulative_variance >= target_variance) + 1
                                    
                                    # If not enough components for target variance, use maximum
                                    if n_components == 1 and len(cumulative_variance) > 1:
                                        n_components = min(max_pca_components, len(cumulative_variance))
                                        
                                    pca_text.text(f"Components for {pca_variance}% variance: {n_components}")
                                    pca_progress.progress(0.6)
                                    
                                    # Apply PCA with optimal components
                                    max_components = min(n_components, max_pca_components)
                                    pca = PCA(n_components=max_components)
                                    embeddings_reduced = pca.fit_transform(embeddings)
                                    
                                    pca_progress.progress(1.0)
                                    pca_text.text(f"✅ PCA applied: {max_components} dimensions ({pca_variance}% variance)")
                                except Exception as e:
                                    st.error(f"PCA error: {e}")
                                    st.info("Continuing without dimensionality reduction")
                                    embeddings_reduced = embeddings
                            else:
                                embeddings_reduced = embeddings
                                st.info(f"Embeddings dimensionality already appropriate ({embeddings.shape[1]}). No PCA needed.")
                            
                            # Step 4: Perform clustering
                            st.subheader("Advanced Semantic Clustering")
                            clustering_engine = ClusteringEngine(num_clusters=num_clusters)
                            
                            try:
                                cluster_labels = clustering_engine.cluster_cached(
                                    embeddings_reduced, 
                                    num_clusters=num_clusters
                                )
                                df["cluster_id"] = cluster_labels
                                st.success(f"✅ Keywords grouped into {len(df['cluster_id'].unique())} semantic clusters")
                            except Exception as e:
                                st.error(f"Clustering error: {e}")
                                st.info("Trying alternative clustering...")
                                # Basic fallback
                                from sklearn.cluster import KMeans
                                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                                df["cluster_id"] = kmeans.fit_predict(embeddings_reduced) + 1
                                st.success("✅ Clustering completed using K-Means as alternative")
                            
                            # Step 5: Refine clusters
                            st.subheader("Cluster Refinement")
                            df = clustering_engine.refine_clusters(df, embeddings_reduced)
                            final_clusters = len(df['cluster_id'].unique())
                            st.success(f"✅ Refinement completed: {final_clusters} final clusters")
                            
                            # Step 6: Find representative keywords
                            st.subheader("Cluster Analysis")
                            cluster_analyzer = ClusterAnalyzer(gpt_model=gpt_model)
                            clusters_with_representatives = cluster_analyzer.find_representative_keywords(
                                df, embeddings_reduced)
                            
                            # Step 7: Generate cluster names with OpenAI if available
                            if openai_api_key and openai_api_key.strip() != "" and openai_available:
                                st.subheader("Cluster Naming")
                                try:
                                    client = OpenAI(api_key=openai_api_key)
                                    cluster_names = cluster_analyzer.generate_cluster_names(
                                        clusters_with_representatives, 
                                        client,
                                        model=gpt_model
                                    )
                                except Exception as e:
                                    st.error(f"Error generating cluster names: {e}")
                                    cluster_names = {k: (f"Cluster {k}", f"Group of keywords {k}") 
                                                   for k in df['cluster_id'].unique()}
                            else:
                                st.warning("Cannot generate cluster names without OpenAI API Key")
                                cluster_names = {k: (f"Cluster {k}", f"Group of keywords {k}") 
                                               for k in df['cluster_id'].unique()}
                            
                            # Apply results to DataFrame
                            df['cluster_name'] = ''
                            df['cluster_description'] = ''
                            df['representative'] = False
                            
                            for cluster_num, (name, description) in cluster_names.items():
                                df.loc[df['cluster_id'] == cluster_num, 'cluster_name'] = name
                                df.loc[df['cluster_id'] == cluster_num, 'cluster_description'] = description
                                
                                # Mark representative keywords
                                for keyword in clusters_with_representatives.get(cluster_num, []):
                                    matching_indices = df[(df['cluster_id'] == cluster_num) & 
                                                         (df['keyword'] == keyword)].index
                                    if not matching_indices.empty:
                                        df.loc[matching_indices, 'representative'] = True
                            
                            # Step 8: Evaluate cluster quality
                            df = cluster_analyzer.evaluate_cluster_quality(df, embeddings_reduced)
                            
                            # Store results in session state
                            st.session_state.df_results = df
                            st.session_state.embeddings = embeddings_reduced
                            st.session_state.process_complete = True
                            
                            st.markdown("<div class='success-box'>✅ Semantic clustering completed successfully!</div>", 
                                       unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error during processing: {e}")
                            logger.error(f"Processing error: {e}", exc_info=True)
                            return

    # Show results if process is complete
    if st.session_state.process_complete and st.session_state.df_results is not None:
        display_results(st.session_state.df_results, st.session_state.embeddings)
        
        # Reset button
        if st.button("Reset and Start New Analysis", type="secondary", use_container_width=True):
            st.session_state.process_complete = False
            st.session_state.df_results = None
            st.session_state.embeddings = None
            st.rerun()

    # Additional information
    with st.expander("About Advanced Semantic Clustering"):
        st.markdown("""
        ### How this advanced semantic clustering works
        
        1. **Linguistic Preprocessing**: Keywords are analyzed using advanced NLP to extract named entities, relevant bigrams, and significant tokens.
        
        2. **High-Quality Embeddings**: State-of-the-art embedding models are used:
           - OpenAI Embeddings (up to 5000 keywords with API key)
           - SentenceTransformers (free alternative)
           - TF-IDF as last resort
        
        3. **Intelligent Dimensionality Reduction**: Optimized PCA preserves important semantic relationships.
        
        4. **Advanced Clustering**: Algorithms that automatically discover optimal structure:
           - HDBSCAN for detecting natural clusters
           - Optimized hierarchical agglomerative clustering
           - Automatic optimal cluster number determination
        
        5. **Post-Clustering Refinement**: Identifies and corrects problematic assignments:
           - Detection of semantic outliers
           - Merging of very similar clusters
           - Reassignment of misclassified keywords
        
        6. **Multi-Metric Evaluation**: Rigorous quality analysis:
           - Internal semantic coherence
           - Density and compactness
           - Separation between clusters
           - Diagnosis of problematic clusters
        
        ### Tips for better results
        
        - **Keyword quality**: Clustering works best when keywords are related to the same domain or industry.
        
        - **Preprocessing**: Ensure your keywords don't contain spelling errors or strange characters.
        
        - **OpenAI API Key**: Provide an API Key for higher quality embeddings, although SentenceTransformers offers good results at no cost.
        
        - **Number of clusters**: Consider using automatic determination of the optimal number of clusters.
        
        - **Iterative evaluation**: Examine clusters with low coherence and consider adjusting parameters or dividing them.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
            Developed for advanced semantic keyword clustering | Version 3.0 with OpenAI/SentenceTransformers hybrid
        </div>
        """, 
        unsafe_allow_html=True
    )

    return

# BLOCK 6 - END

# BLOCK 7 - START 
# Application Execution

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Unhandled application error")
        
        # Show full traceback in debug mode
        import traceback
        st.write("Detailed error information:")
        st.code(traceback.format_exc())
        
        # Provide recovery options
        st.warning("""
        An unexpected error occurred. You can:
        1. Try refreshing the page
        2. Check that your CSV file is properly formatted
        3. Try with fewer keywords if you have a very large dataset
        """)
        
        # Reset button for recovery
        if st.button("Reset Application"):
            # Clear cache and session state
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# BLOCK 7 - END
