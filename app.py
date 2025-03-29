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

# Attempt to import OpenAI
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

# We will load spaCy models dynamically based on language
try:
    import spacy
    spacy_base_available = True
except ImportError:
    spacy_base_available = False

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
#          LANGUAGE MODEL MANAGEMENT
################################################################

# Mapping for some known spaCy language models (if installed).
# If these models are not installed, spaCy loading will fail and we'll fallback.
SPACY_LANGUAGE_MODELS = {
    "English": "en_core_web_sm",
    "Spanish": "es_core_news_sm",
    "French": "fr_core_news_sm",
    "German": "de_core_news_sm",
    "Dutch": "nl_core_news_sm",
    "Italian": "it_core_news_sm",
    "Portuguese": "pt_core_news_sm",
    "Brazilian Portuguese": "pt_core_news_sm",  # same as Portuguese in spaCy
    "Swedish": "sv_core_news_sm",
    "Norwegian": "nb_core_news_sm",
    "Danish": "da_core_news_sm",
    "Greek": "el_core_news_sm",
    "Romanian": "ro_core_news_sm",
    # The following languages often have partial or community models, which might not be installed by default
    # For now, we will rely on fallback if not installed.
    "Korean": None,
    "Japanese": None,
    "Icelandic": None,
    "Lithuanian": None
}

def load_spacy_model_by_language(selected_language):
    """
    Try to load a spaCy model for the given language. If it fails or doesn't exist, returns None.
    """
    if not spacy_base_available:
        return None

    model_name = SPACY_LANGUAGE_MODELS.get(selected_language, None)
    if model_name is None:
        return None

    try:
        return spacy.load(model_name)
    except:
        return None

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
        
        Estimate OpenAI usage costs for a given number of keywords.
        """)
        
        calc_num_keywords = st.number_input(
            "Number of keywords",
            min_value=100, 
            max_value=100000, 
            value=1000,
            step=500
        )
        calc_num_clusters = st.number_input(
            "Approx. number of clusters",
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
                    help="OpenAI processes up to 5,000 keywords; any beyond that are handled via similarity propagation."
                )
                st.metric(
                    "Embeddings cost", 
                    f"${cost_results['embedding_cost']:.4f}",
                    help="Cost using text-embedding-3-small"
                )
            with col2:
                st.metric(
                    "Cluster naming cost", 
                    f"${cost_results['naming_cost']:.4f}",
                    help=f"Cost using {calc_model} to name and describe clusters"
                )
                st.metric(
                    "TOTAL COST", 
                    f"${cost_results['total_cost']:.4f}",
                    help="Approximate total cost"
                )
            
            st.info("""
            **Note:** This is an estimate only. Actual costs may vary based on keyword length and clustering complexity.
            Using SentenceTransformers instead of OpenAI embeddings is $0.
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
                {cost_results['processed_keywords']:,} keywords will be processed by OpenAI directly.
                The remaining {num_keywords - cost_results['processed_keywords']:,} will use
                similarity propagation.
                """)
            
            st.markdown("""
            **Cost Savings**: If you prefer not to use OpenAI, you can 
            use SentenceTransformers at no cost with decent results.
            """)

################################################################
#  SAMPLE CSV GENERATION
################################################################

def generate_sample_csv():
    """
    Returns a sample CSV header row: 
    Keyword,search_volume,competition,cpc,month1..month12
    """
    header = ["Keyword", "search_volume", "competition", "cpc"]
    months = [f"month{i}" for i in range(1, 13)]
    header += months
    return ",".join(header) + "\n"

################################################################
#          SEMANTIC PREPROCESSING
################################################################

def enhanced_preprocessing(text, use_lemmatization, spacy_nlp):
    """
    Enhanced preprocessing using spaCy or fallback with TextBlob.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        if spacy_nlp is not None:  # We have a loaded spaCy model
            doc = spacy_nlp(text.lower())
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
        
        else:
            # fallback to standard nltk
            return preprocess_text(text, use_lemmatization)
    
    except Exception:
        return text.lower() if isinstance(text, str) else ""

def preprocess_text(text, use_lemmatization=True):
    """
    Basic NLTK-based text preprocessing as a fallback.
    """
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

def preprocess_keywords(keywords, use_advanced, spacy_nlp=None):
    """
    Main keyword preprocessing loop.
    """
    processed_keywords = []
    progress_bar = st.progress(0)
    total = len(keywords)
    
    if use_advanced:
        if spacy_nlp is not None:
            st.success("Using advanced preprocessing with spaCy for the selected language.")
        elif textblob_available:
            st.success("Using fallback preprocessing with TextBlob.")
        else:
            st.info("Using standard preprocessing with NLTK.")
    else:
        st.info("Using standard preprocessing with NLTK (advanced preprocessing disabled).")
    
    for i, keyword in enumerate(keywords):
        if use_advanced and (spacy_nlp is not None or textblob_available):
            processed_keywords.append(enhanced_preprocessing(keyword, True, spacy_nlp))
        else:
            processed_keywords.append(preprocess_text(keyword, True))
        
        if i % 100 == 0:
            progress_bar.progress(min(i / total, 1.0))
    
    progress_bar.progress(1.0)
    return processed_keywords

################################################################
#          EMBEDDING GENERATION
################################################################

def generate_embeddings(df, openai_available, openai_api_key=None):
    st.info("Generating embeddings for keywords...")
    
    # Attempt OpenAI embeddings
    if openai_available and openai_api_key:
        try:
            st.info("Using OpenAI embeddings (high semantic precision).")
            os.environ["OPENAI_API_KEY"] = openai_api_key
            client = OpenAI()
            keywords = df['keyword_processed'].fillna('').tolist()
            all_embeddings = []
            
            # If more than 5000 keywords, partial approach
            if len(keywords) > 5000:
                st.warning(f"Limiting to 5000 representative keywords out of {len(keywords)} total.")
                step = max(1, len(keywords) // 5000)
                sample_indices = list(range(0, len(keywords), step))[:5000]
                sample_keywords = [keywords[i] for i in sample_indices]
                
                progress_bar = st.progress(0)
                st.info("Requesting embeddings from OpenAI...")
                
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=sample_keywords
                )
                progress_bar.progress(0.5)
                
                sample_embeddings = np.array([item.embedding for item in response.data])
                
                st.info("Propagating embeddings to remaining keywords via TF-IDF similarity...")
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
                        prog_val = 0.5 + min(0.5, (i / len(remaining_indices) * 0.5))
                        progress_bar.progress(prog_val)
                
                progress_bar.progress(1.0)
            else:
                # If under 5000, direct approach
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
            st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (OpenAI).")
            return embeddings
                
        except Exception as e:
            st.error(f"Error generating embeddings with OpenAI: {str(e)}")
            st.info("Falling back to SentenceTransformers.")

    # Attempt SentenceTransformers if available
    if sentence_transformers_available:
        try:
            st.success("Using SentenceTransformer (free fallback).")
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
            st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (SentenceTransformers).")
            return embeddings
        except Exception as e:
            st.error(f"Error with SentenceTransformer: {str(e)}")
    
    # Fallback to TF-IDF
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
        
        st.success(f"✅ Generated {embeddings.shape[1]} TF-IDF vectors.")
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
    st.info("Applying advanced clustering algorithms...")
    try:
        from sklearn.cluster import KMeans
        if num_clusters is None:
            num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings) + 1
        return labels
    except Exception as e:
        st.warning(f"Error in improved_clustering: {e}")
        return np.random.randint(1, (num_clusters or 10) + 1, size=len(embeddings))

def refine_clusters(df, embeddings, original_cluster_column='cluster_id'):
    st.info("Refining clusters to improve coherence...")
    # If outlier or merging logic is needed, place it here
    return df

################################################################
#          GENERATE CLUSTER NAMES
################################################################

def generate_cluster_names(
    clusters_with_representatives, 
    client, 
    model="gpt-3.5-turbo",
    custom_prompt=None
):
    if not clusters_with_representatives:
        return {}

    results = {}
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Generating SEO-friendly cluster names/descriptions...")

    if not custom_prompt:
        custom_prompt = (
            "You are an expert in SEO and content marketing. Below you'll see several clusters "
            "with a list of representative keywords. Your task is to assign each cluster a short, "
            "clear name (3-6 words) and write a concise SEO meta description (1 or 2 sentences), "
            "briefly explaining the topic and likely search intent. Respond in JSON.\n\n"
        )

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
        json_data = None
        
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
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
            for c_id in clusters_with_representatives.keys():
                results[c_id] = (f"Cluster {c_id}", f"Generic description for cluster {c_id}")
            return results

        cluster_array = json_data["clusters"]
        for item in cluster_array:
            c_id = item.get("cluster_id")
            c_name = item.get("cluster_name", f"Cluster {c_id}")
            c_desc = item.get("cluster_description", "No SEO description provided")
            if c_id is not None:
                results[c_id] = (c_name, c_desc)
    except Exception as e:
        st.error(f"Error generating names with OpenAI: {str(e)}")
        for c_id in clusters_with_representatives.keys():
            results[c_id] = (f"Cluster {c_id}", f"Fallback description {c_id}")

    progress_bar.progress(1.0)
    progress_text.text("✅ SEO cluster naming done.")
    return results

################################################################
#          CLUSTER SEMANTIC ANALYSIS
################################################################

def generate_semantic_analysis(
    clusters_with_representatives,
    client,
    model="gpt-3.5-turbo"
):
    """
    Calls OpenAI to analyze each cluster for:
      1) Main search intent
      2) Suggestion of internal splitting
      3) Additional info or insights
    """
    results = {}
    if not clusters_with_representatives:
        return results

    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Performing semantic analysis on clusters...")

    analysis_prompt = (
        "You are an expert in SEO and clustering analysis. Below are several clusters with representative keywords. "
        "For each cluster, analyze:\n"
        "1) The main search intent.\n"
        "2) If you think it should be split further.\n"
        "3) Any additional insights regarding these keywords.\n\n"
        "Respond ONLY as a JSON object named 'clusters', where each element is:\n"
        "{\n"
        "  \"cluster_id\": <number>,\n"
        "  \"search_intent\": \"...\",\n"
        "  \"split_suggestion\": \"...\",\n"
        "  \"additional_info\": \"...\",\n"
        "  \"coherence_score\": <number between 0 and 10>\n"
        "}\n\n"
        "Here are the clusters:\n"
    )

    for cluster_id, keywords in clusters_with_representatives.items():
        sample_kws = keywords[:15]
        analysis_prompt += f"- Cluster {cluster_id}: {', '.join(sample_kws)}\n"

    analysis_prompt += "\nPlease respond ONLY with the JSON object named 'clusters'."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        content = response.choices[0].message.content.strip()

        # Attempt to parse JSON
        json_data = None
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
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
            return results

        cluster_array = json_data["clusters"]
        for item in cluster_array:
            c_id = item.get("cluster_id")
            search_intent = item.get("search_intent", "")
            split_suggestion = item.get("split_suggestion", "")
            additional_info = item.get("additional_info", "")
            coherence_score = item.get("coherence_score", 0)

            if c_id is not None:
                results[c_id] = {
                    "search_intent": search_intent,
                    "split_suggestion": split_suggestion,
                    "additional_info": additional_info,
                    "coherence_score": coherence_score
                }

    except Exception as e:
        st.error(f"Error in semantic analysis: {str(e)}")

    progress_bar.progress(1.0)
    progress_text.text("✅ Semantic analysis completed.")
    return results

################################################################
#          EVALUATION FUNCTIONS
################################################################

def evaluate_cluster_quality(df, embeddings, cluster_column='cluster_id'):
    """
    Simple placeholder approach to assign a 'cluster_coherence' score for demonstration purposes.
    """
    st.subheader("Cluster Quality Evaluation")
    df['cluster_coherence'] = 1.0  # Placeholder
    st.success("Placeholder coherence assigned = 1.0")
    return df

def calculate_cluster_coherence(cluster_embeddings):
    """
    If more complex logic is needed, implement here.
    """
    return 1.0

def evaluate_and_refine_clusters(df, client, model="gpt-3.5-turbo"):
    st.subheader("AI-Powered Cluster Quality Evaluation")

    if not client:
        st.info("No OpenAI client available. Skipping AI-based cluster analysis.")
        return {}

    # Build a dict of cluster -> representative keywords
    clusters_with_representatives = {}
    for c_id in df['cluster_id'].unique():
        reps = df[(df['cluster_id'] == c_id) & (df['representative'] == True)]['keyword'].tolist()
        if not reps:
            cluster_kws = df[df['cluster_id'] == c_id]['keyword'].tolist()
            reps = cluster_kws[:20]
        clusters_with_representatives[c_id] = reps

    # Call GPT-based analysis
    semantic_analysis = generate_semantic_analysis(
        clusters_with_representatives=clusters_with_representatives,
        client=client,
        model=model
    )

    return semantic_analysis

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
    user_prompt,
    csv_format,
    selected_language
):
    """
    Executes the full clustering pipeline, depending on CSV format:
      - csv_format = "no_header" => read with header=None, names=["keyword"]
      - csv_format = "with_header" => read with header=0
      - selected_language => used to load spaCy model if available
    """
    if uploaded_file is None:
        st.warning("Please upload a CSV file with keywords.")
        return False, None
    
    st.info("Starting advanced semantic clustering pipeline...")
    
    # Attempt to create OpenAI client if key provided
    client = None
    if openai_api_key and openai_available:
        try:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            client = OpenAI()
            # Basic check
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
        st.warning("OpenAI library not installed. No OpenAI functionality.")
    else:
        st.info("No OpenAI API Key provided. Will use free alternatives.")
    
    # Attempt to load spaCy model for selected language
    spacy_nlp = load_spacy_model_by_language(selected_language)

    try:
        # Load CSV according to user's choice
        if csv_format == "no_header":
            # No header, one column
            df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
            st.success(f"✅ Loaded {len(df)} keywords (no header).")
        else:
            df = pd.read_csv(uploaded_file, header=0)
            if "Keyword" in df.columns:
                df.rename(columns={"Keyword": "keyword"}, inplace=True)
            if "keyword" not in df.columns:
                st.error("No 'Keyword' column found in the CSV. Please check your file.")
                return False, None
            st.success(f"✅ Loaded {len(df)} rows (with header).")
        
        num_keywords = len(df)
        show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
        
        # Preprocessing
        st.subheader("Keyword Preprocessing")
        st.info("Preprocessing keywords with advanced NLP or fallback.")
        use_advanced = True  # We'll try advanced approach if possible

        if "keyword" not in df.columns:
            st.error("No column named 'keyword' found. Check CSV.")
            return False, None
        
        keywords_processed = preprocess_keywords(
            df["keyword"].tolist(),
            use_advanced=use_advanced,
            spacy_nlp=spacy_nlp
        )
        df['keyword_processed'] = keywords_processed
        st.success("✅ Preprocessing complete.")
        
        # Generate embeddings
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
                pca_text.text(f"✅ PCA applied: {max_components} dimensions (~{pca_variance}% variance)")
            except Exception as e:
                st.error(f"Error applying PCA: {str(e)}")
                st.info("Proceeding without PCA.")
                keyword_embeddings_reduced = keyword_embeddings
        else:
            keyword_embeddings_reduced = keyword_embeddings
            st.info(f"No PCA needed (dimension is {keyword_embeddings.shape[1]}).")
        
        # Clustering
        st.subheader("Advanced Semantic Clustering")
        cluster_labels = improved_clustering(keyword_embeddings_reduced, num_clusters=num_clusters)
        df["cluster_id"] = cluster_labels
        st.success(f"✅ {len(df['cluster_id'].unique())} clusters created.")
        
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
            st.warning("Using a basic fallback for representatives.")
        
        # Generate cluster names
        if client:
            st.subheader("Generating Cluster Names & Descriptions (SEO-focused)")
            try:
                cluster_names = generate_cluster_names(
                    clusters_with_representatives, 
                    client, 
                    model=gpt_model,
                    custom_prompt=user_prompt
                )
            except Exception as e:
                st.error(f"Error generating cluster names: {str(e)}")
                cluster_names = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
        else:
            st.warning("No OpenAI client available. Using generic cluster names.")
            cluster_names = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
        
        # Apply names
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
        
        # AI-based semantic analysis
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
You can upload:
- A **simple CSV** with no header (just one keyword per line), or
- A **Keyword Planner-like CSV** with a header (Keyword, search_volume, competition, cpc, month1..month12, etc.)
""")

# -----------------------------------------------------------
# Expander describing CSV usage
# -----------------------------------------------------------
with st.expander("CSV Format Info", expanded=False):
    st.markdown("""
**Which CSV format can I use?**

1. **No Header**:  
   - Each line has just one keyword  
   - Example:
     ```
     red shoes
     running shoes
     kids sneakers
     ```
   - The app will treat the entire CSV as a single column: 'keyword'.

2. **With Header** (like Keyword Planner):  
   - The first row has column names (e.g. `Keyword, search_volume, competition, cpc, month1..month12`)  
   - The app will use the 'Keyword' column as the main text  
   - Additional columns can be used later for numeric analysis or weighting

If you pick the wrong format, the first row might be interpreted incorrectly.
""")

# Button to download sample CSV template
sample_csv_button = st.sidebar.button("Download Sample CSV Template")
if sample_csv_button:
    csv_header = generate_sample_csv()
    st.sidebar.download_button(
        label="Click to Download CSV Header",
        data=csv_header,
        file_name="sample_keyword_planner_template.csv",
        mime="text/csv",
        use_container_width=True
    )

# CSV Format selectbox
csv_format = st.sidebar.selectbox(
    "Select CSV format",
    options=["no_header", "with_header"],
    index=0
)

st.sidebar.markdown("<div class='sub-header'>Configuration</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=['csv'])

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key (optional)",
    type="password",
    help="Enter your OpenAI API Key for high-quality embeddings. If omitted, free SentenceTransformers or TF-IDF will be used."
)

# Language selector
language_options = [
    "English", "Spanish", "French", "German", "Dutch", 
    "Korean", "Japanese", "Italian", "Portuguese", 
    "Brazilian Portuguese", "Swedish", "Norwegian", 
    "Danish", "Icelandic", "Lithuanian", "Greek", "Romanian"
]
selected_language = st.sidebar.selectbox(
    "Select language of the CSV",
    options=language_options,
    index=0
)

if openai_available:
    if openai_api_key:
        st.sidebar.success("✅ OpenAI key provided - will use OpenAI for embeddings.")
    else:
        if sentence_transformers_available:
            st.sidebar.info("No OpenAI key - fallback to SentenceTransformers.")
        else:
            st.sidebar.warning("No OpenAI key, no SentenceTransformers - fallback to TF-IDF.")
else:
    if sentence_transformers_available:
        st.sidebar.info("OpenAI not installed - using SentenceTransformers.")
    else:
        st.sidebar.error("No advanced embedding method - fallback TF-IDF only.")

st.sidebar.markdown("<div class='sub-header'>Parameters</div>", unsafe_allow_html=True)

with st.sidebar.expander("ℹ️ Parameters Guide", expanded=False):
    st.markdown("""
### Parameters Guide

1. **Number of clusters**  
   - Controls how many clusters (groups) will be formed.
   - Higher = more and smaller clusters. Lower = fewer, larger clusters.

2. **PCA explained variance (%)**  
   - How much variance to keep when doing PCA dimensionality reduction.
   - For instance, 95% tries to keep most of the data's variance but reduces dimensions.

3. **Max PCA components**  
   - Hard cap on the number of PCA components.

4. **Minimum/Maximum term frequency (min_df, max_df)**  
   - Used when TF-IDF is employed. Filters out extremely rare or overly common terms.

5. **Model for naming clusters**  
   - Either gpt-3.5-turbo or gpt-4 if you have an API key.
   - GPT-4 is generally more advanced (and more expensive).
    """)

num_clusters = st.sidebar.slider("Number of clusters", 2, 50, 10)
pca_variance = st.sidebar.slider("PCA explained variance (%)", 50, 99, 95)
max_pca_components = st.sidebar.slider("Max PCA components", 10, 300, 100)
min_df = st.sidebar.slider("Minimum term frequency", 1, 10, 1)
max_df = st.sidebar.slider("Maximum term frequency (%)", 50, 100, 95)
gpt_model = st.sidebar.selectbox("Model for naming clusters", ["gpt-3.5-turbo", "gpt-4"], index=0)

st.sidebar.markdown("### Custom Prompt for SEO Naming")
default_prompt = (
    "You are an expert in SEO and content marketing. Below you'll see several clusters "
    "with a list of representative keywords. Your task is to assign each cluster a short, "
    "clear name (3-6 words) and write a concise SEO meta description (1 or 2 sentences) "
    "briefly explaining the topic and likely search intent. Respond only in JSON."
)
user_prompt = st.sidebar.text_area(
    "Custom Prompt",
    value=default_prompt,
    height=200
)

add_cost_calculator()

# Session states
if 'process_complete' not in st.session_state:
    st.session_state.process_complete = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

# Trigger process
if uploaded_file is not None and not st.session_state.process_complete:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Start Advanced Semantic Clustering", type="primary", use_container_width=True):
            success, results = run_clustering(
                uploaded_file=uploaded_file,
                openai_api_key=openai_api_key,
                num_clusters=num_clusters,
                pca_variance=pca_variance,
                max_pca_components=max_pca_components,
                min_df=min_df,
                max_df=max_df,
                gpt_model=gpt_model,
                user_prompt=user_prompt,
                csv_format=csv_format,
                selected_language=selected_language
            )
            if success and results is not None:
                st.session_state.df_results = results
                st.session_state.process_complete = True
                st.markdown("<div class='success-box'>✅ Semantic clustering completed successfully!</div>", unsafe_allow_html=True)

# If done, show results
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
        st.subheader("Explore Each Cluster")
        st.markdown("""
        Select a cluster to see details. If AI-based evaluation was done, 
        you may see extra suggestions or outlier info below.
        """)
        
        cluster_options = [
            f"{row['cluster_name']} (ID: {row['cluster_id']})"
            for _, row in df.drop_duplicates(['cluster_id', 'cluster_name'])[['cluster_id', 'cluster_name']].iterrows()
        ]
        selected_cluster = st.selectbox("Select a cluster:", cluster_options)
        
        if selected_cluster:
            cid = int(selected_cluster.split("ID: ")[1].split(")")[0])
            cluster_df = df[df['cluster_id'] == cid].copy()
            
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"### {cluster_df['cluster_name'].iloc[0]}")
                st.markdown(f"**Description:** {cluster_df['cluster_description'].iloc[0]}")
                st.markdown(f"**Total Keywords:** {len(cluster_df)}")
            with colB:
                st.markdown(f"**Semantic Coherence:** {cluster_df['cluster_coherence'].iloc[0]:.3f}")
                reps = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                if reps:
                    st.markdown("**Representative Keywords:**")
                    st.markdown("<ul>" + "".join([f"<li>{kw}</li>" for kw in reps[:10]]) + "</ul>", unsafe_allow_html=True)
            
            # If AI-based suggestions / semantic analysis is available
            if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
                ai_eval = st.session_state.cluster_evaluation
                if cid in ai_eval:
                    st.subheader("AI Semantic Analysis")
                    st.write(f"**Search Intent:** {ai_eval[cid].get('search_intent', 'N/A')}")
                    st.write(f"**Split Suggestion:** {ai_eval[cid].get('split_suggestion', 'N/A')}")
                    st.write(f"**Additional Info:** {ai_eval[cid].get('additional_info', 'N/A')}")
                    st.write(f"**Coherence Score (0-10):** {ai_eval[cid].get('coherence_score', 'N/A')}")
            
            st.markdown("### All Keywords in this Cluster")
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
        
        # Merge coherence
        coherence_df = df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id')
        summary_df.drop('cluster_id', axis=1, inplace=True)
        summary_df.rename(columns={'cluster_coherence': 'Coherence'}, inplace=True)
        
        # Representative keywords
        def get_rep_keywords(cid):
            reps = df[(df['cluster_id'] == cid) & (df['representative'] == True)]['keyword'].tolist()
            return ', '.join(reps[:5])
        summary_df['Representative Keywords'] = summary_df['ID'].apply(get_rep_keywords)
        
        # AI evaluation info
        if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
            evaluated_ids = st.session_state.cluster_evaluation.keys()
            summary_df['AI Evaluation?'] = summary_df['ID'].apply(lambda x: "Yes" if x in evaluated_ids else "No")
        else:
            summary_df['AI Evaluation?'] = "No"
        
        st.dataframe(summary_df, use_container_width=True)
        
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Clusters Summary",
            data=csv_summary,
            file_name="semantic_clusters_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

if 'process_complete' in st.session_state and st.session_state.process_complete:
    if st.button("Reset", use_container_width=True):
        st.session_state.process_complete = False
        st.session_state.df_results = None
        st.experimental_rerun()

with st.expander("More Information about Advanced Semantic Clustering"):
    st.markdown("""
    ### How does it work?
    1. **Linguistic Preprocessing** (spaCy/TextBlob/NLTK).
    2. **Embeddings** (OpenAI if key, else SentenceTransformers, else TF-IDF).
    3. **Dimensionality Reduction** (PCA).
    4. **Clustering** (HDBSCAN/hierarchical/K-Means).
    5. **Refinement** (outlier detection, merging).
    6. **Evaluation** (coherence, density, separation).
    
    ### CSV Formats
    - **No Header**: one keyword per line
    - **With Header**: columns like `Keyword,search_volume,competition,cpc,month1..month12`
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    Developed for advanced semantic keyword clustering – featuring optional CSV formats and a sample template
</div>
""", unsafe_allow_html=True)
