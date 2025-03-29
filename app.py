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
import plotly.graph_objects as go
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
    
    # Sample data for download
    data = "running shoes,5400,0.75,1.25,450,460,470,480,490,500,510,520,530,540,550,560\n"
    data += "nike shoes,8900,0.82,1.78,700,720,740,760,780,800,820,840,860,880,900,920\n"
    data += "adidas sneakers,3200,0.65,1.12,260,270,280,290,300,310,320,330,340,350,360,370\n"
    data += "hiking boots,2800,0.45,0.89,230,240,250,260,270,280,290,300,310,320,330,340\n"
    data += "women's running shoes,4100,0.68,1.35,340,350,360,370,380,390,400,410,420,430,440,450\n"
    
    return ",".join(header) + "\n" + data

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
            client = OpenAI(api_key=openai_api_key)
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
    """
    Generate SEO-friendly names and descriptions for clusters using OpenAI.
    Fixed to better handle JSON parsing and error recovery.
    """
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
            "briefly explaining the topic and likely search intent."
        )

    # Simplified prompt structure to reduce JSON parsing errors
    naming_prompt = custom_prompt.strip() + "\n\n"
    naming_prompt += (
        "FOR EACH CLUSTER, you must provide:\n"
        "1. A clear, concise name (3-6 words)\n"
        "2. A brief description (1-2 sentences)\n\n"
        "FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS - this is crucial:\n\n"
        "```json\n"
        "{\n"
        '  "clusters": [\n'
        "    {\n"
        '      "cluster_id": 1,\n'
        '      "cluster_name": "Example Cluster Name",\n'
        '      "cluster_description": "Example description of what this cluster represents."\n'
        "    },\n"
        "    {\n"
        '      "cluster_id": 2,\n'
        '      "cluster_name": "Another Cluster Name",\n'
        '      "cluster_description": "Another description example."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "DO NOT INCLUDE ANY OTHER TEXT OR EXPLANATION besides this JSON object.\n\n"
        "Here are the clusters:\n"
    )
    
    for cluster_id, keywords in clusters_with_representatives.items():
        sample_kws = keywords[:15]
        naming_prompt += f"- Cluster {cluster_id}: {', '.join(sample_kws)}\n"
    
    naming_prompt += "\nRemember to follow the exact JSON format shown above and include ALL clusters in your response."
    
    num_retries = 3
    for attempt in range(num_retries):
        try:
            progress_text.text(f"Generating cluster names (attempt {attempt+1}/{num_retries})...")
            
            # First try with JSON response format - works with newer models
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": naming_prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}  # This is the key parameter
                )
                
                content = response.choices[0].message.content.strip()
                
                # Debug the response directly in the app (can be removed in production)
                if attempt == 0:  # Only show debug on first attempt
                    st.write("Debug - Raw API Response:")
                    st.code(content[:500] + ("..." if len(content) > 500 else ""), language="json")
                
                json_data = json.loads(content)
                
                if "clusters" in json_data and isinstance(json_data["clusters"], list):
                    for item in json_data["clusters"]:
                        c_id = item.get("cluster_id")
                        if c_id is not None:
                            try:
                                c_id = int(c_id)  # Ensure it's an integer
                                c_name = item.get("cluster_name", f"Cluster {c_id}")
                                c_desc = item.get("cluster_description", "No description provided")
                                results[c_id] = (c_name, c_desc)
                            except (ValueError, TypeError):
                                st.warning(f"Invalid cluster_id format: {c_id}")
                    
                    # If we got results, break the retry loop
                    if results:
                        break
                
            except Exception as json_format_error:
                # If response_format parameter fails, fall back to standard completion
                st.warning(f"JSON response format failed: {json_format_error}. Trying standard completion...")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": naming_prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from markdown code blocks if present
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                json_matches = re.findall(json_pattern, content)
                
                if json_matches:
                    content = json_matches[0]  # Take the first JSON code block
                
                # Try to parse JSON
                try:
                    json_data = json.loads(content)
                    
                    if "clusters" in json_data and isinstance(json_data["clusters"], list):
                        for item in json_data["clusters"]:
                            c_id = item.get("cluster_id")
                            if c_id is not None:
                                try:
                                    c_id = int(c_id)
                                    c_name = item.get("cluster_name", f"Cluster {c_id}")
                                    c_desc = item.get("cluster_description", "No description provided")
                                    results[c_id] = (c_name, c_desc)
                                except (ValueError, TypeError):
                                    pass
                        
                        if results:
                            break
                except json.JSONDecodeError:
                    # If JSON parsing fails, try regex extraction
                    try:
                        cluster_pattern = r'cluster_id["\s:]+(\d+)["\s,}]+\s*cluster_name["\s:]+([^"]+)["\s,}]+\s*cluster_description["\s:]+([^"]+)'
                        matches = re.findall(cluster_pattern, content)
                        
                        for match in matches:
                            try:
                                c_id = int(match[0])
                                c_name = match[1].strip()
                                c_desc = match[2].strip()
                                results[c_id] = (c_name, c_desc)
                            except (ValueError, IndexError):
                                pass
                        
                        if results:
                            break
                    except Exception as regex_err:
                        st.warning(f"Regex extraction failed: {regex_err}")
        
        except Exception as e:
            st.error(f"Error in API call (attempt {attempt+1}): {str(e)}")
            time.sleep(1)  # Wait briefly before retrying
    
    # If we still have no results, use generic names
    if not results:
        st.warning("Could not generate cluster names via API. Using generic names.")
        for c_id in clusters_with_representatives.keys():
            results[c_id] = (f"Cluster {c_id}", f"This is a group of related keywords (cluster {c_id}).")

    progress_bar.progress(1.0)
    progress_text.text("✅ Cluster naming completed.")
    return results

################################################################
#          SEARCH INTENT CLASSIFICATION
################################################################

def classify_search_intent(keywords, search_intent_description):
    """
    Classifies search intent into one of the four main categories based on the AI's
    search intent description and a sample of keywords.
    
    Categories:
    - Informational
    - Navigational
    - Transactional
    - Commercial
    
    Returns both the primary intent and confidence scores.
    """
    # Create patterns to match keywords and descriptions indicative of each intent
    informational_patterns = [
        r'\bhow\b', r'\bwhat\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b',
        r'\bguide\b', r'\btutorial\b', r'\blearn\b', r'\bexplain\b', r'\bmeaning\b',
        r'information', r'knowledge', r'understanding', r'definition', r'examples'
    ]
    
    navigational_patterns = [
        r'\blogin\b', r'\bsign in\b', r'\bwebsite\b', r'\bofficial\b', r'\bportal\b',
        r'\bhomepage\b', r'\bdownload\b', r'\baccount\b', r'\blog in\b', r'\bsite\b',
        r'contact', r'address', r'location', r'directions', r'map'
    ]
    
    transactional_patterns = [
        r'\bbuy\b', r'\bpurchase\b', r'\bshop\b', r'\bsale\b', r'\bdiscount\b',
        r'\bprice\b', r'\bcheap\b', r'\bfree\b', r'\bdeals\b', r'\borderb',
        r'coupon', r'shipping', r'payment', r'checkout', r'subscribe'
    ]
    
    commercial_patterns = [
        r'\bbest\b', r'\btop\b', r'\breview\b', r'\bcomparison\b', r'\bcompare\b',
        r'\brating\b', r'\branking\b', r'\bversus\b', r'\bvs\b', r'\balternative\b',
        r'recommended', r'suggestion', r'opinion', r'evaluation', r'pros and cons'
    ]
    
    # Combine keywords and description for analysis
    text_to_analyze = " ".join(keywords[:10]).lower() + " " + search_intent_description.lower()
    
    # Check matches for each intent
    info_matches = sum(1 for pattern in informational_patterns if re.search(pattern, text_to_analyze))
    nav_matches = sum(1 for pattern in navigational_patterns if re.search(pattern, text_to_analyze))
    trans_matches = sum(1 for pattern in transactional_patterns if re.search(pattern, text_to_analyze))
    comm_matches = sum(1 for pattern in commercial_patterns if re.search(pattern, text_to_analyze))
    
    # Calculate total matches and convert to percentages
    total_matches = max(1, info_matches + nav_matches + trans_matches + comm_matches)
    info_score = (info_matches / total_matches) * 100
    nav_score = (nav_matches / total_matches) * 100
    trans_score = (trans_matches / total_matches) * 100
    comm_score = (comm_matches / total_matches) * 100
    
    # Find primary intent
    scores = {
        "Informational": info_score,
        "Navigational": nav_score,
        "Transactional": trans_score,
        "Commercial": comm_score
    }
    
    primary_intent = max(scores, key=scores.get)
    
    # Also check the description directly for mention of intent
    intent_mentions = {
        "Informational": any(re.search(r'\binformational\b|\binformation\b|\blearn\b', search_intent_description, re.IGNORECASE)),
        "Navigational": any(re.search(r'\bnavigational\b|\bnavigate\b|\bfind\b', search_intent_description, re.IGNORECASE)),
        "Transactional": any(re.search(r'\btransactional\b|\bbuy\b|\bpurchase\b', search_intent_description, re.IGNORECASE)),
        "Commercial": any(re.search(r'\bcommercial\b|\bresearch\b|\bcompare\b', search_intent_description, re.IGNORECASE))
    }
    
    # Boost score if explicitly mentioned
    for intent, mentioned in intent_mentions.items():
        if mentioned:
            scores[intent] += 20  # Give significant boost for explicit mention
    
    # Recalculate primary intent after boosts
    primary_intent = max(scores, key=scores.get)
    
    return {
        "primary_intent": primary_intent,
        "scores": scores
    }

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
      2) Suggestion of internal splitting with specific subclusters
      3) Additional SEO-focused insights
      4) Coherence score
    """
    results = {}
    if not clusters_with_representatives:
        return results

    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Performing semantic analysis on clusters...")

    # Enhanced prompt for more SEO-focused analysis
    analysis_prompt = (
        "You are an expert in SEO and clustering analysis. Below are several clusters with representative keywords. "
        "For each cluster, provide a detailed analysis including:\n"
        "1) The main search intent - describe why users would search for these terms and what they're looking for.\n"
        "2) If you think it should be split further, suggest 2-3 specific subclusters with names and a few keywords for each.\n"
        "3) SEO insights and opportunities - discuss keyword difficulty, search volume potential, content ideas, etc.\n"
        "4) Assign a coherence score from 0-10 where 10 means perfectly coherent semantically related keywords.\n\n"
        "FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS - this is crucial:\n\n"
        "```json\n"
        "{\n"
        '  "clusters": [\n'
        "    {\n"
        '      "cluster_id": 1,\n'
        '      "search_intent": "Detailed description of user intent",\n'
        '      "split_suggestion": "Yes/No and if yes, provide specific subclusters with names and sample keywords",\n'
        '      "additional_info": "SEO-focused analysis with content suggestions and opportunities",\n'
        '      "coherence_score": 8,\n'
        '      "subclusters": [\n'
        '        {"name": "Subcluster 1 name", "keywords": ["keyword1", "keyword2", "keyword3"]},\n'
        '        {"name": "Subcluster 2 name", "keywords": ["keyword4", "keyword5", "keyword6"]}\n'
        '      ]\n'
        "    },\n"
        "    {\n"
        '      "cluster_id": 2,\n'
        '      "search_intent": "Another detailed intent description",\n'
        '      "split_suggestion": "No",\n'
        '      "additional_info": "SEO insights for this cluster",\n'
        '      "coherence_score": 6,\n'
        '      "subclusters": []\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "DO NOT INCLUDE ANY OTHER TEXT OR EXPLANATION besides this JSON object.\n\n"
        "For each cluster where splitting is not recommended, use an empty array [] for subclusters.\n\n"
        "Here are the clusters:\n"
    )

    for cluster_id, keywords in clusters_with_representatives.items():
        sample_kws = keywords[:15]
        analysis_prompt += f"- Cluster {cluster_id}: {', '.join(sample_kws)}\n"

    analysis_prompt += "\nRemember to follow the exact JSON format shown above and include ALL clusters in your response."

    num_retries = 3
    for attempt in range(num_retries):
        try:
            progress_text.text(f"Analyzing clusters (attempt {attempt+1}/{num_retries})...")
            
            # First try with JSON response format
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                
                # Debug the response (can be removed in production)
                if attempt == 0:  # Only show debug on first attempt
                    st.write("Debug - Raw API Response for Semantic Analysis:")
                    st.code(content[:500] + ("..." if len(content) > 500 else ""), language="json")
                
                json_data = json.loads(content)
                
                if "clusters" in json_data and isinstance(json_data["clusters"], list):
                    for item in json_data["clusters"]:
                        c_id = item.get("cluster_id")
                        if c_id is not None:
                            try:
                                c_id = int(c_id)
                                search_intent = item.get("search_intent", "")
                                split_suggestion = item.get("split_suggestion", "")
                                additional_info = item.get("additional_info", "")
                                coherence_score = item.get("coherence_score", 5)
                                subclusters = item.get("subclusters", [])
                                
                                # Classify the search intent
                                intent_classification = classify_search_intent(
                                    clusters_with_representatives.get(c_id, []),
                                    search_intent
                                )
                                
                                results[c_id] = {
                                    "search_intent": search_intent,
                                    "split_suggestion": split_suggestion,
                                    "additional_info": additional_info,
                                    "coherence_score": coherence_score,
                                    "subclusters": subclusters,
                                    "intent_classification": intent_classification
                                }
                            except (ValueError, TypeError):
                                st.warning(f"Invalid cluster_id format in analysis: {c_id}")
                    
                    # If we got results, break the retry loop
                    if results:
                        break
            
            except Exception as json_format_error:
                # If response_format parameter fails, try standard completion
                st.warning(f"JSON response format failed: {json_format_error}. Trying standard completion...")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from markdown code blocks if present
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                json_matches = re.findall(json_pattern, content)
                
                if json_matches:
                    content = json_matches[0]  # Take the first JSON code block
                
                # Try to parse JSON
                try:
                    json_data = json.loads(content)
                    
                    if "clusters" in json_data and isinstance(json_data["clusters"], list):
                        for item in json_data["clusters"]:
                            c_id = item.get("cluster_id")
                            if c_id is not None:
                                try:
                                    c_id = int(c_id)
                                    search_intent = item.get("search_intent", "")
                                    split_suggestion = item.get("split_suggestion", "")
                                    additional_info = item.get("additional_info", "")
                                    coherence_score = item.get("coherence_score", 5)
                                    subclusters = item.get("subclusters", [])
                                    
                                    # Classify the search intent
                                    intent_classification = classify_search_intent(
                                        clusters_with_representatives.get(c_id, []),
                                        search_intent
                                    )
                                    
                                    results[c_id] = {
                                        "search_intent": search_intent,
                                        "split_suggestion": split_suggestion,
                                        "additional_info": additional_info,
                                        "coherence_score": coherence_score,
                                        "subclusters": subclusters,
                                        "intent_classification": intent_classification
                                    }
                                except (ValueError, TypeError):
                                    pass
                        
                        if results:
                            break
                except json.JSONDecodeError:
                    # Last resort: Try regex for basic fields
                    try:
                        cluster_pattern = r'cluster_id["\s:]+(\d+)["\s,}]+'
                        search_pattern = r'search_intent["\s:]+([^"]+)["\s,}]+'
                        split_pattern = r'split_suggestion["\s:]+([^"]+)["\s,}]+'
                        info_pattern = r'additional_info["\s:]+([^"]+)["\s,}]+'
                        score_pattern = r'coherence_score["\s:]+(\d+)'
                        
                        cluster_ids = re.findall(cluster_pattern, content)
                        search_intents = re.findall(search_pattern, content)
                        split_sugs = re.findall(split_pattern, content)
                        add_infos = re.findall(info_pattern, content)
                        scores = re.findall(score_pattern, content)
                        
                        # If we found some cluster_ids
                        for i, c_id_str in enumerate(cluster_ids):
                            try:
                                c_id = int(c_id_str)
                                search_intent = search_intents[i] if i < len(search_intents) else ""
                                split_sug = split_sugs[i] if i < len(split_sugs) else ""
                                add_info = add_infos[i] if i < len(add_infos) else ""
                                score = int(scores[i]) if i < len(scores) else 5
                                
                                # Classify the search intent
                                intent_classification = classify_search_intent(
                                    clusters_with_representatives.get(c_id, []),
                                    search_intent
                                )
                                
                                results[c_id] = {
                                    "search_intent": search_intent,
                                    "split_suggestion": split_sug,
                                    "additional_info": add_info,
                                    "coherence_score": score,
                                    "subclusters": [],  # Default empty as regex extraction of nested objects is complex
                                    "intent_classification": intent_classification
                                }
                            except (ValueError, IndexError):
                                pass
                        
                        if results:
                            break
                    except Exception as regex_err:
                        st.warning(f"Regex extraction failed: {regex_err}")
        
        except Exception as e:
            st.error(f"Error in API call (attempt {attempt+1}): {str(e)}")
            time.sleep(1)
    
    # If we still have no results after all retries, create default results
    if not results:
        st.warning("Could not generate semantic analysis via API. Using default values.")
        for c_id in clusters_with_representatives.keys():
            intent_classification = {
                "primary_intent": "Unknown",
                "scores": {
                    "Informational": 25,
                    "Navigational": 25,
                    "Transactional": 25,
                    "Commercial": 25
                }
            }
            
            results[c_id] = {
                "search_intent": "No search intent data available",
                "split_suggestion": "No split suggestion available",
                "additional_info": "No SEO information available",
                "coherence_score": 5,  # Neutral middle score
                "subclusters": [],
                "intent_classification": intent_classification
            }

    progress_bar.progress(1.0)
    progress_text.text("✅ Semantic analysis completed.")
    return results

################################################################
#          EVALUATION FUNCTIONS
################################################################

def evaluate_cluster_quality(df, embeddings, cluster_column='cluster_id'):
    """
    Improved approach to assign a 'cluster_coherence' score based on distances within clusters.
    """
    st.subheader("Cluster Quality Evaluation")
    
    try:
        # Create a DataFrame to store coherence scores
        df['cluster_coherence'] = 1.0  # Default value
        
        # Get unique clusters
        unique_clusters = df[cluster_column].unique()
        
        with st.spinner("Calculating cluster coherence scores..."):
            progress_bar = st.progress(0)
            
            for i, cluster_id in enumerate(unique_clusters):
                # Get indices for this cluster
                cluster_indices = df[df[cluster_column] == cluster_id].index.tolist()
                
                if len(cluster_indices) > 1:  # Need at least 2 points for coherence
                    # Get embeddings for this cluster
                    cluster_embeddings = embeddings[cluster_indices]
                    
                    # Calculate coherence (using cosine similarity)
                    coherence = calculate_cluster_coherence(cluster_embeddings)
                    
                    # Assign to all rows in this cluster
                    df.loc[cluster_indices, 'cluster_coherence'] = coherence
                
                progress_bar.progress((i + 1) / len(unique_clusters))
            
            progress_bar.progress(1.0)
        
        st.success(f"✅ Coherence scores calculated for {len(unique_clusters)} clusters.")
    except Exception as e:
        st.error(f"Error calculating coherence: {str(e)}")
        st.warning("Using default coherence value of 1.0")
        df['cluster_coherence'] = 1.0
    
    return df

def calculate_cluster_coherence(cluster_embeddings):
    """
    Calculate coherence score based on cosine similarity within clusters.
    Higher score = better coherence (more similar documents within cluster).
    """
    try:
        # Calculate mean embedding (centroid)
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        
        # Calculate cosine similarity between each point and the centroid
        similarities = []
        for embedding in cluster_embeddings:
            # Normalize the embedding
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding = embedding / emb_norm
            
            # Calculate similarity
            similarity = np.dot(embedding, centroid)
            similarities.append(similarity)
        
        # Return average similarity (coherence score)
        coherence = np.mean(similarities)
        
        # Scale to a nice 0-1 range (could adjust this scaling if needed)
        coherence = max(0.0, min(1.0, coherence))
        
        return coherence
    except Exception as e:
        # If anything goes wrong, return default value
        return 1.0

def evaluate_and_refine_clusters(df, client, model="gpt-3.5-turbo"):
    """
    Performs AI-powered analysis of clusters using OpenAI's API.
    Returns a dictionary of analysis results by cluster ID.
    """
    st.subheader("AI-Powered Cluster Quality Evaluation")

    if not client:
        st.info("No OpenAI client available. Skipping AI-based cluster analysis.")
        return {}

    try:
        # Build a dict of cluster -> representative keywords
        clusters_with_representatives = {}
        
        for c_id in df['cluster_id'].unique():
            # First try to get marked representative keywords
            reps = df[(df['cluster_id'] == c_id) & (df['representative'] == True)]['keyword'].tolist()
            
            # If none found, just take the first 20 keywords from this cluster
            if not reps:
                cluster_kws = df[df['cluster_id'] == c_id]['keyword'].tolist()
                reps = cluster_kws[:min(20, len(cluster_kws))]
            
            clusters_with_representatives[c_id] = reps

        # Call GPT-based analysis with retry logic
        semantic_analysis = generate_semantic_analysis(
            clusters_with_representatives=clusters_with_representatives,
            client=client,
            model=model
        )

        # Check if we got results
        if semantic_analysis:
            st.success(f"✅ AI analysis completed for {len(semantic_analysis)} clusters.")
        else:
            st.warning("No AI analysis results were generated.")

        return semantic_analysis
    
    except Exception as e:
        st.error(f"Error in cluster evaluation: {str(e)}")
        return {}

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
            client = OpenAI(api_key=openai_api_key)  # Explicitly set API key
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
                # Generate cluster names with improved error handling
                cluster_names = generate_cluster_names(
                    clusters_with_representatives, 
                    client, 
                    model=gpt_model,
                    custom_prompt=user_prompt
                )
                if not cluster_names:
                    st.warning("Cluster naming function returned empty results. Using fallback names.")
                    cluster_names = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
            except Exception as e:
                st.error(f"Error during cluster naming: {str(e)}")
                st.info("Using fallback generic cluster names.")
                cluster_names = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
        else:
            st.warning("No OpenAI client available. Using generic cluster names.")
            cluster_names = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
        
        # Apply names with error handling
        df['cluster_name'] = ''
        df['cluster_description'] = ''
        df['representative'] = False
        
        try:
            for cnum, (name, desc) in cluster_names.items():
                # Safety check - ensure cluster exists in dataframe
                if cnum in df['cluster_id'].values:
                    df.loc[df['cluster_id'] == cnum, 'cluster_name'] = name
                    df.loc[df['cluster_id'] == cnum, 'cluster_description'] = desc
                    
                    # Mark representative keywords
                    for kw in clusters_with_representatives.get(cnum, []):
                        match_idx = df[(df['cluster_id'] == cnum) & (df['keyword'] == kw)].index
                        if not match_idx.empty:
                            df.loc[match_idx, 'representative'] = True
        except Exception as e:
            st.error(f"Error applying cluster names: {str(e)}")
            st.info("Using fallback approach for cluster names")
            
            # Fallback approach - simple sequential naming
            for cnum in df['cluster_id'].unique():
                df.loc[df['cluster_id'] == cnum, 'cluster_name'] = f"Cluster {cnum}"
                df.loc[df['cluster_id'] == cnum, 'cluster_description'] = f"Group of related keywords (cluster {cnum})"
        
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
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Advanced semantic keyword clustering tool using NLP and OpenAI.'
    }
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
    .intent-box {
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .intent-info {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .intent-nav {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .intent-trans {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .intent-comm {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }
    .subcluster-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
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
        label="Click to Download CSV Template",
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
    "briefly explaining the topic and likely search intent."
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
        st.markdown("""
        This graph shows how semantically related the keywords within each cluster are. 
        Higher coherence scores (closer to 1.0) indicate clusters with more closely related keywords. 
        Clusters with lower coherence might contain more diverse topics and could be candidates for further splitting.
        """)
        
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
        
        # Visualization based on AI Coherence Scores
        if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
            eval_data = st.session_state.cluster_evaluation
            ai_coherence_data = []
            
            for c_id, data in eval_data.items():
                coherence_score = data.get('coherence_score', 5)
                cluster_name = df[df['cluster_id'] == c_id]['cluster_name'].iloc[0] if not df[df['cluster_id'] == c_id].empty else f"Cluster {c_id}"
                count = len(df[df['cluster_id'] == c_id])
                
                primary_intent = data.get('intent_classification', {}).get('primary_intent', 'Unknown')
                
                ai_coherence_data.append({
                    'cluster_id': c_id,
                    'cluster_name': cluster_name,
                    'coherence_score': coherence_score,
                    'count': count,
                    'primary_intent': primary_intent
                })
            
            if ai_coherence_data:
                ai_df = pd.DataFrame(ai_coherence_data)
                ai_df['label'] = ai_df.apply(lambda x: f"{x['cluster_name']} (ID: {x['cluster_id']})", axis=1)
                
                # Color map for intent types
                intent_colors = {
                    'Informational': '#2196f3',
                    'Navigational': '#4caf50',
                    'Transactional': '#ff9800',
                    'Commercial': '#9c27b0',
                    'Unknown': '#9e9e9e'
                }
                
                st.subheader("AI-Evaluated Cluster Coherence")
                
                fig3 = px.scatter(
                    ai_df,
                    x='coherence_score',
                    y='count',
                    color='primary_intent',
                    size='count',
                    hover_name='label',
                    labels={
                        'coherence_score': 'AI Coherence Score (0-10)',
                        'count': 'Number of Keywords',
                        'primary_intent': 'Search Intent'
                    },
                    title='Clusters by Coherence, Size, and Search Intent',
                    color_discrete_map=intent_colors
                )
                
                fig3.update_layout(
                    xaxis=dict(range=[0, 10]),
                    height=600
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Add explanation of the visualization
                st.markdown("""
                **About this chart:**
                - **X-Axis**: AI-evaluated semantic coherence score (0-10)
                - **Y-Axis**: Number of keywords in the cluster 
                - **Bubble Size**: Proportional to number of keywords
                - **Color**: Represents the primary search intent of the cluster
                
                The most valuable clusters are typically those with high coherence scores (right side) and substantial keyword volume (upper area).
                Clusters with low coherence might benefit from being split into more focused sub-clusters.
                """)
    
    with st.expander("Explore Clusters", expanded=True):
        st.subheader("Explore Each Cluster")
        st.markdown("""
        Select a cluster to see details, search intent analysis, and potential sub-cluster suggestions.
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
                
                # Show total search volume if available
                if 'search_volume' in cluster_df.columns:
                    total_search_volume = cluster_df['search_volume'].sum()
                    st.markdown(f"**Total Search Volume:** {total_search_volume:,}")
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
                    st.markdown("---")
                    st.subheader("AI Semantic Analysis")
                    
                    # Search intent classification
                    intent_classification = ai_eval[cid].get('intent_classification', {})
                    primary_intent = intent_classification.get('primary_intent', 'Unknown')
                    scores = intent_classification.get('scores', {})
                    
                    # Format CSS class based on intent
                    intent_class = ""
                    if primary_intent == "Informational":
                        intent_class = "intent-info"
                    elif primary_intent == "Navigational":
                        intent_class = "intent-nav"
                    elif primary_intent == "Transactional":
                        intent_class = "intent-trans"
                    elif primary_intent == "Commercial":
                        intent_class = "intent-comm"
                    
                    # Display search intent with formatting
                    st.markdown(f"""
                    <div class="intent-box {intent_class}">
                        <strong>Primary Search Intent:</strong> {primary_intent}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show all scores as a visualization
                    if scores:
                        intents = list(scores.keys())
                        values = list(scores.values())
                        
                        fig_intent = px.bar(
                            x=intents, 
                            y=values,
                            labels={'x': 'Intent Type', 'y': 'Confidence Score (%)'},
                            title='Search Intent Distribution',
                            color=intents,
                            color_discrete_map={
                                'Informational': '#2196f3',
                                'Navigational': '#4caf50',
                                'Transactional': '#ff9800',
                                'Commercial': '#9c27b0'
                            }
                        )
                        fig_intent.update_layout(yaxis_range=[0, 100])
                        st.plotly_chart(fig_intent)
                    
                    # Search intent description
                    st.write(f"**Search Intent Details:** {ai_eval[cid].get('search_intent', 'N/A')}")
                    
                    # Split suggestion
                    split_suggestion = ai_eval[cid].get('split_suggestion', '')
                    if split_suggestion.lower().startswith('yes'):
                        st.markdown("""
                        <div style="background-color: #fff3cd; padding: 10px; border-left: 5px solid #ffc107; margin-bottom: 10px;">
                        <strong>Split Recommendation:</strong> This cluster could be divided into more focused sub-clusters.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show suggested subclusters
                        subclusters = ai_eval[cid].get('subclusters', [])
                        if subclusters:
                            st.markdown("### Suggested Sub-clusters")
                            
                            for i, subcluster in enumerate(subclusters):
                                subcluster_name = subcluster.get('name', f"Subcluster {i+1}")
                                subcluster_keywords = subcluster.get('keywords', [])
                                
                                st.markdown(f"""
                                <div class="subcluster-box">
                                    <h4>{subcluster_name}</h4>
                                    <p><strong>Sample Keywords:</strong> {', '.join(subcluster_keywords)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #d1e7dd; padding: 10px; border-left: 5px solid #198754; margin-bottom: 10px;">
                        <strong>Split Recommendation:</strong> This cluster appears to be coherent and focused.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show full split suggestion text
                    with st.expander("View full split analysis", expanded=False):
                        st.write(split_suggestion)
                    
                    # SEO Insights
                    st.markdown("### SEO Insights")
                    st.write(ai_eval[cid].get('additional_info', 'No additional information available'))
                    
                    # Coherence Score
                    coherence_score = ai_eval[cid].get('coherence_score', 'N/A')
                    st.metric(label="AI Coherence Score (0-10)", value=coherence_score)
            
            st.markdown("### All Keywords in this Cluster")
            if 'search_volume' in cluster_df.columns:
                # If search volume exists, show it
                st.dataframe(cluster_df[['keyword', 'search_volume']].sort_values(by='search_volume', ascending=False), use_container_width=True)
            else:
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
        
        # Add search volume if it exists
        if 'search_volume' in df.columns:
            volume_df = df.groupby('cluster_id')['search_volume'].sum().reset_index()
            summary_df = summary_df.merge(volume_df, left_on='ID', right_on='cluster_id')
            summary_df.drop('cluster_id', axis=1, inplace=True)
            summary_df.rename(columns={'search_volume': 'Total Search Volume'}, inplace=True)
        
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
            
            # Add primary search intent
            def get_search_intent(cid):
                if cid in st.session_state.cluster_evaluation:
                    intent_data = st.session_state.cluster_evaluation[cid].get('intent_classification', {})
                    return intent_data.get('primary_intent', 'Unknown')
                return 'Unknown'
            
            summary_df['Primary Intent'] = summary_df['ID'].apply(get_search_intent)
        else:
            summary_df['AI Evaluation?'] = "No"
            summary_df['Primary Intent'] = "Unknown"
        
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
    4. **Clustering** (K-Means).
    5. **Refinement** (outlier detection, merging).
    6. **Evaluation** (coherence, density, separation).
    
    ### CSV Formats
    - **No Header**: one keyword per line
    - **With Header**: columns like `Keyword,search_volume,competition,cpc,month1..month12`
    
    ### Search Intent Categories
    - **Informational search intent:** Users looking for information or answers ("how to", "what is", etc.)
    - **Navigational search intent:** Users trying to locate a specific website or page (brand names, specific sites)
    - **Transactional search intent:** Users ready to make a purchase or engage in activities leading to transactions ("buy", "discount", etc.)
    - **Commercial search intent:** Users researching options before making a purchase ("best", "reviews", "vs", etc.)
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    Developed for advanced semantic keyword clustering – featuring optional CSV formats and a sample template
</div>
""", unsafe_allow_html=True)
