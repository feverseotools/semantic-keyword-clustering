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
from collections import Counter

# Helper function for safe numeric formatting
def safe_numeric_formatting(value, format_str="{:,.0f}"):
    """
    Safely format a numeric value, returning a fallback string if the value 
    cannot be formatted as specified.
    """
    try:
        if pd.isna(value):
            return "N/A"
        return format_str.format(value)
    except:
        return "N/A"

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
#          SEARCH INTENT CLASSIFICATION PATTERNS
################################################################

# Search intent classification patterns
# These are comprehensive patterns based on SEO industry standards
SEARCH_INTENT_PATTERNS = {
    "Informational": {
        "prefixes": [
            "how", "what", "why", "when", "where", "who", "which",
            "can", "does", "is", "are", "will", "should", "do", "did",
            "guide", "tutorial", "learn", "understand", "explain"
        ],
        "suffixes": ["definition", "meaning", "examples", "ideas", "guide", "tutorial"],
        "exact_matches": [
            "guide to", "how-to", "tutorial", "resources", "information", "knowledge",
            "examples of", "definition of", "explanation", "steps to", "learn about",
            "facts about", "history of", "benefits of", "causes of", "types of"
        ],
        "keyword_patterns": [
            r'\bhow\s+to\b', r'\bwhat\s+is\b', r'\bwhy\s+is\b', r'\bwhen\s+to\b', 
            r'\bwhere\s+to\b', r'\bwho\s+is\b', r'\bwhich\b.*\bbest\b',
            r'\bdefinition\b', r'\bmeaning\b', r'\bexamples?\b', r'\btips\b',
            r'\btutorials?\b', r'\bguide\b', r'\blearn\b', r'\bsteps?\b',
            r'\bversus\b', r'\bvs\b', r'\bcompared?\b', r'\bdifference\b'
        ],
        "weight": 1.0
    },
    
    "Navigational": {
        "prefixes": ["go to", "visit", "website", "homepage", "home page", "sign in", "login"],
        "suffixes": ["login", "website", "homepage", "official", "online"],
        "exact_matches": [
            "login", "sign in", "register", "create account", "download", "official website",
            "official site", "homepage", "contact", "support", "customer service", "app"
        ],
        "keyword_patterns": [
            r'\blogin\b', r'\bsign\s+in\b', r'\bwebsite\b', r'\bhomepage\b', r'\bportal\b',
            r'\baccount\b', r'\bofficial\b', r'\bdashboard\b', r'\bdownload\b.*\bfrom\b',
            r'\bcontact\b', r'\baddress\b', r'\blocation\b', r'\bdirections?\b',
            r'\bmap\b', r'\btrack\b.*\border\b', r'\bmy\s+\w+\s+account\b'
        ],
        "brand_indicators": True,  # Presence of brand names indicates navigational intent
        "weight": 1.2  # Navigational intent is often more clear-cut
    },
    "Transactional": {
        "prefixes": ["buy", "purchase", "order", "shop", "get"],
        "suffixes": [
            "for sale", "discount", "deal", "coupon", "price", "cost", "cheap", "online", 
            "free", "download", "subscription", "trial"
        ],
        "exact_matches": [
            "buy", "purchase", "order", "shop", "subscribe", "download", "free trial",
            "coupon code", "discount", "deal", "sale", "cheap", "best price", "near me",
            "shipping", "delivery", "in stock", "available", "pay", "checkout"
        ],
        "keyword_patterns": [
            r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bshop\b', r'\bstores?\b',
            r'\bprice\b', r'\bcost\b', r'\bcheap\b', r'\bdiscount\b', r'\bdeal\b',
            r'\bsale\b', r'\bcoupon\b', r'\bpromo\b', r'\bfree\s+shipping\b',
            r'\bnear\s+me\b', r'\bshipping\b', r'\bdelivery\b', r'\bcheck\s*out\b',
            r'\bin\s+stock\b', r'\bavailable\b', r'\bsubscribe\b', r'\bdownload\b',
            r'\binstall\b', r'\bfor\s+sale\b', r'\bhire\b', r'\brent\b'
        ],
        "weight": 1.5  # Strong transactional signals are highly valuable
    },
    
    "Commercial": {
        "prefixes": ["best", "top", "review", "compare", "vs", "versus"],
        "suffixes": [
            "review", "reviews", "comparison", "vs", "versus", "alternative", "alternatives", 
            "recommendation", "recommendations", "comparison", "guide"
        ],
        "exact_matches": [
            "best", "top", "vs", "versus", "comparison", "compare", "review", "reviews", 
            "rating", "ratings", "ranked", "recommended", "alternative", "alternatives",
            "pros and cons", "features", "worth it", "should i buy", "is it good"
        ],
        "keyword_patterns": [
            r'\bbest\b', r'\btop\b', r'\breview\b', r'\bcompare\b', r'\bcompari(son|ng)\b', 
            r'\bvs\b', r'\bversus\b', r'\balternatives?\b', r'\brated\b', r'\branking\b',
            r'\bworth\s+it\b', r'\bshould\s+I\s+buy\b', r'\bis\s+it\s+good\b',
            r'\bpros\s+and\s+cons\b', r'\badvantages?\b', r'\bdisadvantages?\b',
            r'\bfeatures\b', r'\bspecifications?\b', r'\bwhich\s+(is\s+)?(the\s+)?best\b'
        ],
        "weight": 1.2  # Commercial intent signals future transactions
    }
}
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
    data += "best running shoes 2025,3100,0.78,1.52,280,290,300,310,320,330,340,350,360,370,380,390\n"
    data += "how to choose running shoes,2500,0.42,0.95,220,230,240,250,260,270,280,290,300,310,320,330\n"
    data += "running shoes for flat feet,1900,0.56,1.28,170,180,190,200,210,220,230,240,250,260,270,280\n"
    data += "trail running shoes reviews,1700,0.64,1.42,150,160,170,180,190,200,210,220,230,240,250,260\n"
    data += "buy nike air zoom,1500,0.87,1.95,130,140,150,160,170,180,190,200,210,220,230,240\n"
    
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

    # Process clusters in smaller batches
    cluster_ids = list(clusters_with_representatives.keys())
    batch_size = 5  # Process 5 clusters at a time
    
    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(cluster_ids))
        batch_cluster_ids = cluster_ids[batch_start:batch_end]
        
        # Create a simplified prompt for just this batch
        batch_prompt = custom_prompt.strip() + "\n\n"
        batch_prompt += (
            "FOR EACH CLUSTER, provide:\n"
            "1. A clear, concise name (3-6 words)\n"
            "2. A brief description (1-2 sentences)\n\n"
            "FORMAT YOUR RESPONSE AS FOLLOWS:\n\n"
            "```json\n"
            "{\n"
            '  "clusters": [\n'
            "    {\n"
            '      "cluster_id": 1,\n'
            '      "cluster_name": "Example Cluster Name",\n'
            '      "cluster_description": "Example description of what this cluster represents."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Here are the clusters:\n"
        )
        
        for cluster_id in batch_cluster_ids:
            sample_kws = clusters_with_representatives[cluster_id][:10]  # Limit to 10 keywords
            batch_prompt += f"- Cluster {cluster_id}: {', '.join(sample_kws)}\n"
        
        num_retries = 3
        batch_results = {}
        
        for attempt in range(num_retries):
            try:
                progress_text.text(f"Generating names for clusters {batch_start+1}-{batch_end} (attempt {attempt+1}/{num_retries})...")
                
                # Try API call with error handling
                try:
                    # Try with response_format first
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": batch_prompt}],
                            temperature=0.3,
                            response_format={"type": "json_object"},
                            max_tokens=1000
                        )
                    except:
                        # Fallback without response_format
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": batch_prompt + "\nRespond only with the JSON."}],
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
                                        batch_results[c_id] = (c_name, c_desc)
                                    except (ValueError, TypeError):
                                        continue
                            
                            # If we got good results, break the retry loop
                            if batch_results:
                                break
                    except json.JSONDecodeError:
                        # Try regex extraction as fallback
                        for cluster_id in batch_cluster_ids:
                            # Look for cluster ID patterns
                            name_pattern = rf'cluster_id["\s:]+{cluster_id}["\s,}}]+\s*cluster_name["\s:]+([^"]+)["\s,}}]+'
                            desc_pattern = rf'cluster_id["\s:]+{cluster_id}["\s,}}]+.*?cluster_description["\s:]+([^"]+)["\s,}}]+'
                            
                            name_matches = re.findall(name_pattern, content)
                            desc_matches = re.findall(desc_pattern, content, re.DOTALL)
                            
                            if name_matches:
                                c_name = name_matches[0].strip()
                                c_desc = desc_matches[0].strip() if desc_matches else f"Group of related keywords (cluster {cluster_id})"
                                batch_results[cluster_id] = (c_name, c_desc)
                
                except Exception as api_error:
                    progress_text.text(f"API error: {str(api_error)[:100]}...")
                    
                    # Final fallback on last attempt
                    if attempt == num_retries - 1:
                        for cluster_id in batch_cluster_ids:
                            if cluster_id not in batch_results:
                                # Generate a generic name
                                kws = clusters_with_representatives[cluster_id][:3]
                                c_name = f"{kws[0]} {kws[1] if len(kws) > 1 else ''}"
                                c_desc = f"A collection of keywords related to {', '.join(kws[:3])}"
                                batch_results[cluster_id] = (c_name, c_desc)
            
            except Exception as e:
                progress_text.text(f"Error in naming attempt {attempt+1}: {str(e)[:100]}...")
                time.sleep(1)  # Wait before retrying
        
        # Add batch results to overall results
        results.update(batch_results)
        
        # Update progress
        progress_bar.progress(min(1.0, (batch_end) / len(cluster_ids)))
    
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

def extract_features_for_intent(keyword, search_intent_description=""):
    """
    Extracts features for search intent classification based on keyword patterns.
    Returns a dictionary of features that can be used for classification.
    
    This is a more sophisticated approach than the previous classify_search_intent.
    """
    # Features to extract
    features = {
        "keyword_length": len(keyword.split()),
        "keyword_lower": keyword.lower(),
        "has_informational_prefix": False,
        "has_navigational_prefix": False,
        "has_transactional_prefix": False,
        "has_commercial_prefix": False,
        "has_informational_suffix": False,
        "has_navigational_suffix": False,
        "has_transactional_suffix": False,
        "has_commercial_suffix": False,
        "is_informational_exact_match": False,
        "is_navigational_exact_match": False,
        "is_transactional_exact_match": False,
        "is_commercial_exact_match": False,
        "informational_pattern_matches": 0,
        "navigational_pattern_matches": 0,
        "transactional_pattern_matches": 0,
        "commercial_pattern_matches": 0,
        "includes_brand": False,
        "includes_product_modifier": False,
        "includes_price_modifier": False,
        "local_intent": False,
        "modal_verbs": False  # signals a question typically
    }
    
    keyword_lower = keyword.lower()
    
    # Check prefixes
    words = keyword_lower.split()
    if words:
        first_word = words[0]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if any(first_word == prefix.lower() for prefix in patterns["prefixes"]):
                features[f"has_{intent_type.lower()}_prefix"] = True
    
    # Check suffixes 
    if words and len(words) > 1:
        last_word = words[-1]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if any(last_word == suffix.lower() for suffix in patterns["suffixes"]):
                features[f"has_{intent_type.lower()}_suffix"] = True
    
    # Check exact matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        for exact_match in patterns["exact_matches"]:
            if exact_match.lower() in keyword_lower:
                features[f"is_{intent_type.lower()}_exact_match"] = True
                break
    
    # Check pattern matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        match_count = 0
        for pattern in patterns["keyword_patterns"]:
            if re.search(pattern, keyword_lower):
                match_count += 1
        features[f"{intent_type.lower()}_pattern_matches"] = match_count
    
    # Additional features
    features["local_intent"] = any(term in keyword_lower for term in ["near me", "nearby", "in my area", "close to me", "closest", "local"])
    features["modal_verbs"] = any(modal in keyword_lower.split() for modal in ["can", "could", "should", "would", "will", "may", "might"])
    features["includes_price_modifier"] = any(term in keyword_lower for term in ["price", "cost", "cheap", "expensive", "affordable", "discount", "offer", "deal", "coupon"])
    features["includes_product_modifier"] = any(term in keyword_lower for term in ["best", "top", "cheap", "premium", "quality", "new", "used", "refurbished", "alternative"])
    
    # Include any brand names detection here if needed
    
    return features

def classify_search_intent_ml(keywords, search_intent_description="", cluster_name=""):
    """
    Enhanced search intent classification using a ML-inspired approach
    with weighted feature scoring rather than simple pattern matching.
    
    As per the SEJ article, this implements a more sophisticated classification
    system that considers multiple signals and weights them appropriately.
    """
    if not keywords:
        return {
            "primary_intent": "Unknown",
            "scores": {
                "Informational": 25,
                "Navigational": 25,
                "Transactional": 25,
                "Commercial": 25
            },
            "evidence": {}
        }
    
    # Extract features for all keywords
    all_features = []
    for keyword in keywords[:min(len(keywords), 20)]:  # Limit to first 20 keywords for performance
        features = extract_features_for_intent(keyword, search_intent_description)
        all_features.append(features)
    
    # Aggregate features
    informational_signals = []
    navigational_signals = []
    transactional_signals = []
    commercial_signals = []
    
    # Count pattern matches across all features
    for features in all_features:
        # Informational signals
        if features["has_informational_prefix"]:
            informational_signals.append("Has informational prefix")
        if features["has_informational_suffix"]:
            informational_signals.append("Has informational suffix")
        if features["is_informational_exact_match"]:
            informational_signals.append("Contains informational phrase")
        if features["informational_pattern_matches"] > 0:
            informational_signals.append(f"Matches {features['informational_pattern_matches']} informational patterns")
        if features["modal_verbs"]:
            informational_signals.append("Contains question-like modal verb")
            
        # Navigational signals
        if features["has_navigational_prefix"]:
            navigational_signals.append("Has navigational prefix")
        if features["has_navigational_suffix"]:
            navigational_signals.append("Has navigational suffix")
        if features["is_navigational_exact_match"]:
            navigational_signals.append("Contains navigational phrase")
        if features["navigational_pattern_matches"] > 0:
            navigational_signals.append(f"Matches {features['navigational_pattern_matches']} navigational patterns")
        if features["includes_brand"]:
            navigational_signals.append("Includes brand name")
            
        # Transactional signals
        if features["has_transactional_prefix"]:
            transactional_signals.append("Has transactional prefix")
        if features["has_transactional_suffix"]:
            transactional_signals.append("Has transactional suffix")
        if features["is_transactional_exact_match"]:
            transactional_signals.append("Contains transactional phrase")
        if features["transactional_pattern_matches"] > 0:
            transactional_signals.append(f"Matches {features['transactional_pattern_matches']} transactional patterns")
        if features["includes_price_modifier"]:
            transactional_signals.append("Includes price-related term")
        if features["local_intent"]:
            transactional_signals.append("Shows local intent (near me, etc.)")
            
        # Commercial signals
        if features["has_commercial_prefix"]:
            commercial_signals.append("Has commercial prefix")
        if features["has_commercial_suffix"]:
            commercial_signals.append("Has commercial suffix")
        if features["is_commercial_exact_match"]:
            commercial_signals.append("Contains commercial phrase")
        if features["commercial_pattern_matches"] > 0:
            commercial_signals.append(f"Matches {features['commercial_pattern_matches']} commercial patterns")
        if features["includes_product_modifier"]:
            commercial_signals.append("Includes product comparison term")
    
    # Calculate scores based on unique signals
    info_signals = set(informational_signals)
    nav_signals = set(navigational_signals)
    trans_signals = set(transactional_signals)
    comm_signals = set(commercial_signals)
    
    # Calculate relative proportions (with weighting)
    info_weight = SEARCH_INTENT_PATTERNS["Informational"]["weight"]
    nav_weight = SEARCH_INTENT_PATTERNS["Navigational"]["weight"]
    trans_weight = SEARCH_INTENT_PATTERNS["Transactional"]["weight"]
    comm_weight = SEARCH_INTENT_PATTERNS["Commercial"]["weight"]
    
    info_score = len(info_signals) * info_weight
    nav_score = len(nav_signals) * nav_weight
    trans_score = len(trans_signals) * trans_weight
    comm_score = len(comm_signals) * comm_weight
    
    # Check description for explicit mentions
    if search_intent_description:
        desc_lower = search_intent_description.lower()
        if re.search(r'\binformational\b|\binformation\s+intent\b|\binformation\s+search\b|\bleaning\b|\bquestion\b', desc_lower):
            info_score += 5
        if re.search(r'\bnavigational\b|\bnavigate\b|\bfind\s+\w+\s+website\b|\bfind\s+\w+\s+page\b|\baccess\b', desc_lower):
            nav_score += 5
        if re.search(r'\btransactional\b|\bbuy\b|\bpurchase\b|\bshopping\b|\bsale\b|\btransaction\b', desc_lower):
            trans_score += 5
        if re.search(r'\bcommercial\b|\bcompar(e|ing|ison)\b|\breview\b|\balternative\b|\bbest\b', desc_lower):
            comm_score += 5
    
    # Check cluster name for signals
    if cluster_name:
        name_lower = cluster_name.lower()
        if re.search(r'\bhow\b|\bwhat\b|\bwhy\b|\bwhen\b|\bguide\b|\btutorial\b', name_lower):
            info_score += 3
        if re.search(r'\bwebsite\b|\bofficial\b|\blogin\b|\bportal\b|\bdownload\b', name_lower):
            nav_score += 3
        if re.search(r'\bbuy\b|\bshop\b|\bpurchase\b|\bsale\b|\bdiscount\b|\bcost\b|\bprice\b', name_lower):
            trans_score += 3
        if re.search(r'\bbest\b|\btop\b|\breview\b|\bcompare\b|\bvs\b|\balternative\b', name_lower):
            comm_score += 3
    
    # Normalize to percentages
    total_score = max(1, info_score + nav_score + trans_score + comm_score)
    info_pct = (info_score / total_score) * 100
    nav_pct = (nav_score / total_score) * 100
    trans_pct = (trans_score / total_score) * 100
    comm_pct = (comm_score / total_score) * 100
    
    # Prepare scores
    scores = {
        "Informational": info_pct,
        "Navigational": nav_pct,
        "Transactional": trans_pct,
        "Commercial": comm_pct
    }
    
    # Find primary intent (highest score)
    primary_intent = max(scores, key=scores.get)
    
    # If the highest score is less than 30%, consider it mixed intent
    max_score = max(scores.values())
    if max_score < 30:
        # Check if there's a close second
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1] < 10):
            primary_intent = "Mixed Intent"
    
    # Collect evidence for the primary intent
    evidence = {
        "Informational": list(info_signals),
        "Navigational": list(nav_signals),
        "Transactional": list(trans_signals),
        "Commercial": list(comm_signals)
    }
    
    return {
        "primary_intent": primary_intent,
        "scores": scores,
        "evidence": evidence
    }

def analyze_cluster_for_intent_flow(df, cluster_id):
    """
    Following SEJ's recommendation to map customer journey through analysis of
    intent distribution within a cluster - this helps understand if the cluster 
    represents a part of the customer journey.
    """
    # Get keywords for this cluster
    cluster_keywords = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
    
    if not cluster_keywords:
        return None
    
    # Classify each keyword individually
    keyword_intents = []
    for keyword in cluster_keywords:
        intent_data = classify_search_intent_ml([keyword])
        keyword_intents.append({
            "keyword": keyword,
            "primary_intent": intent_data["primary_intent"],
            "scores": intent_data["scores"]
        })
    
    # Calculate distribution of intents
    intent_counts = Counter([item["primary_intent"] for item in keyword_intents])
    total = len(keyword_intents)
    
    # Calculate average scores across all keywords
    avg_scores = {
        "Informational": sum(item["scores"]["Informational"] for item in keyword_intents) / total,
        "Navigational": sum(item["scores"]["Navigational"] for item in keyword_intents) / total,
        "Transactional": sum(item["scores"]["Transactional"] for item in keyword_intents) / total,
        "Commercial": sum(item["scores"]["Commercial"] for item in keyword_intents) / total
    }
    
    # Analyze if this represents a customer journey phase
    # Typically, customer journey: Info -> Commercial -> Transactional
    journey_phase = None
    
    # Simple journey phase detection
    info_pct = (intent_counts.get("Informational", 0) / total) * 100
    comm_pct = (intent_counts.get("Commercial", 0) / total) * 100
    trans_pct = (intent_counts.get("Transactional", 0) / total) * 100
    
    if info_pct > 50:
        journey_phase = "Early (Research Phase)"
    elif comm_pct > 50:
        journey_phase = "Middle (Consideration Phase)"
    elif trans_pct > 50:
        journey_phase = "Late (Purchase Phase)"
    elif info_pct > 25 and comm_pct > 25:
        journey_phase = "Research-to-Consideration Transition"
    elif comm_pct > 25 and trans_pct > 25:
        journey_phase = "Consideration-to-Purchase Transition"
    else:
        journey_phase = "Mixed Journey Stages"
    
    return {
        "intent_distribution": {intent: (count / total) * 100 for intent, count in intent_counts.items()},
        "avg_scores": avg_scores,
        "journey_phase": journey_phase,
        "keyword_sample": [{"keyword": k["keyword"], "intent": k["primary_intent"]} for k in keyword_intents[:10]]
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

    # Validate client first
    if not client:
        st.warning("No valid OpenAI client provided. Using default values.")
        return results

    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Performing semantic analysis on clusters...")
    
    # Process clusters in smaller batches to avoid context limitations
    cluster_ids = list(clusters_with_representatives.keys())
    batch_size = 5  # Process 5 clusters at a time
    
    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(cluster_ids))
        batch_cluster_ids = cluster_ids[batch_start:batch_end]
        
        # Create a simplified prompt for just this batch
        batch_prompt = (
            "You are an expert in SEO and clustering analysis. Analyze each keyword cluster below by providing:\n"
            "1) Search intent: Describe why users would search these terms\n"
            "2) Split suggestion: Yes/No and if yes, suggest 2-3 subclusters\n"
            "3) SEO insights: Keyword difficulty, content ideas, etc.\n"
            "4) Coherence score: 0-10 where 10 means perfectly coherent\n\n"
            "Format your response as a valid JSON object with this structure for EACH cluster:\n"
            "{\n"
            '  "clusters": [\n'
            "    {\n"
            '      "cluster_id": 1,\n'
            '      "search_intent": "Intent description",\n'
            '      "split_suggestion": "Yes or No with explanation",\n'
            '      "additional_info": "SEO insights",\n'
            '      "coherence_score": 7,\n'
            '      "subclusters": [{"name": "Name 1", "keywords": ["kw1", "kw2"]}]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Here are the clusters to analyze:\n"
        )
        
        for cluster_id in batch_cluster_ids:
            sample_kws = clusters_with_representatives[cluster_id][:10]  # Limit to 10 keywords
            batch_prompt += f"Cluster {cluster_id}: {', '.join(sample_kws)}\n"
        
        num_retries = 3
        batch_results = {}
        
        for attempt in range(num_retries):
            try:
                progress_text.text(f"Analyzing clusters {batch_start+1}-{batch_end} (attempt {attempt+1}/{num_retries})...")
                
                # Try API call with error handling
                try:
                    # Try to use response_format parameter if model supports it
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": batch_prompt}],
                            temperature=0.3,
                            response_format={"type": "json_object"},
                            max_tokens=2000
                        )
                    except:
                        # Fallback if response_format isn't supported
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": batch_prompt + "\nPlease respond with valid JSON only."}],
                            temperature=0.3,
                            max_tokens=2000
                        )
                    
                    content = response.choices[0].message.content.strip()
                    
                    # Debug the response (only on first attempt)
                    if attempt == 0:
                        progress_text.text(f"Processing API response...")
                    
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
                                    # Robust cleaning and conversion of cluster_id 
                                    try:
                                        # If it's a string, try to clean and convert it
                                        if isinstance(c_id, str):
                                            # Remove spaces and non-numeric characters
                                            c_id_clean = ''.join(filter(str.isdigit, c_id.strip()))
                                            if c_id_clean:
                                                c_id = int(c_id_clean)
                                            else:
                                                st.warning(f"Invalid cluster ID: '{c_id}' - no digits")
                                                continue
                                        # If it's already a number, use it directly
                                        elif isinstance(c_id, (int, float)):
                                            c_id = int(c_id)
                                        else:
                                            st.warning(f"Unsupported cluster_id type: {type(c_id)}")
                                            continue
                                        
                                        # Verify it's a valid and existing cluster ID
                                        if c_id not in clusters_with_representatives:
                                            st.warning(f"Cluster ID {c_id} doesn't exist in the data")
                                            continue
                                        
                                        search_intent = item.get("search_intent", "")
                                        split_suggestion = item.get("split_suggestion", "")
                                        additional_info = item.get("additional_info", "")
                                        coherence_score = item.get("coherence_score", 5)
                                        subclusters = item.get("subclusters", [])
                                        
                                        # Use our enhanced ML-based classifier
                                        cluster_name = f"Cluster {c_id}"  # Default name
                                        intent_classification = classify_search_intent_ml(
                                            clusters_with_representatives.get(c_id, []),
                                            search_intent,
                                            cluster_name
                                        )
                                        
                                        )
                                    
                                    batch_results[cluster_id] = {
                                        "search_intent": search_intent,
                                        "split_suggestion": "Unable to determine from API response",
                                        "additional_info": "Unable to extract from API response",
                                        "coherence_score": coherence_score,
                                        "subclusters": [],
                                        "intent_classification": intent_classification
                                    }
                
                except Exception as api_error:
                    progress_text.text(f"API error: {str(api_error)[:100]}... Retrying with simpler prompt.")
                    
                    # Simplified fallback on last attempt
                    if attempt == num_retries - 1:
                        # Try with an extremely simple prompt as last resort
                        try:
                            for cluster_id in batch_cluster_ids:
                                # Get just a few keywords
                                kws = clusters_with_representatives[cluster_id][:5]
                                simple_prompt = f"Analyze these keywords: {', '.join(kws)}. Give a one sentence description of search intent."
                                
                                simple_response = client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": simple_prompt}],
                                    temperature=0.3,
                                    max_tokens=200
                                )
                                
                                simple_content = simple_response.choices[0].message.content.strip()
                                
                                # Just use this as search intent and our ML classifier for the rest
                                intent_classification = classify_search_intent_ml(
                                    clusters_with_representatives.get(cluster_id, []),
                                    simple_content,
                                    f"Cluster {cluster_id}"
                                )
                                
                                batch_results[cluster_id] = {
                                    "search_intent": simple_content,
                                    "split_suggestion": "No split suggestion available",
                                    "additional_info": "No SEO information available",
                                    "coherence_score": 5,
                                    "subclusters": [],
                                    "intent_classification": intent_classification
                                }
                        except Exception as e:
                            progress_text.text(f"Final fallback also failed: {str(e)[:100]}")
            
            except Exception as outer_error:
                progress_text.text(f"Outer error: {str(outer_error)[:100]}...")
                time.sleep(1)  # Wait briefly before retrying
        
        # Add batch results to overall results
        results.update(batch_results)
        
        # Update progress
        progress_bar.progress(min(1.0, (batch_end) / len(cluster_ids)))
    
    # If we still have no results, create default ones
    if not results:
        st.warning("Could not generate semantic analysis via API. Using default values.")
        for c_id in clusters_with_representatives.keys():
            intent_classification = classify_search_intent_ml(
                clusters_with_representatives.get(c_id, []),
                "No search intent data available",
                f"Cluster {c_id}"
            )
            
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

        # Process intent flow (customer journey) for each cluster
        for c_id in semantic_analysis:
            intent_flow = analyze_cluster_for_intent_flow(df, c_id)
            if intent_flow:
                semantic_analysis[c_id]['intent_flow'] = intent_flow

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
                
            # Convert numeric columns to proper numeric types with error handling
            for col in df.columns:
                if col != 'keyword' and col.lower() not in ['cluster_name', 'cluster_description']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass  # Keep as is if not convertible
                        
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
    .intent-mixed {
        background-color: #f5f5f5;
        border-left: 5px solid #9e9e9e;
    }
    .subcluster-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-early {
        background-color: #e8f5e9; 
        border-left: 5px solid #43a047;
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-middle {
        background-color: #e3f2fd; 
        border-left: 5px solid #1e88e5;
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-late {
        background-color: #fff3e0; 
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-transition {
        background-color: #f3e5f5; 
        border-left: 5px solid #8e24aa;
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-mixed {
        background-color: #f5f5f5;
        border-left: 5px solid #9e9e9e;
        padding: 10px;
        margin-bottom: 10px;
    }
    .evidence-list {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
        margin-left: 20px;
    }
    
    .keyword-example {
        display: inline-block;
        background-color: #f5f5f5;
        border-radius: 3px;
        padding: 3px 6px;
        margin: 2px;
        font-size: 0.85em;
    }
    
    .info-tag {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    
    .commercial-tag {
        background-color: #f3e5f5;
        color: #4a148c;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    
    .transactional-tag {
        background-color: #fff3e0;
        color: #e65100;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
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
