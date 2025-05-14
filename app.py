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

# Check if PDF export is available
try:
    from export_pdf import add_pdf_export_button
    pdf_export_available = True
except ImportError:
    pdf_export_available = False

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
    "Polish": "pl_core_news_sm",
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

def calculate_api_cost(num_keywords, selected_model="gpt-4.1-nano", num_clusters=10):
    """
    Calculates the estimated cost of using the OpenAI API based on the number of keywords.
    """
    # Updated prices (May 2025) - Always check OpenAI's official pricing
    EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small per 1K tokens
    
    # GPT-4.1-Nano costs
    GPT41NANO_INPUT_COST_PER_1K = 0.15  # $0.15 per 1K input tokens
    GPT41NANO_OUTPUT_COST_PER_1K = 0.60  # $0.60 per 1K output tokens
    
    # GPT-4.1-Mini costs
    GPT41MINI_INPUT_COST_PER_1K = 0.30  # $0.30 per 1K input tokens
    GPT41MINI_OUTPUT_COST_PER_1K = 1.20  # $1.20 per 1K output tokens
    
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
    
    if selected_model == "gpt-4.1-nano":
        input_cost = (estimated_input_tokens / 1000) * GPT41NANO_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT41NANO_OUTPUT_COST_PER_1K
    else:  # GPT-4.1-Mini
        input_cost = (estimated_input_tokens / 1000) * GPT41MINI_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT41MINI_OUTPUT_COST_PER_1K
    
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
            options=["gpt-4.1-nano", "gpt-4.1-mini"],
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

def show_csv_cost_estimate(num_keywords, selected_model="gpt-4.1-nano", num_clusters=10):
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

def enhanced_preprocessing(text, use_lemmatization, spacy_nlp, language="english"):
    """
    Enhanced preprocessing using spaCy or fallback with TextBlob.
    Now with support for Polish.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        if spacy_nlp is not None:  # We use the loaded spaCy model
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
            
            # Get stopwords according to language
            try:
                stop_words = set(stopwords.words(language))
            except:
                stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
                # Add Polish stopwords if needed
                if language == "polish":
                    polish_stopwords = {
                        'i', 'w', 'na', 'z', 'do', 'jest', 'to', 'nie', 'się', 'że', 'dla', 'a', 'od', 'jak',
                        'po', 'co', 'tak', 'za', 'przez', 'o', 'lub', 'ale', 'już', 'być', 'ten', 'który',
                        'też', 'tylko', 'jeszcze', 'przy', 'może', 'ich', 'bardzo', 'tam', 'jako', 'wszystko',
                        'gdy', 'więc', 'zawsze', 'aby', 'kiedy', 'można', 'jeśli', 'bez', 'gdzie', 'czy'
                    }
                    stop_words.update(polish_stopwords)
            
            words = [w for w in blob.words if len(w) > 1 and w.lower() not in stop_words]
            
            if use_lemmatization:
                if language == "polish":
                    # Simplify Polish words (basic approach)
                    lemmas = []
                    for w in words:
                        if len(w) > 4:
                            for suffix in ['ów', 'ami', 'ach', 'om', 'owi', 'em', 'ie', 'ego', 'emu', 'ymi', 'ich', 'imi']:
                                if w.endswith(suffix) and len(w) - len(suffix) > 3:
                                    w = w[:-len(suffix)]
                                    break
                        lemmas.append(w)
                else:
                    # For other languages use standard lemmatizer
                    lemmatizer = WordNetLemmatizer()
                    lemmas = [lemmatizer.lemmatize(w) for w in words]
                
                processed_parts = lemmas + noun_phrases
            else:
                processed_parts = words + noun_phrases
            
            return " ".join(processed_parts)
        
        else:
            # fallback to standard NLTK
            return preprocess_text(text, use_lemmatization, language)
    
    except Exception as e:
        print(f"Error in enhanced_preprocessing: {str(e)}")
        return text.lower() if isinstance(text, str) else ""

def preprocess_text(text, use_lemmatization=True, language="english"):
    """
    Basic NLTK-based text preprocessing as a fallback.
    Supports multiple languages including Polish.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        text = text.lower()
        
        # Language-adapted tokenization
        try:
            tokens = word_tokenize(text, language=language)
        except:
            tokens = word_tokenize(text)  # Fallback to default tokenizer
        
        # Language-specific stopwords handling
        try:
            stop_words = set(stopwords.words(language))
        except:
            # Default stopwords if NLTK fails
            stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
            
            # Polish-specific stopwords if that language is selected
            if language == "polish":
                polish_stopwords = {
                    'i', 'w', 'na', 'z', 'do', 'jest', 'to', 'nie', 'się', 'że', 'dla', 'a', 'od', 'jak',
                    'po', 'co', 'tak', 'za', 'przez', 'o', 'lub', 'ale', 'już', 'być', 'ten', 'który',
                    'też', 'tylko', 'jeszcze', 'przy', 'może', 'ich', 'bardzo', 'tam', 'jako', 'wszystko',
                    'gdy', 'więc', 'zawsze', 'aby', 'kiedy', 'można', 'jeśli', 'bez', 'gdzie', 'czy'
                }
                stop_words.update(polish_stopwords)
        
        # Filter tokens that are words and not stopwords
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        # Lemmatization according to language
        if use_lemmatization:
            if language == "polish":
                # For Polish, try to use spaCy if available
                try:
                    import spacy
                    try:
                        nlp = spacy.load("pl_core_news_sm")
                        doc = nlp(" ".join(tokens))
                        tokens = [token.lemma_ for token in doc]
                    except:
                        # Fallback to simple approach for Polish if spaCy is not available
                        # Remove common Polish endings (simplified)
                        simplified_tokens = []
                        for t in tokens:
                            # Remove some common plural and case endings
                            if len(t) > 4:
                                for suffix in ['ów', 'ami', 'ach', 'om', 'owi', 'em', 'ie', 'ego', 'emu', 'ymi', 'ich', 'imi']:
                                    if t.endswith(suffix) and len(t) - len(suffix) > 3:
                                        t = t[:-len(suffix)]
                                        break
                            simplified_tokens.append(t)
                        tokens = simplified_tokens
                except ImportError:
                    # If spaCy is not installed, use the simplified approach
                    simplified_tokens = []
                    for t in tokens:
                        if len(t) > 4:
                            for suffix in ['ów', 'ami', 'ach', 'om', 'owi', 'em', 'ie', 'ego', 'emu', 'ymi', 'ich', 'imi']:
                                if t.endswith(suffix) and len(t) - len(suffix) > 3:
                                    t = t[:-len(suffix)]
                                    break
                        simplified_tokens.append(t)
                    tokens = simplified_tokens
            else:
                # For other languages, use WordNetLemmatizer
                try:
                    lemmatizer = WordNetLemmatizer()
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
                except:
                    pass
        
        return " ".join(tokens)
    except Exception as e:
        print(f"Error in preprocess_text: {str(e)}")
        return text.lower() if isinstance(text, str) else ""

def preprocess_keywords(keywords, use_advanced, spacy_nlp=None, selected_language="English"):
    """
    Main keyword preprocessing loop with language support.
    """
    processed_keywords = []
    progress_bar = st.progress(0)
    total = len(keywords)
    
    # Map the selected language to NLTK codes
    language_map = {
        "English": "english",
        "Spanish": "spanish",
        "French": "french",
        "German": "german",
        "Italian": "italian",
        "Portuguese": "portuguese",
        "Polish": "polish",  # Add mapping for Polish
        # You can add more mappings as needed
    }
    nltk_language = language_map.get(selected_language, "english")
    
    if use_advanced:
        if spacy_nlp is not None:
            st.success(f"Using advanced preprocessing with spaCy for {selected_language}")
        elif textblob_available:
            st.success(f"Using fallback preprocessing with TextBlob for {selected_language}")
        else:
            st.info(f"Using standard preprocessing with NLTK for {selected_language}")
    else:
        st.info(f"Using standard preprocessing with NLTK for {selected_language} (advanced preprocessing disabled)")
    
    for i, keyword in enumerate(keywords):
        if use_advanced and (spacy_nlp is not None or textblob_available):
            processed_keywords.append(enhanced_preprocessing(keyword, True, spacy_nlp, nltk_language))
        else:
            processed_keywords.append(preprocess_text(keyword, True, nltk_language))
        
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
    model="gpt-4.1-nano",
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
    model="gpt-4.1-nano"
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
                                    # Limpieza y conversión robusta de cluster_id
                                    try:
                                        # Si es una cadena, intentamos limpiarla y convertirla
                                        if isinstance(c_id, str):
                                            # Eliminar espacios y caracteres no numéricos
                                            c_id_clean = ''.join(filter(str.isdigit, c_id.strip()))
                                            if c_id_clean:
                                                c_id = int(c_id_clean)
                                            else:
                                                st.warning(f"Cluster ID no válido: '{c_id}' - no contiene dígitos")
                                                continue
                                        # Si ya es un número, usarlo directamente
                                        elif isinstance(c_id, (int, float)):
                                            c_id = int(c_id)
                                        else:
                                            st.warning(f"Tipo de cluster_id no soportado: {type(c_id)}")
                                            continue
                                        
                                        # Verificar que sea un ID de cluster válido y existente
                                        if c_id not in clusters_with_representatives:
                                            st.warning(f"ID de cluster {c_id} no existe en los datos")
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
                                        
                                        batch_results[c_id] = {
                                            "search_intent": search_intent,
                                            "split_suggestion": split_suggestion,
                                            "additional_info": additional_info,
                                            "coherence_score": coherence_score,
                                            "subclusters": subclusters,
                                            "intent_classification": intent_classification
                                        }
                                    except Exception as e:
                                        st.warning(f"Error al procesar cluster_id: {str(e)}")
                                        continue
                            
                            # If we got good results, break the retry loop
                            if batch_results:
                                break
                    except json.JSONDecodeError:
                        # JSON parsing failed, try regex extraction
                        if attempt == num_retries - 1:  # Only on last attempt
                            for cluster_id in batch_cluster_ids:
                                cluster_pattern = rf"(?:cluster|cluster_id)[^0-9]*{cluster_id}[^0-9]"
                                if re.search(cluster_pattern, content, re.IGNORECASE):
                                    # Extract basic data with regex
                                    search_intent = "Extracted from partial response"
                                    coherence_score = 5  # Default
                                    
                                    # Try to extract coherence score with regex
                                    score_match = re.search(r'coherence_score["\s:]+(\d+)', content)
                                    if score_match:
                                        try:
                                            coherence_score = int(score_match.group(1))
                                        except:
                                            pass
                                    
                                    # Use our ML classifier regardless
                                    intent_classification = classify_search_intent_ml(
                                        clusters_with_representatives.get(cluster_id, []),
                                        search_intent,
                                        f"Cluster {cluster_id}"
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

def extract_entities_for_cluster(df, cluster_id, spacy_nlp):
    """
    Extract named entities from keywords in a cluster.
    Returns a dictionary of entity types and their frequency.
    """
    if not spacy_nlp:
        return {}
        
    keywords = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
    entities = {}
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text(f"Extracting entities for cluster {cluster_id}...")
    
    try:
        for i, keyword in enumerate(keywords):
            doc = spacy_nlp(keyword)
            for ent in doc.ents:
                entity_type = ent.label_
                entity_text = ent.text.lower()
                
                if entity_type not in entities:
                    entities[entity_type] = {}
                if entity_text not in entities[entity_type]:
                    entities[entity_type][entity_text] = 0
                entities[entity_type][entity_text] += 1
                
           # Update progress every 10 keywords
            if i % 10 == 0:
                progress_bar.progress(min(i / len(keywords), 1.0))
        
        progress_bar.progress(1.0)
        progress_text.text(f"✅ Entity extraction complete for cluster {cluster_id}.")
        
        # Sort entities by frequency
        for entity_type in entities:
            entities[entity_type] = dict(sorted(
                entities[entity_type].items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
            
        return entities
    except Exception as e:
        progress_text.text(f"Error extracting entities: {str(e)}")
        return {}

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

def evaluate_and_refine_clusters(df, client, model="gpt-4.1-nano"):
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

def calculate_topic_density(embeddings):
    """
    Calculate topic density based on semantic similarity and variance.
    Returns a score between 0-1 where higher values indicate stronger topical focus.
    """
    try:
        if len(embeddings) < 2:
            return 0.5  # Default value for very small clusters
            
        # Calculate the centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Normalize the centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
            
        # Calculate pairwise similarities within cluster
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                # Normalize embeddings
                emb_i = embeddings[i] / np.linalg.norm(embeddings[i])
                emb_j = embeddings[j] / np.linalg.norm(embeddings[j])
                
                # Calculate cosine similarity
                sim = np.dot(emb_i, emb_j)
                similarities.append(sim)
        
        # Topic density is the average of pairwise similarities
        density = np.mean(similarities) if similarities else 0.5
        
        return max(0.0, min(1.0, density))
    except Exception as e:
        st.warning(f"Error calculating topic density: {str(e)}")
        return 0.5

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
                    model="gpt-4.1-nano",
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
        st.info(f"Preprocessing keywords with advanced NLP or fallback for {selected_language}")
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
#          EXPORT FUNCTIONS
################################################################

def generate_html_report(df, cluster_evaluation=None):
    """Generate a full HTML report with all analysis and visualizations"""
    
    from datetime import datetime
    import plotly.io as pio
    
    # Start HTML with styles
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Semantic Keyword Clustering Report</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
            .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            h3 {{ color: #2980b9; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .info-box {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .cluster-box {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
            .intent-info {{ background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px; margin: 10px 0; }}
            .intent-nav {{ background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 10px; margin: 10px 0; }}
            .intent-trans {{ background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; margin: 10px 0; }}
            .intent-comm {{ background-color: #f3e5f5; border-left: 5px solid #9c27b0; padding: 10px; margin: 10px 0; }}
            .chart {{ width: 100%; height: 500px; margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Semantic Keyword Clustering Analysis</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total keywords: {len(df)}</p>
                <p>Number of clusters: {len(df['cluster_id'].unique())}</p>
                {'<p>Total search volume: ' + format(df['search_volume'].sum(), ',') + '</p>' if 'search_volume' in df.columns else ''}
            </div>
    """
    
    # Add visualizations
    html += """
            <div class="section">
                <h2>Cluster Distribution</h2>
                <div id="cluster-distribution" class="chart"></div>
            </div>
            
            <div class="section">
                <h2>Semantic Coherence</h2>
                <div id="coherence-chart" class="chart"></div>
            </div>
    """
    
    # If there is intent data, add visualizations
    if cluster_evaluation:
        html += """
            <div class="section">
                <h2>Search Intent Analysis</h2>
                <div id="intent-distribution" class="chart"></div>
                <div id="intent-by-cluster" class="chart"></div>
            </div>
            
            <div class="section">
                <h2>Customer Journey Analysis</h2>
                <div id="journey-phases" class="chart"></div>
            </div>
        """
    
    # Cluster details completed
    html += """
        <div class="section">
            <h2>Detailed Cluster Analysis</h2>
    """
    
    # For each cluster, generate a specific section
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_df = df[df['cluster_id'] == cluster_id]
        cluster_name = cluster_df['cluster_name'].iloc[0] if not cluster_df['cluster_name'].empty else f"Cluster {cluster_id}"
        cluster_desc = cluster_df['cluster_description'].iloc[0] if not cluster_df['cluster_description'].empty else ""
        
        html += f"""
            <div class="cluster-box">
                <h3>{cluster_name} (ID: {cluster_id})</h3>
                <p><strong>Description:</strong> {cluster_desc}</p>
                <p><strong>Total Keywords:</strong> {len(cluster_df)}</p>
                <p><strong>Coherence Score:</strong> {cluster_df['cluster_coherence'].mean():.3f}</p>
        """
        
        # Add AI analysis information if it's available
        if cluster_evaluation and cluster_id in cluster_evaluation:
            ai_eval = cluster_evaluation[cluster_id]
            
            # Search Intent
            intent_data = ai_eval.get('intent_classification', {})
            primary_intent = intent_data.get('primary_intent', 'Unknown')
            intent_class = ""
            if primary_intent == "Informational": intent_class = "intent-info"
            elif primary_intent == "Navigational": intent_class = "intent-nav"
            elif primary_intent == "Transactional": intent_class = "intent-trans"
            elif primary_intent == "Commercial": intent_class = "intent-comm"
            
            html += f"""
                <div class="{intent_class}">
                    <p><strong>Primary Search Intent:</strong> {primary_intent}</p>
                    <p><strong>Search Intent Details:</strong> {ai_eval.get('search_intent', 'N/A')}</p>
                </div>
            """
            
            # Journey phase
            if 'intent_flow' in ai_eval and 'journey_phase' in ai_eval['intent_flow']:
                html += f"""
                    <p><strong>Customer Journey Phase:</strong> {ai_eval['intent_flow']['journey_phase']}</p>
                """
            
            # Split suggestion
            html += f"""
                <p><strong>Split Suggestion:</strong> {ai_eval.get('split_suggestion', 'N/A')}</p>
                <p><strong>SEO Insights:</strong> {ai_eval.get('additional_info', 'N/A')}</p>
            """
        
        # Representative keywords table
        rep_kws = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
        if rep_kws:
            html += """
                <h4>Representative Keywords</h4>
                <ul>
            """
            for kw in rep_kws[:10]:
                html += f"<li>{kw}</li>"
            html += "</ul>"
        
        # Table with all keywords with search volume if they're available
        html += """
            <h4>All Keywords</h4>
            <table>
                <tr>
                    <th>Keyword</th>
        """
        
        if 'search_volume' in cluster_df.columns:
            html += "<th>Search Volume</th>"
        
        html += "</tr>"
        
        # Limit to 100 keywords per cluster to avoid making the file too much bigger
        display_df = cluster_df.sort_values(by='search_volume', ascending=False) if 'search_volume' in cluster_df.columns else cluster_df
        display_df = display_df.head(100)
        
        for _, row in display_df.iterrows():
            html += f"<tr><td>{row['keyword']}</td>"
            if 'search_volume' in cluster_df.columns:
                html += f"<td>{row['search_volume']:,}</td>"
            html += "</tr>"
        
        html += """
                </table>
            </div>
        """
    
    # JS code for visualizations
    html += """
        </div>
        <script>
    """
    
    # Generate data for graphs
    cluster_sizes = df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
    cluster_sizes['label'] = cluster_sizes.apply(lambda x: f"{x['cluster_name'][:25]}{'...' if len(x['cluster_name']) > 25 else ''} (ID: {x['cluster_id']})", axis=1)
    
    # Cluster distribution graph
    html += f"""
        var clusterSizes = {{
            x: {cluster_sizes['label'].tolist()},
            y: {cluster_sizes['count'].tolist()},
            type: 'bar',
            marker: {{ color: 'royalblue' }}
        }};
        
        Plotly.newPlot('cluster-distribution', [clusterSizes], {{
            title: 'Size of Each Cluster',
            xaxis: {{ title: 'Cluster', tickangle: -45 }},
            yaxis: {{ title: 'Number of Keywords' }},
            margin: {{ l: 50, r: 50, t: 70, b: 200 }}
        }});
    """
    
    # Coherence graph
    coherence_data = df.groupby(['cluster_id', 'cluster_name'])['cluster_coherence'].mean().reset_index()
    coherence_data['label'] = coherence_data.apply(lambda x: f"{x['cluster_name'][:25]}{'...' if len(x['cluster_name']) > 25 else ''} (ID: {x['cluster_id']})", axis=1)
    
    html += f"""
        var coherenceData = {{
            x: {coherence_data['label'].tolist()},
            y: {coherence_data['cluster_coherence'].tolist()},
            type: 'bar',
            marker: {{ 
                color: {coherence_data['cluster_coherence'].tolist()},
                colorscale: 'Viridis'
            }}
        }};
        
        Plotly.newPlot('coherence-chart', [coherenceData], {{
            title: 'Semantic Coherence by Cluster',
            xaxis: {{ title: 'Cluster', tickangle: -45 }},
            yaxis: {{ title: 'Coherence' }},
            margin: {{ l: 50, r: 50, t: 70, b: 200 }}
        }});
    """
    
    # If there's intent data, generate visualizations
    if cluster_evaluation:
        # Compile intent data
        intent_data = []
        for c_id, data in cluster_evaluation.items():
            if 'intent_classification' in data:
                cluster_name = df[df['cluster_id'] == c_id]['cluster_name'].iloc[0] if not df[df['cluster_id'] == c_id].empty else f"Cluster {c_id}"
                primary_intent = data['intent_classification'].get('primary_intent', 'Unknown')
                count = len(df[df['cluster_id'] == c_id])
                
                intent_data.append({
                    'cluster_id': c_id,
                    'cluster_name': cluster_name,
                    'primary_intent': primary_intent,
                    'count': count
                })
        
        if intent_data:
            # Crear distribución de intent
            intent_counts = {}
            for item in intent_data:
                intent_counts[item['primary_intent']] = intent_counts.get(item['primary_intent'], 0) + item['count']
            
            html += f"""
                var intentData = {{
                    values: {list(intent_counts.values())},
                    labels: {list(intent_counts.keys())},
                    type: 'pie',
                    marker: {{
                        colors: ['rgb(33, 150, 243)', 'rgb(76, 175, 80)', 'rgb(255, 152, 0)', 'rgb(156, 39, 176)', 'rgb(158, 158, 158)']
                    }}
                }};
                
                Plotly.newPlot('intent-distribution', [intentData], {{
                    title: 'Search Intent Distribution',
                    margin: {{ l: 50, r: 50, t: 70, b: 50 }}
                }});
            """
    
    html += """
        </script>
        </div>
    </body>
    </html>
    """
    
    return html

def generate_excel_report(df, cluster_evaluation=None):
    """Generate a detailed Excel report with multiple tabs"""
    
    # Require openpyxl library
    try:
        import pandas as pd
        from io import BytesIO
        
        output = BytesIO()
        
        # Create ExcelWriter
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Tab 1: All data
            df.to_excel(writer, sheet_name='All Keywords', index=False)
            
            # Tab 2: Cluster resume
            summary_df = df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
            summary_df.columns = ['ID', 'Name', 'Description', 'Number of Keywords']
            
            # Add coherence
            coherence_df = df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
            summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id')
            summary_df.drop('cluster_id', axis=1, inplace=True)
            summary_df.rename(columns={'cluster_coherence': 'Coherence'}, inplace=True)
            
            # Add volume if it exists
            if 'search_volume' in df.columns:
                volume_df = df.groupby('cluster_id')['search_volume'].sum().reset_index()
                summary_df = summary_df.merge(volume_df, left_on='ID', right_on='cluster_id')
                summary_df.drop('cluster_id', axis=1, inplace=True)
                summary_df.rename(columns={'search_volume': 'Total Search Volume'}, inplace=True)
            
            # Add intent if it's available
            if cluster_evaluation:
                def get_search_intent(cid):
                    if cid in cluster_evaluation:
                        intent_data = cluster_evaluation[cid].get('intent_classification', {})
                        return intent_data.get('primary_intent', 'Unknown')
                    return 'Unknown'
                
                summary_df['Primary Intent'] = summary_df['ID'].apply(get_search_intent)
                
                # Add journey phase
                def get_journey_phase(cid):
                    if cid in cluster_evaluation and 'intent_flow' in cluster_evaluation[cid]:
                        return cluster_evaluation[cid]['intent_flow'].get('journey_phase', 'Unknown')
                    return 'Unknown'
                
                summary_df['Customer Journey Phase'] = summary_df['ID'].apply(get_journey_phase)
            
            summary_df.to_excel(writer, sheet_name='Clusters Summary', index=False)
            
            # Tab 3: Intent analysis if it's available
            if cluster_evaluation:
                intent_rows = []
                
                for c_id, data in cluster_evaluation.items():
                    if 'intent_classification' in data:
                        cluster_name = df[df['cluster_id'] == c_id]['cluster_name'].iloc[0] if not df[df['cluster_id'] == c_id].empty else f"Cluster {c_id}"
                        
                        intent_data = data['intent_classification']
                        scores = intent_data.get('scores', {})
                        
                        row = {
                            'Cluster ID': c_id,
                            'Cluster Name': cluster_name,
                            'Primary Intent': intent_data.get('primary_intent', 'Unknown'),
                            'Informational Score': scores.get('Informational', 0),
                            'Navigational Score': scores.get('Navigational', 0),
                            'Transactional Score': scores.get('Transactional', 0),
                            'Commercial Score': scores.get('Commercial', 0),
                            'Search Intent': data.get('search_intent', ''),
                            'Coherence Score': data.get('coherence_score', 0),
                            'Split Suggestion': data.get('split_suggestion', ''),
                            'SEO Insights': data.get('additional_info', '')
                        }
                        
                        # Add journey phase if it's available
                        if 'intent_flow' in data:
                            row['Journey Phase'] = data['intent_flow'].get('journey_phase', 'Unknown')
                            row['Intent Distribution'] = str(data['intent_flow'].get('intent_distribution', {}))
                        
                        intent_rows.append(row)
                
                if intent_rows:
                    intent_df = pd.DataFrame(intent_rows)
                    intent_df.to_excel(writer, sheet_name='Intent Analysis', index=False)
            
            # Tab 4+: One tab per cluster
            for cluster_id in df['cluster_id'].unique():
                cluster_df = df[df['cluster_id'] == cluster_id]
                sheet_name = f"Cluster {cluster_id}"
                
                #  Sort by volume if it's available
                if 'search_volume' in cluster_df.columns:
                    cluster_df = cluster_df.sort_values('search_volume', ascending=False)
                
                columns_to_include = ['keyword']
                if 'search_volume' in cluster_df.columns:
                    columns_to_include.append('search_volume')
                if 'competition' in cluster_df.columns:
                    columns_to_include.append('competition')
                if 'cpc' in cluster_df.columns:
                    columns_to_include.append('cpc')
                
                cluster_df[columns_to_include].to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel limita los nombres de hoja a 31 caracteres
        
        # Return buffer
        output.seek(0)
        return output
        
    except ImportError:
        st.error("Excel export requires openpyxl. Install with: pip install openpyxl")
        return None
        
def generate_json_report(df, cluster_evaluation=None):
    """Generate a complete JSON report with all data and analysis"""
    
    from datetime import datetime
    import json
    
    # Create dictionary for data
    report_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_keywords": len(df),
            "total_clusters": len(df['cluster_id'].unique()),
            "search_volume_available": 'search_volume' in df.columns,
            "ai_analysis_available": cluster_evaluation is not None and len(cluster_evaluation) > 0
        },
        "clusters": []
    }
    
    # Collect data for each cluster
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        cluster_data = {
            "cluster_id": int(cluster_id),
            "cluster_name": cluster_df['cluster_name'].iloc[0],
            "cluster_description": cluster_df['cluster_description'].iloc[0],
            "keyword_count": len(cluster_df),
            "coherence_score": float(cluster_df['cluster_coherence'].mean()),
            "keywords": []
        }
        
        # Add total volume if available
        if 'search_volume' in cluster_df.columns:
            cluster_data["total_search_volume"] = int(cluster_df['search_volume'].sum())
        
        # Add AI analysis data if available
        if cluster_evaluation and cluster_id in cluster_evaluation:
            ai_eval = cluster_evaluation[cluster_id]
            
            cluster_data["ai_analysis"] = {
                "search_intent": ai_eval.get('search_intent', ''),
                "split_suggestion": ai_eval.get('split_suggestion', ''),
                "additional_info": ai_eval.get('additional_info', ''),
                "coherence_score": ai_eval.get('coherence_score', 0)
            }
            
            # Add intent classification
            if 'intent_classification' in ai_eval:
                cluster_data["ai_analysis"]["intent_classification"] = ai_eval['intent_classification']
            
            # Add journey analysis
            if 'intent_flow' in ai_eval:
                cluster_data["ai_analysis"]["intent_flow"] = ai_eval['intent_flow']
            
            # Añadir análisis de trayectos
            if 'subclusters' in ai_eval:
                cluster_data["ai_analysis"]["subclusters"] = ai_eval['subclusters']
        
        # Add keywords (all or only the representative ones)
        representative_kws = cluster_df[cluster_df['representative'] == True]
        
        # Add representative keywords
        if not representative_kws.empty:
            cluster_data["representative_keywords"] = []
            
            for _, row in representative_kws.iterrows():
                kw_data = {"keyword": row['keyword']}
                
                # Add metrics if available
                if 'search_volume' in row:
                    kw_data["search_volume"] = int(row['search_volume']) if not pd.isna(row['search_volume']) else 0
                if 'competition' in row:
                    kw_data["competition"] = float(row['competition']) if not pd.isna(row['competition']) else 0
                if 'cpc' in row:
                    kw_data["cpc"] = float(row['cpc']) if not pd.isna(row['cpc']) else 0
                
                cluster_data["representative_keywords"].append(kw_data)
        
        # Add all keywords
        for _, row in cluster_df.iterrows():
            kw_data = {"keyword": row['keyword']}
            
            # Add metrics if available
            if 'search_volume' in row:
                kw_data["search_volume"] = int(row['search_volume']) if not pd.isna(row['search_volume']) else 0
            if 'competition' in row:
                kw_data["competition"] = float(row['competition']) if not pd.isna(row['competition']) else 0
            if 'cpc' in row:
                kw_data["cpc"] = float(row['cpc']) if not pd.isna(row['cpc']) else 0
            
            cluster_data["keywords"].append(kw_data)
        
        report_data["clusters"].append(cluster_data)
    
    return json.dumps(report_data, indent=2)

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
    "Danish", "Icelandic", "Lithuanian", "Greek", "Romanian",
    "Polish"
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
   - Either gpt-4.1-nano or gpt-4.1-turbo if you have an API key.
   - GPT-4.1-mini is generally more advanced (and more expensive).
    """)

num_clusters = st.sidebar.slider("Number of clusters", 2, 50, 10)
pca_variance = st.sidebar.slider("PCA explained variance (%)", 50, 99, 95)
max_pca_components = st.sidebar.slider("Max PCA components", 10, 300, 100)
min_df = st.sidebar.slider("Minimum term frequency", 1, 10, 1)
max_df = st.sidebar.slider("Maximum term frequency (%)", 50, 100, 95)
gpt_model = st.sidebar.selectbox("Model for naming clusters", ["gpt-4.1-nano", "gpt-4.1-mini"], index=0, help="Choose the GPT model for generating cluster names, descriptions, and performing analysis. GPT-4.1-mini is more capable but more expensive than GPT-4.1-nano.")

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
                
                # Get search intent from our enhanced classifier
                primary_intent = data.get('intent_classification', {}).get('primary_intent', 'Unknown')
                
                # Get search volume if available
                if 'search_volume' in df.columns:
                        df['search_volume'] = pd.to_numeric(df['search_volume'], errors='coerce')
                        search_volume = df[df['cluster_id'] == c_id]['search_volume'].sum()
                else:
                    search_volume = count * 100  # Default estimate for visualization scaling
                
                # Get journey phase if available
                if 'intent_flow' in data:
                    journey_phase = data['intent_flow'].get('journey_phase', 'Unknown')
                else:
                    journey_phase = 'Unknown'
                
                ai_coherence_data.append({
                    'cluster_id': c_id,
                    'cluster_name': cluster_name,
                    'coherence_score': coherence_score,
                    'count': count,
                    'search_volume': search_volume,
                    'primary_intent': primary_intent,
                    'journey_phase': journey_phase
                })
            
            if ai_coherence_data:
                ai_df = pd.DataFrame(ai_coherence_data)
                
                # Apply shortened labels for readability
                ai_df['short_name'] = ai_df['cluster_name'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
                ai_df['label'] = ai_df.apply(lambda x: f"{x['short_name']} (ID: {x['cluster_id']})", axis=1)
                
                # Color map for intent types
                intent_colors = {
                    'Informational': '#2196f3',
                    'Navigational': '#4caf50',
                    'Transactional': '#ff9800',
                    'Commercial': '#9c27b0',
                    'Mixed Intent': '#9e9e9e',
                    'Unknown': '#9e9e9e'
                }
                
                # Create tabs for different visualizations
                intent_viz_tabs = st.tabs(["Search Intent & Coherence", "Customer Journey", "Intent Distribution"])
                
                with intent_viz_tabs[0]:
                    st.subheader("Clusters by Search Intent & Coherence")
                    
                    ai_df['search_volume'] = pd.to_numeric(ai_df['search_volume'], errors='coerce')
                    
                    fig3 = px.scatter(
                        ai_df,
                        x='coherence_score',
                        y='count',
                        color='primary_intent',
                        size='search_volume',
                        hover_name='label',
                        labels={
                            'coherence_score': 'AI Coherence Score (0-10)',
                            'count': 'Number of Keywords',
                            'primary_intent': 'Search Intent',
                            'search_volume': 'Search Volume'
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
                    - **Bubble Size**: Proportional to estimated search volume
                    - **Color**: Represents the primary search intent of the cluster
                    
                    The most valuable clusters are typically those with high coherence scores (right side) and substantial keyword volume (upper area).
                    Clusters with low coherence might benefit from being split into more focused sub-clusters.
                    """)
                
                with intent_viz_tabs[1]:
                    st.subheader("Customer Journey Analysis")
                    
                    # Count clusters in each journey phase
                    phase_counts = Counter(ai_df['journey_phase'])
                    
                    # Create journey phase visualization
                    phase_order = [
                        "Early (Research Phase)", 
                        "Research-to-Consideration Transition",
                        "Middle (Consideration Phase)", 
                        "Consideration-to-Purchase Transition",
                        "Late (Purchase Phase)",
                        "Mixed Journey Stages",
                        "Unknown"
                    ]
                    
                    # Filter to only phases that exist in our data
                    phase_order = [phase for phase in phase_order if phase in phase_counts]
                    
                    phase_colors = {
                        "Early (Research Phase)": "#43a047",
                        "Research-to-Consideration Transition": "#26a69a",
                        "Middle (Consideration Phase)": "#1e88e5",
                        "Consideration-to-Purchase Transition": "#7b1fa2",
                        "Late (Purchase Phase)": "#ff9800",
                        "Mixed Journey Stages": "#757575",
                        "Unknown": "#9e9e9e"
                    }
                    
                    # Create dataframe for visualization
                    journey_df = pd.DataFrame({
                        'phase': list(phase_counts.keys()),
                        'count': list(phase_counts.values())
                    })
                    
                    # Order phases
                    journey_df['phase_order'] = journey_df['phase'].apply(
                        lambda x: phase_order.index(x) if x in phase_order else len(phase_order)
                    )
                    journey_df = journey_df.sort_values('phase_order')
                    
                    fig_journey = px.bar(
                        journey_df,
                        x='phase',
                        y='count',
                        color='phase',
                        labels={'phase': 'Customer Journey Phase', 'count': 'Number of Clusters'},
                        title='Distribution of Clusters Across Customer Journey Phases',
                        color_discrete_map={phase: color for phase, color in phase_colors.items() if phase in journey_df['phase'].values}
                    )
                    
                    st.plotly_chart(fig_journey, use_container_width=True)
                    
                    # Journey sankey diagram - shows flow from intent to journey phase
                    from collections import defaultdict
                    
                    # Create source-target pairs for Sankey
                    intent_to_phase = defaultdict(lambda: defaultdict(int))
                    for _, row in ai_df.iterrows():
                        intent_to_phase[row['primary_intent']][row['journey_phase']] += 1
                    
                    # Create Sankey data
                    source = []
                    target = []
                    value = []
                    
                    # Create node labels
                    intents = list(set(ai_df['primary_intent']))
                    phases = list(set(ai_df['journey_phase']))
                    
                    node_labels = intents + phases
                    
                    # Create source-target indices
                    for i, intent in enumerate(intents):
                        for phase in phases:
                            if intent_to_phase[intent][phase] > 0:
                                source.append(i)
                                target.append(len(intents) + phases.index(phase))
                                value.append(intent_to_phase[intent][phase])
                    
                    # Create color array matching node_labels
                    node_colors = []
                    for label in node_labels:
                        if label in intent_colors:
                            node_colors.append(intent_colors[label])
                        elif label in phase_colors:
                            node_colors.append(phase_colors[label])
                        else:
                            node_colors.append('#9e9e9e')  # Default gray
                    
                    # Create Sankey diagram
                        if source and target and value:  # Only if we have data
                            fig_sankey = go.Figure(data=[go.Sankey(
                                node=dict(
                                    pad=15,
                                    thickness=20,
                                    line=dict(color="black", width=0.5),
                                    label=node_labels,
                                    color=node_colors
                                ),
                                link=dict(
                                    source=source,
                                    target=target,
                                    value=value
                            )
                        )])
                        
                        fig_sankey.update_layout(
                            title_text="Flow from Search Intent to Customer Journey Phase",
                            font_size=12,
                            height=500
                        )
                        
                        st.plotly_chart(fig_sankey, use_container_width=True)
                    
                    st.markdown("""
                    **About the Customer Journey Analysis:**
                    
                    This analysis helps you understand where your keywords fit in the customer journey:
                    
                    - **Early (Research Phase)**: Users seeking information, learning about products/services
                    - **Middle (Consideration Phase)**: Users comparing options, reading reviews
                    - **Late (Purchase Phase)**: Users ready to make a purchase
                    - **Transition Phases**: Keywords that bridge multiple journey stages
                    
                    Mapping your content to these journey phases helps create targeted content that meets users where they are.
                    """)
                
                with intent_viz_tabs[2]:
                    st.subheader("Search Intent Distribution")
                    
                    # Create data for pie chart
                    intent_counts = Counter(ai_df['primary_intent'])
                    intent_df = pd.DataFrame({
                        'intent': list(intent_counts.keys()),
                        'count': list(intent_counts.values())
                    })
                    
                    # Create pie chart
                    fig_pie = px.pie(
                        intent_df,
                        names='intent',
                        values='count',
                        title='Distribution of Search Intent Across Clusters',
                        color='intent',
                        color_discrete_map=intent_colors
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Show intent counts by cluster
                    st.subheader("Intent Scores by Cluster")
                    
                    # Prepare data for stacked bar chart
                    intent_score_data = []
                    for c_id, data in eval_data.items():
                        cluster_name = df[df['cluster_id'] == c_id]['cluster_name'].iloc[0] if not df[df['cluster_id'] == c_id].empty else f"Cluster {c_id}"
                        scores = data.get('intent_classification', {}).get('scores', {})
                        
                        if scores:
                            intent_score_data.append({
                                'cluster_id': c_id,
                                'cluster_name': cluster_name,
                                'Informational': scores.get('Informational', 0),
                                'Navigational': scores.get('Navigational', 0),
                                'Transactional': scores.get('Transactional', 0),
                                'Commercial': scores.get('Commercial', 0)
                            })
                    
                    if intent_score_data:
                        intent_scores_df = pd.DataFrame(intent_score_data)
                        intent_scores_df['short_name'] = intent_scores_df['cluster_name'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)
                        intent_scores_df['label'] = intent_scores_df.apply(lambda x: f"{x['short_name']} (ID: {x['cluster_id']})", axis=1)
                        
                        fig_score = px.bar(
                            intent_scores_df,
                            x='label',
                            y=['Informational', 'Navigational', 'Transactional', 'Commercial'],
                            title='Search Intent Score Distribution by Cluster',
                            labels={'value': 'Intent Score (%)', 'label': 'Cluster', 'variable': 'Intent Type'},
                            color_discrete_map={
                                'Informational': intent_colors['Informational'],
                                'Navigational': intent_colors['Navigational'],
                                'Transactional': intent_colors['Transactional'],
                                'Commercial': intent_colors['Commercial']
                            }
                        )
                        
                        st.plotly_chart(fig_score, use_container_width=True)
    
    with st.expander("Explore Clusters", expanded=True):
        st.subheader("Explore Each Cluster")
        st.markdown("""
        Select a cluster to see details, search intent analysis, customer journey mapping, and potential sub-cluster suggestions.
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
                    
                    # Create tabs for different analysis views
                    analysis_tabs = st.tabs(["Search Intent", "Customer Journey", "Cluster Analysis", "SEO Insights"])
                    
                    # Tab 1: Search Intent Analysis
                    with analysis_tabs[0]:
                        st.subheader("Search Intent Analysis")
                        
                        # Get intent classification
                        intent_classification = ai_eval[cid].get('intent_classification', {})
                        primary_intent = intent_classification.get('primary_intent', 'Unknown')
                        scores = intent_classification.get('scores', {})
                        evidence = intent_classification.get('evidence', {})
                        
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
                        elif primary_intent == "Mixed Intent":
                            intent_class = "intent-mixed"
                        
                        # Display search intent with formatting
                        st.markdown(f"""
                        <div class="intent-box {intent_class}">
                            <strong>Primary Search Intent:</strong> {primary_intent}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show search intent description
                        st.write(f"**Search Intent Details:** {ai_eval[cid].get('search_intent', 'N/A')}")
                        
                        # Show evidence for the classification
                        if evidence and primary_intent in evidence and evidence[primary_intent]:
                            st.markdown("**Evidence for this classification:**")
                            evidence_list = "<ul class='evidence-list'>"
                            for e in evidence[primary_intent][:5]:  # Show top 5 pieces of evidence
                                evidence_list += f"<li>{e}</li>"
                            evidence_list += "</ul>"
                            st.markdown(evidence_list, unsafe_allow_html=True)
                        
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
                            
                        # Show keyword examples for each intent
                        st.markdown("### Example Keywords by Intent")
                        
                        # Classify individual keywords
                        keyword_examples = {}
                        for intent_type in ['Informational', 'Commercial', 'Transactional', 'Navigational']:
                            keyword_examples[intent_type] = []
                        
                        # Get a sample of keywords from the cluster
                        sample_keywords = cluster_df['keyword'].sample(min(20, len(cluster_df))).tolist()
                        for keyword in sample_keywords:
                            # Classify individual keyword
                            kw_intent = classify_search_intent_ml([keyword])
                            kw_primary = kw_intent['primary_intent']
                            if kw_primary in keyword_examples:
                                keyword_examples[kw_primary].append(keyword)
                        
                        # Display examples
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("<span class='info-tag'>Informational</span>", unsafe_allow_html=True)
                            if keyword_examples['Informational']:
                                for kw in keyword_examples['Informational'][:5]:
                                    st.markdown(f"<span class='keyword-example'>{kw}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("No clear examples found")
                                
                            st.markdown("<span class='commercial-tag'>Commercial</span>", unsafe_allow_html=True)
                            if keyword_examples['Commercial']:
                                for kw in keyword_examples['Commercial'][:5]:
                                    st.markdown(f"<span class='keyword-example'>{kw}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("No clear examples found")
                        
                        with col2:
                            st.markdown("<span class='transactional-tag'>Transactional</span>", unsafe_allow_html=True)
                            if keyword_examples['Transactional']:
                                for kw in keyword_examples['Transactional'][:5]:
                                    st.markdown(f"<span class='keyword-example'>{kw}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("No clear examples found")
                                
                            st.markdown("<span class='info-tag' style='background-color: #e8f5e9; color: #2e7d32;'>Navigational</span>", unsafe_allow_html=True)
                            if keyword_examples['Navigational']:
                                for kw in keyword_examples['Navigational'][:5]:
                                    st.markdown(f"<span class='keyword-example'>{kw}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("No clear examples found")
                    
                    # Tab 2: Customer Journey Analysis
                    with analysis_tabs[1]:
                        st.subheader("Customer Journey Analysis")
                        
                        # Get customer journey analysis
                        intent_flow = ai_eval[cid].get('intent_flow', None)
                        
                        if intent_flow:
                            # Get journey phase
                            journey_phase = intent_flow.get('journey_phase', 'Unknown')
                            
                            # Format journey phase display based on phase
                            journey_class = "journey-mixed"
                            if "Early" in journey_phase:
                                journey_class = "journey-early"
                            elif "Middle" in journey_phase:
                                journey_class = "journey-middle"
                            elif "Late" in journey_phase:
                                journey_class = "journey-late"
                            elif "Transition" in journey_phase:
                                journey_class = "journey-transition"
                            
                            # Display journey phase
                            st.markdown(f"""
                            <div class="{journey_class}">
                                <strong>Customer Journey Phase:</strong> {journey_phase}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display intent distribution
                            intent_dist = intent_flow.get('intent_distribution', {})
                            if intent_dist:
                                # Create pie chart of intent distribution
                                intent_dist_df = pd.DataFrame({
                                    'Intent': list(intent_dist.keys()),
                                    'Percentage': list(intent_dist.values())
                                })
                                
                                fig_dist = px.pie(
                                    intent_dist_df,
                                    names='Intent',
                                    values='Percentage',
                                    title='Keyword Intent Distribution in this Cluster',
                                    color='Intent',
                                    color_discrete_map={
                                        'Informational': '#2196f3',
                                        'Navigational': '#4caf50',
                                        'Transactional': '#ff9800',
                                        'Commercial': '#9c27b0',
                                        'Mixed Intent': '#9e9e9e',
                                        'Unknown': '#9e9e9e'
                                    }
                                )
                                
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Show keyword examples with intents
                            keyword_sample = intent_flow.get('keyword_sample', [])
                            if keyword_sample:
                                st.markdown("### Sample Keywords with Intent")
                                
                                sample_df = pd.DataFrame(keyword_sample)
                                st.dataframe(sample_df, use_container_width=True)
                            
                            st.markdown("""
                            ### Understanding Customer Journey
                            
                            The journey typically flows through these stages:
                            
                            1. **Research Phase** (Informational): Users are learning about solutions to their problems
                            2. **Consideration Phase** (Commercial): Users are comparing options and evaluating alternatives
                            3. **Purchase Phase** (Transactional): Users are ready to make a purchase
                            
                            Content should be created to match the journey phase of your target audience.
                            """)
                        else:
                            st.info("Customer journey analysis not available for this cluster.")
                    
                    # Tab 3: Cluster Analysis (Split Suggestions)
                    with analysis_tabs[2]:
                        st.subheader("Cluster Analysis")
                        
                        # Coherence Score
                        coherence_score = ai_eval[cid].get('coherence_score', 'N/A')
                        st.metric(label="AI Coherence Score (0-10)", value=coherence_score)
                        
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
                        st.markdown("**Full Split Analysis:**")
                        st.markdown(f"{split_suggestion}")
                        
                    
                    # Tab 4: SEO Insights
                    with analysis_tabs[3]:
                        st.subheader("SEO Insights & Opportunities")
                        
                        # Show SEO insights
                        additional_info = ai_eval[cid].get('additional_info', 'No additional information available')
                        
                        # Parse and highlight key SEO insights
                        # Look for common SEO patterns and highlight them
                        info_with_highlights = additional_info
                        
                        # Highlight SERP features
                        serp_features = [
                            'featured snippet', 'people also ask', 'knowledge panel', 
                            'local pack', 'image pack', 'video results', 'news results',
                            'top stories', 'recipes', 'shopping results', 'job listings'
                        ]
                        
                        for feature in serp_features:
                            if feature.lower() in info_with_highlights.lower():
                                info_with_highlights = re.sub(
                                    r'(?i)(' + re.escape(feature) + ')', 
                                    r'<span style="background-color: #e8f5e9; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\1</span>', 
                                    info_with_highlights
                                )
                        
                        # Highlight competitive insights
                        competitive_terms = [
                            'low competition', 'high competition', 'competitive', 
                            'search volume', 'traffic potential', 'keyword difficulty',
                            'ranking opportunity', 'low hanging fruit', 'long tail'
                        ]
                        
                        for term in competitive_terms:
                            if term.lower() in info_with_highlights.lower():
                                info_with_highlights = re.sub(
                                    r'(?i)(' + re.escape(term) + ')', 
                                    r'<span style="background-color: #fff3e0; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\1</span>', 
                                    info_with_highlights
                                )
                        
                        # Highlight content suggestions
                        content_terms = [
                            'content ideas', 'blog post', 'article', 'guide', 'comparison',
                            'review', 'tutorial', 'how-to', 'listicle', 'pillar page',
                            'topic cluster', 'content strategy'
                        ]
                        
                        for term in content_terms:
                            if term.lower() in info_with_highlights.lower():
                                info_with_highlights = re.sub(
                                    r'(?i)(' + re.escape(term) + ')', 
                                    r'<span style="background-color: #e3f2fd; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\1</span>', 
                                    info_with_highlights
                                )
                        
                        st.markdown(info_with_highlights, unsafe_allow_html=True)
                        
                        # Display search intent-based content recommendations
                        st.markdown("### Content Recommendations by Search Intent")
                        
                        primary_intent = ai_eval[cid].get('intent_classification', {}).get('primary_intent', 'Unknown')
                        
                        if primary_intent == "Informational":
                            st.markdown("""
                            **Recommended Content Types:**
                            - How-to guides and tutorials
                            - Explanatory articles and blog posts
                            - FAQ pages
                            - Infographics and visual explanations
                            - Educational videos
                            
                            **SEO Targets:**
                            - Featured snippets
                            - People Also Ask boxes
                            - Knowledge panels
                            - Video carousels (for YouTube content)
                            """)
                        elif primary_intent == "Commercial":
                            st.markdown("""
                            **Recommended Content Types:**
                            - Product comparisons
                            - Best-of lists
                            - Detailed reviews
                            - Buying guides
                            - Expert roundups and opinions
                            
                            **SEO Targets:**
                            - Rich results with star ratings
                            - Featured snippets for comparison tables
                            - Image packs for product visuals
                            """)
                        elif primary_intent == "Transactional":
                            st.markdown("""
                            **Recommended Content Types:**
                            - Product/service pages
                            - Pricing pages
                            - Special offers and deals
                            - Category pages
                            - Local landing pages (if applicable)
                            
                            **SEO Targets:**
                            - Shopping results
                            - Local packs (for local businesses)
                            - Site links
                            - Structured data for products
                            """)
                        elif primary_intent == "Navigational":
                            st.markdown("""
                            **Recommended Content Types:**
                            - Brand/service landing pages
                            - Contact and location pages
                            - Download/resource pages
                            - Login/account pages
                            
                            **SEO Targets:**
                            - Brand SERP features
                            - Site links
                            - Knowledge panels
                            - App install buttons (if applicable)
                            """)
                        else:
                            st.markdown("""
                            **Recommended Content Types:**
                            - Mix of informational and commercial content
                            - Content that addresses multiple user needs
                            - Topic hubs with different content types
                            
                            **SEO Targets:**
                            - Various SERP features depending on specific keywords
                            """)
            
            st.markdown("### All Keywords in this Cluster")
            if 'search_volume' in cluster_df.columns:
                # If search volume exists, show it
                st.dataframe(cluster_df[['keyword', 'search_volume']].sort_values(by='search_volume', ascending=False), use_container_width=True)
            else:
                st.dataframe(cluster_df[['keyword']], use_container_width=True)
    
with st.expander("Download Results", expanded=True):
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Basic Data Export")
        
        # CSV de datos básicos
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_data,
            file_name="semantic_clustered_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # CSV de resumen de clusters
        summary_df = df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
        summary_df.columns = ['ID', 'Name', 'Description', 'Number of Keywords']
        
        # Merge coherence
        coherence_df = df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id')
        summary_df.drop('cluster_id', axis=1, inplace=True)
        summary_df.rename(columns={'cluster_coherence': 'Coherence'}, inplace=True)
        
        # Add search volume if it exists
        if 'search_volume' in df.columns:
            volume_df = df.groupby('cluster_id')['search_volume'].sum().reset_index()
            summary_df = summary_df.merge(volume_df, left_on='ID', right_on='cluster_id')
            summary_df.drop('cluster_id', axis=1, inplace=True)
            summary_df.rename(columns={'search_volume': 'Total Search Volume'}, inplace=True)
        
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Clusters Summary (CSV)",
            data=csv_summary,
            file_name="semantic_clusters_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("### Complete Analysis Export")
        
        # Obtener evaluación de clusters si está disponible
        cluster_evaluation = st.session_state.cluster_evaluation if 'cluster_evaluation' in st.session_state else None
        
        # Exportar HTML interactivo
        if st.button("Generate HTML Report", use_container_width=True):
            with st.spinner("Generating HTML report..."):
                html_report = generate_html_report(df, cluster_evaluation)
                
                # Crear botón de descarga para el HTML
                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name="semantic_clustering_report.html",
                    mime="text/html",
                    use_container_width=True
                )
        
        # Exportar Excel
        if st.button("Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel report..."):
                try:
                    excel_buffer = generate_excel_report(df, cluster_evaluation)
                    if excel_buffer:
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_buffer,
                            file_name="semantic_clustering_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    else:
                        st.error("Excel export failed. Install openpyxl with: pip install openpyxl")
                except Exception as e:
                    st.error(f"Error generating Excel report: {str(e)}")
        
        # Exportar JSON
        if st.button("Generate JSON Data", use_container_width=True):
            with st.spinner("Generating JSON data..."):
                json_data = generate_json_report(df, cluster_evaluation)
                
                st.download_button(
                    label="Download JSON Data",
                    data=json_data,
                    file_name="semantic_clustering_data.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # PDF export (si está disponible)
    if pdf_export_available:
        st.markdown("### PDF Report")
        st.markdown("Generate a PDF report with visualizations, search intent analysis and cluster details.")
        
        # Código existente para PDF export
        cluster_evaluation = st.session_state.cluster_evaluation if 'cluster_evaluation' in st.session_state else None
        add_pdf_export_button(df, cluster_evaluation)
    else:
        st.warning("PDF export is not available. Make sure to install the additional requirements: reportlab, pillow and kaleido.")
        
 # Add PDF export button if available
    if pdf_export_available:
        st.markdown("---")
        st.markdown("### Export Complete Report")
        st.markdown("Generate a PDF report with visualizations, search intent analysis and cluster details.")
        
        cluster_evaluation = st.session_state.cluster_evaluation if 'cluster_evaluation' in st.session_state else None
        add_pdf_export_button(df, cluster_evaluation)
    else:
        st.warning("PDF export is not available. Make sure to install the additional requirements: reportlab, pillow and kaleido.")

        
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
            
            # Add journey phase if available
            def get_journey_phase(cid):
                if cid in st.session_state.cluster_evaluation and 'intent_flow' in st.session_state.cluster_evaluation[cid]:
                    return st.session_state.cluster_evaluation[cid]['intent_flow'].get('journey_phase', 'Unknown')
                return 'Unknown'
            
            summary_df['Customer Journey Phase'] = summary_df['ID'].apply(get_journey_phase)
        else:
            summary_df['AI Evaluation?'] = "No"
            summary_df['Primary Intent'] = "Unknown"
            summary_df['Customer Journey Phase'] = "Unknown"
        
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
    
    ### Customer Journey Mapping
    This tool helps you map keywords to the customer journey:
    
    1. **Research Phase**: Users seeking information (mostly informational keywords)
    2. **Consideration Phase**: Users comparing options (mostly commercial keywords)
    3. **Purchase Phase**: Users ready to buy (mostly transactional keywords)
    
    Understanding where your keywords fit in this journey helps create targeted content.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    Developed for advanced semantic keyword clustering – featuring intent analysis and customer journey mapping
</div>
""", unsafe_allow_html=True)
