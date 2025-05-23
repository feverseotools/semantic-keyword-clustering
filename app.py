import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import re
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import tempfile
import hashlib
import gc  
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, IncrementalPCA  
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from collections import Counter
import asyncio
from functools import lru_cache
import warnings

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize session state early
if 'memory_monitor' not in st.session_state:
    st.session_state.memory_monitor = {'last_check': time.time(), 'peak_memory': 0}

# Resource monitoring
def monitor_resources():
    """Monitor and log resource usage with improved thresholds"""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        if memory_mb > st.session_state.memory_monitor['peak_memory']:
            st.session_state.memory_monitor['peak_memory'] = memory_mb
        
        # Adjusted thresholds for Streamlit Cloud
        if memory_mb > 800:  # Warning at 800MB
            st.warning(f"⚠️ High memory usage: {memory_mb:.1f}MB")
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            
            # Suggest memory optimization
            if memory_mb > 900:  # Critical threshold
                st.error(f"🚨 Critical memory usage: {memory_mb:.1f}MB. Consider reducing dataset size.")
                # Trigger garbage collection
                import gc
                gc.collect()
        
        st.session_state.memory_monitor['last_check'] = time.time()
        
    except Exception as e:
        logger.error(f"Error monitoring resources: {str(e)}")

# Safe import with fallbacks and resource caching
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_optional_libraries():
    """Load optional libraries with proper error handling"""
    libraries = {
        'openai_available': False,
        'sentence_transformers_available': False,
        'spacy_base_available': False,
        'textblob_available': False,
        'hdbscan_available': False,
        'html_export_available': False,
        'excel_export_available': False,
        'pdf_export_available': False
    }
    
    # OpenAI
    try:
        import openai
        libraries['openai_available'] = True
        logger.info("OpenAI library loaded successfully")
    except ImportError:
        logger.warning("OpenAI library not available")
    
    # SentenceTransformers
    try:
        import sentence_transformers
        libraries['sentence_transformers_available'] = True
        logger.info("SentenceTransformers library loaded successfully")
    except ImportError:
        logger.warning("SentenceTransformers library not available")
    
    # spaCy
    try:
        import spacy
        libraries['spacy_base_available'] = True
        logger.info("spaCy library loaded successfully")
    except ImportError:
        logger.warning("spaCy library not available")
    
    # TextBlob
    try:
        import textblob
        libraries['textblob_available'] = True
        logger.info("TextBlob library loaded successfully")
    except ImportError:
        logger.warning("TextBlob library not available")
    
    # HDBSCAN
    try:
        import hdbscan
        libraries['hdbscan_available'] = True
        logger.info("HDBSCAN library loaded successfully")
    except ImportError:
        logger.warning("HDBSCAN library not available")
    
    # Export modules - Check if files exist instead of trying to import
    export_modules = {
        'html_export': 'html_export_available',
        'excel_export': 'excel_export_available', 
        'export_pdf': 'pdf_export_available'
    }
    
    for module_name, lib_key in export_modules.items():
        try:
            # Check if file exists first
            import os
            if os.path.exists(f"{module_name}.py"):
                __import__(module_name)
                libraries[lib_key] = True
                logger.info(f"{module_name} module loaded successfully")
            else:
                logger.warning(f"{module_name}.py file not found")
        except ImportError as e:
            logger.warning(f"{module_name} module not available: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading {module_name}: {str(e)}")
    
    return libraries

# Load libraries once
LIBRARIES = load_optional_libraries()

# Safe NLTK downloads with error handling
@st.cache_resource
def download_nltk_resources():
    """Download NLTK resources safely"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        logger.info("NLTK resources downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
        return False

# Download NLTK resources
NLTK_AVAILABLE = download_nltk_resources()

# Configuration constants
MAX_KEYWORDS = int(os.getenv('MAX_KEYWORDS', '25000'))
OPENAI_TIMEOUT = float(os.getenv('OPENAI_TIMEOUT', '60.0'))
OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '3'))
MAX_MEMORY_WARNING = int(os.getenv('MAX_MEMORY_MB', '800')) 
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))

################################################################
#          SEARCH INTENT CLASSIFICATION PATTERNS
################################################################

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
        "brand_indicators": True,
        "weight": 1.2
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
        "weight": 1.5
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
        "weight": 1.2
    }
}

################################################################
#          LANGUAGE MODEL MANAGEMENT
################################################################

SPACY_LANGUAGE_MODELS = {
    "English": "en_core_web_sm",
    "Spanish": "es_core_news_sm",
    "French": "fr_core_news_sm",
    "German": "de_core_news_sm",
    "Dutch": "nl_core_news_sm",
    "Italian": "it_core_news_sm",
    "Portuguese": "pt_core_news_sm",
    "Brazilian Portuguese": "pt_core_news_sm",
    "Swedish": "sv_core_news_sm",
    "Norwegian": "nb_core_news_sm",
    "Danish": "da_core_news_sm",
    "Greek": "el_core_news_sm",
    "Romanian": "ro_core_news_sm",
    "Polish": "pl_core_news_sm",
    "Korean": None,
    "Japanese": None,
    "Icelandic": None,
    "Lithuanian": None
}

@st.cache_resource(ttl=7200)  # Cache for 2 hours
def load_spacy_model_by_language(selected_language):
    """Try to load a spaCy model for the given language with caching"""
    if not LIBRARIES['spacy_base_available']:
        logger.warning("spaCy not available")
        return None

    model_name = SPACY_LANGUAGE_MODELS.get(selected_language, None)
    if model_name is None:
        logger.info(f"No spaCy model available for {selected_language}")
        return None

    try:
        import spacy
        model = spacy.load(model_name)
        logger.info(f"Loaded spaCy model {model_name} for {selected_language}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load spaCy model {model_name}: {str(e)}")
        return None

################################################################
#          COST CALCULATION AND SUPPORT FUNCTIONS
################################################################

@lru_cache(maxsize=128)
def calculate_api_cost(num_keywords, selected_model="gpt-4o-mini", num_clusters=10):
    """Calculate estimated OpenAI API costs with caching"""
    try:
        # Updated prices (December 2024) - Always check OpenAI's official pricing
        EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small per 1K tokens
        
        # GPT-4o-mini costs (more cost-effective)
        GPT4O_MINI_INPUT_COST_PER_1K = 0.15  # $0.15 per 1K input tokens
        GPT4O_MINI_OUTPUT_COST_PER_1K = 0.60  # $0.60 per 1K output tokens
        
        # GPT-4o costs
        GPT4O_INPUT_COST_PER_1K = 2.50  # $2.50 per 1K input tokens
        GPT4O_OUTPUT_COST_PER_1K = 10.00  # $10.00 per 1K output tokens
        
        results = {
            "embedding_cost": 0,
            "naming_cost": 0,
            "total_cost": 0,
            "processed_keywords": 0
        }
        
        # 1. Embedding cost (limited to 5000 keywords for memory efficiency)
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
        
        if selected_model == "gpt-4o-mini":
            input_cost = (estimated_input_tokens / 1000) * GPT4O_MINI_INPUT_COST_PER_1K
            output_cost = (estimated_output_tokens / 1000) * GPT4O_MINI_OUTPUT_COST_PER_1K
        else:  # GPT-4o
            input_cost = (estimated_input_tokens / 1000) * GPT4O_INPUT_COST_PER_1K
            output_cost = (estimated_output_tokens / 1000) * GPT4O_OUTPUT_COST_PER_1K
        
        results["naming_cost"] = input_cost + output_cost
        results["total_cost"] = results["embedding_cost"] + results["naming_cost"]
        
        logger.info(f"Calculated API cost for {num_keywords} keywords: ${results['total_cost']:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error calculating API cost: {str(e)}")
        return {"embedding_cost": 0, "naming_cost": 0, "total_cost": 0, "processed_keywords": 0}

def add_cost_calculator():
    """Add cost calculator with improved error handling"""
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
            step=500,
            help="Number of keywords to process"
        )
        calc_num_clusters = st.number_input(
            "Approx. number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            step=1,
            help="Expected number of clusters to generate"
        )
        calc_model = st.radio(
            "Model for naming clusters",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0,
            horizontal=True,
            help="gpt-4o-mini is more cost-effective"
        )
        
        if st.button("Calculate Estimated Cost", use_container_width=True):
            try:
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
            except Exception as e:
                st.error(f"Error calculating cost: {str(e)}")
                logger.error(f"Error in cost calculator: {str(e)}")

def show_csv_cost_estimate(num_keywords, selected_model="gpt-4o-mini", num_clusters=10):
    """Show cost estimate for current CSV with error handling"""
    if num_keywords > 0:
        try:
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
        except Exception as e:
            st.sidebar.error(f"Error calculating cost estimate: {str(e)}")
            logger.error(f"Error in CSV cost estimate: {str(e)}")

################################################################
#  SAMPLE CSV GENERATION
################################################################

@st.cache_data
def generate_sample_csv():
    """Generate sample CSV with caching"""
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
    """Enhanced preprocessing with proper error handling"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        if spacy_nlp is not None:  # We have a loaded spaCy model
            doc = spacy_nlp(text.lower())
            entities = [ent.text for ent in doc.ents if len(ent.text) > 1]
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
        
        elif LIBRARIES['textblob_available']:
            from textblob import TextBlob
            blob = TextBlob(text.lower())
            noun_phrases = list(blob.noun_phrases)
            
            if NLTK_AVAILABLE:
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
            else:
                stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
            
            words = [w for w in blob.words if len(w) > 1 and w.lower() not in stop_words]
            
            if use_lemmatization and NLTK_AVAILABLE:
                try:
                    lemmatizer = WordNetLemmatizer()
                    lemmas = [lemmatizer.lemmatize(w) for w in words]
                    processed_parts = lemmas + noun_phrases
                except:
                    processed_parts = words + noun_phrases
            else:
                processed_parts = words + noun_phrases
            
            return " ".join(processed_parts)
        
        else:
            # fallback to standard nltk
            return preprocess_text(text, use_lemmatization)
    
    except Exception as e:
        logger.warning(f"Error in enhanced preprocessing: {str(e)}")
        return text.lower() if isinstance(text, str) else ""

def preprocess_text(text, use_lemmatization=True):
    """Basic NLTK-based text preprocessing with error handling"""
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        text = text.lower()
        
        if NLTK_AVAILABLE:
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
        else:
            # Basic fallback without NLTK
            import string
            tokens = text.translate(str.maketrans('', '', string.punctuation)).split()
            stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
            tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        
        return " ".join(tokens)
    except Exception as e:
        logger.warning(f"Error in basic preprocessing: {str(e)}")
        return text.lower() if isinstance(text, str) else ""

def preprocess_keywords(keywords, use_advanced, spacy_nlp=None):
    """Main keyword preprocessing with progress tracking and error handling"""
    processed_keywords = []
    progress_bar = st.progress(0, text="Preprocessing keywords...")
    total = len(keywords)
    
    # Monitor memory during preprocessing
    monitor_resources()
    
    if use_advanced:
        if spacy_nlp is not None:
            st.success("✅ Using advanced preprocessing with spaCy for the selected language.")
        elif LIBRARIES['textblob_available']:
            st.success("✅ Using fallback preprocessing with TextBlob.")
        else:
            st.info("ℹ️ Using standard preprocessing with NLTK.")
    else:
        st.info("ℹ️ Using standard preprocessing with NLTK (advanced preprocessing disabled).")
    
    batch_size = BATCH_SIZE
    for i in range(0, len(keywords), batch_size):
        batch_end = min(i + batch_size, len(keywords))
        batch = keywords[i:batch_end]
        
        try:
            for j, keyword in enumerate(batch):
                if use_advanced and (spacy_nlp is not None or LIBRARIES['textblob_available']):
                    processed_keywords.append(enhanced_preprocessing(keyword, True, spacy_nlp))
                else:
                    processed_keywords.append(preprocess_text(keyword, True))
            
            # Update progress
            progress = min((i + len(batch)) / total, 1.0)
            progress_bar.progress(progress, text=f"Processed {i + len(batch)}/{total} keywords")
            
            # Memory check every 1000 keywords
            if (i + len(batch)) % 1000 == 0:
                monitor_resources()
                
        except Exception as e:
            logger.error(f"Error processing batch {i}-{batch_end}: {str(e)}")
            # Add fallback for failed batch
            for keyword in batch:
                processed_keywords.append(keyword.lower() if isinstance(keyword, str) else "")
    
    progress_bar.progress(1.0, text="✅ Preprocessing completed!")
    return processed_keywords

################################################################
#          EMBEDDING GENERATION
################################################################

@st.cache_resource(ttl=3600)  # Cache SentenceTransformer model for 1 hour
def load_sentence_transformer():
    """Load SentenceTransformer model with caching"""
    try:
        if LIBRARIES['sentence_transformers_available']:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SentenceTransformer model loaded successfully")
            return model
        else:
            logger.warning("SentenceTransformers not available")
            return None
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer: {str(e)}")
        return None

def create_openai_client_with_timeout(api_key):
    """Create OpenAI client with proper timeout and error handling"""
    try:
        if not LIBRARIES['openai_available']:
            return None
            
        # Use secrets management if available in Streamlit Cloud
        if api_key is None:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", None)
            except:
                pass
        
        if not api_key:
            logger.warning("No OpenAI API key provided")
            return None
            
        from openai import OpenAI
        
        # Config Timeout
        OPENAI_TIMEOUT = float(os.getenv('OPENAI_TIMEOUT', '60.0'))
        OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '3'))
        
        # Create client with configurable timeout settings
        client = OpenAI(
            api_key=api_key,
            timeout=OPENAI_TIMEOUT, 
            max_retries=OPENAI_MAX_RETRIES 
        )
        
# Test connection with minimal request
        try:
            client.models.list()
            logger.info("OpenAI client created and tested successfully")
            return client
        except Exception as e:
            try:
                from openai import OpenAIError, APIError, RateLimitError, AuthenticationError
                if isinstance(e, AuthenticationError):
                    logger.error(f"OpenAI authentication error: Invalid API key")
                elif isinstance(e, RateLimitError):
                    logger.error(f"OpenAI rate limit error: {str(e)}")
                elif isinstance(e, APIError):
                    logger.error(f"OpenAI API error: {str(e)}")
                elif isinstance(e, OpenAIError):
                    logger.error(f"OpenAI specific error: {str(e)}")
                else:
                    logger.error(f"Unexpected error testing OpenAI client: {str(e)}")
            except ImportError:
                logger.error(f"OpenAI client test failed: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error creating OpenAI client: {str(e)}")
        return None

def generate_embeddings_with_retry(df, openai_available, openai_api_key=None, max_retries=3):
    """Generate embeddings with retry logic and proper error handling"""
    st.info("🔄 Generating embeddings for keywords...")
    monitor_resources()
    
    # Attempt OpenAI embeddings first
    if openai_available and openai_api_key:
        client = create_openai_client_with_timeout(openai_api_key)
        if client:
            try:
                return generate_openai_embeddings(df, client, max_retries)
            except Exception as e:
                logger.error(f"OpenAI embeddings failed: {str(e)}")
                st.warning(f"⚠️ OpenAI embeddings failed: {str(e)}")
                st.info("🔄 Falling back to SentenceTransformers...")

    # Attempt SentenceTransformers
    sentence_model = load_sentence_transformer()
    if sentence_model:
        try:
            return generate_sentence_transformer_embeddings(df, sentence_model)
        except Exception as e:
            logger.error(f"SentenceTransformer embeddings failed: {str(e)}")
            st.warning(f"⚠️ SentenceTransformer embeddings failed: {str(e)}")
            st.info("🔄 Falling back to TF-IDF...")
    
    # Fallback to TF-IDF
    st.warning("⚠️ Using TF-IDF as a last resort (less semantic precision).")
    return generate_tfidf_embeddings(df['keyword_processed'].fillna(''))

def generate_openai_embeddings(df, client, max_retries=3):
    """Generate OpenAI embeddings with proper error handling and memory management"""
    st.info("🤖 Using OpenAI embeddings (high semantic precision).")
    keywords = df['keyword_processed'].fillna('').tolist()
    
    # Memory optimization: limit keywords
    if len(keywords) > 5000:
        st.warning(f"⚠️ Limiting to 5000 representative keywords out of {len(keywords)} total for memory optimization.")
        return generate_openai_embeddings_with_propagation(df, client, keywords, max_retries)
    else:
        return generate_openai_embeddings_direct(keywords, client, max_retries)

def generate_openai_embeddings_direct(keywords, client, max_retries):
    """Direct OpenAI embeddings for smaller datasets"""
    all_embeddings = []
    batch_size = BATCH_SIZE * 10
    
    progress_bar = st.progress(0, text="Requesting embeddings from OpenAI...")
    
    for i in range(0, len(keywords), batch_size):
        batch_end = min(i + batch_size, len(keywords))
        batch = keywords[i:batch_end]
        
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                try:
                    from openai import OpenAIError, APIError, RateLimitError, AuthenticationError
                    if isinstance(e, RateLimitError):
                        logger.warning(f"Rate limit hit on attempt {attempt + 1}, retrying in {2 ** attempt}s...")
                    elif isinstance(e, AuthenticationError):
                        logger.error(f"Authentication error: {str(e)}")
                        raise e  # No retry for auth errors
                    elif isinstance(e, APIError):
                        logger.warning(f"OpenAI API error on attempt {attempt + 1}: {str(e)}, retrying...")
                    elif isinstance(e, OpenAIError):
                        logger.warning(f"OpenAI specific error on attempt {attempt + 1}: {str(e)}, retrying...")
                    else:
                        logger.warning(f"Unexpected error on attempt {attempt + 1}: {str(e)}, retrying...")
                except ImportError:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                
                time.sleep(2 ** attempt)  # Exponential backoff
        
        progress = min(1.0, batch_end / len(keywords))
        progress_bar.progress(progress, text=f"Processed {batch_end}/{len(keywords)} keywords")
        
        # Memory monitoring
        monitor_resources()
    
    embeddings = np.array(all_embeddings)
    st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (OpenAI).")
    return embeddings

def generate_openai_embeddings_with_propagation(df, client, keywords, max_retries):
    """Generate embeddings with similarity propagation for large datasets"""
    step = max(1, len(keywords) // 5000)
    sample_indices = list(range(0, len(keywords), step))[:5000]
    sample_keywords = [keywords[i] for i in sample_indices]
    
    progress_bar = st.progress(0, text="Requesting embeddings from OpenAI...")
    
    # Get embeddings for sample
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=sample_keywords
            )
            sample_embeddings = np.array([item.embedding for item in response.data])
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # 🆕 Logging específico mejorado
            try:
                from openai import OpenAIError, APIError, RateLimitError, AuthenticationError
                if isinstance(e, RateLimitError):
                    logger.warning(f"Rate limit hit during embedding generation, attempt {attempt + 1}, retrying in {2 ** attempt}s...")
                elif isinstance(e, AuthenticationError):
                    logger.error(f"Authentication error during embeddings: {str(e)}")
                    raise e  # No retry for auth errors
                elif isinstance(e, APIError):
                    logger.warning(f"OpenAI API error during embeddings, attempt {attempt + 1}: {str(e)}, retrying...")
                elif isinstance(e, OpenAIError):
                    logger.warning(f"OpenAI specific error during embeddings, attempt {attempt + 1}: {str(e)}, retrying...")
                else:
                    logger.warning(f"Unexpected error during embeddings, attempt {attempt + 1}: {str(e)}, retrying...")
            except ImportError:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
            
            time.sleep(2 ** attempt)
    
    progress_bar.progress(0.5, text="Propagating embeddings via TF-IDF similarity...")
    
    # Propagate to remaining keywords
    vectorizer = TfidfVectorizer(max_features=1000)  # Reduced features for memory
    tfidf_matrix = vectorizer.fit_transform(keywords)
    
    all_embeddings = np.zeros((len(keywords), len(sample_embeddings[0])))
    for i, idx in enumerate(sample_indices):
        all_embeddings[idx] = sample_embeddings[i]
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(3, len(sample_indices)))
    nn.fit(tfidf_matrix[sample_indices])
    
    remaining_indices = [i for i in range(len(keywords)) if i not in sample_indices]
    
    batch_size = 1000
    for batch_start in range(0, len(remaining_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_indices))
        batch_indices = remaining_indices[batch_start:batch_end]
        
        for local_i, idx in enumerate(batch_indices):
            distances, neighbors = nn.kneighbors(tfidf_matrix[idx:idx+1])
            weights = 1.0 / (1.0 + distances[0])
            weights = weights / weights.sum()
            
            weighted_embedding = np.zeros_like(sample_embeddings[0])
            for j, weight in zip(neighbors[0], weights):
                similar_idx = sample_indices[j]
                weighted_embedding += weight * all_embeddings[similar_idx]
            
            all_embeddings[idx] = weighted_embedding
        
        progress = 0.5 + min(0.5, (batch_end / len(remaining_indices) * 0.5))
        progress_bar.progress(progress, text=f"Propagated to {batch_end}/{len(remaining_indices)} remaining keywords")
        
        # Memory check
        if batch_start % 5000 == 0:
            monitor_resources()
    
    progress_bar.progress(1.0, text="✅ OpenAI embeddings completed!")
    st.success(f"✅ Generated embeddings with {all_embeddings.shape[1]} dimensions (OpenAI + propagation).")
    return all_embeddings

def generate_sentence_transformer_embeddings(df, model):
    """Generate SentenceTransformer embeddings with memory optimization"""
    st.success("🤖 Using SentenceTransformer (free alternative).")
    
    keywords = df['keyword_processed'].fillna('').tolist()
    batch_size = BATCH_SIZE * 5
    all_embeddings = []
    
    progress_bar = st.progress(0, text="Generating semantic embeddings...")
    
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        try:
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error in SentenceTransformer batch {i}: {str(e)}")
            # Fallback for problematic batch
            fallback_embeddings = np.random.rand(len(batch), 384)  # Default dimension
            all_embeddings.extend(fallback_embeddings)
        
        progress = min(1.0, (i + batch_size) / len(keywords))
        progress_bar.progress(progress, text=f"Processed {min(i + batch_size, len(keywords))}/{len(keywords)} keywords")
        
        # Memory monitoring
        if i % 2000 == 0:
            monitor_resources()
    
    embeddings = np.array(all_embeddings)
    progress_bar.progress(1.0, text="✅ SentenceTransformer embeddings completed!")
    st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (SentenceTransformers).")
    return embeddings

def generate_tfidf_embeddings(texts, min_df=1, max_df=0.95):
    """Generate TF-IDF embeddings with improved error handling"""
    st.info("📊 Generating TF-IDF vectors for keywords...")
    progress_bar = st.progress(0, text="Initializing TF-IDF vectorizer...")
    
    try:
        # Optimized parameters for memory efficiency
        vectorizer = TfidfVectorizer(
            max_features=300,  # Reduced for memory
            min_df=min_df,
            max_df=max_df,
            stop_words='english' if NLTK_AVAILABLE else None
        )
        
        clean_texts = [t if isinstance(t, str) and t.strip() else "unknown" for t in texts]
        
        progress_bar.progress(0.3, text="Fitting TF-IDF vectorizer...")
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
        
        progress_bar.progress(0.8, text="Converting to dense array...")
        embeddings = tfidf_matrix.toarray()
        
        progress_bar.progress(1.0, text="✅ TF-IDF embeddings completed!")
        st.success(f"✅ Generated {embeddings.shape[1]} TF-IDF features.")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating TF-IDF embeddings: {str(e)}")
        st.warning("⚠️ TF-IDF failed, generating random vectors as fallback.")
        # Emergency fallback
        random_embeddings = np.random.rand(len(texts), 100)
        return random_embeddings

def generate_embeddings(df, openai_available, openai_api_key=None):
    """Wrapper function for backward compatibility"""
    return generate_embeddings_with_retry(df, openai_available, openai_api_key)

################################################################
#          CLUSTERING ALGORITHMS
################################################################

def improved_clustering_with_monitoring(embeddings, num_clusters=None, min_cluster_size=5):
    """Apply clustering with resource monitoring"""
    st.info("🔄 Applying advanced clustering algorithms...")
    monitor_resources()
    
    try:
        from sklearn.cluster import KMeans
        if num_clusters is None:
            num_clusters = min(10, max(2, len(embeddings) // 100))  # Adaptive cluster count
        
        # Use mini-batch KMeans for large datasets
        if len(embeddings) > MAX_KEYWORDS // 2.5:
            from sklearn.cluster import MiniBatchKMeans
            st.info(f"Using MiniBatch KMeans for {len(embeddings)} samples (memory optimization)")
            kmeans = MiniBatchKMeans(
                n_clusters=num_clusters, 
                random_state=42, 
                batch_size=BATCH_SIZE * 10,
                max_iter=100
            )
        else:
            kmeans = KMeans(
                n_clusters=num_clusters, 
                random_state=42, 
                n_init=10,
                max_iter=300
            )
        
        with st.spinner(f"Clustering {len(embeddings)} keywords into {num_clusters} groups..."):
            labels = kmeans.fit_predict(embeddings)
        
        # Convert to 1-based indexing
        labels = labels + 1
        
        st.success(f"✅ Successfully created {len(np.unique(labels))} clusters")
        return labels
        
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        st.warning(f"⚠️ Clustering failed: {str(e)}, using random assignment")
        # Emergency fallback
        num_clusters = num_clusters or 10
        return np.random.randint(1, num_clusters + 1, size=len(embeddings))

def refine_clusters_with_monitoring(df, embeddings, original_cluster_column='cluster_id'):
    """Refine clusters with resource monitoring"""
    st.info("🔧 Refining clusters to improve coherence...")
    monitor_resources()
    
    try:
        # Simple outlier detection based on distance to centroid
        unique_clusters = df[original_cluster_column].unique()
        outlier_threshold = 2.0  # Standard deviations
        
        for cluster_id in unique_clusters:
            cluster_mask = df[original_cluster_column] == cluster_id
            cluster_indices = df[cluster_mask].index.tolist()
            
            if len(cluster_indices) > 3:  # Only refine clusters with enough members
                cluster_embeddings = embeddings[cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distances to centroid
                distances = [np.linalg.norm(embeddings[idx] - centroid) for idx in cluster_indices]
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                
                # Mark outliers
                outlier_threshold_dist = mean_dist + outlier_threshold * std_dist
                for i, idx in enumerate(cluster_indices):
                    if distances[i] > outlier_threshold_dist:
                        # Reassign outlier to nearest cluster
                        min_dist = float('inf')
                        best_cluster = cluster_id
                        
                        for other_cluster in unique_clusters:
                            if other_cluster != cluster_id:
                                other_mask = df[original_cluster_column] == other_cluster
                                other_indices = df[other_mask].index.tolist()
                                if other_indices:
                                    other_centroid = np.mean(embeddings[other_indices], axis=0)
                                    dist = np.linalg.norm(embeddings[idx] - other_centroid)
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_cluster = other_cluster
                        
                        df.loc[idx, original_cluster_column] = best_cluster
        
        st.success("✅ Cluster refinement completed")
        return df
        
    except Exception as e:
        logger.error(f"Error in cluster refinement: {str(e)}")
        st.warning("⚠️ Cluster refinement failed, proceeding with original clusters")
        return df

################################################################
#          GENERATE CLUSTER NAMES
################################################################

def generate_cluster_names_with_retry(
    clusters_with_representatives, 
    client, 
    model="gpt-4o-mini",
    custom_prompt=None,
    max_retries=3
):
    """Generate cluster names with proper retry logic and error handling"""
    if not clusters_with_representatives:
        return {}

    results = {}
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("🤖 Generating SEO-friendly cluster names and descriptions...")

    if not custom_prompt:
        custom_prompt = (
            "You are an expert in SEO and content marketing. Below you'll see several clusters "
            "with a list of representative keywords. Your task is to assign each cluster a short, "
            "clear name (3-6 words) and write a concise SEO meta description (1 or 2 sentences), "
            "briefly explaining the topic and likely search intent."
        )

    cluster_ids = list(clusters_with_representatives.keys())
    batch_size = 2  # Smaller batch size for better reliability
    
    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(cluster_ids))
        batch_cluster_ids = cluster_ids[batch_start:batch_end]
        
        batch_prompt = custom_prompt.strip() + "\n\n"
        batch_prompt += (
            "FOR EACH CLUSTER, provide:\n"
            "1. A clear, concise name (3-6 words)\n"
            "2. A brief description (1-2 sentences)\n\n"
            "FORMAT YOUR RESPONSE AS VALID JSON:\n\n"
            "{\n"
            '  "clusters": [\n'
            "    {\n"
            '      "cluster_id": 1,\n'
            '      "cluster_name": "Example Cluster Name",\n'
            '      "cluster_description": "Example description of what this cluster represents."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Here are the clusters:\n"
        )
        
        for cluster_id in batch_cluster_ids:
            sample_kws = clusters_with_representatives[cluster_id][:8]  # Reduced sample
            batch_prompt += f"- Cluster {cluster_id}: {', '.join(sample_kws)}\n"
        
        batch_results = {}
        
        for attempt in range(max_retries):
            try:
                progress_text.text(f"🤖 Generating names for clusters {batch_start+1}-{batch_end} (attempt {attempt+1}/{max_retries})")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.2,
                    max_tokens=2000,
                    timeout=30  # 30 second timeout per request
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from markdown if present
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                json_matches = re.findall(json_pattern, content)
                
                if json_matches:
                    content = json_matches[0]
                
                # Parse JSON
                json_data = json.loads(content)
                
                if "clusters" in json_data and isinstance(json_data["clusters"], list):
                    for item in json_data["clusters"]:
                        c_id = item.get("cluster_id")
                        if c_id is not None:
                            try:
                                c_id = int(c_id)
                                c_name = item.get("cluster_name", f"Cluster {c_id}")
                                c_desc = item.get("cluster_description", "No description provided")
                                
                                # Sanitize names and descriptions
                                c_name = sanitize_text(c_name)
                                c_desc = sanitize_text(c_desc)
                                
                                batch_results[c_id] = (c_name, c_desc)
                            except (ValueError, TypeError):
                                continue
                    
                    if batch_results:
                        break
                        
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error in attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    # Fallback to generic names
                    for cluster_id in batch_cluster_ids:
                        if cluster_id not in batch_results:
                            kws = clusters_with_representatives[cluster_id][:3]
                            c_name = f"{kws[0]} Related"
                            c_desc = f"A collection of keywords related to {', '.join(kws[:2])}"
                            batch_results[cluster_id] = (c_name, c_desc)
                            
            except Exception as e:
                try:
                    from openai import OpenAIError, APIError, RateLimitError, AuthenticationError
                    if isinstance(e, RateLimitError):
                        logger.warning(f"Rate limit during cluster naming, attempt {attempt+1}")
                    elif isinstance(e, AuthenticationError):
                        logger.error(f"Authentication error during cluster naming: {str(e)}")
                        break  # No point retrying auth errors
                    elif isinstance(e, APIError):
                        logger.warning(f"OpenAI API error during cluster naming, attempt {attempt+1}: {str(e)}")
                    elif isinstance(e, OpenAIError):
                        logger.warning(f"OpenAI specific error during cluster naming, attempt {attempt+1}: {str(e)}")
                    else:
                        logger.warning(f"Unexpected error during cluster naming, attempt {attempt+1}: {str(e)}")
                except ImportError:
                    logger.warning(f"API error in attempt {attempt+1}: {str(e)}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Final fallback
                    for cluster_id in batch_cluster_ids:
                        if cluster_id not in batch_results:
                            kws = clusters_with_representatives[cluster_id][:2]
                            c_name = " ".join(kws) if len(kws) > 1 else kws[0]
                            c_desc = f"A group of related keywords (cluster {cluster_id})"
                            batch_results[cluster_id] = (c_name, c_desc)
        
        results.update(batch_results)
        progress_bar.progress(min(1.0, batch_end / len(cluster_ids)))
    
    # Ensure all clusters have names
    for c_id in clusters_with_representatives.keys():
        if c_id not in results:
            results[c_id] = (f"Cluster {c_id}", f"Keyword group {c_id}")

    progress_bar.progress(1.0)
    progress_text.text("✅ Cluster naming completed successfully!")
    return results

def sanitize_text(text):
    """Sanitize text to prevent potential security issues"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove potential malicious content
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s\-.,!?()]+', '', text)  # Keep only safe characters
    text = text.strip()
    
    return text[:200]  # Limit length

################################################################
#          SEARCH INTENT CLASSIFICATION
################################################################

def extract_features_for_intent(keyword, search_intent_description=""):
    """Extract features for search intent classification with improved error handling"""
    try:
        features = {
            "keyword_length": len(keyword.split()) if isinstance(keyword, str) else 0,
            "keyword_lower": keyword.lower() if isinstance(keyword, str) else "",
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
            "modal_verbs": False
        }
        
        if not isinstance(keyword, str) or not keyword.strip():
            return features
        
        keyword_lower = keyword.lower()
        words = keyword_lower.split()
        
        if not words:
            return features
        
        # Check prefixes
        first_word = words[0]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if any(first_word == prefix.lower() for prefix in patterns["prefixes"]):
                features[f"has_{intent_type.lower()}_prefix"] = True
        
        # Check suffixes
        if len(words) > 1:
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
                try:
                    if re.search(pattern, keyword_lower):
                        match_count += 1
                except re.error:
                    continue
            features[f"{intent_type.lower()}_pattern_matches"] = match_count
        
        # Additional features
        features["local_intent"] = any(term in keyword_lower for term in ["near me", "nearby", "in my area", "close to me", "closest", "local"])
        features["modal_verbs"] = any(modal in keyword_lower.split() for modal in ["can", "could", "should", "would", "will", "may", "might"])
        features["includes_price_modifier"] = any(term in keyword_lower for term in ["price", "cost", "cheap", "expensive", "affordable", "discount", "offer", "deal", "coupon"])
        features["includes_product_modifier"] = any(term in keyword_lower for term in ["best", "top", "cheap", "premium", "quality", "new", "used", "refurbished", "alternative"])
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features for keyword '{keyword}': {str(e)}")
        return features

def classify_search_intent_ml(keywords, search_intent_description="", cluster_name=""):
    """Enhanced search intent classification with proper error handling"""
    try:
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
        
        # Process limited number of keywords for performance
        sample_keywords = keywords[:min(len(keywords), 15)]
        all_features = []
        
        for keyword in sample_keywords:
            features = extract_features_for_intent(keyword, search_intent_description)
            all_features.append(features)
        
        # Aggregate signals
        signal_collections = {
            "informational": set(),
            "navigational": set(),
            "transactional": set(),
            "commercial": set()
        }
        
        for features in all_features:
            # Collect evidence for each intent type
            for intent_type in signal_collections.keys():
                if features.get(f"has_{intent_type}_prefix", False):
                    signal_collections[intent_type].add(f"Has {intent_type} prefix")
                if features.get(f"has_{intent_type}_suffix", False):
                    signal_collections[intent_type].add(f"Has {intent_type} suffix")
                if features.get(f"is_{intent_type}_exact_match", False):
                    signal_collections[intent_type].add(f"Contains {intent_type} phrase")
                
                pattern_matches = features.get(f"{intent_type}_pattern_matches", 0)
                if pattern_matches > 0:
                    signal_collections[intent_type].add(f"Matches {pattern_matches} {intent_type} patterns")
            
            # Special signals
            if features.get("modal_verbs", False):
                signal_collections["informational"].add("Contains question-like modal verb")
            if features.get("includes_price_modifier", False):
                signal_collections["transactional"].add("Includes price-related term")
            if features.get("local_intent", False):
                signal_collections["transactional"].add("Shows local intent")
            if features.get("includes_product_modifier", False):
                signal_collections["commercial"].add("Includes product comparison term")
        
        # Calculate weighted scores
        weights = {intent: SEARCH_INTENT_PATTERNS[intent.title()]["weight"] for intent in signal_collections.keys()}
        
        scores = {}
        for intent, signals in signal_collections.items():
            scores[intent] = len(signals) * weights[intent]
        
        # Add context bonuses
        if search_intent_description:
            desc_lower = search_intent_description.lower()
            for intent in scores.keys():
                if intent in desc_lower:
                    scores[intent] += 3
        
        if cluster_name:
            name_lower = cluster_name.lower()
            for intent in scores.keys():
                if intent in name_lower:
                    scores[intent] += 2
        
        # Normalize to percentages
        total_score = max(1, sum(scores.values()))
        normalized_scores = {
            "Informational": (scores["informational"] / total_score) * 100,
            "Navigational": (scores["navigational"] / total_score) * 100,
            "Transactional": (scores["transactional"] / total_score) * 100,
            "Commercial": (scores["commercial"] / total_score) * 100
        }
        
        # Determine primary intent
        primary_intent = max(normalized_scores, key=normalized_scores.get)
        max_score = max(normalized_scores.values())
        
        # Check for mixed intent
        if max_score < 30:
            sorted_scores = sorted(normalized_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1] < 10):
                primary_intent = "Mixed Intent"
        
        # Prepare evidence
        evidence = {
            "Informational": list(signal_collections["informational"]),
            "Navigational": list(signal_collections["navigational"]),
            "Transactional": list(signal_collections["transactional"]),
            "Commercial": list(signal_collections["commercial"])
        }
        
        return {
            "primary_intent": primary_intent,
            "scores": normalized_scores,
            "evidence": evidence
        }
        
    except Exception as e:
        logger.error(f"Error in search intent classification: {str(e)}")
        # Return default classification
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

def analyze_cluster_for_intent_flow(df, cluster_id):
    """Analyze customer journey flow for cluster with error handling"""
    try:
        cluster_keywords = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
        
        if not cluster_keywords:
            return None
        
        # Limit keywords for performance
        sample_keywords = cluster_keywords[:min(len(cluster_keywords), 20)]
        keyword_intents = []
        
        for keyword in sample_keywords:
            try:
                intent_data = classify_search_intent_ml([keyword])
                keyword_intents.append({
                    "keyword": keyword,
                    "primary_intent": intent_data["primary_intent"],
                    "scores": intent_data["scores"]
                })
            except Exception as e:
                logger.warning(f"Error classifying intent for keyword '{keyword}': {str(e)}")
                continue
        
        if not keyword_intents:
            return None
        
        # Calculate intent distribution
        intent_counts = Counter([item["primary_intent"] for item in keyword_intents])
        total = len(keyword_intents)
        
        # Calculate average scores
        avg_scores = {
            "Informational": sum(item["scores"]["Informational"] for item in keyword_intents) / total,
            "Navigational": sum(item["scores"]["Navigational"] for item in keyword_intents) / total,
            "Transactional": sum(item["scores"]["Transactional"] for item in keyword_intents) / total,
            "Commercial": sum(item["scores"]["Commercial"] for item in keyword_intents) / total
        }
        
        # Determine journey phase
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
        
    except Exception as e:
        logger.error(f"Error analyzing intent flow for cluster {cluster_id}: {str(e)}")
        return None

################################################################
#          CLUSTER SEMANTIC ANALYSIS
################################################################

def generate_semantic_analysis_with_retry(
    clusters_with_representatives,
    client,
    model="gpt-4o-mini",
    max_retries=3
):
    """Generate semantic analysis with proper error handling and retry logic"""
    results = {}
    if not clusters_with_representatives:
        return results

    if not client:
        st.warning("⚠️ No valid OpenAI client provided. Using default values.")
        return create_default_semantic_analysis(clusters_with_representatives)

    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("🔍 Performing semantic analysis on clusters...")
    
    cluster_ids = list(clusters_with_representatives.keys())
    batch_size = 3  # Small batch size for reliability
    
    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(cluster_ids))
        batch_cluster_ids = cluster_ids[batch_start:batch_end]
        
        batch_prompt = (
            "You are an expert in SEO and clustering analysis. Analyze each keyword cluster below:\n"
            "1) Search intent: Describe why users would search these terms\n"
            "2) Split suggestion: Yes/No and if yes, suggest 2-3 subclusters\n"
            "3) SEO insights: Keyword difficulty, content ideas, etc.\n"
            "4) Coherence score: 0-10 where 10 means perfectly coherent\n\n"
            "Respond with valid JSON:\n"
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
            "Clusters to analyze:\n"
        )
        
        for cluster_id in batch_cluster_ids:
            sample_kws = clusters_with_representatives[cluster_id][:8]  # Reduced sample
            batch_prompt += f"Cluster {cluster_id}: {', '.join(sample_kws)}\n"
        
        batch_results = {}
        
        for attempt in range(max_retries):
            try:
                progress_text.text(f"🔍 Analyzing clusters {batch_start+1}-{batch_end} (attempt {attempt+1}/{max_retries})")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                    timeout=45  # 45 second timeout
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                json_matches = re.findall(json_pattern, content)
                
                if json_matches:
                    content = json_matches[0]
                
                json_data = json.loads(content)
                
                if "clusters" in json_data and isinstance(json_data["clusters"], list):
                    for item in json_data["clusters"]:
                        c_id = item.get("cluster_id")
                        if c_id is not None:
                            try:
                                c_id = int(c_id)
                                if c_id not in clusters_with_representatives:
                                    continue
                                
                                search_intent = sanitize_text(item.get("search_intent", ""))
                                split_suggestion = sanitize_text(item.get("split_suggestion", ""))
                                additional_info = sanitize_text(item.get("additional_info", ""))
                                coherence_score = max(0, min(10, int(item.get("coherence_score", 5))))
                                subclusters = item.get("subclusters", [])
                                
                                # Use enhanced ML classifier
                                cluster_name = f"Cluster {c_id}"
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
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error processing cluster {c_id}: {str(e)}")
                                continue
                    
                    if batch_results:
                        break
                        
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error in semantic analysis attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    # Create fallback results
                    for cluster_id in batch_cluster_ids:
                        if cluster_id not in batch_results:
                            batch_results[cluster_id] = create_default_cluster_analysis(
                                cluster_id, clusters_with_representatives.get(cluster_id, [])
                            )
                            
            except Exception as e:
                logger.warning(f"API error in semantic analysis attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    # Final fallback
                    for cluster_id in batch_cluster_ids:
                        if cluster_id not in batch_results:
                            batch_results[cluster_id] = create_default_cluster_analysis(
                                cluster_id, clusters_with_representatives.get(cluster_id, [])
                            )
        
        results.update(batch_results)
        progress_bar.progress(min(1.0, batch_end / len(cluster_ids)))
    
    # Ensure all clusters have analysis
    for c_id in clusters_with_representatives.keys():
        if c_id not in results:
            results[c_id] = create_default_cluster_analysis(
                c_id, clusters_with_representatives.get(c_id, [])
            )

    progress_bar.progress(1.0)
    progress_text.text("✅ Semantic analysis completed successfully!")
    return results

def create_default_semantic_analysis(clusters_with_representatives):
    """Create default semantic analysis when API is unavailable"""
    results = {}
    for c_id, keywords in clusters_with_representatives.items():
        results[c_id] = create_default_cluster_analysis(c_id, keywords)
    return results

def create_default_cluster_analysis(cluster_id, keywords):
    """Create default analysis for a single cluster"""
    try:
        intent_classification = classify_search_intent_ml(
            keywords,
            "No search intent data available",
            f"Cluster {cluster_id}"
        )
        
        return {
            "search_intent": f"Keywords related to {', '.join(keywords[:3])}",
            "split_suggestion": "Unable to determine without AI analysis",
            "additional_info": "No SEO information available - API analysis failed",
            "coherence_score": 5,
            "subclusters": [],
            "intent_classification": intent_classification
        }
    except Exception as e:
        logger.error(f"Error creating default analysis for cluster {cluster_id}: {str(e)}")
        return {
            "search_intent": "Analysis unavailable",
            "split_suggestion": "No suggestion available",
            "additional_info": "Analysis failed",
            "coherence_score": 5,
            "subclusters": [],
            "intent_classification": {
                "primary_intent": "Unknown",
                "scores": {"Informational": 25, "Navigational": 25, "Transactional": 25, "Commercial": 25},
                "evidence": {}
            }
        }

################################################################
#          EVALUATION FUNCTIONS
################################################################

def evaluate_cluster_quality_with_monitoring(df, embeddings, cluster_column='cluster_id'):
    """Improved cluster quality evaluation with resource monitoring"""
    st.subheader("📊 Cluster Quality Evaluation")
    monitor_resources()
    
    try:
        df['cluster_coherence'] = 1.0  # Default value
        unique_clusters = df[cluster_column].unique()
        
        with st.spinner("Calculating cluster coherence scores..."):
            progress_bar = st.progress(0)
            
            for i, cluster_id in enumerate(unique_clusters):
                try:
                    cluster_indices = df[df[cluster_column] == cluster_id].index.tolist()
                    
                    if len(cluster_indices) > 1:
                        cluster_embeddings = embeddings[cluster_indices]
                        coherence = calculate_cluster_coherence_safe(cluster_embeddings)
                        df.loc[cluster_indices, 'cluster_coherence'] = coherence
                
                except Exception as e:
                    logger.warning(f"Error calculating coherence for cluster {cluster_id}: {str(e)}")
                    # Keep default value
                
                progress_bar.progress((i + 1) / len(unique_clusters))
            
            progress_bar.progress(1.0)
        
        st.success(f"✅ Coherence scores calculated for {len(unique_clusters)} clusters.")
        return df
        
    except Exception as e:
        logger.error(f"Error in cluster quality evaluation: {str(e)}")
        st.warning("⚠️ Using default coherence values due to calculation error")
        df['cluster_coherence'] = 1.0
        return df

def calculate_cluster_coherence_safe(cluster_embeddings):
    """Calculate coherence with improved error handling"""
    try:
        if len(cluster_embeddings) < 2:
            return 1.0
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        else:
            return 1.0
        
        # Calculate similarities
        similarities = []
        for embedding in cluster_embeddings:
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding = embedding / emb_norm
                similarity = np.dot(embedding, centroid)
                similarities.append(max(0, min(1, similarity)))  # Clamp to [0,1]
            else:
                similarities.append(0.5)  # Neutral similarity for zero vectors
        
        if not similarities:
            return 1.0
        
        coherence = np.mean(similarities)
        return max(0.0, min(1.0, coherence))
        
    except Exception as e:
        logger.warning(f"Error calculating coherence: {str(e)}")
        return 1.0

def evaluate_and_refine_clusters_with_monitoring(df, client, model="gpt-4o-mini"):
    """AI-powered cluster analysis with proper monitoring"""
    st.subheader("🤖 AI-Powered Cluster Quality Evaluation")
    monitor_resources()

    if not client:
        st.info("ℹ️ No OpenAI client available. Skipping AI-based cluster analysis.")
        return create_default_semantic_analysis({})

    try:
        # Build cluster representatives dictionary
        clusters_with_representatives = {}
        
        for c_id in df['cluster_id'].unique():
            # Get representative keywords
            reps = df[(df['cluster_id'] == c_id) & (df['representative'] == True)]['keyword'].tolist()
            
            if not reps:
                cluster_kws = df[df['cluster_id'] == c_id]['keyword'].tolist()
                reps = cluster_kws[:min(15, len(cluster_kws))]  # Reduced for memory
            
            clusters_with_representatives[c_id] = reps

        # Generate semantic analysis
        semantic_analysis = generate_semantic_analysis_with_retry(
            clusters_with_representatives=clusters_with_representatives,
            client=client,
            model=model
        )

        # Add intent flow analysis
        for c_id in semantic_analysis:
            try:
                intent_flow = analyze_cluster_for_intent_flow(df, c_id)
                if intent_flow:
                    semantic_analysis[c_id]['intent_flow'] = intent_flow
            except Exception as e:
                logger.warning(f"Error analyzing intent flow for cluster {c_id}: {str(e)}")

        if semantic_analysis:
            st.success(f"✅ AI analysis completed for {len(semantic_analysis)} clusters.")
        else:
            st.warning("⚠️ No AI analysis results were generated.")

        return semantic_analysis
    
    except Exception as e:
        logger.error(f"Error in AI cluster evaluation: {str(e)}")
        st.error(f"Error in cluster evaluation: {str(e)}")
        return create_default_semantic_analysis({})

################################################################
#          INPUT VALIDATION
################################################################

def validate_csv_content(df):
    """Validate CSV content for security and format issues"""
    try:
        # Check for minimum requirements
        if len(df) == 0:
            return False, "CSV file is empty"
        
        if len(df) > MAX_KEYWORDS * 2:  
            return False, f"CSV file too large ({len(df)} rows). Maximum {MAX_KEYWORDS * 2:,} rows allowed for memory constraints."
        
        # Check for required columns
        if 'keyword' not in df.columns:
            return False, "No 'keyword' column found in CSV"
        
        # Check for malicious content patterns
        malicious_patterns = [
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'\.\./\.\.',
            r'file://',
            r'data:.*base64'
        ]
        
        for _, row in df.head(100).iterrows():  # Check first 100 rows for performance
            keyword = str(row.get('keyword', ''))
            for pattern in malicious_patterns:
                if re.search(pattern, keyword, re.IGNORECASE):
                    return False, f"Potentially malicious content detected in keyword: {keyword[:50]}..."
        
        # Check for valid keyword content
        valid_keywords = 0
        for keyword in df['keyword'].head(1000):  # Sample check
            if isinstance(keyword, str) and len(keyword.strip()) > 0:
                valid_keywords += 1
        
        if valid_keywords < len(df) * 0.5:  # At least 50% should be valid
            return False, "Too many invalid or empty keywords in CSV"
        
        return True, "CSV validation passed"
        
    except Exception as e:
        logger.error(f"Error validating CSV: {str(e)}")
        return False, f"CSV validation error: {str(e)}"

def sanitize_csv_data(df):
    """Sanitize CSV data to prevent potential issues"""
    try:
        # Clean keyword column
        if 'keyword' in df.columns:
            df['keyword'] = df['keyword'].astype(str)
            df['keyword'] = df['keyword'].apply(lambda x: sanitize_text(x) if isinstance(x, str) else "")
            df = df[df['keyword'].str.len() > 0]  # Remove empty keywords
        
        # Clean search volume if present
        if 'search_volume' in df.columns:
            df['search_volume'] = pd.to_numeric(df['search_volume'], errors='coerce')
            df['search_volume'] = df['search_volume'].fillna(0)
        
        # Limit dataset size for memory management
        if len(df) > MAX_KEYWORDS:  
            st.warning(f"⚠️ Dataset reduced from {len(df)} to {MAX_KEYWORDS:,} rows for memory optimization")
            df = df.head(MAX_KEYWORDS)
        
        return df
        
    except Exception as e:
        logger.error(f"Error sanitizing CSV data: {str(e)}")
        return df

################################################################
#          MAIN CLUSTERING PIPELINE
################################################################

def run_clustering_with_monitoring(
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
    """Execute clustering pipeline with comprehensive monitoring and error handling"""
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file with keywords.")
        return False, None
    
    st.info("🚀 Starting advanced semantic clustering pipeline...")
    monitor_resources()
    
    # Create OpenAI client with proper error handling
    client = None
    if openai_api_key and LIBRARIES['openai_available']:
        client = create_openai_client_with_timeout(openai_api_key)
        if client:
            st.success("✅ Connected to OpenAI successfully.")
        else:
            st.warning("⚠️ OpenAI connection failed. Using free alternatives.")
    elif not LIBRARIES['openai_available']:
        st.warning("⚠️ OpenAI library not installed. No OpenAI functionality.")
    else:
        st.info("ℹ️ No OpenAI API Key provided. Will use free alternatives.")
    
    # Load spaCy model
    spacy_nlp = load_spacy_model_by_language(selected_language)

    try:
        # Load and validate CSV
        st.subheader("📁 Loading and Validating CSV")
        
        if csv_format == "no_header":
            df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
            st.success(f"✅ Loaded {len(df)} keywords (no header format).")
        else:
            df = pd.read_csv(uploaded_file, header=0)
            if "Keyword" in df.columns:
                df.rename(columns={"Keyword": "keyword"}, inplace=True)
            if "keyword" not in df.columns:
                st.error("❌ No 'Keyword' column found in the CSV. Please check your file.")
                return False, None
            st.success(f"✅ Loaded {len(df)} rows (with header format).")
        
        # Validate CSV content
        is_valid, validation_message = validate_csv_content(df)
        if not is_valid:
            st.error(f"❌ CSV validation failed: {validation_message}")
            return False, None
        
        st.success(f"✅ {validation_message}")
        
        # Sanitize data
        df = sanitize_csv_data(df)
        num_keywords = len(df)
        
        # Show cost estimate
        show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
        
        # Preprocessing
        st.subheader("🔧 Keyword Preprocessing")
        st.info("Preprocessing keywords with advanced NLP techniques...")
        
        keywords_processed = preprocess_keywords(
            df["keyword"].tolist(),
            use_advanced=True,
            spacy_nlp=spacy_nlp
        )
        df['keyword_processed'] = keywords_processed
        st.success("✅ Preprocessing completed successfully!")
        
        # Generate embeddings
        st.subheader("🧠 Generating Semantic Vectors (Embeddings)")
        keyword_embeddings = generate_embeddings_with_retry(df, LIBRARIES['openai_available'], openai_api_key)
        
        # Memory cleanup after embeddings
        gc.collect()
        monitor_resources()

        # Monitor resources after embeddings
        monitor_resources()
        
        # Dimensionality reduction (PCA)
        if keyword_embeddings.shape[1] > max_pca_components:
            st.subheader("📉 Dimensionality Reduction (PCA)")
            try:
                keyword_embeddings_reduced = apply_pca_with_monitoring(
                    keyword_embeddings, pca_variance, max_pca_components
                )
            except Exception as e:
                logger.error(f"PCA failed: {str(e)}")
                st.warning(f"⚠️ PCA failed: {str(e)}. Proceeding without PCA.")
                keyword_embeddings_reduced = keyword_embeddings
        else:
            keyword_embeddings_reduced = keyword_embeddings
            st.info(f"ℹ️ No PCA needed (dimension is {keyword_embeddings.shape[1]}).")

        # Memory cleanup after PCA
        gc.collect()
        monitor_resources()
        
        # Clustering
        st.subheader("🔗 Advanced Semantic Clustering")
        cluster_labels = improved_clustering_with_monitoring(keyword_embeddings_reduced, num_clusters=num_clusters)
        df["cluster_id"] = cluster_labels
        st.success(f"✅ {len(df['cluster_id'].unique())} clusters created successfully!")
        
        # Memory cleanup after clustering
        gc.collect()
        monitor_resources()

        # Refinement
        st.subheader("🔧 Cluster Refinement")
        df = refine_clusters_with_monitoring(df, keyword_embeddings_reduced)
        final_clusters = len(df['cluster_id'].unique())
        st.success(f"✅ Refinement completed: {final_clusters} final clusters.")
        
        # Representative keywords
        st.subheader("⭐ Identifying Representative Keywords")
        clusters_with_representatives = find_representative_keywords_with_monitoring(
            df, keyword_embeddings_reduced
        )
        
        # Generate cluster names
        if client:
            st.subheader("🏷️ Generating Cluster Names & Descriptions")
            try:
                cluster_names = generate_cluster_names_with_retry(
                    clusters_with_representatives, 
                    client, 
                    model=gpt_model,
                    custom_prompt=user_prompt
                )
                if not cluster_names:
                    st.warning("⚠️ Cluster naming returned empty results. Using fallback names.")
                    cluster_names = create_fallback_cluster_names(df)
            except Exception as e:
                logger.error(f"Cluster naming failed: {str(e)}")
                st.warning(f"⚠️ Cluster naming failed: {str(e)}. Using fallback names.")
                cluster_names = create_fallback_cluster_names(df)
        else:
            st.warning("⚠️ No OpenAI client available. Using generic cluster names.")
            cluster_names = create_fallback_cluster_names(df)
        
        # Evaluate cluster quality
        df = evaluate_cluster_quality_with_monitoring(df, keyword_embeddings_reduced)
        
        # AI-based semantic analysis
        if client:
            try:
                eval_results = evaluate_and_refine_clusters_with_monitoring(df, client, model=gpt_model)
                st.session_state.cluster_evaluation = eval_results
            except Exception as e:
                logger.error(f"AI evaluation failed: {str(e)}")
                st.warning(f"⚠️ AI evaluation failed: {str(e)}")
                st.session_state.cluster_evaluation = create_default_semantic_analysis(clusters_with_representatives)
        
        # Final memory cleanup
        gc.collect()        
        monitor_resources()
        
        return True, df
    
    except Exception as e:
        logger.error(f"Critical error in clustering pipeline: {str(e)}")
        st.error(f"❌ Critical error in the clustering pipeline: {str(e)}")
        return False, None

def apply_pca_with_monitoring(keyword_embeddings, pca_variance, max_pca_components):
    """Apply PCA with monitoring and error handling"""
    try:
        pca_progress = st.progress(0)
        pca_text = st.empty()
        pca_text.text("🔍 Analyzing PCA explained variance...")
        
        # Use incremental PCA for large datasets
        if len(keyword_embeddings) > 10000:
            from sklearn.decomposition import IncrementalPCA
            pca = IncrementalPCA(n_components=min(max_pca_components, keyword_embeddings.shape[1]))
            
            batch_size = 1000
            for i in range(0, len(keyword_embeddings), batch_size):
                batch = keyword_embeddings[i:i+batch_size]
                pca.partial_fit(batch)
            
            keyword_embeddings_reduced = pca.transform(keyword_embeddings)
            pca_progress.progress(1.0)
            pca_text.text(f"✅ Incremental PCA applied: {pca.n_components} dimensions")
            
        else:
            # Standard PCA for smaller datasets
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
        
        return keyword_embeddings_reduced
        
    except Exception as e:
        logger.error(f"PCA error: {str(e)}")
        raise e

def find_representative_keywords_with_monitoring(df, keyword_embeddings_reduced):
    """Find representative keywords with proper monitoring"""
    rep_progress = st.progress(0)
    rep_text = st.empty()
    rep_text.text("🔍 Finding representative keywords...")
    clusters_with_representatives = {}
    
    try:
        unique_cluster_ids = df['cluster_id'].unique()
        for i, cnum in enumerate(unique_cluster_ids):
            csize = len(df[df['cluster_id'] == cnum])
            n_rep = min(max(5, BATCH_SIZE // 7), csize)
            indices = df[df['cluster_id'] == cnum].index.tolist()
            
            if len(indices) > 0:
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
        return clusters_with_representatives
        
    except Exception as e:
        logger.error(f"Error finding representative keywords: {str(e)}")
        st.warning("⚠️ Using fallback method for representative keywords.")
        # Fallback method
        for cnum in df['cluster_id'].unique():
            cluster_kws = df[df['cluster_id'] == cnum]['keyword'].tolist()
            clusters_with_representatives[cnum] = cluster_kws[:min(15, len(cluster_kws))]
        return clusters_with_representatives

def create_fallback_cluster_names(df):
    """Create fallback cluster names when AI naming fails"""
    cluster_names = {}
    for cnum in df['cluster_id'].unique():
        cluster_kws = df[df['cluster_id'] == cnum]['keyword'].tolist()
        if cluster_kws:
            # Use first few keywords to create name
            name_parts = cluster_kws[0].split()[:2]  # First 2 words of first keyword
            cluster_name = " ".join(name_parts).title()
            cluster_desc = f"Keywords related to {cluster_kws[0]}"
        else:
            cluster_name = f"Cluster {cnum}"
            cluster_desc = f"Keyword group {cnum}"
        
        cluster_names[cnum] = (cluster_name, cluster_desc)
    
    return cluster_names

def apply_cluster_names_safely(df, cluster_names, clusters_with_representatives):
    """Apply cluster names with comprehensive error handling"""
    try:
        # Initialize columns if they don't exist
        if 'cluster_name' not in df.columns:
            df['cluster_name'] = ''
        if 'cluster_description' not in df.columns:
            df['cluster_description'] = ''
        if 'representative' not in df.columns:
            df['representative'] = False
        
        # Enhanced data columns (if using GPT-4 enhanced analysis)
        if hasattr(st.session_state, 'enhanced_cluster_data') and st.session_state.enhanced_cluster_data:
            if 'primary_intent' not in df.columns:
                df['primary_intent'] = ''
            if 'business_value' not in df.columns:
                df['business_value'] = ''
            if 'content_strategy' not in df.columns:
                df['content_strategy'] = ''
        
        for cnum, (name, desc) in cluster_names.items():
            # Safety check - ensure cluster exists in dataframe
            if cnum in df['cluster_id'].values:
                mask = df['cluster_id'] == cnum
                df.loc[mask, 'cluster_name'] = name
                df.loc[mask, 'cluster_description'] = desc
                
                # Add enhanced data if available
                if hasattr(st.session_state, 'enhanced_cluster_data') and st.session_state.enhanced_cluster_data:
                    enhanced = st.session_state.enhanced_cluster_data.get(cnum, {})
                    if enhanced:
                        df.loc[mask, 'primary_intent'] = enhanced.get('primary_intent', '')
                        df.loc[mask, 'business_value'] = enhanced.get('business_value', '')
                        df.loc[mask, 'content_strategy'] = enhanced.get('content_strategy', '')
                
                # Mark representative keywords safely
                representative_keywords = clusters_with_representatives.get(cnum, [])
                for kw in representative_keywords:
                    try:
                        # Create boolean mask for matching keywords in this cluster
                        keyword_mask = (df['cluster_id'] == cnum) & (df['keyword'] == kw)
                        df.loc[keyword_mask, 'representative'] = True
                    except Exception as e:
                        logger.warning(f"Error marking representative keyword '{kw}': {str(e)}")
        
        st.success("✅ Cluster names applied successfully!")
        
    except Exception as e:
        logger.error(f"Error applying cluster names: {str(e)}")
        st.warning(f"⚠️ Error applying cluster names: {str(e)}. Using fallback approach.")
        
        # Emergency fallback
        try:
            # Ensure required columns exist
            for col in ['cluster_name', 'cluster_description', 'representative']:
                if col not in df.columns:
                    if col == 'representative':
                        df[col] = False
                    else:
                        df[col] = ''
            
            for cnum in df['cluster_id'].unique():
                mask = df['cluster_id'] == cnum
                df.loc[mask, 'cluster_name'] = f"Cluster {cnum}"
                df.loc[mask, 'cluster_description'] = f"Group of related keywords (cluster {cnum})"
        except Exception as fallback_error:
            logger.error(f"Even fallback failed: {str(fallback_error)}")
    
    return df

################################################################
#          MAIN STREAMLIT APP
################################################################

# Page configuration with error handling
try:
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
except Exception as e:
    logger.warning(f"Error setting page config: {str(e)}")

# CSS styling with improved accessibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f1f1f;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0066cc;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #dc3545;
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
    .memory-indicator {
        position: fixed;
        top: 60px;
        right: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8em;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<div class='main-header'>🔍 Advanced Semantic Keyword Clustering</div>", unsafe_allow_html=True)

# System status indicator
try:
    peak_memory = st.session_state.memory_monitor.get('peak_memory', 0)
    if peak_memory > 0:
        memory_color = "🔴" if peak_memory > 800 else "🟡" if peak_memory > 500 else "🟢"
        st.markdown(f"""
        <div class='memory-indicator'>
            {memory_color} Memory: {peak_memory:.0f}MB
        </div>
        """, unsafe_allow_html=True)
except:
    pass

# Application description
st.markdown("""
This application clusters semantically similar keywords using advanced NLP and clustering methods.
You can upload:
- A **simple CSV** with no header (just one keyword per line), or
- A **Keyword Planner-like CSV** with a header (Keyword, search_volume, competition, cpc, month1..month12, etc.)

### ✨ Key Features:
- 🤖 **OpenAI-powered embeddings** for superior semantic understanding
- 🔍 **Advanced search intent classification** (Informational, Commercial, Transactional, Navigational)
- 🗺️ **Customer journey mapping** to understand user behavior
- 📊 **Multiple export formats** (CSV, Excel, HTML, PDF)
- 🌍 **Multi-language support** with spaCy integration
- ⚡ **Optimized for Streamlit Cloud** with resource monitoring
""")

# CSV Format Info expander
with st.expander("📁 CSV Format Information", expanded=False):
    st.markdown("""
    ### Which CSV format should I use?

    #### 1. **No Header Format** (Simple):
    - Each line contains just one keyword  
    - Example:
      ```
      red shoes
      running shoes
      kids sneakers
      ```
    - The app will treat the entire CSV as a single column: 'keyword'

    #### 2. **With Header Format** (Keyword Planner style):
    - The first row contains column names (e.g., `Keyword,search_volume,competition,cpc,month1..month12`)  
    - The app will use the 'Keyword' column as the main text  
    - Additional columns can be used for search volume analysis and weighting

    ⚠️ **Important**: If you select the wrong format, the first row might be interpreted incorrectly.
    
    ### 📊 Data Limits:
    - **Maximum rows**: 25,000 (for memory optimization)
    - **OpenAI processing**: Up to 5,000 keywords directly processed
    - **File size**: Recommend keeping under 10MB
    """)

# Sidebar configuration
st.sidebar.markdown("### 🔧 Configuration")

# Sample CSV download
sample_csv_button = st.sidebar.button("📥 Download Sample CSV Template", use_container_width=True)
if sample_csv_button:
    try:
        csv_header = generate_sample_csv()
        st.sidebar.download_button(
            label="📄 Click to Download CSV Template",
            data=csv_header,
            file_name="sample_keyword_planner_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.sidebar.error(f"Error generating sample CSV: {str(e)}")

# CSV Format selection
csv_format = st.sidebar.selectbox(
    "📋 Select CSV format",
    options=["no_header", "with_header"],
    format_func=lambda x: "No Header (Simple)" if x == "no_header" else "With Header (Keyword Planner)",
    index=0,
    help="Choose based on your CSV structure"
)

# File upload with validation
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload your CSV file", 
    type=['csv'],
    help="Upload a CSV file containing keywords for clustering"
)

# API Key input with security
openai_api_key = st.sidebar.text_input(
    "🔑 OpenAI API Key (optional)",
    type="password",
    help="Enter your OpenAI API Key for high-quality embeddings. If omitted, free alternatives will be used.",
    placeholder="sk-..."
)

# Language selection
language_options = [
    "English", "Spanish", "French", "German", "Dutch", 
    "Korean", "Japanese", "Italian", "Portuguese", 
    "Brazilian Portuguese", "Swedish", "Norwegian", 
    "Danish", "Icelandic", "Lithuanian", "Greek", "Romanian",
    "Polish"
]
selected_language = st.sidebar.selectbox(
    "🌍 Select language of the CSV",
    options=language_options,
    index=0,
    help="Choose the primary language of your keywords for better preprocessing"
)

# Library status indicators
st.sidebar.markdown("### 📚 Available Libraries")
status_indicators = []

if LIBRARIES['openai_available']:
    if openai_api_key:
        status_indicators.append("🤖 OpenAI: ✅ Ready")
    else:
        status_indicators.append("🤖 OpenAI: ⚠️ No API key")
else:
    status_indicators.append("🤖 OpenAI: ❌ Not installed")

if LIBRARIES['sentence_transformers_available']:
    status_indicators.append("🧠 SentenceTransformers: ✅ Available")
else:
    status_indicators.append("🧠 SentenceTransformers: ❌ Not available")

if LIBRARIES['spacy_base_available']:
    status_indicators.append("🔤 spaCy: ✅ Available")
else:
    status_indicators.append("🔤 spaCy: ❌ Not available")

for indicator in status_indicators:
    st.sidebar.text(indicator)

# Parameters section
st.sidebar.markdown("### ⚙️ Clustering Parameters")

# Parameters guide
with st.sidebar.expander("ℹ️ Parameters Guide", expanded=False):
    st.markdown("""
    ### 📖 Parameters Guide

    **Number of clusters**  
    Controls how many groups will be formed. Higher = more granular, Lower = broader groups.

    **PCA explained variance (%)**  
    How much data variance to preserve during dimensionality reduction. 95% keeps most information.

    **Max PCA components**  
    Hard limit on dimensions after PCA reduction.

    **Term frequency filters**  
    Used for TF-IDF. Filters extremely rare or common terms.

    **GPT Model**  
    - `gpt-4o-mini`: Cost-effective, good quality
    - `gpt-4o`: Higher quality, more expensive
    """)

# Parameter inputs with validation
num_clusters = st.sidebar.slider(
    "🎯 Number of clusters", 
    min_value=2, max_value=50, value=10,
    help="Target number of clusters to create"
)

pca_variance = st.sidebar.slider(
    "📉 PCA explained variance (%)", 
    min_value=50, max_value=99, value=95,
    help="Percentage of variance to preserve in PCA"
)

max_pca_components = st.sidebar.slider(
    "🔢 Max PCA components", 
    min_value=10, max_value=300, value=100,
    help="Maximum number of components after PCA"
)

min_df = st.sidebar.slider(
    "📊 Minimum term frequency", 
    min_value=1, max_value=10, value=1,
    help="Minimum frequency for TF-IDF terms"
)

max_df = st.sidebar.slider(
    "📈 Maximum term frequency (%)", 
    min_value=50, max_value=100, value=95,
    help="Maximum frequency for TF-IDF terms"
)

gpt_model = st.sidebar.selectbox(
    "🤖 GPT Model for naming clusters", 
    options=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4"], 
    index=1,  # Default to gpt-4.1-mini
    help="gpt-4.1-nano is fastest/cheapest, gpt-4.1-mini offers balanced performance, gpt-4 is the legacy stable option."
)

# Custom prompt section
st.sidebar.markdown("### 📝 Custom Prompt for SEO Naming")
default_prompt = """You are an expert SEO strategist and semantic clustering specialist. Your task is to analyze keyword clusters and provide actionable insights for content strategy.

For each cluster, provide:

1. **CLUSTER NAME** (3-6 words): A clear, SEO-friendly name that captures the core theme
2. **META DESCRIPTION** (1-2 sentences): Concise description explaining the search intent and content opportunity
3. **SEARCH INTENT ANALYSIS**: Primary intent (Informational/Commercial/Transactional/Navigational) with confidence reasoning
4. **CONTENT STRATEGY**: Specific content type recommendations
5. **SEO OPPORTUNITY**: Difficulty assessment and ranking potential

**IMPORTANT GUIDELINES:**
- Focus on user search intent and business value
- Consider keyword difficulty and search volume patterns
- Suggest content that matches the user journey stage
- Provide actionable, specific recommendations
- Ensure cluster names are brandable and memorable

**OUTPUT FORMAT:** Respond with valid JSON only, no additional text."""

user_prompt = st.sidebar.text_area(
    "✏️ Custom Prompt",
    value=default_prompt,
    height=150,
    help="Customize how AI generates cluster names and descriptions"
)

# Cost calculator
add_cost_calculator()

# Helper Functions for Cluster Analysis Display
def display_search_intent_analysis(cluster_data):
    """Display search intent analysis for a cluster with enhanced error handling"""
    try:
        intent_classification = cluster_data.get('intent_classification', {})
        primary_intent = intent_classification.get('primary_intent', 'Unknown')
        scores = intent_classification.get('scores', {})
        evidence = intent_classification.get('evidence', {})
        
        # Intent display with styling
        intent_class = get_intent_css_class(primary_intent)
        st.markdown(f"""
        <div class="intent-box {intent_class}">
            <strong>🎯 Primary Search Intent:</strong> {primary_intent}
        </div>
        """, unsafe_allow_html=True)
        
        # Search intent details
        search_intent_detail = cluster_data.get('search_intent', 'N/A')
        st.write(f"**📝 Search Intent Details:** {search_intent_detail}")
        
        # Evidence display with validation
        if evidence and primary_intent in evidence and evidence[primary_intent]:
            st.markdown("**🔍 Evidence for this classification:**")
            evidence_items = evidence[primary_intent][:5]  # Limit to 5 items
            for e in evidence_items:
                if e and isinstance(e, str):  # Validate evidence item
                    st.markdown(f"• {e}")
        
        # Scores visualization with validation
        if scores and isinstance(scores, dict) and len(scores) > 0:
            # Ensure all values are numeric
            clean_scores = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
            
            if clean_scores:
                fig_intent = px.bar(
                    x=list(clean_scores.keys()), 
                    y=list(clean_scores.values()),
                    labels={'x': 'Intent Type', 'y': 'Confidence Score (%)'},
                    title='Search Intent Distribution'
                )
                fig_intent.update_layout(
                    yaxis_range=[0, 100], 
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_intent, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying search intent analysis: {str(e)}")
        logger.error(f"Error in display_search_intent_analysis: {str(e)}", exc_info=True)

def display_customer_journey_analysis(cluster_data):
    """Display customer journey analysis for a cluster with improved structure"""
    try:
        intent_flow = cluster_data.get('intent_flow', None)
        
        if not intent_flow:
            st.info("Customer journey analysis not available for this cluster.")
            return
        
        journey_phase = intent_flow.get('journey_phase', 'Unknown')
        
        # Journey phase display
        journey_class = get_journey_css_class(journey_phase)
        st.markdown(f"""
        <div class="{journey_class}">
            <strong>🗺️ Customer Journey Phase:</strong> {journey_phase}
        </div>
        """, unsafe_allow_html=True)
        
        # Intent distribution visualization
        intent_dist = intent_flow.get('intent_distribution', {})
        if intent_dist and isinstance(intent_dist, dict) and len(intent_dist) > 0:
            # Clean and validate data
            clean_dist = {k: v for k, v in intent_dist.items() if isinstance(v, (int, float)) and v > 0}
            
            if clean_dist:
                fig_dist = px.pie(
                    names=list(clean_dist.keys()),
                    values=list(clean_dist.values()),
                    title='Keyword Intent Distribution in this Cluster',
                    hole=0.3  # Create donut chart for better visibility
                )
                fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Keyword samples display
        keyword_sample = intent_flow.get('keyword_sample', [])
        if keyword_sample and isinstance(keyword_sample, list) and len(keyword_sample) > 0:
            st.markdown("### 📝 Sample Keywords with Intent")
            
            # Ensure it's a list of dicts for DataFrame creation
            if all(isinstance(item, dict) for item in keyword_sample):
                sample_df = pd.DataFrame(keyword_sample)
                # Add styling to dataframe
                st.dataframe(
                    sample_df, 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("Keyword sample data format is invalid.")
            
    except Exception as e:
        st.error(f"Error displaying customer journey analysis: {str(e)}")
        logger.error(f"Error in display_customer_journey_analysis: {str(e)}", exc_info=True)

def display_ai_insights(cluster_data):
    """Display AI-generated insights for a cluster with enhanced formatting"""
    try:
        # Coherence score with validation
        coherence_score = cluster_data.get('coherence_score', 'N/A')
        if isinstance(coherence_score, (int, float)):
            score_color = "🟢" if coherence_score >= 7 else "🟡" if coherence_score >= 5 else "🔴"
            st.metric(
                label="🧠 AI Coherence Score (0-10)", 
                value=f"{score_color} {coherence_score:.1f}"
            )
        else:
            st.metric(label="🧠 AI Coherence Score", value="N/A")
        
        # Split suggestion analysis
        split_suggestion = cluster_data.get('split_suggestion', '')
        if not split_suggestion:
            st.info("No split analysis available for this cluster.")
            return
            
        # Determine if split is recommended
        should_split = split_suggestion.lower().startswith('yes')
        
        if should_split:
            st.warning("💡 **Split Recommendation:** This cluster could be divided into more focused sub-clusters.")
            
            # Display subclusters if available
            subclusters = cluster_data.get('subclusters', [])
            if subclusters and isinstance(subclusters, list):
                st.markdown("### 🔄 Suggested Sub-clusters")
                
                for i, subcluster in enumerate(subclusters):
                    if isinstance(subcluster, dict):
                        subcluster_name = subcluster.get('name', f"Subcluster {i+1}")
                        subcluster_keywords = subcluster.get('keywords', [])
                        
                        # Ensure keywords is a list
                        if isinstance(subcluster_keywords, list):
                            keywords_display = ', '.join(str(kw) for kw in subcluster_keywords[:5])
                            if len(subcluster_keywords) > 5:
                                keywords_display += f" (+{len(subcluster_keywords) - 5} more)"
                        else:
                            keywords_display = "No keywords available"
                        
                        st.markdown(f"""
                        <div class="subcluster-box">
                            <h4>{subcluster_name}</h4>
                            <p><strong>Sample Keywords:</strong> {keywords_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.success("✅ **Split Recommendation:** This cluster appears to be coherent and focused.")
        
        # Display full analysis
        st.markdown("**📊 Full Analysis:**")
        st.markdown(split_suggestion)
        
    except Exception as e:
        st.error(f"Error displaying AI insights: {str(e)}")
        logger.error(f"Error in display_ai_insights: {str(e)}", exc_info=True)

def display_seo_recommendations(cluster_data, cluster_df):
    """Display SEO recommendations for a cluster with structured content"""
    try:
        # Additional info section
        additional_info = cluster_data.get('additional_info', '')
        if additional_info:
            st.markdown("### 📋 Cluster Analysis")
            st.markdown(additional_info)
        
        # Intent-based recommendations
        intent_classification = cluster_data.get('intent_classification', {})
        primary_intent = intent_classification.get('primary_intent', 'Unknown')
        
        recommendations = get_intent_based_recommendations(primary_intent)
        st.markdown("### 💡 Content Recommendations by Search Intent")
        st.markdown(recommendations)
        
        # Add keyword-specific recommendations if available
        if len(cluster_df) > 0 and 'search_volume' in cluster_df.columns:
            high_volume_keywords = cluster_df.nlargest(5, 'search_volume')[['keyword', 'search_volume']]
            if len(high_volume_keywords) > 0:
                st.markdown("### 🎯 High-Priority Keywords")
                st.markdown("Focus content creation on these high-volume keywords:")
                st.dataframe(high_volume_keywords, hide_index=True, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying SEO recommendations: {str(e)}")
        logger.error(f"Error in display_seo_recommendations: {str(e)}", exc_info=True)

def get_intent_css_class(intent):
    """Get CSS class for intent styling with safe fallback"""
    intent_classes = {
        "Informational": "intent-info",
        "Navigational": "intent-nav", 
        "Transactional": "intent-trans",
        "Commercial": "intent-comm",
        "Mixed Intent": "intent-mixed",
        "Unknown": "intent-unknown"
    }
    return intent_classes.get(intent, "intent-unknown")

def get_journey_css_class(journey_phase):
    """Get CSS class for journey phase styling with comprehensive matching"""
    journey_phase_lower = journey_phase.lower() if journey_phase else ""
    
    if "early" in journey_phase_lower or "research" in journey_phase_lower:
        return "journey-early"
    elif "middle" in journey_phase_lower or "consideration" in journey_phase_lower:
        return "journey-middle"
    elif "late" in journey_phase_lower or "purchase" in journey_phase_lower:
        return "journey-late"
    elif "transition" in journey_phase_lower:
        return "journey-transition"
    else:
        return "journey-mixed"

def get_intent_based_recommendations(primary_intent):
    """Get content recommendations based on search intent with comprehensive coverage"""
    recommendations = {
        "Informational": """
        **📚 Recommended Content Types:**
        - How-to guides and tutorials
        - Explanatory articles and blog posts
        - FAQ pages and knowledge bases
        - Educational videos and infographics
        - Case studies and whitepapers
        
        **🎯 SEO Targets:**
        - Featured snippets
        - People Also Ask boxes
        - Knowledge panels
        - Video carousels
        
        **💡 Best Practices:**
        - Use clear headings and subheadings
        - Include step-by-step instructions
        - Add relevant images and diagrams
        - Answer related questions comprehensively
        """,
        "Commercial": """
        **🛍️ Recommended Content Types:**
        - Product comparisons and reviews
        - Best-of lists and buying guides
        - Expert roundups and opinions
        - Detailed feature breakdowns
        - Alternative/competitor comparisons
        
        **🎯 SEO Targets:**
        - Rich results with ratings
        - Comparison tables in featured snippets
        - Product carousels
        - Review snippets
        
        **💡 Best Practices:**
        - Include pros and cons lists
        - Add comparison tables
        - Use schema markup for reviews
        - Include authentic user testimonials
        """,
        "Transactional": """
        **💰 Recommended Content Types:**
        - Product and service pages
        - Pricing and package pages
        - Special offers and deals
        - Local landing pages
        - Category pages
        
        **🎯 SEO Targets:**
        - Shopping results
        - Local pack listings
        - Structured data for products
        - Merchant listings
        
        **💡 Best Practices:**
        - Clear CTAs above the fold
        - Include trust signals
        - Optimize for mobile conversions
        - Use urgency and scarcity tactfully
        """,
        "Navigational": """
        **🧭 Recommended Content Types:**
        - Brand and service landing pages
        - Contact and location pages
        - Download and resource pages
        - Account and login pages
        - Company information pages
        
        **🎯 SEO Targets:**
        - Brand SERP features
        - Site links
        - Knowledge panels
        - Brand boxes
        
        **💡 Best Practices:**
        - Ensure brand consistency
        - Optimize page load speed
        - Include clear navigation
        - Implement proper redirects
        """,
        "Mixed Intent": """
        **📝 Recommended Content Types:**
        - Comprehensive topic hubs
        - Multi-purpose landing pages
        - Resource centers
        - Content that addresses multiple user needs
        
        **🎯 SEO Targets:**
        - Multiple SERP features
        - Broad keyword coverage
        - Various rich results
        
        **💡 Best Practices:**
        - Create content clusters
        - Use internal linking strategically
        - Address different user intents in sections
        - Implement clear content organization
        """
    }
    
    return recommendations.get(primary_intent, recommendations["Mixed Intent"])

def display_enhanced_cluster_insights(cluster_id):
    """Display enhanced cluster insights from GPT-4 analysis with rich formatting"""
    if not hasattr(st.session_state, 'enhanced_cluster_data'):
        st.info("🤖 Enhanced GPT-4 analysis not available. Run clustering with GPT-4 models to see strategic insights.")
        return
    
    enhanced_data = st.session_state.enhanced_cluster_data.get(cluster_id)
    if not enhanced_data:
        st.info("🤖 No enhanced analysis available for this cluster.")
        return
    
    st.markdown("### 🎯 GPT-4 Strategic Analysis")
    st.markdown("Advanced AI-powered insights for strategic content planning and SEO optimization.")
    
    # Main metrics with enhanced display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        intent_color = {
            "Informational": "🔵",
            "Commercial": "🟣", 
            "Transactional": "🟠",
            "Navigational": "🟢",
            "Mixed Intent": "⚪",
            "Unknown": "⚪"
        }.get(enhanced_data.get('primary_intent', 'Unknown'), "⚪")
        
        st.metric(
            label="🔍 Search Intent",
            value=f"{intent_color} {enhanced_data.get('primary_intent', 'Unknown')}"
        )
    
    with col2:
        confidence = enhanced_data.get('intent_confidence', 'Medium')
        confidence_color = {
            "High": "🟢",
            "Medium": "🟡", 
            "Low": "🔴"
        }.get(confidence, "🟡")
        
        st.metric(
            label="📊 Confidence",
            value=f"{confidence_color} {confidence}"
        )
    
    with col3:
        business_value = enhanced_data.get('business_value', 'Medium')
        value_color = {
            "High": "🟢",
            "Medium": "🟡",
            "Low": "🔴"
        }.get(business_value, "🟡")
        
        st.metric(
            label="💼 Business Value",
            value=f"{value_color} {business_value}"
        )
    
    # Detailed analysis in organized sections
    col1, col2 = st.columns(2)
    
    with col1:
        if enhanced_data.get('intent_reasoning'):
            st.markdown("#### 🧠 Intent Analysis")
            st.info(enhanced_data['intent_reasoning'])
        
        if enhanced_data.get('seo_opportunity'):
            st.markdown("#### 🚀 SEO Opportunity")
            st.warning(enhanced_data['seo_opportunity'])
    
    with col2:
        if enhanced_data.get('content_strategy'):
            st.markdown("#### 📝 Content Strategy")
            st.success(enhanced_data['content_strategy'])
    
    # Recommended Actions
    if enhanced_data.get('recommended_actions') and isinstance(enhanced_data['recommended_actions'], list):
        st.markdown("#### ✅ Recommended Actions")
        st.markdown("Strategic next steps for this cluster:")
        
        for i, action in enumerate(enhanced_data['recommended_actions'], 1):
            if action:  # Ensure action is not empty
                st.markdown(f"**{i}.** {action}")
    
    # Additional Insights in expandable format
    with st.expander("📊 Detailed Analysis", expanded=False):
        st.markdown("**🎯 Strategic Positioning:**")
        
        business_value_text = enhanced_data.get('business_value', 'Medium').lower()
        confidence_text = enhanced_data.get('intent_confidence', 'Medium').lower()
        
        st.write(f"This cluster represents a **{business_value_text}** business value opportunity "
                f"with **{confidence_text}** confidence in our intent classification.")
        
        # Intent-specific guidance
        intent_guidance = {
            'Informational': "**💡 Content Focus:** Educational content, how-to guides, and comprehensive resources that establish authority and capture early-stage users.",
            'Commercial': "**💡 Content Focus:** Comparison content, reviews, and evaluative resources that help users make informed decisions.",
            'Transactional': "**💡 Content Focus:** Product pages, pricing information, and conversion-optimized content for ready-to-buy users.",
            'Navigational': "**💡 Content Focus:** Brand pages, specific product/service pages, and direct navigation content.",
            'Mixed Intent': "**💡 Content Focus:** Multi-purpose content that addresses various user intents within a single comprehensive resource."
        }
        
        current_intent = enhanced_data.get('primary_intent', 'Mixed Intent')
        st.markdown(intent_guidance.get(current_intent, intent_guidance['Mixed Intent']))
        
        st.markdown("---")
        st.markdown("*Analysis generated by GPT-4 with advanced semantic understanding and strategic insights.*")

def create_cluster_summary(df):
    """Create a comprehensive summary dataframe of clusters with error handling"""
    try:
        # Validate input
        if df is None or df.empty:
            logger.warning("Empty dataframe provided to create_cluster_summary")
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['cluster_id', 'cluster_name', 'cluster_description', 'keyword']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Basic summary
        summary_df = df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
        summary_df.columns = ['ID', 'Name', 'Description', 'Number of Keywords']
        
        # Add search volume if available
        if 'search_volume' in df.columns:
            volume_df = df.groupby('cluster_id')['search_volume'].agg(['sum', 'mean']).reset_index()
            volume_df.columns = ['cluster_id', 'Total Search Volume', 'Avg Search Volume']
            summary_df = summary_df.merge(volume_df, left_on='ID', right_on='cluster_id', how='left')
            summary_df.drop('cluster_id', axis=1, inplace=True)
            
            # Format volume columns
            summary_df['Total Search Volume'] = summary_df['Total Search Volume'].fillna(0).astype(int)
            summary_df['Avg Search Volume'] = summary_df['Avg Search Volume'].fillna(0).round(0).astype(int)
        
        # Add coherence score
        if 'cluster_coherence' in df.columns:
            coherence_df = df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
            coherence_df.columns = ['cluster_id', 'Coherence']
            summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id', how='left')
            summary_df.drop('cluster_id', axis=1, inplace=True)
            summary_df['Coherence'] = summary_df['Coherence'].round(3)
        
        # Add representative keywords
        def get_rep_keywords(cid):
            try:
                cluster_data = df[df['cluster_id'] == cid]
                if 'representative' in df.columns:
                    reps = cluster_data[cluster_data['representative'] == True]['keyword'].tolist()
                else:
                    # Fallback: get top keywords by search volume or first few
                    if 'search_volume' in df.columns:
                        reps = cluster_data.nlargest(5, 'search_volume')['keyword'].tolist()
                    else:
                        reps = cluster_data['keyword'].head(5).tolist()
                
                return ', '.join(str(kw) for kw in reps[:5]) if reps else 'None'
            except Exception as e:
                logger.error(f"Error getting representative keywords for cluster {cid}: {str(e)}")
                return 'Error'
        
        summary_df['Representative Keywords'] = summary_df['ID'].apply(get_rep_keywords)
        
        # Add AI evaluation info if available
        if hasattr(st.session_state, 'cluster_evaluation') and st.session_state.cluster_evaluation:
            evaluated_ids = set(st.session_state.cluster_evaluation.keys())
            summary_df['AI Analysis'] = summary_df['ID'].apply(
                lambda x: "✅ Complete" if x in evaluated_ids else "❌ Not Available"
            )
            
            def get_search_intent(cid):
                try:
                    if cid in st.session_state.cluster_evaluation:
                        intent_data = st.session_state.cluster_evaluation[cid].get('intent_classification', {})
                        return intent_data.get('primary_intent', 'Unknown')
                    return 'Not Analyzed'
                except Exception:
                    return 'Error'
            
            summary_df['Primary Intent'] = summary_df['ID'].apply(get_search_intent)
        
        # Sort by number of keywords (largest clusters first)
        summary_df = summary_df.sort_values('Number of Keywords', ascending=False)
        
        # Truncate long descriptions for display
        summary_df['Description'] = summary_df['Description'].apply(
            lambda x: x[:100] + '...' if len(str(x)) > 100 else x
        )
        
        return summary_df
        
    except Exception as e:
        logger.error(f"Error creating cluster summary: {str(e)}", exc_info=True)
        return pd.DataFrame()

# Initialize session state with comprehensive error handling
def initialize_session_state():
    """Initialize all required session state variables with safe defaults"""
    try:
        defaults = {
            'process_complete': False,
            'df_results': None,
            'cluster_evaluation': {},
            'enhanced_cluster_data': {},
            'memory_monitor': {
                'last_check': time.time(), 
                'peak_memory': 0,
                'warnings_shown': 0
            },
            'export_history': [],
            'user_preferences': {
                'theme': 'light',
                'auto_export': False
            }
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
        return True
        
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}", exc_info=True)
        st.error("Failed to initialize application state. Please refresh the page.")
        return False

# Call initialization
if not initialize_session_state():
    st.stop()
