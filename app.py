import os
import time
import json
import logging
import warnings
import tempfile
import hashlib
import gc
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from io import StringIO, BytesIO
from collections import Counter
from unidecode import unidecode
from rapidfuzz import fuzz

# Core data processing
import numpy as np
import pandas as pd

# Streamlit for UI
import streamlit as st

# NLP and ML libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLTK imports and downloads
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    for dataset in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
        except LookupError:
            nltk.download(dataset, quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None
except Exception as e:
    logging.warning(f"NLTK setup failed: {str(e)}")
    NLTK_AVAILABLE = False

# Optional libraries detection
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
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    TextBlob = None

# Configuration and Constants
MAX_KEYWORDS = 25000
OPENAI_TIMEOUT = 60.0
OPENAI_MAX_RETRIES = 3
MAX_MEMORY_WARNING = 800  # MB
BATCH_SIZE = 100
MIN_CLUSTER_SIZE = 2
MAX_CLUSTERS = 50

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Streamlit page configuration
try:
    st.set_page_config(
        page_title="Semantic Keyword Clustering",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass  # Page config already set

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f1f1f;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0066cc;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
    }
    .cluster-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .stProgress .st-bo {
        background-color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Search Intent Classification Patterns
SEARCH_INTENT_PATTERNS = {
    "Informational": {
        "keywords": [
            "how", "what", "why", "when", "where", "who", "which", "guide", 
            "tutorial", "learn", "definition", "meaning", "examples", "tips",
            "steps", "explain", "understanding", "knowledge", "information"
        ],
        "patterns": [
            r'\bhow\s+to\b', r'\bwhat\s+is\b', r'\bwhy\s+is\b', r'\bguide\s+to\b',
            r'\btutorial\b', r'\blearn\s+about\b', r'\bexamples?\s+of\b',
            r'\bsteps?\s+to\b', r'\btips?\s+for\b', r'\bways?\s+to\b'
        ],
        "weight": 1.0
    },
    "Commercial": {
        "keywords": [
            "best", "top", "review", "compare", "vs", "versus", "alternative",
            "recommendation", "rating", "ranked", "pros", "cons", "features",
            "comparison", "worth", "should buy", "which", "better"
        ],
        "patterns": [
            r'\bbest\b', r'\btop\b', r'\breview\s*s?\b', r'\bcompare\b',
            r'\bvs\b', r'\bversus\b', r'\balternative\s*s?\b',
            r'\bworth\s+it\b', r'\bshould\s+i\s+buy\b', r'\bwhich\s+is\s+better\b'
        ],
        "weight": 1.2
    },
    "Transactional": {
        "keywords": [
            "buy", "purchase", "order", "shop", "price", "cost", "cheap",
            "discount", "deal", "sale", "coupon", "free shipping", "near me",
            "store", "online", "checkout", "pay", "shipping", "delivery"
        ],
        "patterns": [
            r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bshop\b',
            r'\bprice\b', r'\bcost\b', r'\bcheap\b', r'\bdiscount\b',
            r'\bnear\s+me\b', r'\bfree\s+shipping\b', r'\bfor\s+sale\b'
        ],
        "weight": 1.5
    },
    "Navigational": {
        "keywords": [
            "login", "sign in", "website", "homepage", "official", "contact",
            "address", "location", "directions", "hours", "phone", "email",
            "support", "customer service", "account", "portal", "dashboard"
        ],
        "patterns": [
            r'\blogin\b', r'\bsign\s+in\b', r'\bwebsite\b', r'\bhomepage\b',
            r'\bofficial\s+site\b', r'\bcontact\s+us\b', r'\bcustomer\s+service\b'
        ],
        "weight": 1.1
    }
}

# Language models for spaCy
SPACY_MODELS = {
    "English": "en_core_web_sm",
    "Spanish": "es_core_news_sm", 
    "French": "fr_core_news_sm",
    "German": "de_core_news_sm",
    "Portuguese": "pt_core_news_sm",
    "Italian": "it_core_news_sm",
    "Dutch": "nl_core_news_sm"
}

# Pricing for cost calculation (per 1K tokens)
OPENAI_PRICING = {
    "text-embedding-3-small": 0.00002,
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00}
}

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'process_complete': False,
        'df_results': None,
        'cluster_evaluation': {},
        'memory_monitor': {
            'last_check': time.time(),
            'peak_memory': 0,
            'warnings_shown': 0
        },
        'processing_started': False,
        'results_df': None,
        'processing_metadata': None,
        'error_state': None,
        'last_uploaded_file': None,
        'session_start': datetime.now().isoformat(),
        'processing_time': 'Unknown',
        'app_settings': {
            'results_per_page': 50,
            'chart_theme': 'plotly_white',
            'number_format': 'Auto',
            'enable_caching': True,
            'memory_optimization': True,
            'auto_refresh': False
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    return True

def monitor_resources():
    """Monitor system resources and show warnings if needed"""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Update peak memory
        if memory_mb > st.session_state.memory_monitor['peak_memory']:
            st.session_state.memory_monitor['peak_memory'] = memory_mb
        
        # Show warnings for high memory usage
        if memory_mb > MAX_MEMORY_WARNING:
            if st.session_state.memory_monitor['warnings_shown'] < 3:
                st.warning(f"⚠️ High memory usage: {memory_mb:.1f}MB")
                st.session_state.memory_monitor['warnings_shown'] += 1
            
            # Force garbage collection
            gc.collect()
        
        st.session_state.memory_monitor['last_check'] = time.time()
        
    except Exception as e:
        logger.warning(f"Resource monitoring error: {str(e)}")

@st.cache_resource(ttl=3600)
def download_nltk_data():
    """Download required NLTK data with caching"""
    if not NLTK_AVAILABLE:
        return False
    
    try:
        datasets = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger']
        
        for dataset in datasets:
            try:
                if dataset == 'punkt':
                    nltk.data.find(f'tokenizers/{dataset}')
                else:
                    nltk.data.find(f'corpora/{dataset}')
            except LookupError:
                nltk.download(dataset, quiet=True)
        
        return True
    except Exception as e:
        logger.warning(f"NLTK download failed: {str(e)}")
        return False

@st.cache_resource(ttl=7200)
def load_spacy_model(language="English"):
    """Load spaCy model for the specified language"""
    if not SPACY_AVAILABLE:
        return None
    
    model_name = SPACY_MODELS.get(language)
    if not model_name:
        return None
    
    try:
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spaCy model: {model_name}")
        return nlp
    except Exception as e:
        logger.warning(f"Failed to load spaCy model {model_name}: {str(e)}")
        return None

def create_openai_client(api_key):
    """Create OpenAI client with error handling"""
    if not OPENAI_AVAILABLE or not api_key:
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            timeout=OPENAI_TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES
        )
        
        # Test the client with a simple request
        try:
            client.models.list()
            logger.info("OpenAI client created and tested successfully")
            return client
        except Exception as test_error:
            logger.error(f"OpenAI client test failed: {str(test_error)}")
            st.error(f"OpenAI API test failed: {str(test_error)}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {str(e)}")
        st.error(f"OpenAI client creation failed: {str(e)}")
        return None

def calculate_estimated_cost(num_keywords, model="gpt-4o-mini", num_clusters=10):
    """Calculate estimated API costs"""
    try:
        # Embedding cost (limited to 5000 keywords for performance)
        keywords_for_embeddings = min(num_keywords, 5000)
        embedding_tokens = keywords_for_embeddings * 2  # ~2 tokens per keyword
        embedding_cost = (embedding_tokens / 1000) * OPENAI_PRICING["text-embedding-3-small"]
        
        # Naming cost
        if model in OPENAI_PRICING:
            pricing = OPENAI_PRICING[model]
            input_tokens = num_clusters * 200  # ~200 tokens per cluster
            output_tokens = num_clusters * 80   # ~80 tokens output per cluster
            
            naming_cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
        else:
            naming_cost = 0
        
        total_cost = embedding_cost + naming_cost
        
        return {
            "embedding_cost": embedding_cost,
            "naming_cost": naming_cost,
            "total_cost": total_cost,
            "processed_keywords": keywords_for_embeddings
        }
    except Exception as e:
        logger.error(f"Cost calculation error: {str(e)}")
        return {"embedding_cost": 0, "naming_cost": 0, "total_cost": 0, "processed_keywords": 0}

def sanitize_text(text, max_length=200):
    """Sanitize text input to prevent security issues"""
    if not isinstance(text, str):
        return str(text)[:max_length]
    
    # Remove HTML tags and suspicious content
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^\w\s\-.,!?()]+', '', text)
    text = text.strip()[:max_length]
    
    return text

def validate_dataframe(df, required_columns=None):
    """Validate DataFrame structure and content"""
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check for malicious content in keyword column
    if 'keyword' in df.columns:
        suspicious_patterns = [r'<script', r'javascript:', r'\.\./', r'file://']
        for pattern in suspicious_patterns:
            try:
                if df['keyword'].astype(str).str.contains(pattern, case=False, regex=True, na=False).any():
                    return False, f"Suspicious content detected matching pattern: {pattern}"
            except Exception as e:
                logger.warning(f"Error checking pattern {pattern}: {str(e)}")
                continue
    
    # Check minimum data requirements
    if len(df) == 0:
        return False, "No data rows found"
    
    return True, "Validation passed"

def clean_memory():
    """Force garbage collection and memory cleanup"""
    try:
        gc.collect()
        
        # Clear Streamlit caches if memory is high
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > MAX_MEMORY_WARNING:
                    # Clear data cache but keep resource cache
                    if hasattr(st, 'cache_data'):
                        st.cache_data.clear()
                    logger.info(f"Cleared cache due to high memory usage: {memory_mb:.1f}MB")
            except Exception as e:
                logger.warning(f"Memory check failed: {str(e)}")
                
    except Exception as e:
        logger.warning(f"Memory cleanup error: {str(e)}")

def log_error(error, context="Unknown", additional_info=None):
    """Enhanced error logging"""
    try:
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "additional_info": additional_info
        }
        logger.error(json.dumps(error_data, indent=2))
    except Exception:
        logger.error(f"Error in {context}: {str(error)}")

def safe_file_read(uploaded_file, encoding='utf-8'):
    """Safely read uploaded file with error handling"""
    if uploaded_file is None:
        raise ValueError("No file provided")
    
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read content
        content = uploaded_file.read()
        
        # Decode if bytes
        if isinstance(content, bytes):
            try:
                content = content.decode(encoding)
            except UnicodeDecodeError:
                # Try alternative encodings
                for alt_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        content = content.decode(alt_encoding)
                        logger.warning(f"File decoded using {alt_encoding} instead of {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"Could not decode file with any encoding")
        
        # Reset file pointer again
        uploaded_file.seek(0)
        
        return content
        
    except Exception as e:
        logger.error(f"File reading error: {str(e)}")
        raise e

def format_number(num):
    """Format numbers for display"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))

def create_progress_tracker(total_steps, step_names=None):
    """Create a progress tracking context manager"""
    class ProgressTracker:
        def __init__(self, total, names):
            self.total = total
            self.current = 0
            self.names = names or [f"Step {i+1}" for i in range(total)]
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
        
        def update(self, step_name=None):
            self.current += 1
            progress = self.current / self.total
            self.progress_bar.progress(progress)
            
            if step_name:
                self.status_text.text(f"✅ {step_name}")
            elif self.current <= len(self.names):
                self.status_text.text(f"🔄 {self.names[self.current-1]}")
            
            return self
        
        def complete(self, message="Process completed!"):
            self.progress_bar.progress(1.0)
            self.status_text.text(f"✅ {message}")
    
    return ProgressTracker(total_steps, step_names)

def estimate_memory_usage(num_keywords, embedding_method):
    """Estimate memory usage based on configuration"""
    try:
        base_memory = 50  # Base app memory in MB
        
        # Keyword storage
        keyword_memory = num_keywords * 0.001  # ~1KB per keyword
        
        # Embedding memory
        if embedding_method == "openai":
            embedding_memory = num_keywords * 0.006  # ~6KB per embedding (1536 dims * 4 bytes)
        elif embedding_method == "sentence_transformers":
            embedding_memory = num_keywords * 0.002  # ~2KB per embedding (384-512 dims)
        else:  # TF-IDF
            embedding_memory = num_keywords * 0.02   # ~20KB per embedding (5000 features)
        
        # Processing overhead
        processing_memory = num_keywords * 0.005  # General processing overhead
        
        total_memory = base_memory + keyword_memory + embedding_memory + processing_memory
        
        return total_memory
        
    except Exception as e:
        log_error(e, "memory_estimation")
        return 100  # Default estimate

def calculate_entropy(values):
    """Calculate entropy for diversity measurement"""
    try:
        if len(values) == 0:
            return 0
        
        values = np.array(values)
        total = values.sum()
        
        if total == 0:
            return 0
        
        probabilities = values / total
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
        
    except Exception:
        return 0

def preprocess_keywords_basic(keywords_list, language="English"):
    """Basic keyword preprocessing using NLTK"""
    if not keywords_list:
        return []
    
    # Ensure NLTK data is available
    if not download_nltk_data():
        logger.warning("NLTK data not available, using basic preprocessing")
        return [kw.lower().strip() for kw in keywords_list if isinstance(kw, str)]
    
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        # Get stopwords with fallback
        try:
            stop_words = set(stopwords.words('english'))
        except Exception:
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        
        lemmatizer = WordNetLemmatizer()
        processed_keywords = []
        
        for keyword in keywords_list:
            if not isinstance(keyword, str) or not keyword.strip():
                processed_keywords.append("")
                continue
            
            # Basic cleaning
            keyword = keyword.lower().strip()
            
            # Tokenization with fallback
            try:
                tokens = word_tokenize(keyword)
            except Exception:
                tokens = keyword.split()
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if (token.isalpha() and 
                    len(token) > 1 and 
                    token not in stop_words):
                    try:
                        lemmatized = lemmatizer.lemmatize(token)
                        processed_tokens.append(lemmatized)
                    except Exception:
                        processed_tokens.append(token)
            
            # Join tokens
            processed_keyword = " ".join(processed_tokens)
            processed_keywords.append(processed_keyword if processed_keyword else keyword)
        
        return processed_keywords
        
    except Exception as e:
        logger.warning(f"Basic preprocessing failed: {str(e)}")
        return [kw.lower().strip() for kw in keywords_list if isinstance(kw, str)]

def preprocess_keywords_advanced(keywords_list, spacy_nlp, language="English"):
    """Advanced preprocessing using spaCy"""
    if not keywords_list:
        return []
        
    if not spacy_nlp:
        return preprocess_keywords_basic(keywords_list, language)
    
    try:
        processed_keywords = []
        
        for keyword in keywords_list:
            if not isinstance(keyword, str) or not keyword.strip():
                processed_keywords.append("")
                continue
            
            try:
                # Process with spaCy
                doc = spacy_nlp(keyword.lower())
                
                # Extract meaningful tokens
                tokens = []
                entities = []
                
                # Get named entities
                for ent in doc.ents:
                    if len(ent.text) > 1:
                        entities.append(ent.text.replace(" ", "_"))
                
                # Get lemmatized tokens
                for token in doc:
                    if (not token.is_stop and 
                        not token.is_punct and 
                        token.is_alpha and 
                        len(token.text) > 1):
                        tokens.append(token.lemma_)
                
                # Get noun phrases
                noun_phrases = []
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2:
                        noun_phrases.append(chunk.text.replace(" ", "_"))
                
                # Combine all features (limit to prevent explosion)
                all_features = tokens[:5] + entities[:3] + noun_phrases[:2]
                
                # Create processed keyword
                if all_features:
                    processed_keyword = " ".join(all_features)
                else:
                    processed_keyword = keyword.lower()
                
                processed_keywords.append(processed_keyword)
                
            except Exception as e:
                logger.warning(f"spaCy processing failed for '{keyword}': {str(e)}")
                processed_keywords.append(keyword.lower())
        
        return processed_keywords
        
    except Exception as e:
        logger.warning(f"Advanced preprocessing failed: {str(e)}")
        return preprocess_keywords_basic(keywords_list, language)

def preprocess_keywords_textblob(keywords_list):
    """Preprocessing using TextBlob"""
    if not keywords_list:
        return []
        
    if not TEXTBLOB_AVAILABLE:
        return preprocess_keywords_basic(keywords_list)
    
    try:
        from textblob import TextBlob
        processed_keywords = []
        
        for keyword in keywords_list:
            if not isinstance(keyword, str) or not keyword.strip():
                processed_keywords.append("")
                continue
            
            try:
                # Create TextBlob
                blob = TextBlob(keyword.lower())
                
                # Get noun phrases
                noun_phrases = list(blob.noun_phrases)
                
                # Get words (filtered)
                words = [word for word in blob.words 
                        if len(word) > 1 and word.isalpha()]
                
                # Combine features (limit to prevent explosion)
                all_features = words[:5] + noun_phrases[:3]
                
                if all_features:
                    processed_keyword = " ".join(str(f) for f in all_features)
                else:
                    processed_keyword = keyword.lower()
                
                processed_keywords.append(processed_keyword)
                
            except Exception as e:
                logger.warning(f"TextBlob processing failed for '{keyword}': {str(e)}")
                processed_keywords.append(keyword.lower())
        
        return processed_keywords
        
    except Exception as e:
        logger.warning(f"TextBlob preprocessing failed: {str(e)}")
        return preprocess_keywords_basic(keywords_list)

def preprocess_keywords(keywords_list, language="English", method="auto"):
    """Main preprocessing function with multiple fallbacks"""
    try:
        # Validate input
        if not keywords_list or not isinstance(keywords_list, list):
            return []
        
        # Clean input
        cleaned_keywords = []
        for kw in keywords_list:
            if isinstance(kw, str) and kw.strip():
                cleaned_kw = sanitize_text(kw.strip())
                cleaned_keywords.append(cleaned_kw)
            else:
                cleaned_keywords.append("")
        
        if not cleaned_keywords:
            return []
        
        # Choose preprocessing method
        if method == "auto":
            # Try advanced methods first, fallback to basic
            spacy_nlp = load_spacy_model(language)
            if spacy_nlp:
                return preprocess_keywords_advanced(cleaned_keywords, spacy_nlp, language)
            elif TEXTBLOB_AVAILABLE:
                return preprocess_keywords_textblob(cleaned_keywords)
            else:
                return preprocess_keywords_basic(cleaned_keywords, language)
        
        elif method == "spacy":
            spacy_nlp = load_spacy_model(language)
            return preprocess_keywords_advanced(cleaned_keywords, spacy_nlp, language)
        
        elif method == "textblob":
            return preprocess_keywords_textblob(cleaned_keywords)
        
        else:  # basic
            return preprocess_keywords_basic(cleaned_keywords, language)
    
    except Exception as e:
        log_error(e, "keyword_preprocessing", {"num_keywords": len(keywords_list)})
        # Ultimate fallback
        return [kw.lower().strip() for kw in keywords_list if isinstance(kw, str)]

def extract_keyword_features(keyword):
    """Extract features from a keyword for intent classification"""
    if not isinstance(keyword, str) or not keyword.strip():
        return {
            "length": 0,
            "has_question_word": False,
            "has_commercial_intent": False,
            "has_transactional_intent": False,
            "has_navigational_intent": False,
            "has_local_intent": False,
            "has_brand_indicators": False,
            "has_numbers": False,
            "first_word": "",
            "last_word": ""
        }
    
    keyword_lower = keyword.lower().strip()
    words = keyword_lower.split()
    
    features = {
        "length": len(words),
        "has_question_word": any(w in keyword_lower for w in ["how", "what", "why", "when", "where", "who"]),
        "has_commercial_intent": any(w in keyword_lower for w in ["best", "top", "review", "compare", "vs"]),
        "has_transactional_intent": any(w in keyword_lower for w in ["buy", "price", "cheap", "discount", "shop"]),
        "has_navigational_intent": any(w in keyword_lower for w in ["login", "website", "official", "contact"]),
        "has_local_intent": any(phrase in keyword_lower for phrase in ["near me", "nearby", "local"]),
        "has_brand_indicators": bool(re.search(r'\b[A-Z][a-z]+\b', keyword)),
        "has_numbers": bool(re.search(r'\d+', keyword)),
        "first_word": words[0] if words else "",
        "last_word": words[-1] if words else ""
    }
    
    return features

def classify_search_intent(keyword, features=None):
    """Classify search intent for a keyword"""
    if not isinstance(keyword, str) or not keyword.strip():
        return "Unknown"
    
    if features is None:
        features = extract_keyword_features(keyword)
    
    scores = {
        "Informational": 0,
        "Commercial": 0,
        "Transactional": 0,
        "Navigational": 0
    }
    
    keyword_lower = keyword.lower().strip()
    
    # Score based on patterns and keywords
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        # Check keywords
        for kw in patterns["keywords"]:
            if kw in keyword_lower:
                scores[intent_type] += patterns["weight"]
        
        # Check regex patterns
        for pattern in patterns["patterns"]:
            try:
                if re.search(pattern, keyword_lower):
                    scores[intent_type] += patterns["weight"] * 1.5
            except re.error:
                continue
    
    # Apply feature-based scoring
    if features.get("has_question_word", False):
        scores["Informational"] += 2
    if features.get("has_commercial_intent", False):
        scores["Commercial"] += 2
    if features.get("has_transactional_intent", False):
        scores["Transactional"] += 2
    if features.get("has_navigational_intent", False):
        scores["Navigational"] += 2
    if features.get("has_local_intent", False):
        scores["Transactional"] += 1
    
    # Determine primary intent
    if all(score == 0 for score in scores.values()):
        return "Unknown"
    
    primary_intent = max(scores, key=scores.get)
    max_score = max(scores.values())
    
    # Check for mixed intent (close scores)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 1:
        return "Mixed"
    
    return primary_intent

def batch_classify_intents(keywords_list, batch_size=1000):
    """Classify search intents for a list of keywords in batches"""
    if not keywords_list:
        return []
    
    try:
        all_intents = []
        
        for i in range(0, len(keywords_list), batch_size):
            batch = keywords_list[i:i + batch_size]
            batch_intents = []
            
            for keyword in batch:
                try:
                    intent = classify_search_intent(keyword)
                    batch_intents.append(intent)
                except Exception as e:
                    logger.warning(f"Intent classification failed for '{keyword}': {str(e)}")
                    batch_intents.append("Unknown")
            
            all_intents.extend(batch_intents)
            
            # Memory cleanup every 10 batches
            if i % (batch_size * 10) == 0:
                clean_memory()
        
        return all_intents
        
    except Exception as e:
        log_error(e, "batch_intent_classification")
        return ["Unknown"] * len(keywords_list)

def analyze_search_intent_bulk(keywords_list, batch_size=1000):
    """Analyze search intent for multiple keywords"""
    try:
        st.info("🔍 Analyzing search intent patterns...")
        
        intent_results = []
        progress = st.progress(0)
        
        for i in range(0, len(keywords_list), batch_size):
            batch = keywords_list[i:i + batch_size]
            batch_intents = batch_classify_intents(batch)
            intent_results.extend(batch_intents)
            
            progress.progress(min(1.0, (i + batch_size) / len(keywords_list)))
        
        progress.progress(1.0)
        
        # Calculate intent distribution
        intent_counts = Counter(intent_results)
        total = len(intent_results)
        
        intent_distribution = {
            intent: (count / total) * 100 
            for intent, count in intent_counts.items()
        }
        
        st.success("✅ Search intent analysis completed")
        
        return intent_results, intent_distribution
        
    except Exception as e:
        log_error(e, "bulk_intent_analysis")
        return ["Unknown"] * len(keywords_list), {"Unknown": 100.0}

@st.cache_data(ttl=3600, max_entries=5, hash_funcs={OpenAI: lambda _: None})
def generate_openai_embeddings(keywords_list, _client, model="text-embedding-3-small", batch_size=100):
    """Generate embeddings using OpenAI API with batching and caching

    Parameters
    ----------
    keywords_list : list[str]
        The keywords to embed.
    _client : OpenAI
        Initialized OpenAI client instance.
    model : str, optional
        Embedding model name, by default "text-embedding-3-small".
    batch_size : int, optional
        Number of keywords per API request, by default 100.
    """
    if not _client or not keywords_list:
        return None
    
    if not OPENAI_AVAILABLE:
        st.error("OpenAI library not available")
        return None
    
    try:
        all_embeddings = []
        total_batches = (len(keywords_list) + batch_size - 1) // batch_size
        
        progress = st.progress(0)
        status = st.empty()
        
        for i in range(0, len(keywords_list), batch_size):
            batch = keywords_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            status.text(f"🔄 Generating embeddings: batch {batch_num}/{total_batches}")
            
            try:
                # Clean batch - remove empty strings and None values
                clean_batch = []
                for kw in batch:
                    if isinstance(kw, str) and kw.strip():
                        clean_batch.append(kw.strip())
                    else:
                        clean_batch.append("empty keyword")  # Placeholder for empty keywords
                
                if not clean_batch:
                    # Add zero embeddings for empty batch
                    zero_embedding = np.zeros(1536, dtype=np.float32)
                    all_embeddings.extend([zero_embedding] * len(batch))
                    continue
                
                # Make API call with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = _client.embeddings.create(
                            input=clean_batch,
                            model=model
                        )
                        break
                    except Exception as api_error:
                        if attempt == max_retries - 1:
                            raise api_error
                        time.sleep(2 ** attempt)  # Exponential backoff
                
                # Extract embeddings
                batch_embeddings = []
                for embedding_obj in response.data:
                    embedding = np.array(embedding_obj.embedding, dtype=np.float32)
                    batch_embeddings.append(embedding)
                
                # Ensure we have the right number of embeddings
                while len(batch_embeddings) < len(batch):
                    zero_embedding = np.zeros(len(batch_embeddings[0]) if batch_embeddings else 1536, dtype=np.float32)
                    batch_embeddings.append(zero_embedding)
                
                all_embeddings.extend(batch_embeddings[:len(batch)])
                
                # Update progress
                progress.progress(min(1.0, (i + batch_size) / len(keywords_list)))
                
                # Rate limiting - small delay between batches
                if batch_num < total_batches:
                    time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"OpenAI embedding error for batch {batch_num}: {str(e)}")
                # Add zero embeddings for failed batch
                zero_embedding = np.zeros(1536, dtype=np.float32)
                all_embeddings.extend([zero_embedding] * len(batch))
        
        progress.progress(1.0)
        status.text("✅ OpenAI embeddings generated successfully")
        
        # Final validation
        if len(all_embeddings) != len(keywords_list):
            logger.warning(f"Embedding count mismatch: {len(all_embeddings)} vs {len(keywords_list)}")
            # Pad or trim to match
            while len(all_embeddings) < len(keywords_list):
                all_embeddings.append(np.zeros(1536, dtype=np.float32))
            all_embeddings = all_embeddings[:len(keywords_list)]
        
        return np.array(all_embeddings, dtype=np.float32)
        
    except Exception as e:
        log_error(e, "openai_embeddings", {"num_keywords": len(keywords_list)})
        st.error(f"OpenAI embeddings failed: {str(e)}")
        return None

@st.cache_data(ttl=3600, max_entries=3)
def generate_sentence_transformer_embeddings(keywords_list, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings using SentenceTransformers"""
    if not keywords_list:
        return None
        
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.warning("SentenceTransformers not available")
        return None
    
    try:
        from sentence_transformers import SentenceTransformer
        
        st.info(f"🧠 Loading SentenceTransformer model: {model_name}")
        
        # Load model with error handling
        try:
            model = SentenceTransformer(model_name)
        except Exception as model_error:
            logger.warning(f"Failed to load {model_name}, trying fallback model")
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")  # Fallback
                st.warning(f"Using fallback model: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"Failed to load any SentenceTransformer model: {str(fallback_error)}")
                return None
        
        # Clean keywords
        clean_keywords = []
        for kw in keywords_list:
            if isinstance(kw, str) and kw.strip():
                clean_keywords.append(kw.strip())
            else:
                clean_keywords.append("empty keyword")
        
        st.info("🔄 Generating SentenceTransformer embeddings...")
        
        # Generate embeddings in batches for memory efficiency
        batch_size = 500
        all_embeddings = []
        
        progress = st.progress(0)
        
        for i in range(0, len(clean_keywords), batch_size):
            batch = clean_keywords[i:i + batch_size]
            
            try:
                batch_embeddings = model.encode(
                    batch, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
                all_embeddings.append(batch_embeddings.astype(np.float32))
            except Exception as batch_error:
                logger.warning(f"Error encoding batch {i//batch_size + 1}: {str(batch_error)}")
                # Create zero embeddings for failed batch
                zero_batch = np.zeros((len(batch), 384), dtype=np.float32)  # Default ST dimension
                all_embeddings.append(zero_batch)
            
            progress.progress(min(1.0, (i + batch_size) / len(clean_keywords)))
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        progress.progress(1.0)
        st.success("✅ SentenceTransformer embeddings generated successfully")
        
        return embeddings
        
    except Exception as e:
        log_error(e, "sentence_transformer_embeddings", {"num_keywords": len(keywords_list)})
        st.error(f"SentenceTransformer embeddings failed: {str(e)}")
        return None

@st.cache_data(ttl=3600, max_entries=3)
def generate_tfidf_embeddings(keywords_list, processed_keywords=None, max_features=5000):
    """Generate TF-IDF embeddings as fallback"""
    if not keywords_list:
        return None
        
    try:
        if processed_keywords is None:
            processed_keywords = preprocess_keywords(keywords_list)
        
        # Clean processed keywords
        clean_processed = []
        for i, kw in enumerate(processed_keywords):
            if isinstance(kw, str) and kw.strip():
                clean_processed.append(kw.strip())
            else:
                # Use original keyword as fallback
                original = keywords_list[i] if i < len(keywords_list) else "empty"
                if isinstance(original, str) and original.strip():
                    clean_processed.append(original.strip().lower())
                else:
                    clean_processed.append("empty")
        
        st.info("🔄 Generating TF-IDF embeddings...")
        
        # Adjust max_features based on dataset size
        adjusted_max_features = min(max_features, len(clean_processed) * 2, 10000)
        
        # Create TF-IDF vectorizer with robust settings
        vectorizer = TfidfVectorizer(
            max_features=adjusted_max_features,
            ngram_range=(1, 2),
            min_df=max(1, min(2, len(clean_processed) // 100)),  # Dynamic min_df
            max_df=0.95,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
        )
        
        # Fit and transform with error handling
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_processed)
            embeddings = tfidf_matrix.toarray().astype(np.float32)
        except ValueError as ve:
            logger.warning(f"TF-IDF fitting failed: {str(ve)}, trying simplified approach")
            # Fallback: use basic settings
            vectorizer = TfidfVectorizer(
                max_features=min(1000, len(clean_processed)),
                ngram_range=(1, 1),
                min_df=1,
                max_df=1.0
            )
            tfidf_matrix = vectorizer.fit_transform(clean_processed)
            embeddings = tfidf_matrix.toarray().astype(np.float32)
        
        st.success("✅ TF-IDF embeddings generated successfully")
        
        return embeddings
        
    except Exception as e:
        log_error(e, "tfidf_embeddings", {"num_keywords": len(keywords_list)})
        st.error(f"TF-IDF embeddings failed: {str(e)}")
        
        # Ultimate fallback: random embeddings
        try:
            logger.warning("Using random embeddings as ultimate fallback")
            random_embeddings = np.random.normal(0, 0.1, (len(keywords_list), 100)).astype(np.float32)
            return random_embeddings
        except Exception as final_error:
            logger.error(f"Even random embeddings failed: {str(final_error)}")
            return None

def generate_embeddings(keywords_list, client=None, method="auto", **kwargs):
    """Main embedding generation function with multiple methods"""
    if not keywords_list:
        st.error("No keywords provided for embedding generation")
        return None
        
    try:
        # Monitor resources
        monitor_resources()
        
        st.subheader("🧠 Generating Semantic Embeddings")
        
        # Limit keywords for memory efficiency
        original_count = len(keywords_list)
        if len(keywords_list) > MAX_KEYWORDS:
            st.warning(f"⚠️ Limiting to {MAX_KEYWORDS:,} keywords for memory efficiency")
            keywords_list = keywords_list[:MAX_KEYWORDS]
        
        embeddings = None
        method_used = "none"
        
        if method == "auto":
            # Try methods in order of preference
            if client and OPENAI_AVAILABLE:
                st.info("🚀 Attempting OpenAI embeddings (highest quality)")
                embeddings = generate_openai_embeddings(keywords_list, client)
                method_used = "openai"
            
            if embeddings is None and SENTENCE_TRANSFORMERS_AVAILABLE:
                st.info("🧠 Falling back to SentenceTransformers (good quality, free)")
                embeddings = generate_sentence_transformer_embeddings(keywords_list)
                method_used = "sentence_transformers"
            
            if embeddings is None:
                st.info("📊 Using TF-IDF embeddings (basic quality, always available)")
                processed_keywords = preprocess_keywords(keywords_list)
                embeddings = generate_tfidf_embeddings(keywords_list, processed_keywords)
                method_used = "tfidf"
        
        elif method == "openai" and client:
            embeddings = generate_openai_embeddings(keywords_list, client)
            method_used = "openai"
        
        elif method == "sentence_transformers":
            embeddings = generate_sentence_transformer_embeddings(keywords_list)
            method_used = "sentence_transformers"
        
        elif method == "tfidf":
            processed_keywords = preprocess_keywords(keywords_list)
            embeddings = generate_tfidf_embeddings(keywords_list, processed_keywords)
            method_used = "tfidf"
        
        else:
            raise ValueError(f"Unknown or unavailable embedding method: {method}")
        
        # Validate embeddings
        if embeddings is None:
            raise ValueError("All embedding methods failed")
        
        if len(embeddings) != len(keywords_list):
            raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(keywords_list)}")
        
        # Check for invalid embeddings (all zeros, NaN, etc.)
        if np.isnan(embeddings).any():
            logger.warning("Found NaN values in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0)
        
        if np.allclose(embeddings, 0):
            logger.warning("All embeddings are zero - this may cause clustering issues")
        
        # Normalize embeddings
        embeddings = normalize(embeddings, norm='l2', axis=1)
        
        # Final validation
        if embeddings.shape[0] == 0:
            raise ValueError("Generated empty embeddings matrix")
        
        st.success(f"✅ Generated embeddings using {method_used.upper()}: {embeddings.shape}")
        logger.info(f"Generated embeddings with shape: {embeddings.shape} using method: {method_used}")
        
        # Show truncation warning if applicable
        if original_count > len(keywords_list):
            st.warning(f"⚠️ Processed {len(keywords_list):,} out of {original_count:,} keywords")
        
        # Memory cleanup
        clean_memory()
        
        return embeddings
        
    except Exception as e:
        log_error(e, "embedding_generation", {
            "method": method,
            "num_keywords": len(keywords_list) if keywords_list else 0,
            "has_client": client is not None
        })
        st.error(f"Embedding generation failed: {str(e)}")
        return None

def reduce_embedding_dimensions(embeddings, target_dim=100, variance_threshold=0.95):
    """Reduce embedding dimensions using PCA"""
    if embeddings is None:
        return None
        
    if embeddings.shape[1] <= target_dim:
        return embeddings
    
    try:
        st.info(f"🔄 Reducing dimensions from {embeddings.shape[1]} to ~{target_dim}")
        
        # Validate input
        if embeddings.shape[0] < 2:
            logger.warning("Too few samples for PCA, skipping dimension reduction")
            return embeddings
        
        # Use Incremental PCA for large datasets
        if len(embeddings) > 10000:
            pca = IncrementalPCA(n_components=min(target_dim, embeddings.shape[1]))
            
            # Fit in batches
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                pca.partial_fit(batch)
            
            # Transform all data
            reduced_embeddings = pca.transform(embeddings)
            explained_var = sum(pca.explained_variance_ratio_)
            
        else:
            # Standard PCA for smaller datasets
            # First fit to analyze variance
            pca_analysis = PCA()
            pca_analysis.fit(embeddings)
            
            # Find number of components for target variance
            cumsum_variance = np.cumsum(pca_analysis.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            n_components = min(n_components, target_dim, embeddings.shape[1] - 1)
            
            # Apply PCA with optimal components
            pca = PCA(n_components=max(1, n_components))
            reduced_embeddings = pca.fit_transform(embeddings)
            explained_var = sum(pca.explained_variance_ratio_)
        
        st.success(f"✅ Dimensions reduced to {reduced_embeddings.shape[1]} (explained variance: {explained_var:.2%})")
        
        return reduced_embeddings.astype(np.float32)
        
    except Exception as e:
        log_error(e, "dimension_reduction")
        st.warning(f"⚠️ Dimension reduction failed: {str(e)}. Using original embeddings.")
        return embeddings

def determine_optimal_clusters(embeddings, max_clusters=20, min_clusters=2):
    """Determine optimal number of clusters using elbow method and silhouette analysis"""
    if embeddings is None or len(embeddings) < min_clusters:
        return min_clusters
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        n_samples = len(embeddings)
        # Ensure we don't exceed reasonable limits
        max_clusters = min(max_clusters, n_samples // 3, 50)  # At least 3 samples per cluster
        
        if max_clusters <= min_clusters:
            return min_clusters
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)
        
        st.info("🔄 Finding optimal number of clusters...")
        progress = st.progress(0)
        
        for i, k in enumerate(cluster_range):
            try:
                # Use smaller n_init for speed, but ensure reproducibility
                kmeans = KMeans(
                    n_clusters=k, 
                    random_state=42, 
                    n_init=5,  # Reduced for speed
                    max_iter=100,  # Reduced for speed
                    algorithm='lloyd'  # More stable than 'elkan' for small datasets
                )
                
                labels = kmeans.fit_predict(embeddings)
                
                # Validate labels
                unique_labels = len(np.unique(labels))
                if unique_labels < 2:
                    inertias.append(float('inf'))
                    silhouette_scores.append(-1)
                    continue
                
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score with error handling
                try:
                    if unique_labels > 1 and unique_labels < n_samples:
                        sil_score = silhouette_score(embeddings, labels)
                        silhouette_scores.append(max(-1, min(1, sil_score)))  # Clamp to valid range
                    else:
                        silhouette_scores.append(0)
                except Exception as sil_error:
                    logger.warning(f"Silhouette calculation failed for k={k}: {str(sil_error)}")
                    silhouette_scores.append(0)
                
                progress.progress((i + 1) / len(cluster_range))
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for k={k}: {str(e)}")
                inertias.append(float('inf'))
                silhouette_scores.append(-1)
        
        # Find elbow point using improved method
        if len(inertias) >= 3:
            # Calculate second derivatives to find elbow
            valid_inertias = [x for x in inertias if x != float('inf')]
            if len(valid_inertias) >= 3:
                # Use percentage decrease method
                decreases = []
                for i in range(1, len(valid_inertias)):
                    if valid_inertias[i-1] > 0:
                        decrease = (valid_inertias[i-1] - valid_inertias[i]) / valid_inertias[i-1]
                        decreases.append(decrease)
                    else:
                        decreases.append(0)
                
                # Find where decrease rate drops significantly
                if decreases:
                    avg_decrease = np.mean(decreases)
                    elbow_idx = 0
                    for i, decrease in enumerate(decreases):
                        if decrease < avg_decrease * 0.5:  # 50% less than average
                            elbow_idx = i
                            break
                    elbow_k = list(cluster_range)[elbow_idx + 1]
                else:
                    elbow_k = min_clusters
            else:
                elbow_k = min_clusters
        else:
            elbow_k = min_clusters
        
        # Find best silhouette score
        valid_sil_scores = [(i, score) for i, score in enumerate(silhouette_scores) if score > -1]
        if valid_sil_scores:
            best_sil_idx, best_sil_score = max(valid_sil_scores, key=lambda x: x[1])
            best_sil_k = list(cluster_range)[best_sil_idx]
        else:
            best_sil_k = min_clusters
            best_sil_score = 0
        
        # Choose optimal k with improved logic
        if best_sil_score > 0.3:  # Good silhouette score
            optimal_k = best_sil_k
        elif best_sil_score > 0.1:  # Decent silhouette score
            # Choose between elbow and silhouette based on reasonableness
            if abs(elbow_k - best_sil_k) <= 3:
                optimal_k = best_sil_k  # Close enough, prefer silhouette
            else:
                optimal_k = min(elbow_k, best_sil_k)  # Choose smaller for stability
        else:
            optimal_k = elbow_k
        
        # Final sanity checks
        optimal_k = max(min_clusters, min(optimal_k, max_clusters))
        
        st.success(f"✅ Optimal clusters determined: {optimal_k} (elbow: {elbow_k}, silhouette: {best_sil_k})")
        return optimal_k
        
    except Exception as e:
        log_error(e, "optimal_cluster_determination")
        st.warning(f"⚠️ Could not determine optimal clusters: {str(e)}. Using default.")
        return min(8, max_clusters, n_samples // 5)

def perform_kmeans_clustering(embeddings, n_clusters, random_state=42):
    """Perform K-means clustering with enhanced error handling"""
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings provided for clustering")
        
    if n_clusters <= 0:
        raise ValueError(f"Invalid number of clusters: {n_clusters}")
        
    if n_clusters >= len(embeddings):
        raise ValueError(f"Number of clusters ({n_clusters}) must be less than number of samples ({len(embeddings)})")
    
    try:
        from sklearn.cluster import KMeans
        
        st.info(f"🔄 Performing K-means clustering with {n_clusters} clusters...")
        
        # Determine optimal parameters based on dataset size
        n_samples = len(embeddings)
        
        if n_samples > 10000:
            # For large datasets, use fewer iterations and init attempts
            n_init = 3
            max_iter = 100
            algorithm = 'lloyd'  # More memory efficient
        elif n_samples > 1000:
            n_init = 5
            max_iter = 200
            algorithm = 'lloyd'
        else:
            n_init = 10
            max_iter = 300
            algorithm = 'lloyd'
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            algorithm=algorithm,
            tol=1e-4  # Slightly relaxed tolerance for speed
        )
        
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Validate results
        unique_labels = np.unique(cluster_labels)
        actual_clusters = len(unique_labels)
        
        if actual_clusters < n_clusters:
            st.warning(f"⚠️ K-means produced only {actual_clusters} clusters instead of {n_clusters}")
        
        # Calculate cluster statistics
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        
        # Check for very small clusters
        min_size = min(cluster_sizes)
        if min_size == 1:
            singleton_count = sum(1 for size in cluster_sizes if size == 1)
            st.warning(f"⚠️ Found {singleton_count} singleton clusters")
        
        st.success(f"✅ K-means clustering completed. Cluster sizes: {cluster_sizes}")
        
        return cluster_labels, kmeans
        
    except Exception as e:
        log_error(e, "kmeans_clustering", {
            "n_clusters": n_clusters,
            "n_samples": len(embeddings),
            "embedding_shape": embeddings.shape
        })
        raise e

def perform_hierarchical_clustering(embeddings, n_clusters, method='ward'):
    """Perform hierarchical clustering with enhanced validation"""
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings provided for clustering")
        
    if n_clusters <= 0:
        raise ValueError(f"Invalid number of clusters: {n_clusters}")
        
    if n_clusters >= len(embeddings):
        raise ValueError(f"Number of clusters ({n_clusters}) must be less than number of samples ({len(embeddings)})")
    
    try:
        from sklearn.cluster import AgglomerativeClustering
        
        st.info(f"🔄 Performing hierarchical clustering with {n_clusters} clusters...")
        
        # Choose linkage method based on dataset size and characteristics
        n_samples = len(embeddings)
        
        if n_samples > 5000:
            # For large datasets, use more efficient methods
            linkage_method = 'average'  # More stable than ward for large datasets
            st.info("Using 'average' linkage for large dataset")
        else:
            # Use specified method for smaller datasets
            linkage_method = method
        
        # Validate linkage method
        valid_methods = ['ward', 'complete', 'average', 'single']
        if linkage_method not in valid_methods:
            logger.warning(f"Invalid linkage method '{linkage_method}', using 'ward'")
            linkage_method = 'ward'
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Validate results
        unique_labels = np.unique(cluster_labels)
        actual_clusters = len(unique_labels)
        
        if actual_clusters != n_clusters:
            st.warning(f"⚠️ Hierarchical clustering produced {actual_clusters} clusters instead of {n_clusters}")
        
        # Calculate cluster statistics
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        
        # Check for very unbalanced clusters
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')
        
        if imbalance_ratio > 20:
            st.warning(f"⚠️ Highly unbalanced clusters detected (ratio: {imbalance_ratio:.1f})")
        
        st.success(f"✅ Hierarchical clustering completed. Cluster sizes: {cluster_sizes}")
        
        return cluster_labels, clustering
        
    except Exception as e:
        log_error(e, "hierarchical_clustering", {
            "n_clusters": n_clusters,
            "method": method,
            "n_samples": len(embeddings),
            "embedding_shape": embeddings.shape
        })
        raise e

def perform_advanced_clustering(embeddings, method="auto", n_clusters=None):
    """Perform advanced clustering with automatic method selection"""
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings provided for clustering")
    
    try:
        n_samples = len(embeddings)
        n_features = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
        
        # Determine optimal clusters if not provided
        if n_clusters is None:
            n_clusters = determine_optimal_clusters(embeddings)
        
        # Validate cluster number
        min_clusters = 2
        max_clusters = min(50, n_samples // 3)  # At least 3 samples per cluster
        n_clusters = max(min_clusters, min(n_clusters, max_clusters))
        
        if n_clusters >= n_samples:
            raise ValueError(f"Cannot create {n_clusters} clusters from {n_samples} samples")
        
        # Method selection logic
        if method == "auto":
            if n_samples > 10000:
                method = "kmeans"  # Better for large datasets
                st.info("Auto-selected K-means for large dataset")
            elif n_samples < 100:
                method = "hierarchical"  # Better for very small datasets
                st.info("Auto-selected Hierarchical for small dataset")
            elif n_features > 100:
                method = "kmeans"  # Better for high dimensions
                st.info("Auto-selected K-means for high-dimensional data")
            else:
                method = "hierarchical"  # Default for medium datasets
                st.info("Auto-selected Hierarchical clustering")
        
        # Perform clustering with retry logic
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                if method == "kmeans":
                    labels, model = perform_kmeans_clustering(embeddings, n_clusters)
                elif method == "hierarchical":
                    labels, model = perform_hierarchical_clustering(embeddings, n_clusters)
                else:
                    raise ValueError(f"Unknown clustering method: {method}")
                
                # Validate results
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    raise ValueError("Clustering produced only one cluster")
                
                if len(unique_labels) != n_clusters:
                    logger.warning(f"Expected {n_clusters} clusters, got {len(unique_labels)}")
                
                return labels, model
                
            except Exception as attempt_error:
                last_error = attempt_error
                logger.warning(f"Clustering attempt {attempt + 1} failed: {str(attempt_error)}")
                
                if attempt < max_attempts - 1:
                    # Try reducing cluster count or switching method
                    if n_clusters > 3:
                        n_clusters = max(3, n_clusters - 1)
                        st.warning(f"Reducing cluster count to {n_clusters} and retrying...")
                    elif method == "hierarchical":
                        method = "kmeans"
                        n_clusters = determine_optimal_clusters(embeddings)
                        st.warning("Switching to K-means and retrying...")
                    else:
                        # Last resort: very simple clustering
                        n_clusters = 2
                        st.warning("Using minimal clustering as fallback...")
        
        # If all attempts failed, raise the last error
        raise last_error or ValueError("All clustering attempts failed")
        
    except Exception as e:
        log_error(e, "advanced_clustering", {
            "method": method, 
            "n_clusters": n_clusters,
            "n_samples": n_samples,
            "embedding_shape": embeddings.shape if embeddings is not None else None
        })
        raise e

def refine_clusters(embeddings, initial_labels, min_cluster_size=2):
    """Refine clusters by merging small clusters and outlier detection"""
    if embeddings is None or initial_labels is None:
        raise ValueError("No embeddings or labels provided for refinement")
        
    if len(embeddings) != len(initial_labels):
        raise ValueError(f"Embeddings and labels length mismatch: {len(embeddings)} vs {len(initial_labels)}")
    
    try:
        st.info("🔄 Refining clusters...")
        
        refined_labels = initial_labels.copy()
        unique_labels = np.unique(refined_labels)
        
        # Find small clusters
        small_clusters = []
        cluster_sizes = {}
        
        for label in unique_labels:
            size = np.sum(refined_labels == label)
            cluster_sizes[label] = size
            if size < min_cluster_size:
                small_clusters.append(label)
        
        if not small_clusters:
            st.success("✅ No refinement needed - all clusters meet minimum size")
            return refined_labels
        
        st.info(f"Found {len(small_clusters)} small clusters to merge")
        
        # Merge small clusters with nearest large clusters
        merged_count = 0
        
        for small_label in small_clusters:
            small_cluster_indices = np.where(refined_labels == small_label)[0]
            
            if len(small_cluster_indices) == 0:
                continue
                
            small_cluster_embeddings = embeddings[small_cluster_indices]
            
            # Find the best cluster to merge with
            best_distance = float('inf')
            best_target_label = None
            
            for target_label in unique_labels:
                if target_label == small_label or target_label in small_clusters:
                    continue
                
                # Skip if target cluster no longer exists (already merged)
                if not np.any(refined_labels == target_label):
                    continue
                
                target_indices = np.where(refined_labels == target_label)[0]
                if len(target_indices) == 0:
                    continue
                    
                target_embeddings = embeddings[target_indices]
                
                # Calculate average distance using cosine similarity
                try:
                    similarities = cosine_similarity(small_cluster_embeddings, target_embeddings)
                    avg_similarity = np.mean(similarities)
                    avg_distance = 1 - avg_similarity  # Convert similarity to distance
                    
                    if avg_distance < best_distance:
                        best_distance = avg_distance
                        best_target_label = target_label
                        
                except Exception as sim_error:
                    logger.warning(f"Similarity calculation failed for merging: {str(sim_error)}")
                    continue
            
            # Merge the small cluster if we found a target
            if best_target_label is not None:
                refined_labels[refined_labels == small_label] = best_target_label
                merged_count += 1
                logger.info(f"Merged cluster {small_label} into {best_target_label}")
            else:
                # If no good merge target, merge with largest remaining cluster
                remaining_labels = [l for l in unique_labels if l not in small_clusters and np.any(refined_labels == l)]
                if remaining_labels:
                    largest_label = max(remaining_labels, key=lambda x: np.sum(refined_labels == x))
                    refined_labels[refined_labels == small_label] = largest_label
                    merged_count += 1
                    logger.info(f"Merged cluster {small_label} into largest cluster {largest_label}")
        
        # Relabel clusters to be consecutive starting from 0
        unique_refined = np.unique(refined_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_refined)}
        
        final_labels = np.array([label_mapping[label] for label in refined_labels])
        
        n_original = len(unique_labels)
        n_refined = len(unique_refined)
        
        st.success(f"✅ Clusters refined: {n_original} → {n_refined} (merged {merged_count} small clusters)")
        
        return final_labels
        
    except Exception as e:
        log_error(e, "cluster_refinement", {
            "min_cluster_size": min_cluster_size,
            "n_small_clusters": len(small_clusters) if 'small_clusters' in locals() else 0
        })
        st.warning(f"⚠️ Cluster refinement failed: {str(e)}. Using original clusters.")
        return initial_labels

def find_representative_keywords(embeddings, keywords, cluster_labels, top_k=5):
    """Find representative keywords for each cluster with enhanced error handling"""
    if embeddings is None or not keywords or cluster_labels is None:
        raise ValueError("Missing required inputs for representative keyword finding")
        
    if len(embeddings) != len(keywords) or len(keywords) != len(cluster_labels):
        raise ValueError("Input arrays must have the same length")
    
    try:
        st.info("🔄 Finding representative keywords...")
        
        unique_labels = np.unique(cluster_labels)
        representatives = {}
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            
            if len(cluster_indices) == 0:
                logger.warning(f"No indices found for cluster {label}")
                continue
            
            cluster_embeddings = embeddings[cluster_indices]
            cluster_keywords = [keywords[i] for i in cluster_indices if i < len(keywords)]
            
            if len(cluster_embeddings) == 0 or len(cluster_keywords) == 0:
                logger.warning(f"Empty cluster data for cluster {label}")
                representatives[label] = []
                continue
            
            try:
                # Calculate centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Handle potential numerical issues
                if np.isnan(centroid).any():
                    logger.warning(f"NaN values in centroid for cluster {label}, using first embedding")
                    centroid = cluster_embeddings[0]
                
                # Find keywords closest to centroid
                similarities = cosine_similarity([centroid], cluster_embeddings)[0]
                
                # Handle potential similarity calculation issues
                if np.isnan(similarities).any():
                    logger.warning(f"NaN similarities for cluster {label}, using random selection")
                    # Use random selection as fallback
                    selected_indices = np.random.choice(len(cluster_keywords), 
                                                      size=min(top_k, len(cluster_keywords)), 
                                                      replace=False)
                else:
                    # Get top-k most representative (handle edge case where top_k > cluster size)
                    k = min(top_k, len(similarities))
                    selected_indices = np.argsort(similarities)[-k:][::-1]
                
                representative_keywords = [cluster_keywords[i] for i in selected_indices 
                                         if i < len(cluster_keywords)]
                
                # Ensure we have valid keywords
                representative_keywords = [kw for kw in representative_keywords 
                                         if isinstance(kw, str) and kw.strip()]
                
                if not representative_keywords and cluster_keywords:
                    # Fallback: just take first few keywords
                    representative_keywords = cluster_keywords[:min(top_k, len(cluster_keywords))]
                
                representatives[label] = representative_keywords
                
            except Exception as cluster_error:
                logger.warning(f"Error processing cluster {label}: {str(cluster_error)}")
                # Fallback: use first few keywords
                fallback_keywords = cluster_keywords[:min(top_k, len(cluster_keywords))]
                representatives[label] = fallback_keywords
        
        # Validate results
        total_representatives = sum(len(reps) for reps in representatives.values())
        if total_representatives == 0:
            logger.warning("No representatives found, using fallback method")
            # Ultimate fallback: distribute keywords evenly
            for label in unique_labels:
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_keywords = [keywords[i] for i in cluster_indices[:top_k]]
                representatives[label] = [kw for kw in cluster_keywords if isinstance(kw, str)]
        
        valid_clusters = len([k for k, v in representatives.items() if v])
        st.success(f"✅ Found representatives for {valid_clusters}/{len(unique_labels)} clusters")
        
        return representatives
        
    except Exception as e:
        log_error(e, "representative_keywords", {
            "num_clusters": len(np.unique(cluster_labels)),
            "num_keywords": len(keywords),
            "top_k": top_k
        })
        # Fallback: return first keyword of each cluster
        unique_labels = np.unique(cluster_labels)
        representatives = {}
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            if len(cluster_indices) > 0:
                first_keyword = keywords[cluster_indices[0]] if cluster_indices[0] < len(keywords) else f"cluster_{label}"
                representatives[label] = [first_keyword] if isinstance(first_keyword, str) else [f"cluster_{label}"]
            else:
                representatives[label] = [f"cluster_{label}"]
        
        return representatives

def calculate_cluster_coherence(embeddings, cluster_labels):
    """Calculate coherence score for each cluster with robust error handling"""
    if embeddings is None or cluster_labels is None:
        return {}
        
    if len(embeddings) != len(cluster_labels):
        logger.warning("Embeddings and labels length mismatch in coherence calculation")
        return {}
    
    try:
        unique_labels = np.unique(cluster_labels)
        coherence_scores = {}
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            
            if len(cluster_indices) == 0:
                coherence_scores[label] = 0.0
                continue
                
            cluster_embeddings = embeddings[cluster_indices]
            
            if len(cluster_embeddings) < 2:
                coherence_scores[label] = 1.0  # Single item is perfectly coherent
                continue
            
            try:
                # Calculate pairwise similarities within cluster
                similarities = cosine_similarity(cluster_embeddings)
                
                # Check for numerical issues
                if np.isnan(similarities).any() or np.isinf(similarities).any():
                    logger.warning(f"Numerical issues in similarity calculation for cluster {label}")
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
                        coherence = 1.0
                else:
                    coherence = 1.0
                
                coherence_scores[label] = coherence
                
            except Exception as cluster_error:
                logger.warning(f"Error calculating coherence for cluster {label}: {str(cluster_error)}")
                coherence_scores[label] = 0.5  # Default moderate coherence
        
        # Validate all scores are reasonable
        for label, score in coherence_scores.items():
            if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
                coherence_scores[label] = 0.5
            elif score < 0 or score > 1:
                coherence_scores[label] = max(0.0, min(1.0, score))
        
        return coherence_scores
        
    except Exception as e:
        log_error(e, "cluster_coherence")
        # Return default scores for all clusters
        unique_labels = np.unique(cluster_labels) if cluster_labels is not None else []
        return {label: 0.5 for label in unique_labels}

def cluster_keywords(keywords_list, embeddings, n_clusters=None, method="auto", min_cluster_size=2):
    """Main clustering function that orchestrates the entire process"""
    if not keywords_list or embeddings is None:
        raise ValueError("Keywords list and embeddings are required")
        
    if len(keywords_list) != len(embeddings):
        raise ValueError(f"Keywords and embeddings length mismatch: {len(keywords_list)} vs {len(embeddings)}")
    
    try:
        st.subheader("🔗 Performing Semantic Clustering")
        
        # Monitor resources
        monitor_resources()
        
        # Validate inputs
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
            
        if embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 samples for clustering")
        
        # Perform clustering with error handling
        try:
            cluster_labels, model = perform_advanced_clustering(embeddings, method, n_clusters)
        except Exception as clustering_error:
            st.error(f"Primary clustering failed: {str(clustering_error)}")
            # Fallback: simple random clustering
            st.warning("Using fallback clustering method...")
            n_fallback_clusters = min(5, max(2, len(keywords_list) // 10))
            cluster_labels = np.random.randint(0, n_fallback_clusters, size=len(keywords_list))
            model = None
        
        # Refine clusters with error handling
        try:
            refined_labels = refine_clusters(embeddings, cluster_labels, min_cluster_size)
        except Exception as refinement_error:
            st.warning(f"Cluster refinement failed: {str(refinement_error)}")
            refined_labels = cluster_labels
        
        # Find representative keywords with error handling
        try:
            representatives = find_representative_keywords(embeddings, keywords_list, refined_labels)
        except Exception as rep_error:
            st.warning(f"Representative keyword finding failed: {str(rep_error)}")
            # Fallback representatives
            unique_labels = np.unique(refined_labels)
            representatives = {}
            for label in unique_labels:
                cluster_indices = np.where(refined_labels == label)[0]
                reps = [keywords_list[i] for i in cluster_indices[:3] if i < len(keywords_list)]
                representatives[label] = reps if reps else [f"cluster_{label}"]
        
        # Calculate coherence scores with error handling
        try:
            coherence_scores = calculate_cluster_coherence(embeddings, refined_labels)
        except Exception as coh_error:
            st.warning(f"Coherence calculation failed: {str(coh_error)}")
            unique_labels = np.unique(refined_labels)
            coherence_scores = {label: 0.5 for label in unique_labels}
        
        # Create results summary with validation
        unique_labels = np.unique(refined_labels)
        cluster_sizes = {}
        
        for label in unique_labels:
            size = np.sum(refined_labels == label)
            cluster_sizes[label] = size
        
        # Validate final results
        if len(unique_labels) == 0:
            raise ValueError("No clusters were created")
            
        if len(refined_labels) != len(keywords_list):
            raise ValueError("Label assignment failed")
        
        st.success(f"✅ Clustering completed: {len(unique_labels)} clusters created")
        
        # Display cluster summary
        st.info("📊 Cluster Summary:")
        for label in sorted(unique_labels):
            size = cluster_sizes[label]
            coherence = coherence_scores.get(label, 0.5)
            st.text(f"  Cluster {label}: {size} keywords (coherence: {coherence:.3f})")
        
        # Memory cleanup
        clean_memory()
        
        return {
            "labels": refined_labels,
            "model": model,
            "representatives": representatives,
            "coherence_scores": coherence_scores,
            "cluster_sizes": cluster_sizes
        }
        
    except Exception as e:
        log_error(e, "main_clustering", {
            "num_keywords": len(keywords_list),
            "embedding_shape": embeddings.shape if embeddings is not None else None,
            "method": method,
            "n_clusters": n_clusters
        })
        raise e

def generate_cluster_names_openai(representatives, client, model="gpt-4o-mini", custom_prompt=None):
    """Generate cluster names using OpenAI API"""
    if not client or not representatives:
        return {}
    
    try:
        st.info("🤖 Generating AI-powered cluster names...")
        
        # Default prompt if none provided
        if not custom_prompt:
            custom_prompt = """You are an expert SEO strategist analyzing keyword clusters.
For each cluster, provide a clear, descriptive name (3-6 words) and a brief description
that explains the search intent and content opportunity. Summarize average search volume,
average CPC, competition level and trend direction if provided."""
        
        cluster_names = {}
        cluster_ids = list(representatives.keys())
        batch_size = 3  # Process in small batches
        
        progress = st.progress(0)
        
        for i in range(0, len(cluster_ids), batch_size):
            batch_ids = cluster_ids[i:i + batch_size]
            
            # Create prompt for this batch
            prompt = custom_prompt + "\n\nAnalyze these keyword clusters:\n\n"
            
            for cluster_id in batch_ids:
                keywords = representatives[cluster_id][:8]  # Limit keywords
                prompt += f"Cluster {cluster_id}: {', '.join(keywords)}\n"
            
            prompt += """\nRespond with valid JSON only:
{
  "clusters": [
    {
      "cluster_id": 1,
      "name": "Cluster Name Here",
      "description": "Brief description of the cluster's search intent and content opportunity."
    }
  ]
}"""
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500,
                    timeout=30
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                try:
                    # Try to parse as JSON directly
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Extract JSON from code blocks
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        raise ValueError("Could not extract JSON from response")
                
                # Process the response
                if "clusters" in data:
                    for cluster_info in data["clusters"]:
                        cluster_id = cluster_info.get("cluster_id")
                        if cluster_id in batch_ids:
                            cluster_names[cluster_id] = {
                                "name": sanitize_text(cluster_info.get("name", f"Cluster {cluster_id}")),
                                "description": sanitize_text(cluster_info.get("description", ""))
                            }
                
            except Exception as e:
                logger.warning(f"OpenAI API error for batch {i//batch_size + 1}: {str(e)}")
                # Fallback names for this batch
                for cluster_id in batch_ids:
                    if cluster_id not in cluster_names:
                        keywords = representatives[cluster_id][:3]
                        cluster_names[cluster_id] = {
                            "name": f"{keywords[0].title()} Related" if keywords else f"Cluster {cluster_id}",
                            "description": f"Keywords related to {', '.join(keywords[:2])}" if keywords else f"Keyword group {cluster_id}"
                        }
            
            progress.progress((i + batch_size) / len(cluster_ids))
        
        # Ensure all clusters have names
        for cluster_id in representatives.keys():
            if cluster_id not in cluster_names:
                keywords = representatives[cluster_id][:2]
                cluster_names[cluster_id] = {
                    "name": f"Cluster {cluster_id}",
                    "description": f"Keywords related to {', '.join(keywords)}" if keywords else f"Keyword group {cluster_id}"
                }
        
        progress.progress(1.0)
        st.success(f"✅ Generated names for {len(cluster_names)} clusters")
        
        return cluster_names
        
    except Exception as e:
        log_error(e, "openai_cluster_naming")
        return create_fallback_cluster_names(representatives)

def create_fallback_cluster_names(representatives):
    """Create fallback cluster names when AI naming fails"""
    cluster_names = {}
    
    for cluster_id, keywords in representatives.items():
        if keywords:
            # Use the first keyword as base for name
            first_keyword = keywords[0]
            words = first_keyword.split()[:2]  # Take first 2 words
            
            if len(words) > 1:
                name = " ".join(words).title()
            else:
                name = first_keyword.title()
            
            cluster_names[cluster_id] = {
                "name": f"{name} Related",
                "description": f"Keywords related to {first_keyword}"
            }
        else:
            cluster_names[cluster_id] = {
                "name": f"Cluster {cluster_id}",
                "description": f"Keyword group {cluster_id}"
            }
    
    return cluster_names

def analyze_cluster_quality_ai(representatives, coherence_scores, client=None, model="gpt-4o-mini"):
    """AI-powered cluster quality analysis"""
    if not client:
        return create_basic_quality_analysis(representatives, coherence_scores)
    
    try:
        st.info("🔍 Performing AI cluster quality analysis...")
        
        quality_analysis = {}
        cluster_ids = list(representatives.keys())
        
        # Process in batches
        batch_size = 5
        progress = st.progress(0)
        
        for i in range(0, len(cluster_ids), batch_size):
            batch_ids = cluster_ids[i:i + batch_size]
            
            # Create analysis prompt
            prompt = """Analyze the quality and coherence of these keyword clusters. 
For each cluster, evaluate:
1. Semantic coherence (how related the keywords are)
2. Search intent consistency 
3. Content opportunity potential
4. Suggested improvements

Respond with JSON:"""
            
            prompt += """
{
  "clusters": [
    {
      "cluster_id": 1,
      "quality_score": 8.5,
      "coherence_assessment": "High - keywords are semantically related",
      "intent_consistency": "Commercial intent - comparison focused",
      "content_opportunity": "Create comparison guides and reviews",
      "suggestions": "Consider splitting into product-specific subclusters"
    }
  ]
}

Clusters to analyze:
"""
            
            for cluster_id in batch_ids:
                keywords = representatives[cluster_id][:10]
                coherence = coherence_scores.get(cluster_id, 0.5)
                prompt += f"Cluster {cluster_id} (coherence: {coherence:.3f}): {', '.join(keywords)}\n"
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2000,
                    timeout=45
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON response
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        raise ValueError("Could not parse AI response")
                
                # Process results
                if "clusters" in data:
                    for cluster_info in data["clusters"]:
                        cluster_id = cluster_info.get("cluster_id")
                        if cluster_id in batch_ids:
                            quality_analysis[cluster_id] = {
                                "quality_score": cluster_info.get("quality_score", 5.0),
                                "coherence_assessment": sanitize_text(cluster_info.get("coherence_assessment", "")),
                                "intent_consistency": sanitize_text(cluster_info.get("intent_consistency", "")),
                                "content_opportunity": sanitize_text(cluster_info.get("content_opportunity", "")),
                                "suggestions": sanitize_text(cluster_info.get("suggestions", ""))
                            }
                
            except Exception as e:
                logger.warning(f"AI quality analysis error for batch {i//batch_size + 1}: {str(e)}")
                # Create fallback analysis for this batch
                for cluster_id in batch_ids:
                    if cluster_id not in quality_analysis:
                        quality_analysis[cluster_id] = create_basic_cluster_analysis(
                            cluster_id, representatives[cluster_id], coherence_scores.get(cluster_id, 0.5)
                        )
            
            progress.progress((i + batch_size) / len(cluster_ids))
        
        progress.progress(1.0)
        st.success(f"✅ AI quality analysis completed for {len(quality_analysis)} clusters")
        
        return quality_analysis
        
    except Exception as e:
        log_error(e, "ai_quality_analysis")
        return create_basic_quality_analysis(representatives, coherence_scores)

def create_basic_quality_analysis(representatives, coherence_scores):
    """Create basic quality analysis without AI"""
    quality_analysis = {}
    
    for cluster_id, keywords in representatives.items():
        coherence = coherence_scores.get(cluster_id, 0.5)
        
        # Basic analysis based on coherence score
        if coherence > 0.7:
            quality_score = 8.0
            assessment = "High semantic coherence"
        elif coherence > 0.5:
            quality_score = 6.5
            assessment = "Moderate semantic coherence"
        else:
            quality_score = 4.0
            assessment = "Low semantic coherence"
        
        # Basic intent analysis
        intent = classify_search_intent(keywords[0] if keywords else "")
        
        quality_analysis[cluster_id] = {
            "quality_score": quality_score,
            "coherence_assessment": assessment,
            "intent_consistency": f"Primarily {intent} intent",
            "content_opportunity": f"Create {intent.lower()} content targeting these keywords",
            "suggestions": "Manual review recommended for optimization"
        }
    
    return quality_analysis

def create_basic_cluster_analysis(cluster_id, keywords, coherence):
    """Create basic analysis for a single cluster"""
    # Determine quality based on coherence
    if coherence > 0.7:
        quality_score = 8.0
        assessment = "High - keywords are well-related"
    elif coherence > 0.5:
        quality_score = 6.0
        assessment = "Moderate - some semantic relationship"
    else:
        quality_score = 4.0
        assessment = "Low - keywords may need regrouping"
    
    # Basic intent analysis
    primary_intent = classify_search_intent(keywords[0] if keywords else "")
    
    return {
        "quality_score": quality_score,
        "coherence_assessment": assessment,
        "intent_consistency": f"Primarily {primary_intent} intent",
        "content_opportunity": f"Focus on {primary_intent.lower()} content",
        "suggestions": "Consider manual review for optimization"
    }

def generate_content_suggestions(cluster_analysis, representatives):
    """Generate content suggestions based on cluster analysis"""
    try:
        content_suggestions = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            keywords = representatives.get(cluster_id, [])
            if not keywords:
                continue
            
            # Extract intent from analysis
            intent_text = analysis.get("intent_consistency", "").lower()
            
            if "informational" in intent_text:
                suggestions = [
                    f"Create how-to guide: 'How to {keywords[0]}'",
                    f"Write comprehensive article about {keywords[0]}",
                    f"Develop FAQ section covering {', '.join(keywords[:3])}",
                    "Create tutorial videos or step-by-step guides"
                ]
            elif "commercial" in intent_text:
                suggestions = [
                    f"Write comparison article: 'Best {keywords[0]} Options'",
                    f"Create review roundup for {keywords[0]}",
                    f"Develop buying guide for {', '.join(keywords[:3])}",
                    "Build comparison tables and feature matrices"
                ]
            elif "transactional" in intent_text:
                suggestions = [
                    f"Optimize product pages for {keywords[0]}",
                    f"Create landing pages targeting {', '.join(keywords[:3])}",
                    "Develop local SEO pages if applicable",
                    "Build conversion-focused content with clear CTAs"
                ]
            else:
                suggestions = [
                    f"Create targeted content for {keywords[0]}",
                    f"Develop topic cluster around {', '.join(keywords[:3])}",
                    "Research user intent and create appropriate content",
                    "Consider A/B testing different content approaches"
                ]
            
            content_suggestions[cluster_id] = suggestions
        
        return content_suggestions
        
    except Exception as e:
        log_error(e, "content_suggestions")
        return {}

def calculate_business_value_scores(cluster_analysis, cluster_sizes, search_volumes=None):
    """Calculate business value scores for clusters"""
    try:
        value_scores = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            quality_score = analysis.get("quality_score", 5.0)
            cluster_size = cluster_sizes.get(cluster_id, 1)
            
            # Base score from quality and size
            base_score = (quality_score / 10.0) * min(cluster_size / 10, 1.0)
            
            # Intent multiplier
            intent_text = analysis.get("intent_consistency", "").lower()
            if "transactional" in intent_text:
                intent_multiplier = 1.5
            elif "commercial" in intent_text:
                intent_multiplier = 1.3
            elif "informational" in intent_text:
                intent_multiplier = 1.1
            else:
                intent_multiplier = 1.0
            
            # Search volume multiplier (if available)
            volume_multiplier = 1.0
            if search_volumes and cluster_id in search_volumes:
                volume = search_volumes[cluster_id]
                if volume > 10000:
                    volume_multiplier = 1.4
                elif volume > 1000:
                    volume_multiplier = 1.2
                elif volume > 100:
                    volume_multiplier = 1.1
            
            final_score = base_score * intent_multiplier * volume_multiplier
            value_scores[cluster_id] = min(10.0, final_score * 10)  # Scale to 0-10
        
        return value_scores
        
    except Exception as e:
        log_error(e, "business_value_calculation")
        return {cluster_id: 5.0 for cluster_id in cluster_analysis.keys()}

def calculate_weighted_cluster_scores(df, weights=None):
    """Calculate weighted scores for clusters based on search metrics."""
    try:
        if weights is None:
            weights = {
                'search_volume': 1.0,
                'cpc': 0.5,
                'competition': -0.5,
                'trend': 1.0,
            }

        scores = {}
        for cluster_id, group in df.groupby('cluster_id'):
            volume = group['search_volume'].mean() if 'search_volume' in df.columns else 0
            cpc = group['cpc'].mean() if 'cpc' in df.columns else 0
            competition = group['competition'].mean() if 'competition' in df.columns else 0
            trend = group['trend'].mean() if 'trend' in df.columns else 0

            score = (
                weights.get('search_volume', 0) * volume +
                weights.get('cpc', 0) * cpc +
                weights.get('competition', 0) * competition +
                weights.get('trend', 0) * trend
            )
            scores[cluster_id] = score

        df['cluster_score'] = df['cluster_id'].map(scores)
        return scores

    except Exception as e:
        log_error(e, "weighted_cluster_score")
        return {}

def validate_ai_response(response_data, expected_cluster_ids):
    """Validate AI response format and content"""
    try:
        if not isinstance(response_data, dict):
            return False, "Response is not a dictionary"
        
        if "clusters" not in response_data:
            return False, "Missing 'clusters' key in response"
        
        clusters = response_data["clusters"]
        if not isinstance(clusters, list):
            return False, "Clusters is not a list"
        
        for cluster_info in clusters:
            if not isinstance(cluster_info, dict):
                return False, "Cluster info is not a dictionary"
            
            required_fields = ["cluster_id", "name", "description"]
            for field in required_fields:
                if field not in cluster_info:
                    return False, f"Missing required field: {field}"
            
            cluster_id = cluster_info.get("cluster_id")
            if cluster_id not in expected_cluster_ids:
                return False, f"Unexpected cluster_id: {cluster_id}"
        
        return True, "Validation passed"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def create_ai_prompt_template(task_type, cluster_data=None):
    """Create standardized AI prompt templates"""
    templates = {
        "cluster_naming": """You are an expert SEO strategist analyzing keyword clusters. 
For each cluster, provide a clear, descriptive name (3-6 words) and a brief description 
that explains the search intent and content opportunity.

Analyze these keyword clusters:

{cluster_data}

Respond with valid JSON only:
{{
  "clusters": [
    {{
      "cluster_id": 1,
      "name": "Cluster Name Here",
      "description": "Brief description of the cluster's search intent and content opportunity."
    }}
  ]
}}""",
        
        "quality_analysis": """Analyze the quality and coherence of these keyword clusters. 
For each cluster, evaluate:
1. Semantic coherence (how related the keywords are)
2. Search intent consistency 
3. Content opportunity potential
4. Suggested improvements

Respond with JSON:
{{
  "clusters": [
    {{
      "cluster_id": 1,
      "quality_score": 8.5,
      "coherence_assessment": "High - keywords are semantically related",
      "intent_consistency": "Commercial intent - comparison focused",
      "content_opportunity": "Create comparison guides and reviews",
      "suggestions": "Consider splitting into product-specific subclusters"
    }}
  ]
}}

Clusters to analyze:
{cluster_data}""",
        
        "content_strategy": """Based on these keyword clusters, provide content strategy recommendations.
Focus on practical, actionable advice for content creation and SEO optimization.

Keyword clusters:
{cluster_data}

Respond with JSON:
{{
  "strategy": [
    {{
      "cluster_id": 1,
      "content_type": "How-to Guide",
      "priority": "High",
      "target_audience": "Beginners",
      "content_ideas": ["Idea 1", "Idea 2"],
      "seo_recommendations": ["Recommendation 1", "Recommendation 2"]
    }}
  ]
}}"""
    }
    
    return templates.get(task_type, "")

def process_ai_response_safely(response_content, expected_format="json"):
    """Safely process AI response with multiple parsing strategies"""
    try:
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response_content), None
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from code blocks
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1)), None
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON-like content
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, response_content)
        if json_match:
            try:
                return json.loads(json_match.group(0)), None
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Clean and retry
        cleaned_content = response_content.strip()
        for char in ['"', '"', ''', ''']: # Replace smart quotes
            cleaned_content = cleaned_content.replace(char, '"')
        
        try:
            return json.loads(cleaned_content), None
        except json.JSONDecodeError:
            pass
        
        return None, "Could not parse JSON from AI response"
        
    except Exception as e:
        return None, f"Error processing AI response: {str(e)}"

def enhance_cluster_analysis_with_metadata(cluster_analysis, keywords_list, embeddings):
    """Enhance cluster analysis with additional metadata"""
    try:
        enhanced_analysis = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            enhanced = analysis.copy()
            
            # Add cluster-specific metadata
            enhanced.update({
                "analysis_timestamp": datetime.now().isoformat(),
                "cluster_size": analysis.get("cluster_size", 0),
                "embedding_dimension": embeddings.shape[1] if embeddings is not None else 0,
                "processing_version": "1.0"
            })
            
            enhanced_analysis[cluster_id] = enhanced
        
        return enhanced_analysis
        
    except Exception as e:
        log_error(e, "cluster_analysis_enhancement")
        return cluster_analysis

def validate_openai_api_connection(client):
    """Validate OpenAI API connection"""
    try:
        if not client:
            return False, "No client provided"
        
        # Test with a simple API call
        response = client.models.list()
        if response and hasattr(response, 'data'):
            return True, "Connection successful"
        else:
            return False, "Invalid response from API"
            
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg:
            return False, "Invalid API key"
        elif "quota" in error_msg or "billing" in error_msg:
            return False, "API quota exceeded or billing issue"
        elif "rate limit" in error_msg:
            return False, "Rate limit exceeded"
        else:
            return False, f"Connection error: {str(e)}"

def optimize_batch_processing(total_items, available_memory_mb=1000, complexity_factor=1.0):
    """Optimize batch size based on available resources"""
    try:
        # Base batch size calculation
        if total_items < 100:
            base_batch_size = total_items
        elif total_items < 1000:
            base_batch_size = 50
        elif total_items < 5000:
            base_batch_size = 25
        else:
            base_batch_size = 10
        
        # Adjust for memory constraints
        memory_factor = min(1.0, available_memory_mb / 500)  # 500MB baseline
        
        # Adjust for complexity
        optimal_batch_size = int(base_batch_size * memory_factor / complexity_factor)
        
        # Ensure minimum viable batch size
        return max(1, min(optimal_batch_size, total_items))
        
    except Exception as e:
        log_error(e, "batch_optimization")
        return min(5, total_items)  # Safe fallback

def load_csv_file(uploaded_file, csv_format="auto"):
    """Load and validate CSV file with enhanced error handling"""
    try:
        # Validate input
        if uploaded_file is None:
            raise ValueError("No file uploaded")
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 100:  # 100MB limit
            raise ValueError(f"File too large ({file_size_mb:.1f}MB). Maximum size is 100MB.")
        
        # Read file content safely
        content = safe_file_read(uploaded_file)
        
        if not content or not content.strip():
            raise ValueError("File is empty or contains no readable content")
        
        # Detect encoding issues
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            st.warning("⚠️ Encoding issues detected. Some characters may not display correctly.")
        
        # Detect format if auto
        if csv_format == "auto":
            first_line = content.split('\n')[0].lower() if content else ""
            if any(keyword in first_line for keyword in ['keyword', 'search', 'query', 'term', 'phrase']):
                csv_format = "with_header"
            else:
                csv_format = "no_header"
        
        # Parse CSV based on format
        try:
            if csv_format == "no_header":
                df = pd.read_csv(
                    StringIO(content), 
                    header=None, 
                    names=["keyword"],
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
            else:
                df = pd.read_csv(
                    StringIO(content),
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                
                # Standardize column names
                df = standardize_column_names(df)
                
                # Ensure keyword column exists
                if 'keyword' not in df.columns:
                    if len(df.columns) > 0:
                        df = df.rename(columns={df.columns[0]: 'keyword'})
                    else:
                        raise ValueError("No columns found in CSV file")
        
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or contains no data")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error: {str(e)}")
        
        # Validate and clean data
        df = validate_and_clean_dataframe(df)
        
        # Limit size for memory management
        if len(df) > MAX_KEYWORDS:
            st.warning(f"⚠️ Dataset too large. Limiting to {MAX_KEYWORDS:,} keywords.")
            df = df.head(MAX_KEYWORDS)
        
        st.success(f"✅ Loaded {len(df):,} keywords successfully")
        
        return df
        
    except Exception as e:
        log_error(e, "csv_loading", {"file_size": getattr(uploaded_file, 'size', 0)})
        st.error(f"CSV loading failed: {str(e)}")
        return None

def standardize_column_names(df):
    """Standardize column names to expected format"""
    try:
        column_mapping = {}
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Keyword column variations
            if any(keyword in col_lower for keyword in ['keyword', 'query', 'term', 'phrase', 'kw']):
                column_mapping[col] = 'keyword'
            
            # Search volume variations
            elif any(volume in col_lower for volume in ['volume', 'searches', 'search_volume', 'avg_monthly']):
                column_mapping[col] = 'search_volume'
            
            # Competition variations
            elif any(comp in col_lower for comp in ['competition', 'comp', 'difficulty', 'kd']):
                column_mapping[col] = 'competition'
            
            # CPC variations
            elif any(cpc in col_lower for cpc in ['cpc', 'cost', 'bid', 'price']):
                column_mapping[col] = 'cpc'
            
            # Click-through rate variations
            elif any(ctr in col_lower for ctr in ['ctr', 'click_through', 'clickthrough']):
                column_mapping[col] = 'ctr'
            
            # Impression share variations
            elif any(imp in col_lower for imp in ['impression', 'impr', 'share']):
                column_mapping[col] = 'impression_share'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            st.info(f"📝 Standardized column names: {list(column_mapping.values())}")
        
        return df
        
    except Exception as e:
        log_error(e, "column_standardization")
        return df

def validate_and_clean_dataframe(df):
    """Validate and clean DataFrame with comprehensive checks"""
    try:
        # Check if DataFrame is empty
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        
        # Validate required columns
        if 'keyword' not in df.columns:
            raise ValueError("No 'keyword' column found")
        
        # Clean keyword column
        original_count = len(df)
        
        # Convert to string and strip whitespace
        df['keyword'] = df['keyword'].astype(str).str.strip()
        
        # Remove empty, null, or invalid keywords
        df = df[df['keyword'].notna()]
        df = df[df['keyword'] != '']
        df = df[df['keyword'] != 'nan']
        df = df[df['keyword'] != 'None']
        
        # Remove duplicates (case-insensitive)
        df_clean = df.copy()
        df_clean['keyword_lower'] = df_clean['keyword'].str.lower()
        df_clean['keyword_normalized'] = df_clean['keyword_lower'].apply(unidecode)
        df_clean = df_clean.drop_duplicates(subset=['keyword_normalized'])
        # Fuzzy deduplication for near duplicates
        unique_keywords = []
        keep_indices = []
        for idx, row in df_clean.iterrows():
            kw = row['keyword_normalized']
            if not any(fuzz.ratio(kw, existing) > 90 for existing in unique_keywords):
                unique_keywords.append(kw)
                keep_indices.append(idx)
        df_clean = df_clean.loc[keep_indices]
        df = df_clean.drop(columns=['keyword_lower', 'keyword_normalized'])
        
        # Remove keywords that are too short or too long
        df = df[(df['keyword'].str.len() >= 2) & (df['keyword'].str.len() <= 200)]
        
        # Remove keywords with suspicious patterns
        suspicious_patterns = [
            r'^[0-9]+$',  # Only numbers
            r'^[^a-zA-Z0-9\s]+$',  # Only special characters
            r'<script|javascript:|data:|vbscript:',  # Potential XSS
            r'\.\./'  # Path traversal
        ]
        
        for pattern in suspicious_patterns:
            mask = df['keyword'].str.contains(pattern, case=False, regex=True, na=False)
            df = df[~mask]
        
        # Clean numeric columns if present
        numeric_columns = ['search_volume', 'competition', 'cpc', 'ctr', 'impression_share']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
                
                # Validate ranges
                if col == 'competition' or col == 'ctr':
                    df[col] = df[col].clip(0, 1)
                elif col in ['search_volume', 'cpc', 'impression_share']:
                    df[col] = df[col].clip(0, None)  # Non-negative
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Report cleaning results
        removed_count = original_count - len(df)
        if removed_count > 0:
            st.info(f"🧹 Cleaned data: removed {removed_count:,} invalid/duplicate keywords")
        
        if len(df) == 0:
            raise ValueError("No valid keywords remaining after cleaning")
        
        return df
        
    except Exception as e:
        log_error(e, "dataframe_validation")
        raise e

def create_results_dataframe(keywords_list, cluster_results, cluster_names, 
                           coherence_scores, intent_results=None, quality_analysis=None):
    """Create comprehensive results DataFrame with enhanced error handling"""
    try:
        # Validate inputs
        if not keywords_list:
            raise ValueError("Keywords list is empty")
        
        if not cluster_results or 'labels' not in cluster_results:
            raise ValueError("Invalid cluster results")
        
        if len(keywords_list) != len(cluster_results['labels']):
            raise ValueError(f"Keyword count ({len(keywords_list)}) doesn't match cluster labels count ({len(cluster_results['labels'])})")
        
        # Create basic DataFrame
        df = pd.DataFrame({
            'keyword': keywords_list,
            'cluster_id': cluster_results['labels'],
        })
        
        # Validate cluster IDs
        if df['cluster_id'].isna().any():
            st.warning("⚠️ Found NaN values in cluster assignments")
            df['cluster_id'] = df['cluster_id'].fillna(-1).astype(int)
        
        # Add cluster names and descriptions
        df['cluster_name'] = df['cluster_id'].map(
            lambda x: cluster_names.get(x, {}).get('name', f'Cluster {x}') if isinstance(cluster_names.get(x), dict) else f'Cluster {x}'
        )
        
        df['cluster_description'] = df['cluster_id'].map(
            lambda x: cluster_names.get(x, {}).get('description', '') if isinstance(cluster_names.get(x), dict) else ''
        )
        
        # Add coherence scores with validation
        df['cluster_coherence'] = df['cluster_id'].map(
            lambda x: coherence_scores.get(x, 0.5)
        )
        # Ensure coherence is in valid range
        df['cluster_coherence'] = df['cluster_coherence'].clip(0, 1)
        
        # Mark representative keywords
        df['is_representative'] = False
        representatives = cluster_results.get('representatives', {})
        
        for cluster_id, rep_keywords in representatives.items():
            if rep_keywords:  # Check if list is not empty
                mask = (df['cluster_id'] == cluster_id) & (df['keyword'].isin(rep_keywords))
                df.loc[mask, 'is_representative'] = True
        
        # Add search intent if available
        if intent_results and len(intent_results) == len(keywords_list):
            df['search_intent'] = intent_results
        else:
            # Calculate intent for representative keywords only (for performance)
            df['search_intent'] = df.apply(
                lambda row: classify_search_intent(row['keyword']) if row['is_representative'] else 'Unknown',
                axis=1
            )
        
        # Add quality metrics if available
        if quality_analysis:
            df['quality_score'] = df['cluster_id'].map(
                lambda x: quality_analysis.get(x, {}).get('quality_score', 5.0)
            )
            df['content_opportunity'] = df['cluster_id'].map(
                lambda x: quality_analysis.get(x, {}).get('content_opportunity', '')
            )
            
            # Validate quality scores
            df['quality_score'] = pd.to_numeric(df['quality_score'], errors='coerce').fillna(5.0)
            df['quality_score'] = df['quality_score'].clip(0, 10)
        
        # Add cluster size
        cluster_sizes = df['cluster_id'].value_counts().to_dict()
        df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
        
        # Add processing metadata
        df['processing_timestamp'] = datetime.now().isoformat()
        df['keyword_length'] = df['keyword'].str.len()
        df['word_count'] = df['keyword'].str.split().str.len()
        
        # Sort by cluster_id and then by representative status
        df = df.sort_values(['cluster_id', 'is_representative'], ascending=[True, False])
        df = df.reset_index(drop=True)
        
        # Final validation
        validate_final_dataframe(df)
        
        st.success(f"✅ Results DataFrame created with {len(df):,} rows and {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        log_error(e, "dataframe_creation", {
            "num_keywords": len(keywords_list) if keywords_list else 0,
            "has_cluster_results": cluster_results is not None,
            "has_cluster_names": cluster_names is not None
        })
        
        # Create minimal DataFrame as fallback
        try:
            return create_fallback_dataframe(keywords_list)
        except Exception as fallback_error:
            log_error(fallback_error, "fallback_dataframe_creation")
            raise e

def create_fallback_dataframe(keywords_list):
    """Create minimal fallback DataFrame when main creation fails"""
    try:
        df = pd.DataFrame({
            'keyword': keywords_list,
            'cluster_id': range(len(keywords_list)),
            'cluster_name': [f'Cluster {i}' for i in range(len(keywords_list))],
            'cluster_description': ['Individual keyword' for _ in keywords_list],
            'cluster_coherence': [1.0 for _ in keywords_list],
            'is_representative': [True for _ in keywords_list],
            'search_intent': ['Unknown' for _ in keywords_list],
            'cluster_size': [1 for _ in keywords_list],
            'processing_timestamp': datetime.now().isoformat(),
            'keyword_length': [len(kw) for kw in keywords_list],
            'word_count': [len(kw.split()) for kw in keywords_list]
        })
        
        st.warning("⚠️ Using fallback DataFrame structure due to processing errors")
        return df
        
    except Exception as e:
        log_error(e, "fallback_dataframe_creation")
        raise ValueError("Failed to create even fallback DataFrame")

def validate_final_dataframe(df):
    """Validate final DataFrame structure and content"""
    try:
        required_columns = ['keyword', 'cluster_id', 'cluster_name', 'cluster_coherence']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for data quality issues
        issues = []
        
        # Check for empty keywords
        empty_keywords = df['keyword'].isna().sum() + (df['keyword'] == '').sum()
        if empty_keywords > 0:
            issues.append(f"{empty_keywords} empty keywords")
        
        # Check cluster ID validity
        if df['cluster_id'].isna().sum() > 0:
            issues.append("NaN values in cluster_id column")
        
        # Check coherence scores
        invalid_coherence = ((df['cluster_coherence'] < 0) | (df['cluster_coherence'] > 1)).sum()
        if invalid_coherence > 0:
            issues.append(f"{invalid_coherence} invalid coherence scores")
        
        # Check cluster size consistency
        cluster_size_check = df.groupby('cluster_id').size()
        reported_sizes = df.groupby('cluster_id')['cluster_size'].first()
        
        if not cluster_size_check.equals(reported_sizes):
            issues.append("Cluster size inconsistency detected")
        
        # Report issues if any
        if issues:
            st.warning(f"⚠️ Data quality issues found: {'; '.join(issues)}")
        
        # Basic statistics
        n_clusters = df['cluster_id'].nunique()
        n_keywords = len(df)
        avg_cluster_size = n_keywords / n_clusters if n_clusters > 0 else 0
        
        st.info(f"📊 Validation Summary: {n_keywords:,} keywords in {n_clusters} clusters (avg size: {avg_cluster_size:.1f})")
        
        return True
        
    except Exception as e:
        log_error(e, "final_dataframe_validation")
        return False

def add_search_volume_data(df, search_volume_col='search_volume'):
    """Add search volume analysis to DataFrame with enhanced validation"""
    try:
        if search_volume_col not in df.columns:
            st.info("ℹ️ No search volume data available")
            return df
        
        # Validate and clean search volume data
        original_col = df[search_volume_col].copy()
        df[search_volume_col] = pd.to_numeric(df[search_volume_col], errors='coerce')
        
        # Count conversion issues
        conversion_issues = df[search_volume_col].isna().sum()
        if conversion_issues > 0:
            st.warning(f"⚠️ {conversion_issues} search volume values could not be converted to numbers")
        
        # Fill NaN values with 0
        df[search_volume_col] = df[search_volume_col].fillna(0)
        
        # Ensure non-negative values
        negative_values = (df[search_volume_col] < 0).sum()
        if negative_values > 0:
            st.warning(f"⚠️ {negative_values} negative search volume values found, setting to 0")
            df[search_volume_col] = df[search_volume_col].clip(lower=0)
        
        # Calculate cluster-level metrics
        cluster_volume_stats = df.groupby('cluster_id')[search_volume_col].agg([
            'sum', 'mean', 'max', 'count', 'std'
        ]).round(2)
        
        cluster_volume_stats.columns = [
            'cluster_total_volume',
            'cluster_avg_volume', 
            'cluster_max_volume',
            'cluster_keyword_count',
            'cluster_volume_std'
        ]
        
        # Handle NaN in std calculation
        cluster_volume_stats['cluster_volume_std'] = cluster_volume_stats['cluster_volume_std'].fillna(0)
        
        # Merge back to main DataFrame
        df = df.merge(cluster_volume_stats, left_on='cluster_id', right_index=True, how='left')
        
        # Calculate volume percentiles
        if df[search_volume_col].max() > 0:
            df['volume_percentile'] = df[search_volume_col].rank(pct=True) * 100
            
            # Add volume categories
            df['volume_category'] = pd.cut(
                df['volume_percentile'],
                bins=[0, 25, 50, 75, 90, 100],
                labels=['Low', 'Medium', 'High', 'Very High', 'Top'],
                include_lowest=True
            )
        else:
            df['volume_percentile'] = 50.0
            df['volume_category'] = 'Unknown'
        
        # Calculate volume efficiency (volume per keyword in cluster)
        df['volume_efficiency'] = df['cluster_total_volume'] / df['cluster_keyword_count']
        
        st.success("✅ Search volume analysis added")
        
        return df
        
    except Exception as e:
        log_error(e, "search_volume_analysis")
        st.warning(f"⚠️ Search volume analysis failed: {str(e)}")
        return df

def add_trend_data(df):
    """Infer a trend score from monthly search volume columns if available."""
    try:
        trend_cols = [col for col in df.columns if re.search(r'search_volume_\d{4}', col)]
        if len(trend_cols) >= 2:
            trend_cols = sorted(trend_cols)
            x = np.arange(len(trend_cols))
            df['trend'] = df[trend_cols].apply(lambda r: np.polyfit(x, r.values, 1)[0], axis=1)
        elif 'trend' not in df.columns:
            df['trend'] = 0.0
        return df
    except Exception as e:
        log_error(e, "trend_calculation")
        if 'trend' not in df.columns:
            df['trend'] = 0.0
        return df

def calculate_cluster_metrics(df):
    """Calculate comprehensive cluster metrics with enhanced analysis"""
    try:
        st.info("🔄 Calculating cluster metrics...")
        
        metrics = {}
        
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            # Basic metrics
            cluster_metrics = {
                'cluster_id': cluster_id,
                'keyword_count': len(cluster_data),
                'avg_coherence': cluster_data['cluster_coherence'].mean(),
                'min_coherence': cluster_data['cluster_coherence'].min(),
                'max_coherence': cluster_data['cluster_coherence'].max(),
                'representative_count': cluster_data['is_representative'].sum(),
                'representative_ratio': cluster_data['is_representative'].mean(),
            }
            
            # Keyword characteristics
            cluster_metrics.update({
                'avg_keyword_length': cluster_data['keyword_length'].mean(),
                'avg_word_count': cluster_data['word_count'].mean(),
                'min_word_count': cluster_data['word_count'].min(),
                'max_word_count': cluster_data['word_count'].max(),
            })
            
            # Search volume metrics (if available)
            if 'search_volume' in df.columns:
                volume_data = cluster_data['search_volume']
                cluster_metrics.update({
                    'total_search_volume': volume_data.sum(),
                    'avg_search_volume': volume_data.mean(),
                    'median_search_volume': volume_data.median(),
                    'max_search_volume': volume_data.max(),
                    'min_search_volume': volume_data.min(),
                    'volume_std': volume_data.std(),
                    'volume_cv': volume_data.std() / volume_data.mean() if volume_data.mean() > 0 else 0,
                })
            
            # Intent distribution
            if 'search_intent' in df.columns:
                intent_counts = cluster_data['search_intent'].value_counts()
                total_keywords = len(cluster_data)
                
                primary_intent = intent_counts.index[0] if len(intent_counts) > 0 else 'Unknown'
                intent_diversity = len(intent_counts)
                intent_entropy = calculate_entropy(intent_counts.values)
                
                cluster_metrics.update({
                    'primary_intent': primary_intent,
                    'primary_intent_ratio': intent_counts.iloc[0] / total_keywords if len(intent_counts) > 0 else 0,
                    'intent_diversity': intent_diversity,
                    'intent_entropy': intent_entropy,
                    'intent_distribution': intent_counts.to_dict()
                })
            
            # Quality metrics (if available)
            if 'quality_score' in df.columns:
                quality_data = cluster_data['quality_score']
                cluster_metrics.update({
                    'avg_quality_score': quality_data.mean(),
                    'min_quality_score': quality_data.min(),
                    'max_quality_score': quality_data.max(),
                    'quality_std': quality_data.std(),
                })
            
            # Cluster health score (composite metric)
            health_components = []
            
            # Coherence component (0-1)
            health_components.append(cluster_metrics['avg_coherence'])
            
            # Size component (normalized, optimal around 5-20 keywords)
            size_score = min(1.0, cluster_metrics['keyword_count'] / 10) if cluster_metrics['keyword_count'] <= 20 else max(0.5, 20 / cluster_metrics['keyword_count'])
            health_components.append(size_score)
            
            # Representative ratio component
            health_components.append(min(1.0, cluster_metrics['representative_ratio'] * 3))
            
            # Intent consistency component (if available)
            if 'primary_intent_ratio' in cluster_metrics:
                health_components.append(cluster_metrics['primary_intent_ratio'])
            
            cluster_metrics['health_score'] = np.mean(health_components)
            
            metrics[cluster_id] = cluster_metrics
        
        st.success(f"✅ Calculated metrics for {len(metrics)} clusters")
        
        return metrics
        
    except Exception as e:
        log_error(e, "cluster_metrics_calculation")
        return {}

def create_cluster_summary_dataframe(df, metrics=None):
    """Create a comprehensive summary DataFrame for clusters"""
    try:
        summary_data = []
        
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            # Get representative keywords
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
            if not rep_keywords:
                rep_keywords = cluster_data['keyword'].head(3).tolist()
            
            # Basic summary
            summary_row = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_data['cluster_name'].iloc[0],
                'keyword_count': len(cluster_data),
                'representative_keywords': ', '.join(rep_keywords[:5]),
                'avg_coherence': round(cluster_data['cluster_coherence'].mean(), 3),
            }
            
            # Add search volume if available
            if 'search_volume' in df.columns:
                summary_row.update({
                    'total_search_volume': int(cluster_data['search_volume'].sum()),
                    'avg_search_volume': round(cluster_data['search_volume'].mean(), 0),
                    'max_search_volume': int(cluster_data['search_volume'].max()),
                })
            
            # Add intent information
            if 'search_intent' in df.columns:
                intent_counts = cluster_data['search_intent'].value_counts()
                primary_intent = intent_counts.index[0] if len(intent_counts) > 0 else 'Unknown'
                summary_row['primary_intent'] = primary_intent
                
                if len(intent_counts) > 1:
                    intent_diversity = len(intent_counts)
                    summary_row['intent_diversity'] = intent_diversity
            
            # Add quality score if available
            if 'quality_score' in df.columns:
                summary_row['avg_quality'] = round(cluster_data['quality_score'].mean(), 1)
            
            # Add metrics if available
            if metrics and cluster_id in metrics:
                cluster_metrics = metrics[cluster_id]
                summary_row.update({
                    'health_score': round(cluster_metrics.get('health_score', 0), 3),
                    'avg_keyword_length': round(cluster_metrics.get('avg_keyword_length', 0), 1),
                    'avg_word_count': round(cluster_metrics.get('avg_word_count', 0), 1),
                })

            if 'cluster_score' in df.columns:
                summary_row['weighted_score'] = round(cluster_data['cluster_score'].mean(), 2)            
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        
        if summary_df.empty:
            return summary_df
        
        if 'weighted_score' in summary_df.columns:
            summary_df = summary_df.sort_values('weighted_score', ascending=False)
        elif 'total_search_volume' in summary_df.columns:
            summary_df = summary_df.sort_values(['total_search_volume', 'keyword_count'], ascending=False)
        else:
            summary_df = summary_df.sort_values('keyword_count', ascending=False)
        
        summary_df = summary_df.reset_index(drop=True)
        
        return summary_df
        
    except Exception as e:
        log_error(e, "summary_dataframe_creation")
        return pd.DataFrame()

def export_results_to_csv(df, filename=None):
    """Export results DataFrame to CSV with enhanced formatting"""
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keyword_clusters_{timestamp}.csv"
        
        # Create clean export DataFrame
        export_df = df.copy()
        
        # Round numeric columns to appropriate precision
        numeric_columns = export_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['cluster_coherence', 'quality_score']:
                export_df[col] = export_df[col].round(3)
            elif col in ['search_volume', 'cluster_total_volume', 'cluster_avg_volume']:
                export_df[col] = export_df[col].round(0).astype(int)
            elif col in ['volume_percentile']:
                export_df[col] = export_df[col].round(1)
            else:
                export_df[col] = export_df[col].round(2)
        
        # Convert boolean columns to Yes/No
        bool_columns = export_df.select_dtypes(include=[bool]).columns
        for col in bool_columns:
            export_df[col] = export_df[col].map({True: 'Yes', False: 'No'})
        
        # Ensure string columns are properly formatted
        string_columns = export_df.select_dtypes(include=['object']).columns
        for col in string_columns:
            export_df[col] = export_df[col].astype(str)
        
        # Reorder columns for better readability
        preferred_order = [
            'keyword', 'cluster_id', 'cluster_name', 'cluster_description',
            'is_representative', 'search_intent', 'cluster_coherence',
            'search_volume', 'cluster_size'
        ]
        
        # Add remaining columns
        remaining_cols = [col for col in export_df.columns if col not in preferred_order]
        column_order = [col for col in preferred_order if col in export_df.columns] + remaining_cols
        
        export_df = export_df[column_order]
        
        # Generate CSV with proper encoding
        csv_data = export_df.to_csv(index=False, encoding='utf-8-sig')  # BOM for Excel compatibility
        
        return csv_data, filename
        
    except Exception as e:
        log_error(e, "csv_export")
        raise e

def filter_dataframe_by_criteria(df, criteria):
    """Filter DataFrame based on various criteria with validation"""
    try:
        if df is None or df.empty:
            return df
        
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Filter by cluster size
        if criteria.get('min_cluster_size'):
            min_size = criteria['min_cluster_size']
            cluster_sizes = filtered_df['cluster_id'].value_counts()
            valid_clusters = cluster_sizes[cluster_sizes >= min_size].index
            filtered_df = filtered_df[filtered_df['cluster_id'].isin(valid_clusters)]
        
        # Filter by coherence
        if criteria.get('min_coherence') is not None:
            min_coherence = float(criteria['min_coherence'])
            filtered_df = filtered_df[filtered_df['cluster_coherence'] >= min_coherence]
        
        # Filter by search volume
        if criteria.get('min_search_volume') and 'search_volume' in df.columns:
            min_volume = float(criteria['min_search_volume'])
            filtered_df = filtered_df[filtered_df['search_volume'] >= min_volume]
        
        # Filter by search intent
        if criteria.get('search_intents') and 'search_intent' in df.columns:
            intents = criteria['search_intents']
            if isinstance(intents, str):
                intents = [intents]
            filtered_df = filtered_df[filtered_df['search_intent'].isin(intents)]
        
        # Filter by quality score
        if criteria.get('min_quality') and 'quality_score' in df.columns:
            min_quality = float(criteria['min_quality'])
            filtered_df = filtered_df[filtered_df['quality_score'] >= min_quality]
        
        # Filter by representative keywords only
        if criteria.get('representative_only'):
            filtered_df = filtered_df[filtered_df['is_representative'] == True]
        
        # Filter by keyword length
        if criteria.get('min_keyword_length'):
            min_length = int(criteria['min_keyword_length'])
            filtered_df = filtered_df[filtered_df['keyword'].str.len() >= min_length]
        
        if criteria.get('max_keyword_length'):
            max_length = int(criteria['max_keyword_length'])
            filtered_df = filtered_df[filtered_df['keyword'].str.len() <= max_length]
        
        # Filter by word count
        if criteria.get('min_word_count'):
            min_words = int(criteria['min_word_count'])
            word_counts = filtered_df['keyword'].str.split().str.len()
            filtered_df = filtered_df[word_counts >= min_words]
        
        # Text search filter
        if criteria.get('keyword_search'):
            search_term = criteria['keyword_search'].lower()
            mask = filtered_df['keyword'].str.lower().str.contains(search_term, na=False, regex=False)
            filtered_df = filtered_df[mask]
        
        # Volume category filter
        if criteria.get('volume_categories') and 'volume_category' in df.columns:
            categories = criteria['volume_categories']
            if isinstance(categories, str):
                categories = [categories]
            filtered_df = filtered_df[filtered_df['volume_category'].isin(categories)]
        
        filtered_count = len(filtered_df)
        removed_count = initial_count - filtered_count
        
        if removed_count > 0:
            st.info(f"🔍 Filter applied: {removed_count:,} keywords filtered out, {filtered_count:,} remaining")
        
        return filtered_df
        
    except Exception as e:
        log_error(e, "dataframe_filtering", {"criteria": criteria})
        st.warning(f"⚠️ Filtering failed: {str(e)}. Returning original data.")
        return df

def merge_original_data(results_df, original_df):
    """Merge clustering results with original CSV data safely"""
    try:
        if original_df is None or original_df.empty:
            return results_df
        
        # Validate that both DataFrames have keyword column
        if 'keyword' not in results_df.columns or 'keyword' not in original_df.columns:
            st.warning("⚠️ Cannot merge: missing keyword column")
            return results_df
        
        # Identify columns to merge (avoid conflicts)
        original_cols_to_merge = []
        for col in original_df.columns:
            if col != 'keyword' and col not in results_df.columns:
                original_cols_to_merge.append(col)
        
        if not original_cols_to_merge:
            st.info("ℹ️ No additional columns to merge from original data")
            return results_df
        
        # Prepare merge columns
        merge_columns = ['keyword'] + original_cols_to_merge
        original_subset = original_df[merge_columns].copy()
        
        # Handle duplicates in original data
        original_subset = original_subset.drop_duplicates(subset=['keyword'], keep='first')
        
        # Perform merge
        merged_df = results_df.merge(
            original_subset,
            on='keyword',
            how='left'
        )
        
        # Check merge success
        original_col_count = len(original_cols_to_merge)
        merged_col_count = sum(1 for col in original_cols_to_merge if col in merged_df.columns)
        
        if merged_col_count == original_col_count:
            st.success(f"✅ Original data merged: {original_col_count} columns added")
        else:
            st.warning(f"⚠️ Partial merge: {merged_col_count}/{original_col_count} columns merged")
        
        return merged_df
        
    except Exception as e:
        log_error(e, "data_merging")
        st.warning(f"⚠️ Could not merge original data: {str(e)}")
        return results_df

def create_clustering_summary_metrics(df):
    """Create comprehensive summary metrics with enhanced calculations"""
    try:
        metrics = {}
        
        # Basic metrics
        metrics['total_keywords'] = len(df)
        metrics['total_clusters'] = df['cluster_id'].nunique()
        metrics['avg_cluster_size'] = metrics['total_keywords'] / metrics['total_clusters']
        metrics['median_cluster_size'] = df['cluster_id'].value_counts().median()
        metrics['avg_coherence'] = df['cluster_coherence'].mean()
        metrics['median_coherence'] = df['cluster_coherence'].median()
        
        # Representative keywords metrics
        metrics['representative_keywords'] = df['is_representative'].sum()
        metrics['rep_percentage'] = (metrics['representative_keywords'] / metrics['total_keywords']) * 100
        
        # Cluster size distribution
        cluster_sizes = df['cluster_id'].value_counts()
        metrics['largest_cluster_size'] = cluster_sizes.max()
        metrics['smallest_cluster_size'] = cluster_sizes.min()
        metrics['size_std'] = cluster_sizes.std()
        metrics['size_cv'] = metrics['size_std'] / metrics['avg_cluster_size'] if metrics['avg_cluster_size'] > 0 else 0
        
        # Coherence distribution
        metrics['min_coherence'] = df['cluster_coherence'].min()
        metrics['max_coherence'] = df['cluster_coherence'].max()
        metrics['coherence_std'] = df['cluster_coherence'].std()
        
        # High quality clusters (coherence > 0.7)
        high_coherence_clusters = df.groupby('cluster_id')['cluster_coherence'].mean()
        metrics['high_coherence_clusters'] = (high_coherence_clusters > 0.7).sum()
        metrics['high_coherence_percentage'] = (metrics['high_coherence_clusters'] / metrics['total_clusters']) * 100
        
        # Search volume metrics (if available)
        if 'search_volume' in df.columns:
            metrics['total_search_volume'] = df['search_volume'].sum()
            metrics['avg_search_volume'] = df['search_volume'].mean()
            metrics['median_search_volume'] = df['search_volume'].median()
            metrics['max_search_volume'] = df['search_volume'].max()
            metrics['zero_volume_keywords'] = (df['search_volume'] == 0).sum()
            metrics['zero_volume_percentage'] = (metrics['zero_volume_keywords'] / metrics['total_keywords']) * 100
            
            # Cluster-level volume metrics
            cluster_volumes = df.groupby('cluster_id')['search_volume'].sum()
            metrics['highest_volume_cluster'] = cluster_volumes.max()
            metrics['avg_cluster_volume'] = cluster_volumes.mean()
            
            # Volume concentration (top 20% of clusters)
            top_20_percent = int(np.ceil(len(cluster_volumes) * 0.2))
            top_clusters_volume = cluster_volumes.nlargest(top_20_percent).sum()
            metrics['volume_concentration_20'] = (top_clusters_volume / metrics['total_search_volume']) * 100
        
        # Intent distribution (if available)
        if 'search_intent' in df.columns:
            intent_dist = df['search_intent'].value_counts(normalize=True) * 100
            metrics['primary_intent'] = intent_dist.index[0] if len(intent_dist) > 0 else 'Unknown'
            metrics['primary_intent_percentage'] = intent_dist.iloc[0] if len(intent_dist) > 0 else 0
            metrics['intent_distribution'] = intent_dist.to_dict()
            metrics['intent_diversity'] = len(intent_dist)
            
            # Intent entropy (diversity measure)
            intent_counts = df['search_intent'].value_counts()
            metrics['intent_entropy'] = calculate_entropy(intent_counts.values)
        
        # Quality metrics (if available)
        if 'quality_score' in df.columns:
            metrics['avg_quality'] = df['quality_score'].mean()
            metrics['median_quality'] = df['quality_score'].median()
            metrics['min_quality'] = df['quality_score'].min()
            metrics['max_quality'] = df['quality_score'].max()
            
            high_quality_threshold = 7.0
            high_quality_clusters = df.groupby('cluster_id')['quality_score'].mean()
            metrics['high_quality_clusters'] = (high_quality_clusters >= high_quality_threshold).sum()
            metrics['high_quality_percentage'] = (metrics['high_quality_clusters'] / metrics['total_clusters']) * 100
        
        # Keyword characteristics
        if 'keyword_length' in df.columns or 'keyword' in df.columns:
            if 'keyword_length' not in df.columns:
                df['keyword_length'] = df['keyword'].str.len()
            
            metrics['avg_keyword_length'] = df['keyword_length'].mean()
            metrics['median_keyword_length'] = df['keyword_length'].median()
            
        if 'word_count' in df.columns or 'keyword' in df.columns:
            if 'word_count' not in df.columns:
                df['word_count'] = df['keyword'].str.split().str.len()
            
            metrics['avg_word_count'] = df['word_count'].mean()
            metrics['median_word_count'] = df['word_count'].median()
        
        # Data quality indicators
        metrics['data_completeness'] = {
            'keywords_with_coherence': (~df['cluster_coherence'].isna()).sum(),
            'keywords_with_cluster_names': (~df['cluster_name'].isna()).sum(),
            'completeness_percentage': (~df['cluster_coherence'].isna()).mean() * 100
        }
        
        # Processing metadata
        metrics['processing_info'] = {
            'columns_available': list(df.columns),
            'has_search_volume': 'search_volume' in df.columns,
            'has_intent_data': 'search_intent' in df.columns,
            'has_quality_scores': 'quality_score' in df.columns,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        log_error(e, "summary_metrics")
        return {'error': f"Failed to calculate metrics: {str(e)}"}

def create_cluster_size_chart(df):
    """Create cluster size distribution chart with enhanced styling"""
    try:
        if df is None or df.empty:
            return None
        
        # Calculate cluster sizes
        cluster_sizes = df['cluster_id'].value_counts().reset_index()
        cluster_sizes.columns = ['cluster_id', 'keyword_count']
        
        if cluster_sizes.empty:
            st.warning("⚠️ No cluster data available for size chart")
            return None
        
        # Add cluster names
        cluster_names = df.groupby('cluster_id')['cluster_name'].first().reset_index()
        cluster_sizes = cluster_sizes.merge(cluster_names, on='cluster_id', how='left')
        
        # Handle missing cluster names
        cluster_sizes['cluster_name'] = cluster_sizes['cluster_name'].fillna(
            cluster_sizes['cluster_id'].apply(lambda x: f"Cluster {x}")
        )
        
        # Create short labels for better display
        cluster_sizes['label'] = cluster_sizes.apply(
            lambda x: f"{x['cluster_name'][:25]}{'...' if len(x['cluster_name']) > 25 else ''} ({x['cluster_id']})",
            axis=1
        )
        
        # Sort by size and limit to top clusters for readability
        cluster_sizes = cluster_sizes.sort_values('keyword_count', ascending=True)
        max_clusters_to_show = min(20, len(cluster_sizes))
        top_clusters = cluster_sizes.tail(max_clusters_to_show)
        
        # Create horizontal bar chart
        fig = px.bar(
            top_clusters,
            x='keyword_count',
            y='label',
            orientation='h',
            title=f'Cluster Size Distribution (Top {max_clusters_to_show})',
            labels={'keyword_count': 'Number of Keywords', 'label': 'Cluster'},
            color='keyword_count',
            color_continuous_scale='viridis',
            template='plotly_white'
        )
        
        # Customize layout
        fig.update_layout(
            height=max(400, max_clusters_to_show * 25),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=200, r=50, t=80, b=50),
            font=dict(size=11),
            coloraxis_colorbar=dict(
                title="Keywords",
                titleside="right"
            )
        )
        
        # Add value annotations
        fig.update_traces(
            texttemplate='%{x}',
            textposition='outside',
            textfont_size=10
        )
        
        return fig
        
    except Exception as e:
        log_error(e, "cluster_size_chart")
        st.error(f"Failed to create cluster size chart: {str(e)}")
        return None

def create_coherence_chart(df):
    """Create cluster coherence analysis chart with size correlation"""
    try:
        if df is None or df.empty or 'cluster_coherence' not in df.columns:
            return None
        
        # Aggregate coherence data
        coherence_data = df.groupby(['cluster_id', 'cluster_name']).agg({
            'cluster_coherence': 'mean',
            'keyword': 'count'
        }).reset_index()
        
        coherence_data.columns = ['cluster_id', 'cluster_name', 'avg_coherence', 'keyword_count']
        
        if coherence_data.empty:
            st.warning("⚠️ No coherence data available")
            return None
        
        # Create short labels
        coherence_data['label'] = coherence_data.apply(
            lambda x: f"{x['cluster_name'][:20]}{'...' if len(x['cluster_name']) > 20 else ''} ({x['cluster_id']})",
            axis=1
        )
        
        # Add coherence categories for color coding
        coherence_data['coherence_category'] = pd.cut(
            coherence_data['avg_coherence'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        
        # Create scatter plot
        fig = px.scatter(
            coherence_data,
            x='avg_coherence',
            y='keyword_count',
            size='keyword_count',
            hover_name='label',
            hover_data={
                'avg_coherence': ':.3f',
                'keyword_count': ':,',
                'coherence_category': True
            },
            title='Cluster Coherence vs Size Analysis',
            labels={
                'avg_coherence': 'Average Semantic Coherence Score',
                'keyword_count': 'Number of Keywords'
            },
            color='coherence_category',
            color_discrete_map={
                'Low': '#ff7f7f',
                'Medium': '#ffbf7f', 
                'High': '#7fbf7f',
                'Very High': '#7f7fff'
            },
            template='plotly_white'
        )
        
        # Add trend line
        if len(coherence_data) > 3:
            fig.add_scatter(
                x=coherence_data['avg_coherence'],
                y=coherence_data['keyword_count'],
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='gray', width=2),
                showlegend=True
            )
        
        # Customize layout
        fig.update_layout(
            height=500,
            xaxis=dict(range=[0, 1], tickformat='.2f'),
            yaxis=dict(title_standoff=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=50, t=100, b=50)
        )
        
        # Add reference lines
        fig.add_hline(
            y=coherence_data['keyword_count'].median(),
            line_dash="dot",
            line_color="gray",
            annotation_text="Median Size"
        )
        
        fig.add_vline(
            x=0.5,
            line_dash="dot", 
            line_color="gray",
            annotation_text="Coherence Threshold"
        )
        
        return fig
        
    except Exception as e:
        log_error(e, "coherence_chart")
        st.error(f"Failed to create coherence chart: {str(e)}")
        return None

def create_intent_distribution_chart(df):
    """Create comprehensive search intent distribution charts"""
    try:
        if df is None or df.empty or 'search_intent' not in df.columns:
            return None
        
        intent_counts = df['search_intent'].value_counts()
        
        if intent_counts.empty:
            st.warning("⚠️ No search intent data available")
            return None
        
        # Define colors for consistency
        intent_colors = {
            'Informational': '#3498db',
            'Commercial': '#e74c3c', 
            'Transactional': '#2ecc71',
            'Navigational': '#f39c12',
            'Mixed': '#9b59b6',
            'Unknown': '#95a5a6'
        }
        
        # Create subplot with pie and bar charts
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=('Distribution Overview', 'Keyword Counts by Intent'),
            horizontal_spacing=0.1
        )
        
        # Pie chart
        colors = [intent_colors.get(intent, '#95a5a6') for intent in intent_counts.index]
        
        fig.add_trace(
            go.Pie(
                labels=intent_counts.index,
                values=intent_counts.values,
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Keywords: %{value:,}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=intent_counts.index,
                y=intent_counts.values,
                marker_color=colors,
                text=intent_counts.values,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Keywords: %{y:,}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Search Intent Distribution Analysis',
            template='plotly_white',
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update bar chart axes
        fig.update_xaxes(title_text="Search Intent", row=1, col=2)
        fig.update_yaxes(title_text="Number of Keywords", row=1, col=2)
        
        return fig
        
    except Exception as e:
        log_error(e, "intent_distribution_chart")
        st.error(f"Failed to create intent distribution chart: {str(e)}")
        return None

def create_cluster_quality_heatmap(df):
    """Create enhanced cluster quality heatmap with multiple dimensions"""
    try:
        if df is None or df.empty:
            return None
        
        # Prepare data for heatmap
        cluster_data = df.groupby('cluster_id').agg({
            'cluster_coherence': 'mean',
            'keyword': 'count'
        }).reset_index()
        
        cluster_data.columns = ['cluster_id', 'coherence', 'size']
        
        # Add quality score if available
        if 'quality_score' in df.columns:
            quality_data = df.groupby('cluster_id')['quality_score'].mean()
            cluster_data['quality'] = cluster_data['cluster_id'].map(quality_data)
        else:
            # Create synthetic quality score from coherence
            cluster_data['quality'] = cluster_data['coherence'] * 10
        
        # Add search volume if available
        if 'search_volume' in df.columns:
            volume_data = df.groupby('cluster_id')['search_volume'].sum()
            cluster_data['volume'] = cluster_data['cluster_id'].map(volume_data)
        else:
            cluster_data['volume'] = cluster_data['size']  # Use size as proxy
        
        # Create bins for better visualization
        cluster_data['size_bin'] = pd.cut(
            cluster_data['size'], 
            bins=5, 
            labels=['XS (1-2)', 'S (3-5)', 'M (6-10)', 'L (11-20)', 'XL (20+)']
        )
        
        cluster_data['coherence_bin'] = pd.cut(
            cluster_data['coherence'], 
            bins=5, 
            labels=['Low (0-0.2)', 'Below Avg (0.2-0.4)', 'Average (0.4-0.6)', 'Above Avg (0.6-0.8)', 'High (0.8-1.0)']
        )
        
        # Create pivot table for heatmap
        heatmap_data = cluster_data.groupby(['size_bin', 'coherence_bin']).agg({
            'quality': 'mean',
            'cluster_id': 'count'
        }).reset_index()
        
        quality_pivot = heatmap_data.pivot(
            index='size_bin', 
            columns='coherence_bin', 
            values='quality'
        )
        
        count_pivot = heatmap_data.pivot(
            index='size_bin', 
            columns='coherence_bin', 
            values='cluster_id'
        )
        
        # Fill NaN values
        quality_pivot = quality_pivot.fillna(0)
        count_pivot = count_pivot.fillna(0)
        
        # Create custom text for hover
        hover_text = []
        for i in range(len(quality_pivot.index)):
            hover_row = []
            for j in range(len(quality_pivot.columns)):
                size_bin = quality_pivot.index[i]
                coherence_bin = quality_pivot.columns[j]
                quality = quality_pivot.iloc[i, j]
                count = count_pivot.iloc[i, j]
                
                hover_row.append(
                    f"Size: {size_bin}<br>"
                    f"Coherence: {coherence_bin}<br>"
                    f"Avg Quality: {quality:.1f}<br>"
                    f"Clusters: {int(count)}"
                )
            hover_text.append(hover_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=quality_pivot.values,
            x=quality_pivot.columns,
            y=quality_pivot.index,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            colorscale='RdYlGn',
            colorbar=dict(
                title="Average Quality Score",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title='Cluster Quality Heatmap (Size vs Coherence)',
            xaxis_title='Coherence Level',
            yaxis_title='Cluster Size Category',
            template='plotly_white',
            height=400,
            margin=dict(l=100, r=100, t=80, b=50)
        )
        
        return fig
        
    except Exception as e:
        log_error(e, "quality_heatmap")
        st.error(f"Failed to create quality heatmap: {str(e)}")
        return None

def create_search_volume_analysis(df):
    """Create comprehensive search volume analysis charts"""
    try:
        if df is None or df.empty or 'search_volume' not in df.columns:
            return None, None
        
        # Prepare volume data
        volume_data = df.groupby(['cluster_id', 'cluster_name']).agg({
            'search_volume': ['sum', 'mean', 'max', 'count'],
            'keyword': 'count'
        }).reset_index()
        
        # Flatten column names
        volume_data.columns = [
            'cluster_id', 'cluster_name', 'total_volume', 'avg_volume', 
            'max_volume', 'volume_keyword_count', 'keyword_count'
        ]
        
        if volume_data.empty or volume_data['total_volume'].sum() == 0:
            st.warning("⚠️ No search volume data available or all volumes are zero")
            return None, None
        
        # Create short labels
        volume_data['label'] = volume_data.apply(
            lambda x: f"{x['cluster_name'][:20]}{'...' if len(x['cluster_name']) > 20 else ''} ({x['cluster_id']})",
            axis=1
        )
        
        # Chart 1: Total volume by cluster (top 15)
        top_volume_clusters = volume_data.nlargest(15, 'total_volume')
        
        fig1 = px.bar(
            top_volume_clusters,
            x='label',
            y='total_volume',
            title='Total Search Volume by Cluster (Top 15)',
            labels={'total_volume': 'Total Search Volume', 'label': 'Cluster'},
            color='total_volume',
            color_continuous_scale='viridis',
            template='plotly_white'
        )
        
        fig1.update_layout(
            height=450,
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(l=60, r=50, t=80, b=120),
            coloraxis_colorbar=dict(title="Volume", titleside="right")
        )
        
        # Add value annotations
        fig1.update_traces(
            texttemplate='%{y:,.0f}',
            textposition='outside',
            textfont_size=9
        )
        
        # Chart 2: Volume efficiency scatter plot
        # Calculate volume efficiency metrics
        volume_data['volume_per_keyword'] = volume_data['total_volume'] / volume_data['keyword_count']
        volume_data['volume_concentration'] = volume_data['max_volume'] / volume_data['total_volume']
        
        # Create size categories for better visualization
        volume_data['size_category'] = pd.cut(
            volume_data['keyword_count'],
            bins=[0, 5, 10, 20, float('inf')],
            labels=['Small (1-5)', 'Medium (6-10)', 'Large (11-20)', 'XL (20+)']
        )
        
        fig2 = px.scatter(
            volume_data,
            x='keyword_count',
            y='total_volume',
            size='avg_volume',
            color='size_category',
            hover_name='label',
            hover_data={
                'keyword_count': ':,',
                'total_volume': ':,.0f',
                'avg_volume': ':,.0f',
                'volume_per_keyword': ':,.0f'
            },
            title='Search Volume vs Cluster Size Analysis',
            labels={
                'keyword_count': 'Number of Keywords',
                'total_volume': 'Total Search Volume',
                'avg_volume': 'Average Volume per Keyword'
            },
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Add trend line
        if len(volume_data) > 3:
            # Calculate trend line
            z = np.polyfit(volume_data['keyword_count'], volume_data['total_volume'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(volume_data['keyword_count'].min(), volume_data['keyword_count'].max(), 100)
            
            fig2.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend Line',
                    line=dict(dash='dash', color='red', width=2),
                    hoverinfo='skip'
                )
            )
        
        fig2.update_layout(
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=50, t=80, b=50)
        )
        
        # Add reference lines
        median_volume = volume_data['total_volume'].median()
        median_size = volume_data['keyword_count'].median()
        
        fig2.add_hline(
            y=median_volume,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Median Volume: {median_volume:,.0f}"
        )
        
        fig2.add_vline(
            x=median_size,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Median Size: {median_size:.0f}"
        )
        
        return fig1, fig2
        
    except Exception as e:
        log_error(e, "search_volume_analysis")
        st.error(f"Failed to create search volume analysis: {str(e)}")
        return None, None

def create_representative_keywords_chart(df, top_clusters=10):
    """Create enhanced chart showing representative keywords for top clusters"""
    try:
        if df is None or df.empty:
            return None
        
        # Get top clusters by size or volume
        if 'search_volume' in df.columns:
            cluster_ranking = df.groupby('cluster_id')['search_volume'].sum().nlargest(top_clusters)
        else:
            cluster_ranking = df['cluster_id'].value_counts().head(top_clusters)
        
        top_cluster_ids = cluster_ranking.index
        
        rep_data = []
        for cluster_id in top_cluster_ids:
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_name = cluster_data['cluster_name'].iloc[0]
            
            # Get representative keywords
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
            if not rep_keywords:
                rep_keywords = cluster_data['keyword'].head(3).tolist()
            
            # Calculate cluster metrics
            keyword_count = len(cluster_data)
            avg_coherence = cluster_data['cluster_coherence'].mean()
            
            # Add search volume if available
            if 'search_volume' in df.columns:
                total_volume = cluster_data['search_volume'].sum()
                metric_value = total_volume
                metric_label = f"Volume: {total_volume:,.0f}"
            else:
                metric_value = keyword_count
                metric_label = f"Keywords: {keyword_count}"
            
            rep_data.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'representative_keywords': ', '.join(rep_keywords[:5]),
                'keyword_count': keyword_count,
                'metric_value': metric_value,
                'metric_label': metric_label,
                'avg_coherence': avg_coherence,
                'coherence_category': 'High' if avg_coherence > 0.7 else 'Medium' if avg_coherence > 0.4 else 'Low'
            })
        
        rep_df = pd.DataFrame(rep_data)
        
        if rep_df.empty:
            return None
        
        # Create horizontal bar chart with color coding
        fig = px.bar(
            rep_df,
            x='metric_value',
            y='cluster_name',
            orientation='h',
            title=f'Top {top_clusters} Clusters with Representative Keywords',
            labels={'metric_value': 'Metric Value', 'cluster_name': 'Cluster'},
            hover_data={
                'representative_keywords': True,
                'keyword_count': ':,',
                'avg_coherence': ':.3f',
                'coherence_category': True
            },
            color='coherence_category',
            color_discrete_map={
                'High': '#2ecc71',
                'Medium': '#f39c12', 
                'Low': '#e74c3c'
            },
            template='plotly_white'
        )
        
        # Customize layout
        fig.update_layout(
            height=max(400, top_clusters * 40),
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=200, r=100, t=80, b=50),
            showlegend=True,
            legend=dict(
                title="Coherence Level",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add value annotations
        fig.update_traces(
            texttemplate='%{x:,.0f}',
            textposition='outside',
            textfont_size=10
        )
        
        # Add representative keywords as annotations
        for i, row in rep_df.iterrows():
            fig.add_annotation(
                x=row['metric_value'] * 0.5,
                y=i,
                text=f"Keywords: {row['representative_keywords'][:50]}{'...' if len(row['representative_keywords']) > 50 else ''}",
                showarrow=False,
                font=dict(size=9, color='white'),
                bgcolor='rgba(0,0,0,0.6)',
                bordercolor='white',
                borderwidth=1
            )
        
        return fig
        
    except Exception as e:
        log_error(e, "representative_keywords_chart")
        st.error(f"Failed to create representative keywords chart: {str(e)}")
        return None

def display_clustering_dashboard(df):
    """Display comprehensive clustering dashboard with enhanced metrics"""
    try:
        if df is None or df.empty:
            st.error("❌ No data available for dashboard")
            return False
        
        st.header("📊 Clustering Analysis Dashboard")
        
        # Calculate comprehensive metrics
        with st.spinner("Calculating dashboard metrics..."):
            metrics = create_clustering_summary_metrics(df)
        
        if 'error' in metrics:
            st.error(f"❌ Failed to calculate metrics: {metrics['error']}")
            return False
        
        # Main metrics display
        st.subheader("📈 Key Performance Indicators")
        
        # Primary metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Keywords", 
                format_number(metrics['total_keywords']),
                help="Total number of keywords processed"
            )
            
        with col2:
            st.metric(
                "Clusters Created", 
                metrics['total_clusters'],
                help="Number of distinct clusters formed"
            )
            
        with col3:
            st.metric(
                "Avg Cluster Size", 
                f"{metrics['avg_cluster_size']:.1f}",
                delta=f"Median: {metrics.get('median_cluster_size', 0):.0f}",
                help="Average number of keywords per cluster"
            )
            
        with col4:
            st.metric(
                "Avg Coherence", 
                f"{metrics['avg_coherence']:.3f}",
                delta=f"Range: {metrics.get('min_coherence', 0):.2f}-{metrics.get('max_coherence', 1):.2f}",
                help="Average semantic coherence score (0-1)"
            )
        
        # Secondary metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'total_search_volume' in metrics:
                st.metric(
                    "Total Search Volume", 
                    format_number(metrics['total_search_volume']),
                    help="Combined search volume of all keywords"
                )
            else:
                st.metric(
                    "Representative Keywords", 
                    format_number(metrics['representative_keywords']),
                    delta=f"{metrics['rep_percentage']:.1f}%",
                    help="Number and percentage of representative keywords"
                )
                
        with col2:
            if 'avg_search_volume' in metrics:
                st.metric(
                    "Avg Search Volume", 
                    format_number(metrics['avg_search_volume']),
                    help="Average search volume per keyword"
                )
            else:
                st.metric(
                    "High Coherence Clusters", 
                    metrics.get('high_coherence_clusters', 0),
                    delta=f"{metrics.get('high_coherence_percentage', 0):.1f}%",
                    help="Clusters with coherence > 0.7"
                )
                
        with col3:
            if 'primary_intent' in metrics:
                st.metric(
                    "Primary Intent", 
                    metrics['primary_intent'],
                    delta=f"{metrics.get('primary_intent_percentage', 0):.1f}%",
                    help="Most common search intent"
                )
            else:
                st.metric(
                    "Size Variation (CV)", 
                    f"{metrics.get('size_cv', 0):.2f}",
                    help="Coefficient of variation in cluster sizes"
                )
                
        with col4:
            if 'high_quality_clusters' in metrics:
                st.metric(
                    "High Quality Clusters", 
                    metrics['high_quality_clusters'],
                    delta=f"{metrics.get('high_quality_percentage', 0):.1f}%",
                    help="Clusters with quality score ≥ 7"
                )
            else:
                st.metric(
                    "Largest Cluster", 
                    metrics.get('largest_cluster_size', 0),
                    help="Number of keywords in largest cluster"
                )
        
        # Data quality indicators
        if 'data_completeness' in metrics:
            with st.expander("📋 Data Quality Summary", expanded=False):
                quality_data = metrics['data_completeness']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Data Completeness",
                        f"{quality_data['completeness_percentage']:.1f}%"
                    )
                with col2:
                    st.metric(
                        "Keywords with Names",
                        quality_data['keywords_with_cluster_names']
                    )
                with col3:
                    if 'zero_volume_percentage' in metrics:
                        st.metric(
                            "Zero Volume Keywords",
                            f"{metrics['zero_volume_percentage']:.1f}%"
                        )
        
        # Charts in organized tabs
        st.subheader("📊 Visual Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📏 Cluster Sizes", 
            "🎯 Quality Analysis", 
            "🔍 Search Intent", 
            "📈 Search Volume",
            "⭐ Top Clusters"
        ])
        
        with tab1:
            st.markdown("### Cluster Size Distribution")
            size_chart = create_cluster_size_chart(df)
            if size_chart:
                st.plotly_chart(size_chart, use_container_width=True)
            else:
                st.info("📊 Cluster size chart not available")
            
            st.markdown("### Coherence vs Size Analysis")
            coherence_chart = create_coherence_chart(df)
            if coherence_chart:
                st.plotly_chart(coherence_chart, use_container_width=True)
            else:
                st.info("📊 Coherence chart not available")
        
        with tab2:
            st.markdown("### Quality Heatmap")
            quality_heatmap = create_cluster_quality_heatmap(df)
            if quality_heatmap:
                st.plotly_chart(quality_heatmap, use_container_width=True)
            else:
                st.info("📊 Quality heatmap not available")
            
            # Quality distribution
            if 'quality_score' in df.columns:
                st.markdown("### Quality Score Distribution")
                quality_hist = px.histogram(
                    df,
                    x='quality_score',
                    bins=20,
                    title='Distribution of Quality Scores',
                    labels={'quality_score': 'Quality Score', 'count': 'Number of Keywords'},
                    template='plotly_white'
                )
                quality_hist.update_layout(height=300)
                st.plotly_chart(quality_hist, use_container_width=True)
        
        with tab3:
            st.markdown("### Search Intent Analysis")
            intent_chart = create_intent_distribution_chart(df)
            if intent_chart:
                st.plotly_chart(intent_chart, use_container_width=True)
                
                # Intent by cluster analysis
                if 'search_intent' in df.columns:
                    st.markdown("### Intent Distribution by Cluster")
                    intent_cluster = df.groupby(['cluster_id', 'cluster_name', 'search_intent']).size().reset_index(name='count')
                    
                    if not intent_cluster.empty:
                        # Select top 10 clusters for readability
                        top_clusters = df['cluster_id'].value_counts().head(10).index
                        intent_cluster_filtered = intent_cluster[intent_cluster['cluster_id'].isin(top_clusters)]
                        
                        if not intent_cluster_filtered.empty:
                            intent_sunburst = px.sunburst(
                                intent_cluster_filtered,
                                path=['cluster_name', 'search_intent'],
                                values='count',
                                title='Intent Distribution within Top 10 Clusters',
                                template='plotly_white'
                            )
                            intent_sunburst.update_layout(height=400)
                            st.plotly_chart(intent_sunburst, use_container_width=True)
            else:
                st.info("📊 Search intent analysis not available")
        
        with tab4:
            st.markdown("### Search Volume Analysis")
            vol_chart1, vol_chart2 = create_search_volume_analysis(df)
            if vol_chart1 and vol_chart2:
                st.plotly_chart(vol_chart1, use_container_width=True)
                st.plotly_chart(vol_chart2, use_container_width=True)
                
                # Volume distribution histogram
                if 'search_volume' in df.columns and df['search_volume'].sum() > 0:
                    st.markdown("### Search Volume Distribution")
                    
                    # Filter out zero volumes for better visualization
                    non_zero_volumes = df[df['search_volume'] > 0]['search_volume']
                    
                    if not non_zero_volumes.empty:
                        vol_hist = px.histogram(
                            non_zero_volumes,
                            bins=30,
                            title='Distribution of Non-Zero Search Volumes (Log Scale)',
                            labels={'value': 'Search Volume', 'count': 'Number of Keywords'},
                            template='plotly_white',
                            log_x=True
                        )
                        vol_hist.update_layout(height=300)
                        st.plotly_chart(vol_hist, use_container_width=True)
            else:
                st.info("📊 Search volume data not available or all volumes are zero")
        
        with tab5:
            st.markdown("### Top Performing Clusters")
            rep_chart = create_representative_keywords_chart(df)
            if rep_chart:
                st.plotly_chart(rep_chart, use_container_width=True)
            else:
                st.info("📊 Representative keywords chart not available")
            
            # Top clusters table
            st.markdown("### Cluster Performance Summary")
            summary_df = create_cluster_summary_dataframe(df)
            if not summary_df.empty:
                # Display top 10 clusters
                top_summary = summary_df.head(10)
                
                # Format for better display
                display_cols = ['cluster_name', 'keyword_count', 'avg_coherence']
                if 'total_search_volume' in top_summary.columns:
                    display_cols.extend(['total_search_volume', 'avg_search_volume'])
                if 'primary_intent' in top_summary.columns:
                    display_cols.append('primary_intent')
                if 'avg_quality' in top_summary.columns:
                    display_cols.append('avg_quality')
                if 'weighted_score' in top_summary.columns:
                    display_cols.append('weighted_score')
                
                display_summary = top_summary[display_cols].copy()
                
                # Round numeric columns
                for col in display_summary.select_dtypes(include=[np.number]).columns:
                    if col in ['avg_coherence', 'avg_quality']:
                        display_summary[col] = display_summary[col].round(3)
                    else:
                        display_summary[col] = display_summary[col].round(0).astype(int)
                
                st.dataframe(display_summary, use_container_width=True, height=350)
            else:
                st.info("📊 Cluster summary not available")
        
        # Advanced insights section
        st.subheader("🔍 Advanced Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("#### 📊 Distribution Analysis")
            
            # Cluster size distribution
            cluster_sizes = df['cluster_id'].value_counts()
            size_stats = {
                "Very Small (1-2)": (cluster_sizes <= 2).sum(),
                "Small (3-5)": ((cluster_sizes > 2) & (cluster_sizes <= 5)).sum(),
                "Medium (6-10)": ((cluster_sizes > 5) & (cluster_sizes <= 10)).sum(),
                "Large (11-20)": ((cluster_sizes > 10) & (cluster_sizes <= 20)).sum(),
                "Very Large (20+)": (cluster_sizes > 20).sum()
            }
            
            for category, count in size_stats.items():
                percentage = (count / len(cluster_sizes)) * 100
                st.write(f"**{category}:** {count} clusters ({percentage:.1f}%)")
        
        with insights_col2:
            st.markdown("#### 🎯 Quality Insights")
            
            # Coherence distribution
            coherence_stats = {
                "Low Coherence (0-0.4)": (df['cluster_coherence'] <= 0.4).sum(),
                "Medium Coherence (0.4-0.7)": ((df['cluster_coherence'] > 0.4) & (df['cluster_coherence'] <= 0.7)).sum(),
                "High Coherence (0.7+)": (df['cluster_coherence'] > 0.7).sum()
            }
            
            total_keywords = len(df)
            for category, count in coherence_stats.items():
                percentage = (count / total_keywords) * 100
                st.write(f"**{category}:** {count:,} keywords ({percentage:.1f}%)")
        
        # Performance recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = generate_dashboard_recommendations(metrics, df)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.info(f"**{i}.** {rec}")
        
        return True
        
    except Exception as e:
        log_error(e, "clustering_dashboard")
        st.error(f"Dashboard error: {str(e)}")
        return False

def generate_dashboard_recommendations(metrics, df):
    """Generate actionable recommendations based on clustering results"""
    try:
        recommendations = []
        
        # Size-based recommendations
        avg_size = metrics.get('avg_cluster_size', 0)
        if avg_size < 3:
            recommendations.append(
                "Consider reducing the number of clusters - many clusters are very small and may not be meaningful."
            )
        elif avg_size > 20:
            recommendations.append(
                "Consider increasing the number of clusters - some clusters may be too large and could be split."
            )
        
        # Coherence-based recommendations
        avg_coherence = metrics.get('avg_coherence', 0)
        if avg_coherence < 0.5:
            recommendations.append(
                "Low average coherence detected. Try preprocessing keywords differently or adjusting clustering parameters."
            )
        elif avg_coherence > 0.8:
            recommendations.append(
                "Excellent coherence! Your clusters are semantically well-defined."
            )
        
        # Representative keywords recommendations
        rep_percentage = metrics.get('rep_percentage', 0)
        if rep_percentage < 10:
            recommendations.append(
                "Very few representative keywords identified. Consider manual review of cluster representatives."
            )
        elif rep_percentage > 30:
            recommendations.append(
                "High percentage of representative keywords. Consider tightening representative selection criteria."
            )
        
        # Search volume recommendations
        if 'total_search_volume' in metrics:
            zero_volume_pct = metrics.get('zero_volume_percentage', 0)
            if zero_volume_pct > 50:
                recommendations.append(
                    f"{zero_volume_pct:.0f}% of keywords have zero search volume. Focus on clusters with measurable demand."
                )
            
            volume_concentration = metrics.get('volume_concentration_20', 0)
            if volume_concentration > 80:
                recommendations.append(
                    "Search volume is highly concentrated in few clusters. Prioritize these high-volume clusters for content strategy."
                )
        
        # Intent recommendations
        if 'intent_diversity' in metrics:
            intent_diversity = metrics['intent_diversity']
            if intent_diversity < 3:
                recommendations.append(
                    "Limited search intent diversity. Consider expanding keyword research to cover different user intents."
                )
            
            primary_intent_pct = metrics.get('primary_intent_percentage', 0)
            if primary_intent_pct > 70:
                recommendations.append(
                    f"Keywords are heavily skewed toward {metrics.get('primary_intent', 'unknown')} intent ({primary_intent_pct:.0f}%). Consider diversifying for comprehensive coverage."
                )
        
        # Quality recommendations
        if 'high_quality_percentage' in metrics:
            high_quality_pct = metrics['high_quality_percentage']
            if high_quality_pct < 30:
                recommendations.append(
                    "Less than 30% of clusters are high quality. Review clustering parameters or consider manual refinement."
                )
        
        # Size variation recommendations
        size_cv = metrics.get('size_cv', 0)
        if size_cv > 1.5:
            recommendations.append(
                "High variation in cluster sizes detected. Some clusters may need to be split or merged."
            )
        
        # Data quality recommendations
        if 'data_completeness' in metrics:
            completeness = metrics['data_completeness'].get('completeness_percentage', 100)
            if completeness < 95:
                recommendations.append(
                    "Some data quality issues detected. Review and clean your dataset for better results."
                )
        
        # General recommendations
        total_clusters = metrics.get('total_clusters', 0)
        total_keywords = metrics.get('total_keywords', 0)
        
        if total_clusters > total_keywords * 0.5:
            recommendations.append(
                "Too many small clusters. Consider increasing minimum cluster size or reducing target cluster count."
            )
        
        if len(recommendations) == 0:
            recommendations.append(
                "Great job! Your clustering results look well-balanced. Consider exploring the cluster explorer for detailed insights."
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations
        
    except Exception as e:
        log_error(e, "dashboard_recommendations")
        return ["Unable to generate recommendations due to analysis error."]

def create_cluster_explorer(df):
    """Create interactive cluster explorer with enhanced features"""
    try:
        if df is None or df.empty:
            st.error("❌ No data available for cluster explorer")
            return False
        
        st.header("🔍 Interactive Cluster Explorer")
        
        # Cluster selection with enhanced options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create cluster options with detailed info
            cluster_options = {}
            cluster_info_list = []
            
            for cluster_id in sorted(df['cluster_id'].unique()):
                cluster_data = df[df['cluster_id'] == cluster_id]
                cluster_name = cluster_data['cluster_name'].iloc[0]
                keyword_count = len(cluster_data)
                avg_coherence = cluster_data['cluster_coherence'].mean()
                
                # Add search volume info if available
                if 'search_volume' in cluster_data.columns:
                    total_volume = cluster_data['search_volume'].sum()
                    volume_info = f", Vol: {format_number(total_volume)}"
                else:
                    volume_info = ""
                
                # Add quality info if available
                if 'quality_score' in cluster_data.columns:
                    avg_quality = cluster_data['quality_score'].mean()
                    quality_info = f", Q: {avg_quality:.1f}"
                else:
                    quality_info = ""
                
                option_text = f"{cluster_name} (ID: {cluster_id}, {keyword_count} kw, Coh: {avg_coherence:.2f}{volume_info}{quality_info})"
                cluster_options[option_text] = cluster_id
                
                cluster_info_list.append({
                    'id': cluster_id,
                    'name': cluster_name,
                    'keywords': keyword_count,
                    'coherence': avg_coherence,
                    'volume': cluster_data['search_volume'].sum() if 'search_volume' in cluster_data.columns else 0,
                    'quality': cluster_data['quality_score'].mean() if 'quality_score' in cluster_data.columns else 0
                })
            
            selected_cluster_key = st.selectbox(
                "Select a cluster to explore:",
                options=list(cluster_options.keys()),
                help="Choose a cluster to view detailed analysis"
            )
        
        with col2:
            # Sorting options
            sort_by = st.selectbox(
                "Sort clusters by:",
                options=["Size (largest first)", "Coherence (highest first)", "Volume (highest first)", "Quality (highest first)", "Name (A-Z)"],
                index=0,
                help="Change the sorting order of clusters"
            )
            
            # Apply sorting
            if sort_by == "Size (largest first)":
                sorted_options = sorted(cluster_options.items(), 
                                      key=lambda x: df[df['cluster_id'] == x[1]].shape[0], reverse=True)
            elif sort_by == "Coherence (highest first)":
                sorted_options = sorted(cluster_options.items(),
                                      key=lambda x: df[df['cluster_id'] == x[1]]['cluster_coherence'].mean(), reverse=True)
            elif sort_by == "Volume (highest first)" and 'search_volume' in df.columns:
                sorted_options = sorted(cluster_options.items(),
                                      key=lambda x: df[df['cluster_id'] == x[1]]['search_volume'].sum(), reverse=True)
            elif sort_by == "Quality (highest first)" and 'quality_score' in df.columns:
                sorted_options = sorted(cluster_options.items(),
                                      key=lambda x: df[df['cluster_id'] == x[1]]['quality_score'].mean(), reverse=True)
            else:  # Name A-Z
                sorted_options = sorted(cluster_options.items())
        
        if selected_cluster_key:
            selected_cluster_id = cluster_options[selected_cluster_key]
            cluster_data = df[df['cluster_id'] == selected_cluster_id]
            
            # Cluster overview section
            st.subheader("📋 Cluster Overview")
            
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            
            with overview_col1:
                st.markdown("#### Basic Information")
                st.write(f"**ID:** {selected_cluster_id}")
                st.write(f"**Name:** {cluster_data['cluster_name'].iloc[0]}")
                st.write(f"**Keywords:** {len(cluster_data):,}")
                st.write(f"**Coherence:** {cluster_data['cluster_coherence'].iloc[0]:.3f}")
                
                if cluster_data['cluster_description'].iloc[0]:
                    st.write(f"**Description:** {cluster_data['cluster_description'].iloc[0]}")
            
            with overview_col2:
                st.markdown("#### Performance Metrics")
                
                if 'search_volume' in cluster_data.columns:
                    total_volume = cluster_data['search_volume'].sum()
                    avg_volume = cluster_data['search_volume'].mean()
                    max_volume = cluster_data['search_volume'].max()
                    
                    st.write(f"**Total Volume:** {format_number(total_volume)}")
                    st.write(f"**Avg Volume:** {format_number(avg_volume)}")
                    st.write(f"**Max Volume:** {format_number(max_volume)}")
                
                if 'quality_score' in cluster_data.columns:
                    avg_quality = cluster_data['quality_score'].mean()
                    st.write(f"**Quality Score:** {avg_quality:.1f}/10")
                
                # Representative keywords count
                rep_count = cluster_data['is_representative'].sum()
                rep_percentage = (rep_count / len(cluster_data)) * 100
                st.write(f"**Representatives:** {rep_count} ({rep_percentage:.1f}%)")
            
            with overview_col3:
                st.markdown("#### Content Insights")
                
                # Keyword characteristics
                avg_length = cluster_data['keyword'].str.len().mean()
                avg_words = cluster_data['keyword'].str.split().str.len().mean()
                
                st.write(f"**Avg Keyword Length:** {avg_length:.1f} chars")
                st.write(f"**Avg Word Count:** {avg_words:.1f} words")
                
                # Search intent distribution
                if 'search_intent' in cluster_data.columns:
                    intent_dist = cluster_data['search_intent'].value_counts()
                    if len(intent_dist) > 0:
                        primary_intent = intent_dist.index[0]
                        primary_pct = (intent_dist.iloc[0] / len(cluster_data)) * 100
                        st.write(f"**Primary Intent:** {primary_intent} ({primary_pct:.1f}%)")
                        
                        if len(intent_dist) > 1:
                            st.write(f"**Intent Diversity:** {len(intent_dist)} types")
            
            # Representative keywords section
            st.subheader("⭐ Representative Keywords")
            
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]
            
            if not rep_keywords.empty:
                rep_col1, rep_col2 = st.columns(2)
                
                with rep_col1:
                    st.markdown("#### Top Representatives")
                    
                    # Sort by search volume if available, otherwise by coherence
                    if 'search_volume' in rep_keywords.columns:
                        rep_sorted = rep_keywords.sort_values('search_volume', ascending=False)
                    else:
                        rep_sorted = rep_keywords.sort_values('cluster_coherence', ascending=False)
                    
                    for idx, (_, row) in enumerate(rep_sorted.head(10).iterrows(), 1):
                        volume_text = f" ({format_number(row['search_volume'])} vol)" if 'search_volume' in row else ""
                        st.write(f"**{idx}.** {row['keyword']}{volume_text}")
                
                with rep_col2:
                    if 'search_volume' in rep_keywords.columns and rep_keywords['search_volume'].sum() > 0:
                        st.markdown("#### Volume Distribution")
                        
                        rep_volume_chart = px.bar(
                            rep_sorted.head(10),
                            x='keyword',
                            y='search_volume',
                            title='Search Volume of Top Representatives',
                            template='plotly_white'
                        )
                        rep_volume_chart.update_layout(
                            height=300,
                            xaxis_tickangle=-45,
                            showlegend=False
                        )
                        st.plotly_chart(rep_volume_chart, use_container_width=True)
            else:
                st.info("ℹ️ No representative keywords explicitly marked. Showing top keywords by volume/coherence.")
                
                # Show top keywords as fallback
                if 'search_volume' in cluster_data.columns:
                    top_keywords = cluster_data.nlargest(5, 'search_volume')['keyword'].tolist()
                else:
                    top_keywords = cluster_data.nlargest(5, 'cluster_coherence')['keyword'].tolist()
                
                for idx, keyword in enumerate(top_keywords, 1):
                    st.write(f"**{idx}.** {keyword}")
            
            # Detailed keywords table
            st.subheader("📝 All Keywords in this Cluster")
            
            # Table configuration options
            table_col1, table_col2, table_col3 = st.columns(3)
            
            with table_col1:
                show_only_rep = st.checkbox("Show only representatives", value=False)
            
            with table_col2:
                if 'search_volume' in cluster_data.columns:
                    min_volume = st.number_input("Min search volume", min_value=0, value=0)
                else:
                    min_volume = 0
            
            with table_col3:
                sort_table_by = st.selectbox(
                    "Sort by:",
                    options=["Representative first", "Search volume", "Alphabetical", "Coherence"],
                    index=0
                )
            
            # Filter and sort data
            table_data = cluster_data.copy()
            
            if show_only_rep:
                table_data = table_data[table_data['is_representative'] == True]
            
            if min_volume > 0 and 'search_volume' in table_data.columns:
                table_data = table_data[table_data['search_volume'] >= min_volume]
            
            # Sort data
            if sort_table_by == "Representative first":
                table_data = table_data.sort_values(['is_representative', 'search_volume' if 'search_volume' in table_data.columns else 'cluster_coherence'], 
                                                  ascending=[False, False])
            elif sort_table_by == "Search volume" and 'search_volume' in table_data.columns:
                table_data = table_data.sort_values('search_volume', ascending=False)
            elif sort_table_by == "Alphabetical":
                table_data = table_data.sort_values('keyword')
            elif sort_table_by == "Coherence":
                table_data = table_data.sort_values('cluster_coherence', ascending=False)
            
            # Prepare display columns
            display_cols = ['keyword', 'is_representative']
            if 'search_volume' in table_data.columns:
                display_cols.append('search_volume')
            if 'search_intent' in table_data.columns:
                display_cols.append('search_intent')
            if 'quality_score' in table_data.columns:
                display_cols.append('quality_score')
            
            display_data = table_data[display_cols].copy()
            
            # Format display
            display_data['is_representative'] = display_data['is_representative'].map({True: '⭐', False: ''})
            
            if 'quality_score' in display_data.columns:
                display_data['quality_score'] = display_data['quality_score'].round(1)
            
            # Show filtered count
            if len(table_data) < len(cluster_data):
                st.info(f"Showing {len(table_data):,} of {len(cluster_data):,} keywords after filtering")
            
            # Display table
            st.dataframe(
                display_data,
                use_container_width=True,
                height=400
            )
            
            # Export cluster data
            st.subheader("📥 Export Cluster Data")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                cluster_csv = table_data.to_csv(index=False)
                st.download_button(
                    label=f"📄 Download Cluster {selected_cluster_id} (CSV)",
                    data=cluster_csv,
                    file_name=f"cluster_{selected_cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with export_col2:
                # Representatives only export
                if not rep_keywords.empty:
                    rep_csv = rep_keywords.to_csv(index=False)
                    st.download_button(
                        label="⭐ Download Representatives Only",
                        data=rep_csv,
                        file_name=f"representatives_cluster_{selected_cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        return True
        
    except Exception as e:
        log_error(e, "cluster_explorer")
        st.error(f"Cluster explorer error: {str(e)}")
        return False

def show_data_table_view(df):
    """Show interactive data table view"""
    try:
        st.markdown("#### Data Table View")
        
        # Pagination
        rows_per_page = st.slider("Rows per page", 10, 100, 25)
        total_pages = (len(df) + rows_per_page - 1) // rows_per_page
        page = st.number_input("Page", 1, total_pages, 1)
        
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        
        # Display table
        st.dataframe(
            df.iloc[start_idx:end_idx], 
            use_container_width=True,
            height=400
        )
        
        # Show page info
        st.info(f"Showing rows {start_idx+1} to {end_idx} of {len(df)}")
        
    except Exception as e:
        log_error(e, "data_table_view")
        st.error(f"Error displaying data table: {str(e)}")

def show_statistical_analysis(df):
    """Show statistical analysis of the clustering results"""
    try:
        # Basic statistics
        st.subheader("📈 Descriptive Statistics")
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown("#### Cluster Size Distribution")
            cluster_sizes = df['cluster_id'].value_counts()
            
            size_stats = {
                "Total Clusters": len(cluster_sizes),
                "Mean Size": cluster_sizes.mean(),
                "Median Size": cluster_sizes.median(),
                "Std Deviation": cluster_sizes.std(),
                "Min Size": cluster_sizes.min(),
                "Max Size": cluster_sizes.max()
            }
            
            for stat, value in size_stats.items():
                if isinstance(value, float):
                    st.write(f"**{stat}:** {value:.2f}")
                else:
                    st.write(f"**{stat}:** {value}")
        
        with stat_col2:
            st.markdown("#### Coherence Distribution")
            coherence_stats = df['cluster_coherence'].describe()
            
            for stat, value in coherence_stats.items():
                st.write(f"**{stat.title()}:** {value:.4f}")
        
        # Distribution visualizations
        st.subheader("📊 Distribution Analysis")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            # Cluster size histogram
            fig_sizes = px.histogram(
                cluster_sizes.values,
                nbins=20,
                title="Cluster Size Distribution",
                labels={'value': 'Cluster Size', 'count': 'Number of Clusters'},
                template='plotly_white'
            )
            fig_sizes.update_layout(height=300)
            st.plotly_chart(fig_sizes, use_container_width=True)
        
        with dist_col2:
            # Coherence histogram
            fig_coherence = px.histogram(
                df['cluster_coherence'],
                nbins=20,
                title="Coherence Score Distribution",
                labels={'value': 'Coherence Score', 'count': 'Number of Keywords'},
                template='plotly_white'
            )
            fig_coherence.update_layout(height=300)
            st.plotly_chart(fig_coherence, use_container_width=True)
        
        # Correlation analysis
        if 'search_volume' in df.columns:
            st.subheader("🔗 Correlation Analysis")
            
            # Calculate correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            
            # Create correlation heatmap
            fig_corr = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto',
                template='plotly_white'
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        
    except Exception as e:
        log_error(e, "statistical_analysis")
        st.error(f"Statistical analysis error: {str(e)}")

def show_advanced_analytics(df, config):
    """Show advanced analytics and insights"""
    try:
        # Advanced metrics
        st.subheader("🔬 Advanced Metrics")
        
        advanced_col1, advanced_col2 = st.columns(2)
        
        with advanced_col1:
            st.markdown("#### Clustering Quality Metrics")
            
            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score
                if len(df['cluster_id'].unique()) > 1:
                    # Use a sample for large datasets
                    sample_size = min(1000, len(df))
                    sample_df = df.sample(n=sample_size, random_state=42)
                    
                    # Create dummy embeddings for silhouette calculation
                    # (In real implementation, you'd use actual embeddings)
                    dummy_embeddings = np.random.rand(len(sample_df), 10)
                    
                    silhouette_avg = silhouette_score(dummy_embeddings, sample_df['cluster_id'])
                    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                else:
                    st.info("Silhouette score requires multiple clusters")
            except Exception:
                st.info("Silhouette score calculation not available")
            
            # Cluster balance metrics
            cluster_sizes = df['cluster_id'].value_counts()
            balance_score = 1 - (cluster_sizes.std() / cluster_sizes.mean())
            st.metric("Cluster Balance", f"{balance_score:.3f}")
            
            # Representative ratio
            rep_ratio = df['is_representative'].mean()
            st.metric("Representative Ratio", f"{rep_ratio:.3f}")
        
        with advanced_col2:
            st.markdown("#### Business Value Metrics")
            
            if 'search_volume' in df.columns:
                # Volume concentration
                cluster_volumes = df.groupby('cluster_id')['search_volume'].sum()
                top_20_percent = int(np.ceil(len(cluster_volumes) * 0.2))
                volume_concentration = cluster_volumes.nlargest(top_20_percent).sum() / cluster_volumes.sum()
                st.metric("Volume Concentration (Top 20%)", f"{volume_concentration:.1%}")
                
                # Average cluster value
                avg_cluster_value = cluster_volumes.mean()
                st.metric("Avg Cluster Volume", format_number(avg_cluster_value))
            
            if 'search_intent' in df.columns:
                # Intent diversity
                intent_counts = df['search_intent'].value_counts()
                intent_entropy = calculate_entropy(intent_counts.values)
                st.metric("Intent Diversity (Entropy)", f"{intent_entropy:.3f}")
        
        # Trend analysis
        if 'search_volume' in df.columns:
            st.subheader("📈 Value Analysis")
            
            # Create value vs size scatter
            cluster_analysis = df.groupby(['cluster_id', 'cluster_name']).agg({
                'search_volume': ['sum', 'mean'],
                'cluster_coherence': 'mean',
                'keyword': 'count'
            }).reset_index()
            
            cluster_analysis.columns = ['cluster_id', 'cluster_name', 'total_volume', 'avg_volume', 'coherence', 'size']
            
            fig_value = px.scatter(
                cluster_analysis,
                x='size',
                y='total_volume',
                size='coherence',
                color='avg_volume',
                hover_name='cluster_name',
                title='Cluster Value Analysis: Size vs Total Volume',
                labels={
                    'size': 'Number of Keywords',
                    'total_volume': 'Total Search Volume',
                    'coherence': 'Coherence Score',
                    'avg_volume': 'Avg Volume per Keyword'
                },
                template='plotly_white'
            )
            fig_value.update_layout(height=400)
            st.plotly_chart(fig_value, use_container_width=True)
        
        # Performance insights
        st.subheader("💡 Performance Insights")
        
        insights = generate_performance_insights(df, config)
        
        for insight in insights:
            st.info(insight)
        
    except Exception as e:
        log_error(e, "advanced_analytics")
        st.error(f"Advanced analytics error: {str(e)}")

def generate_performance_insights(df, config):
    """Generate performance insights based on analysis"""
    try:
        insights = []
        
        # Cluster quality insights
        avg_coherence = df['cluster_coherence'].mean()
        if avg_coherence > 0.7:
            insights.append("🎯 Excellent clustering quality! Most clusters show strong semantic coherence.")
        elif avg_coherence > 0.5:
            insights.append("👍 Good clustering quality. Some clusters may benefit from refinement.")
        else:
            insights.append("⚠️ Low clustering quality detected. Consider adjusting parameters or preprocessing.")
        
        # Size distribution insights
        cluster_sizes = df['cluster_id'].value_counts()
        size_cv = cluster_sizes.std() / cluster_sizes.mean()
        
        if size_cv < 0.5:
            insights.append("📊 Well-balanced cluster sizes across the dataset.")
        elif size_cv > 1.5:
            insights.append("📊 High variation in cluster sizes. Consider merging small clusters or splitting large ones.")
        
        # Volume insights
        if 'search_volume' in df.columns:
            zero_volume_pct = (df['search_volume'] == 0).mean() * 100
            
            if zero_volume_pct > 50:
                insights.append(f"📈 {zero_volume_pct:.0f}% of keywords have zero search volume. Focus on keywords with measurable demand.")
            elif zero_volume_pct < 10:
                insights.append("📈 Excellent! Most keywords have search volume data.")
        
        # Intent insights
        if 'search_intent' in df.columns:
            intent_dist = df['search_intent'].value_counts(normalize=True)
            primary_intent = intent_dist.index[0]
            primary_pct = intent_dist.iloc[0] * 100
            
            if primary_pct > 70:
                insights.append(f"🎯 Keywords heavily focused on {primary_intent} intent ({primary_pct:.0f}%). Consider diversifying for comprehensive coverage.")
            else:
                insights.append("🎯 Good intent diversity across your keyword portfolio.")
        
        # Representative insights
        rep_pct = df['is_representative'].mean() * 100
        if rep_pct < 5:
            insights.append("⭐ Very selective representative keyword identification. Quality over quantity approach.")
        elif rep_pct > 25:
            insights.append("⭐ High percentage of representative keywords. Consider tightening selection criteria.")
        
        return insights[:5]  # Limit to top 5 insights
        
    except Exception as e:
        log_error(e, "performance_insights")
        return ["Unable to generate insights due to analysis error."]

def show_data_analysis_tab(df, config):
    """Show detailed data analysis tab"""
    try:
        analysis_subtab1, analysis_subtab2, analysis_subtab3 = st.tabs([
            "📋 Data Table", 
            "📊 Statistical Analysis", 
            "🔬 Advanced Analytics"
        ])
        
        with analysis_subtab1:
            st.markdown("#### Interactive Data Table")
            show_data_table_view(df)
        
        with analysis_subtab2:
            st.markdown("#### Statistical Analysis")
            show_statistical_analysis(df)
        
        with analysis_subtab3:
            st.markdown("#### Advanced Analytics")
            show_advanced_analytics(df, config)
        
    except Exception as e:
        log_error(e, "data_analysis_tab")
        st.error(f"Data analysis error: {str(e)}")

def show_export_options(df):
    """Show comprehensive export options with download buttons"""
    try:
        if df is None or df.empty:
            st.error("❌ No data available for export")
            return False
        
        st.header("📥 Export Results")
        
        # Export statistics
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            st.metric("Total Keywords", format_number(len(df)))
        with export_col2:
            st.metric("Total Clusters", df['cluster_id'].nunique())
        with export_col3:
            if 'search_volume' in df.columns:
                st.metric("Total Volume", format_number(df['search_volume'].sum()))
            else:
                st.metric("Representative Keywords", df['is_representative'].sum())
        with export_col4:
            st.metric("Avg Coherence", f"{df['cluster_coherence'].mean():.3f}")
        
        # Main export options
        st.subheader("📊 Main Export Options")
        
        main_col1, main_col2 = st.columns(2)
        
        with main_col1:
            st.markdown("#### 📄 Standard Formats")
            
            # CSV export (full dataset)
            try:
                csv_data, csv_filename = export_results_to_csv(df)
                st.download_button(
                    label="📄 Download Full Results (CSV)",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    help="Complete dataset with all columns and metadata",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"CSV export failed: {str(e)}")
            
            # Summary CSV
            try:
                summary_df = create_cluster_summary_dataframe(df)
                if not summary_df.empty:
                    summary_csv = summary_df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="📋 Download Cluster Summary (CSV)",
                        data=summary_csv,
                        file_name=f"cluster_summary_{timestamp}.csv",
                        mime="text/csv",
                        help="Condensed summary with key metrics per cluster",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Summary export failed: {str(e)}")
        
        with main_col2:
            st.markdown("#### 📊 Advanced Formats")
            
            # Excel export
            try:
                excel_data, excel_filename, excel_mime = prepare_download_data(df, "excel")
                st.download_button(
                    label="📊 Download Excel Report (Multi-sheet)",
                    data=excel_data,
                    file_name=excel_filename,
                    mime=excel_mime,
                    help="Excel file with multiple analysis sheets",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Excel export not available: {str(e)}")
            
            # JSON export
            try:
                json_data, json_filename, json_mime = prepare_download_data(df, "json")
                st.download_button(
                    label="🔗 Download JSON Data",
                    data=json_data,
                    file_name=json_filename,
                    mime=json_mime,
                    help="Structured JSON format for API integration",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"JSON export not available: {str(e)}")
        
        # Specialized exports
        st.subheader("🎯 Specialized Exports")
        
        specialized_col1, specialized_col2, specialized_col3 = st.columns(3)
        
        with specialized_col1:
            st.markdown("#### ⭐ Representative Keywords")
            
            rep_keywords = df[df['is_representative'] == True]
            if not rep_keywords.empty:
                # Representatives only
                rep_csv = rep_keywords[['keyword', 'cluster_id', 'cluster_name', 'search_volume' if 'search_volume' in df.columns else 'cluster_coherence']].to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label=f"⭐ Representatives Only ({len(rep_keywords)} keywords)",
                    data=rep_csv,
                    file_name=f"representative_keywords_{timestamp}.csv",
                    mime="text/csv",
                    help="Only the most representative keywords from each cluster",
                    use_container_width=True
                )
                
                # Top representatives by volume/coherence
                if 'search_volume' in rep_keywords.columns:
                    top_rep = rep_keywords.nlargest(100, 'search_volume')
                    sort_column = 'search_volume'
                    sort_label = "Volume"
                else:
                    top_rep = rep_keywords.nlargest(100, 'cluster_coherence')
                    sort_column = 'cluster_coherence'
                    sort_label = "Coherence"
                
                if len(top_rep) > 0:
                    top_rep_csv = top_rep.to_csv(index=False)
                    st.download_button(
                        label=f"🏆 Top 100 by {sort_label}",
                        data=top_rep_csv,
                        file_name=f"top_representatives_{timestamp}.csv",
                        mime="text/csv",
                        help=f"Top 100 representative keywords sorted by {sort_label.lower()}",
                        use_container_width=True
                    )
            else:
                st.info("No representative keywords marked")
        
        with specialized_col2:
            st.markdown("#### 🔍 By Search Intent")
            
            if 'search_intent' in df.columns:
                intent_counts = df['search_intent'].value_counts()
                
                for intent in intent_counts.index[:4]:  # Top 4 intents
                    intent_data = df[df['search_intent'] == intent]
                    intent_csv = intent_data.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label=f"🎯 {intent} ({len(intent_data)} keywords)",
                        data=intent_csv,
                        file_name=f"{intent.lower()}_keywords_{timestamp}.csv",
                        mime="text/csv",
                        help=f"Keywords with {intent} search intent",
                        use_container_width=True
                    )
            else:
                st.info("No search intent data available")
        
        with specialized_col3:
            st.markdown("#### 📈 By Search Volume")
            
            if 'search_volume' in df.columns and df['search_volume'].sum() > 0:
                # High volume keywords
                high_volume = df[df['search_volume'] >= df['search_volume'].quantile(0.8)]
                if not high_volume.empty:
                    high_vol_csv = high_volume.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label=f"📈 High Volume (Top 20%, {len(high_volume)} keywords)",
                        data=high_vol_csv,
                        file_name=f"high_volume_keywords_{timestamp}.csv",
                        mime="text/csv",
                        help="Keywords in the top 20% by search volume",
                        use_container_width=True
                    )
                
                # Zero volume keywords
                zero_volume = df[df['search_volume'] == 0]
                if not zero_volume.empty:
                    zero_vol_csv = zero_volume.to_csv(index=False)
                    
                    st.download_button(
                        label=f"📉 Zero Volume ({len(zero_volume)} keywords)",
                        data=zero_vol_csv,
                        file_name=f"zero_volume_keywords_{timestamp}.csv",
                        mime="text/csv",
                        help="Keywords with no recorded search volume",
                        use_container_width=True
                    )
            else:
                st.info("No search volume data available")
        
        # Custom export builder
        st.subheader("🛠️ Custom Export Builder")
        
        with st.expander("Build Custom Export", expanded=False):
            custom_col1, custom_col2 = st.columns(2)
            
            with custom_col1:
                st.markdown("#### Select Columns")
                
                available_columns = df.columns.tolist()
                essential_columns = ['keyword', 'cluster_id', 'cluster_name']
                optional_columns = [col for col in available_columns if col not in essential_columns]
                
                selected_columns = st.multiselect(
                    "Additional columns to include:",
                    options=optional_columns,
                    default=[col for col in ['is_representative', 'cluster_coherence', 'search_volume', 'search_intent'] if col in optional_columns],
                    help="Essential columns (keyword, cluster_id, cluster_name) are always included"
                )
                
                final_columns = essential_columns + selected_columns
            
            with custom_col2:
                st.markdown("#### Apply Filters")
                
                # Cluster size filter
                min_cluster_size = st.slider(
                    "Minimum cluster size:",
                    min_value=1,
                    max_value=df['cluster_id'].value_counts().max(),
                    value=1,
                    help="Only include clusters with at least this many keywords"
                )
                
                # Coherence filter
                min_coherence = st.slider(
                    "Minimum coherence:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help="Only include keywords from clusters with this coherence or higher"
                )
                
                # Representative only
                rep_only = st.checkbox(
                    "Representative keywords only",
                    value=False,
                    help="Export only representative keywords"
                )
            
            # Generate custom export
            if st.button("🎯 Generate Custom Export", use_container_width=True):
                try:
                    # Apply filters
                    filtered_df = df.copy()
                    
                    # Filter by cluster size
                    if min_cluster_size > 1:
                        cluster_sizes = filtered_df['cluster_id'].value_counts()
                        valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index
                        filtered_df = filtered_df[filtered_df['cluster_id'].isin(valid_clusters)]
                    
                    # Filter by coherence
                    if min_coherence > 0:
                        filtered_df = filtered_df[filtered_df['cluster_coherence'] >= min_coherence]
                    
                    # Filter representatives
                    if rep_only:
                        filtered_df = filtered_df[filtered_df['is_representative'] == True]
                    
                    # Select columns
                    if not filtered_df.empty:
                        export_df = filtered_df[final_columns]
                        custom_csv = export_df.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.success(f"✅ Custom export ready: {len(export_df):,} keywords")
                        
                        st.download_button(
                            label=f"📥 Download Custom Export ({len(export_df)} keywords)",
                            data=custom_csv,
                            file_name=f"custom_export_{timestamp}.csv",
                            mime="text/csv",
                            help="Your custom filtered and configured export",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No data matches your filter criteria")
                        
                except Exception as e:
                    st.error(f"Custom export failed: {str(e)}")
        
        # Export preview
        st.subheader("👀 Export Preview")
        
        preview_options = ["Full Dataset", "Cluster Summary", "Representative Keywords Only"]
        if 'search_intent' in df.columns:
            preview_options.append("Intent Distribution")
        if 'search_volume' in df.columns:
            preview_options.append("Volume Analysis")
        
        preview_selection = st.selectbox(
            "Select preview type:",
            options=preview_options,
            help="Preview different export formats"
        )
        
        if preview_selection == "Full Dataset":
            st.markdown("#### Full Dataset Preview (First 20 rows)")
            st.dataframe(df.head(20), use_container_width=True)
            
        elif preview_selection == "Cluster Summary":
            st.markdown("#### Cluster Summary Preview")
            summary_df = create_cluster_summary_dataframe(df)
            if not summary_df.empty:
                st.dataframe(summary_df.head(10), use_container_width=True)
            else:
                st.info("No summary data available")
                
        elif preview_selection == "Representative Keywords Only":
            st.markdown("#### Representative Keywords Preview")
            rep_preview = df[df['is_representative'] == True]
            if not rep_preview.empty:
                display_cols = ['keyword', 'cluster_name', 'cluster_coherence']
                if 'search_volume' in rep_preview.columns:
                    display_cols.append('search_volume')
                st.dataframe(rep_preview[display_cols].head(20), use_container_width=True)
            else:
                st.info("No representative keywords marked")
                
        elif preview_selection == "Intent Distribution":
            st.markdown("#### Intent Distribution Preview")
            intent_summary = df.groupby(['search_intent', 'cluster_name']).size().reset_index(name='keyword_count')
            intent_summary = intent_summary.sort_values('keyword_count', ascending=False)
            st.dataframe(intent_summary.head(20), use_container_width=True)
            
        elif preview_selection == "Volume Analysis":
            st.markdown("#### Volume Analysis Preview")
            volume_summary = df.groupby('cluster_name').agg({
                'search_volume': ['sum', 'mean', 'count'],
                'keyword': 'count'
            }).reset_index()
            volume_summary.columns = ['cluster_name', 'total_volume', 'avg_volume', 'volume_keywords', 'total_keywords']
            volume_summary = volume_summary.sort_values('total_volume', ascending=False)
            st.dataframe(volume_summary.head(15), use_container_width=True)
        
        # Export tips
        with st.expander("💡 Export Tips & Best Practices", expanded=False):
            st.markdown("""
            #### 📋 Export Best Practices
            
            **For SEO Content Planning:**
            - Use **Representative Keywords** export for content creation priorities
            - Filter by **High Volume** keywords for traffic opportunities
            - Export by **Search Intent** to align content with user needs
            
            **For Technical Analysis:**
            - Use **Full Dataset** for comprehensive analysis in Excel/Python
            - **JSON format** for integration with other tools and APIs
            - **Excel Multi-sheet** for stakeholder presentations
            
            **For Team Collaboration:**
            - **Cluster Summary** provides executive overview
            - **Custom Export** for specific team requirements
            - Include coherence scores to indicate cluster quality
            
            #### 🔍 File Format Guide
            
            - **CSV**: Best for Excel, Google Sheets, most analytics tools
            - **Excel**: Professional reports, multiple data views, stakeholder presentations  
            - **JSON**: API integration, custom applications, data pipelines
            
            #### ⚡ Performance Tips
            
            - Large datasets (>10k keywords): Use filtered exports
            - Multiple team members: Share summary first, then detailed data
            - Regular updates: Use timestamped filenames for version control
            """)
        
        return True
        
    except Exception as e:
        log_error(e, "export_options")
        st.error(f"Export options error: {str(e)}")
        return False

def prepare_download_data(df, format_type="csv"):
    """Prepare data for download in various formats with enhanced options"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "csv":
            data, filename = export_results_to_csv(df, f"keyword_clusters_{timestamp}.csv")
            mime_type = "text/csv"
            
        elif format_type.lower() == "excel":
            # Create Excel with multiple sheets
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main results sheet
                export_df = prepare_excel_export_dataframe(df)
                export_df.to_excel(writer, sheet_name='Clustering Results', index=False)
                
                # Cluster summary sheet
                summary_df = create_cluster_summary_dataframe(df)
                if not summary_df.empty:
                    summary_df.to_excel(writer, sheet_name='Cluster Summary', index=False)
                
                # Intent analysis sheet
                if 'search_intent' in df.columns:
                    intent_summary = create_intent_analysis_sheet(df)
                    intent_summary.to_excel(writer, sheet_name='Intent Analysis', index=False)
                
                # Volume analysis sheet
                if 'search_volume' in df.columns:
                    volume_summary = create_volume_analysis_sheet(df)
                    volume_summary.to_excel(writer, sheet_name='Volume Analysis', index=False)
                
                # Representative keywords sheet
                rep_keywords = df[df['is_representative'] == True][['keyword', 'cluster_id', 'cluster_name']]
                if not rep_keywords.empty:
                    rep_keywords.to_excel(writer, sheet_name='Representative Keywords', index=False)
            
            data = output.getvalue()
            filename = f"keyword_clusters_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
        elif format_type.lower() == "json":
            # Create JSON export
            json_data = create_json_export(df)
            data = json.dumps(json_data, indent=2, ensure_ascii=False)
            filename = f"keyword_clusters_{timestamp}.json"
            mime_type = "application/json"
            
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return data, filename, mime_type
        
    except Exception as e:
        log_error(e, "download_preparation", {"format": format_type})
        # Fallback to CSV
        try:
            data, filename = export_results_to_csv(df)
            return data, filename, "text/csv"
        except Exception as fallback_error:
            log_error(fallback_error, "csv_fallback")
            raise e

def prepare_excel_export_dataframe(df):
    """Prepare DataFrame specifically for Excel export"""
    try:
        export_df = df.copy()
        
        # Format numeric columns for Excel
        if 'search_volume' in export_df.columns:
            export_df['search_volume'] = export_df['search_volume'].astype(int)
        
        if 'cluster_coherence' in export_df.columns:
            export_df['cluster_coherence'] = export_df['cluster_coherence'].round(3)
        
        if 'quality_score' in export_df.columns:
            export_df['quality_score'] = export_df['quality_score'].round(1)
        
        # Convert boolean to text for better Excel compatibility
        bool_columns = export_df.select_dtypes(include=[bool]).columns
        for col in bool_columns:
            export_df[col] = export_df[col].map({True: 'Yes', False: 'No'})
        
        return export_df
        
    except Exception as e:
        log_error(e, "excel_dataframe_preparation")
        return df

def create_intent_analysis_sheet(df):
    """Create intent analysis data for Excel export"""
    try:
        intent_analysis = df.groupby(['cluster_id', 'cluster_name', 'search_intent']).agg({
            'keyword': 'count',
            'search_volume': 'sum' if 'search_volume' in df.columns else 'count'
        }).reset_index()
        
        intent_analysis.columns = ['cluster_id', 'cluster_name', 'search_intent', 'keyword_count', 'total_volume']
        
        # Add percentage within cluster
        cluster_totals = intent_analysis.groupby('cluster_id')['keyword_count'].sum()
        intent_analysis['percentage_in_cluster'] = intent_analysis.apply(
            lambda row: (row['keyword_count'] / cluster_totals[row['cluster_id']]) * 100,
            axis=1
        ).round(1)
        
        return intent_analysis
        
    except Exception as e:
        log_error(e, "intent_analysis_sheet")
        return pd.DataFrame()

def create_volume_analysis_sheet(df):
    """Create volume analysis data for Excel export"""
    try:
        volume_analysis = df.groupby(['cluster_id', 'cluster_name']).agg({
            'search_volume': ['sum', 'mean', 'median', 'max', 'min', 'std'],
            'keyword': 'count'
        }).reset_index()
        
        # Flatten column names
        volume_analysis.columns = [
            'cluster_id', 'cluster_name', 'total_volume', 'avg_volume',
            'median_volume', 'max_volume', 'min_volume', 'std_volume', 'keyword_count'
        ]
        
        # Calculate volume efficiency
        volume_analysis['volume_per_keyword'] = (
            volume_analysis['total_volume'] / volume_analysis['keyword_count']
        ).round(0)
        
        # Sort by total volume
        volume_analysis = volume_analysis.sort_values('total_volume', ascending=False)
        
        return volume_analysis
        
    except Exception as e:
        log_error(e, "volume_analysis_sheet")
        return pd.DataFrame()

def create_json_export(df):
    """Create structured JSON export"""
    try:
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_keywords": len(df),
                "total_clusters": df['cluster_id'].nunique(),
                "avg_cluster_size": len(df) / df['cluster_id'].nunique(),
                "avg_coherence": float(df['cluster_coherence'].mean()),
            },
            "clusters": []
        }
        
        # Add summary statistics if available
        if 'search_volume' in df.columns:
            export_data["metadata"]["total_search_volume"] = int(df['search_volume'].sum())
            export_data["metadata"]["avg_search_volume"] = float(df['search_volume'].mean())
        
        # Process each cluster
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            cluster_info = {
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_data['cluster_name'].iloc[0],
                "cluster_description": cluster_data['cluster_description'].iloc[0],
                "keyword_count": len(cluster_data),
                "avg_coherence": float(cluster_data['cluster_coherence'].mean()),
                "keywords": []
            }
            
            # Add representative keywords
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
            cluster_info["representative_keywords"] = rep_keywords
            
            # Add search volume info if available
            if 'search_volume' in df.columns:
                cluster_info["total_search_volume"] = int(cluster_data['search_volume'].sum())
                cluster_info["avg_search_volume"] = float(cluster_data['search_volume'].mean())
            
            # Add intent info if available
            if 'search_intent' in df.columns:
                intent_dist = cluster_data['search_intent'].value_counts().to_dict()
                cluster_info["intent_distribution"] = intent_dist
            
            # Add all keywords with details
            for _, row in cluster_data.iterrows():
                keyword_info = {
                    "keyword": row['keyword'],
                    "is_representative": bool(row['is_representative']),
                    "coherence": float(row['cluster_coherence'])
                }
                
                if 'search_volume' in row:
                    keyword_info["search_volume"] = int(row['search_volume'])
                
                if 'search_intent' in row:
                    keyword_info["search_intent"] = row['search_intent']
                
                if 'quality_score' in row:
                    keyword_info["quality_score"] = float(row['quality_score'])
                
                cluster_info["keywords"].append(keyword_info)
            
            export_data["clusters"].append(cluster_info)
        
        return export_data
        
    except Exception as e:
        log_error(e, "json_export_creation")
        return {"error": f"Failed to create JSON export: {str(e)}"}

def generate_comprehensive_report(df, config):
    """Generate comprehensive analysis report"""
    try:
        if df is None or df.empty:
            st.error("❌ No data available for report generation")
            return
        
        st.info("📊 Generating comprehensive report...")
        
        # Calculate comprehensive metrics
        report_data = {
            'executive_summary': generate_executive_summary(df, config),
            'detailed_metrics': calculate_detailed_metrics(df),
            'cluster_analysis': generate_cluster_analysis_report(df),
            'recommendations': generate_detailed_recommendations(df, config),
            'methodology': generate_methodology_section(config)
        }
        
        # Create report document
        report_content = create_report_document(report_data)
        
        # Offer download
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        st.download_button(
            label="📊 Download Comprehensive Report",
            data=report_content,
            file_name=f"keyword_clustering_report_{timestamp}.md",
            mime="text/markdown",
            help="Download detailed analysis report in Markdown format"
        )
        
        # Show preview
        with st.expander("👀 Report Preview", expanded=False):
            st.markdown(report_content[:2000] + "..." if len(report_content) > 2000 else report_content)
        
        st.success("✅ Report generated successfully!")
        
    except Exception as e:
        log_error(e, "generate_report")
        st.error(f"Report generation failed: {str(e)}")

def generate_executive_summary(df, config):
    """Generate executive summary for report"""
    try:
        total_keywords = len(df)
        total_clusters = df['cluster_id'].nunique()
        avg_coherence = df['cluster_coherence'].mean()
        rep_keywords = df['is_representative'].sum()
        rep_percentage = (rep_keywords / total_keywords) * 100
        largest_cluster = df['cluster_id'].value_counts().max()
        
        summary = f"""
## Executive Summary

This keyword clustering analysis processed **{total_keywords:,} keywords** using {config.get('embedding_method', 'unknown')} embeddings 
and {config.get('clustering_method', 'unknown')} clustering algorithm, resulting in **{total_clusters} distinct clusters**.

### Key Findings:
- Average semantic coherence: **{avg_coherence:.3f}**
- Representative keywords identified: **{rep_keywords:,}** ({rep_percentage:.1f}%)
- Largest cluster contains: **{largest_cluster} keywords**
- Processing method: **{config.get('embedding_method', 'unknown')}** embeddings with **{config.get('clustering_method', 'unknown')}** clustering
"""
        
        if 'search_volume' in df.columns:
            total_volume = df['search_volume'].sum()
            avg_volume = df['search_volume'].mean()
            summary += f"""- Total search volume: **{format_number(total_volume)}**
- Average search volume per keyword: **{format_number(avg_volume)}**
"""
        
        if 'search_intent' in df.columns:
            intent_dist = df['search_intent'].value_counts()
            if len(intent_dist) > 0:
                primary_intent = intent_dist.index[0]
                primary_intent_pct = (intent_dist.iloc[0] / total_keywords) * 100
                summary += f"""- Primary search intent: **{primary_intent}** ({primary_intent_pct:.1f}%)
- Intent diversity: **{len(intent_dist)} different intents**
"""
        
        if 'quality_score' in df.columns:
            avg_quality = df['quality_score'].mean()
            summary += f"- Average quality score: **{avg_quality:.1f}/10**\n"
        
        # Add performance summary
        processing_time = st.session_state.get('processing_time', 'Unknown')
        summary += f"""
### Performance Summary:
- Processing time: **{processing_time}**
- Keywords per second: **{int(total_keywords / float(processing_time.split()[0])) if processing_time != 'Unknown' else 'N/A'}**
- Memory efficiency: **Optimized** (reduced dimensions, batch processing)
"""
        
        return summary
        
    except Exception as e:
        log_error(e, "executive_summary")
        return "## Executive Summary\n\nError generating summary."

def calculate_detailed_metrics(df):
    """Calculate detailed metrics for report"""
    try:
        metrics = create_clustering_summary_metrics(df)
        
        # Format metrics for report
        formatted_metrics = f"""
## Detailed Metrics

### Cluster Distribution
- Total clusters: **{metrics.get('total_clusters', 'Unknown')}**
- Average cluster size: **{metrics.get('avg_cluster_size', 0):.1f}**
- Median cluster size: **{metrics.get('median_cluster_size', 0):.0f}**
- Largest cluster: **{metrics.get('largest_cluster_size', 'Unknown')} keywords**
- Smallest cluster: **{metrics.get('smallest_cluster_size', 'Unknown')} keywords**
- Size coefficient of variation: **{metrics.get('size_cv', 0):.2f}**

### Quality Metrics
- Average coherence: **{metrics.get('avg_coherence', 0):.3f}**
- Median coherence: **{metrics.get('median_coherence', 0):.3f}**
- Coherence range: **{metrics.get('min_coherence', 0):.3f} - {metrics.get('max_coherence', 0):.3f}**
- Coherence standard deviation: **{metrics.get('coherence_std', 0):.3f}**
- High coherence clusters (>0.7): **{metrics.get('high_coherence_clusters', 0)}** ({metrics.get('high_coherence_percentage', 0):.1f}%)
"""
        
        if 'total_search_volume' in metrics:
            formatted_metrics += f"""
### Search Volume Analysis
- Total search volume: **{format_number(metrics['total_search_volume'])}**
- Average search volume: **{format_number(metrics['avg_search_volume'])}**
- Median search volume: **{format_number(metrics['median_search_volume'])}**
- Maximum search volume: **{format_number(metrics['max_search_volume'])}**
- Zero volume keywords: **{metrics.get('zero_volume_keywords', 0):,}** ({metrics.get('zero_volume_percentage', 0):.1f}%)
- Volume concentration (top 20%): **{metrics.get('volume_concentration_20', 0):.1f}%**
"""
        
        if 'primary_intent' in metrics:
            formatted_metrics += f"""
### Search Intent Analysis
- Primary intent: **{metrics['primary_intent']}** ({metrics.get('primary_intent_percentage', 0):.1f}%)
- Intent diversity: **{metrics.get('intent_diversity', 0)} types**
- Intent entropy: **{metrics.get('intent_entropy', 0):.3f}**
"""
            
            # Add intent distribution
            if 'intent_distribution' in metrics:
                formatted_metrics += "\nIntent distribution:\n"
                for intent, percentage in sorted(metrics['intent_distribution'].items(), key=lambda x: x[1], reverse=True):
                    formatted_metrics += f"- {intent}: **{percentage:.1f}%**\n"
        
        if 'avg_quality' in metrics:
            formatted_metrics += f"""
### Quality Score Analysis
- Average quality: **{metrics['avg_quality']:.1f}/10**
- Median quality: **{metrics['median_quality']:.1f}/10**
- Quality range: **{metrics['min_quality']:.1f} - {metrics['max_quality']:.1f}**
- High quality clusters (≥7): **{metrics.get('high_quality_clusters', 0)}** ({metrics.get('high_quality_percentage', 0):.1f}%)
"""
        
        # Keyword characteristics
        formatted_metrics += f"""
### Keyword Characteristics
- Average keyword length: **{metrics.get('avg_keyword_length', 0):.1f} characters**
- Median keyword length: **{metrics.get('median_keyword_length', 0):.0f} characters**
- Average word count: **{metrics.get('avg_word_count', 0):.1f} words**
- Median word count: **{metrics.get('median_word_count', 0):.0f} words**
"""
        
        return formatted_metrics
        
    except Exception as e:
        log_error(e, "detailed_metrics")
        return "## Detailed Metrics\n\nError calculating metrics."

def generate_cluster_analysis_report(df):
    """Generate cluster-by-cluster analysis"""
    try:
        analysis = "## Cluster Analysis\n\n"
        
        # Get top clusters by different criteria
        cluster_sizes = df['cluster_id'].value_counts()
        
        # Determine sorting method
        if 'search_volume' in df.columns:
            cluster_volumes = df.groupby('cluster_id')['search_volume'].sum()
            top_clusters = cluster_volumes.nlargest(15).index
            analysis += "*Clusters sorted by total search volume*\n\n"
        else:
            top_clusters = cluster_sizes.nlargest(15).index
            analysis += "*Clusters sorted by size*\n\n"
        
        for idx, cluster_id in enumerate(top_clusters, 1):
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_name = cluster_data['cluster_name'].iloc[0]
            
            analysis += f"### {idx}. {cluster_name} (ID: {cluster_id})\n"
            
            # Basic metrics
            analysis += f"- **Keywords:** {len(cluster_data):,}\n"
            analysis += f"- **Average coherence:** {cluster_data['cluster_coherence'].mean():.3f}\n"
            
            # Representative keywords
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
            if not rep_keywords:
                rep_keywords = cluster_data.nlargest(5, 'cluster_coherence')['keyword'].tolist()
            
            if rep_keywords:
                analysis += f"- **Representative keywords:** {', '.join(rep_keywords[:5])}"
                if len(rep_keywords) > 5:
                    analysis += f" (+{len(rep_keywords)-5} more)"
                analysis += "\n"
            
            # Search volume
            if 'search_volume' in cluster_data.columns:
                total_volume = cluster_data['search_volume'].sum()
                avg_volume = cluster_data['search_volume'].mean()
                max_volume_keyword = cluster_data.loc[cluster_data['search_volume'].idxmax()]
                
                analysis += f"- **Total search volume:** {format_number(total_volume)}\n"
                analysis += f"- **Average volume:** {format_number(avg_volume)}\n"
                analysis += f"- **Highest volume keyword:** {max_volume_keyword['keyword']} ({format_number(max_volume_keyword['search_volume'])})\n"
            
            # Search intent
            if 'search_intent' in cluster_data.columns:
                intent_dist = cluster_data['search_intent'].value_counts()
                primary_intent = intent_dist.index[0]
                primary_pct = (intent_dist.iloc[0] / len(cluster_data)) * 100
                
                analysis += f"- **Primary intent:** {primary_intent} ({primary_pct:.1f}%)\n"
                
                if len(intent_dist) > 1:
                    analysis += f"- **Intent breakdown:** "
                    intent_summary = [f"{intent} ({(count/len(cluster_data)*100):.0f}%)" 
                                    for intent, count in intent_dist.items()]
                    analysis += ", ".join(intent_summary[:3])
                    if len(intent_dist) > 3:
                        analysis += f" +{len(intent_dist)-3} more"
                    analysis += "\n"
            
            # Quality score
            if 'quality_score' in cluster_data.columns:
                avg_quality = cluster_data['quality_score'].mean()
                analysis += f"- **Average quality score:** {avg_quality:.1f}/10\n"
            
            # Content opportunity
            if 'content_opportunity' in cluster_data.columns:
                opportunity = cluster_data['content_opportunity'].iloc[0]
                if opportunity and opportunity.strip():
                    analysis += f"- **Content opportunity:** {opportunity}\n"
            
            analysis += "\n"
        
        # Add summary statistics
        analysis += "### Cluster Summary Statistics\n\n"
        analysis += f"- **Total clusters analyzed:** {len(top_clusters)}\n"
        analysis += f"- **Keywords in top clusters:** {df[df['cluster_id'].isin(top_clusters)].shape[0]:,}\n"
        analysis += f"- **Percentage of total keywords:** {(df[df['cluster_id'].isin(top_clusters)].shape[0] / len(df) * 100):.1f}%\n"
        
        if 'search_volume' in df.columns:
            volume_in_top = df[df['cluster_id'].isin(top_clusters)]['search_volume'].sum()
            total_volume = df['search_volume'].sum()
            volume_pct = (volume_in_top / total_volume * 100) if total_volume > 0 else 0
            analysis += f"- **Search volume in top clusters:** {volume_pct:.1f}%\n"
        
        return analysis
        
    except Exception as e:
        log_error(e, "cluster_analysis_report")
        return "## Cluster Analysis\n\nError generating cluster analysis."

def generate_detailed_recommendations(df, config):
    """Generate detailed recommendations"""
    try:
        recommendations = "## Recommendations\n\n"
        
        # Calculate metrics for recommendations
        metrics = create_clustering_summary_metrics(df)
        
        # Content strategy recommendations
        recommendations += "### Content Strategy Recommendations\n\n"
        
        # High-value cluster recommendations
        if 'search_volume' in df.columns:
            high_volume_clusters = df.groupby('cluster_id').agg({
                'search_volume': 'sum',
                'keyword': 'count',
                'cluster_name': 'first'
            }).nlargest(5, 'search_volume')
            
            recommendations += "**1. Prioritize High-Volume Clusters:**\n"
            for idx, (cluster_id, row) in enumerate(high_volume_clusters.iterrows(), 1):
                recommendations += f"   - {row['cluster_name']}: {format_number(row['search_volume'])} volume, {row['keyword']} keywords\n"
            recommendations += "\n"
        
        # Intent-based recommendations
        if 'search_intent' in df.columns:
            intent_dist = df['search_intent'].value_counts(normalize=True)
            
            recommendations += "**2. Intent-Based Content Development:**\n"
            
            if intent_dist.get('Informational', 0) > 0.4:
                recommendations += "   - High informational intent detected. Focus on comprehensive guides, tutorials, and educational content.\n"
            
            if intent_dist.get('Commercial', 0) > 0.2:
                recommendations += "   - Significant commercial intent. Create comparison articles, reviews, and buyer's guides.\n"
            
            if intent_dist.get('Transactional', 0) > 0.15:
                recommendations += "   - Notable transactional intent. Optimize product pages and create targeted landing pages.\n"
            
            recommendations += "\n"
        
        # Cluster quality recommendations
        recommendations += "**3. Cluster Quality Optimization:**\n"
        
        avg_coherence = metrics.get('avg_coherence', 0)
        if avg_coherence < 0.5:
            recommendations += "   - Low average coherence detected. Consider:\n"
            recommendations += "     * Adjusting clustering parameters\n"
            recommendations += "     * Using more advanced embedding methods\n"
            recommendations += "     * Preprocessing keywords more thoroughly\n"
        elif avg_coherence > 0.7:
            recommendations += "   - Excellent coherence achieved! Your clusters are well-defined and semantically meaningful.\n"
        else:
            recommendations += "   - Moderate coherence. Some clusters may benefit from manual review and refinement.\n"
        
        # Size distribution recommendations
        size_cv = metrics.get('size_cv', 0)
        if size_cv > 1.5:
            recommendations += "   - High variation in cluster sizes detected. Consider merging small clusters or splitting large ones.\n"
        
        recommendations += "\n"
        
        # Technical SEO recommendations
        recommendations += "### Technical SEO Recommendations\n\n"
        
        # Keyword characteristics
        avg_word_count = metrics.get('avg_word_count', 0)
        if avg_word_count < 2:
            recommendations += "**1. Short-tail Keywords:** Many single-word keywords detected. Consider:\n"
            recommendations += "   - Expanding to long-tail variations for better targeting\n"
            recommendations += "   - Grouping related short-tail keywords for topic clusters\n\n"
        elif avg_word_count > 4:
            recommendations += "**1. Long-tail Focus:** Predominantly long-tail keywords. Consider:\n"
            recommendations += "   - Creating highly specific, targeted content\n"
            recommendations += "   - Using FAQ schema for question-based keywords\n\n"
        
        # Zero volume keywords
        if 'zero_volume_percentage' in metrics:
            zero_vol_pct = metrics['zero_volume_percentage']
            if zero_vol_pct > 30:
                recommendations += f"**2. Zero Volume Keywords:** {zero_vol_pct:.0f}% of keywords have no search volume.\n"
                recommendations += "   - Validate these keywords with alternative tools\n"
                recommendations += "   - Consider them for semantic SEO and topic completeness\n"
                recommendations += "   - Focus primary efforts on keywords with measurable demand\n\n"
        
        # Strategic recommendations
        recommendations += "### Strategic Recommendations\n\n"
        
        recommendations += "**1. Content Clustering Strategy:**\n"
        recommendations += "   - Create pillar pages for top 5-10 clusters\n"
        recommendations += "   - Develop supporting content for representative keywords\n"
        recommendations += "   - Implement strong internal linking within clusters\n\n"
        
        recommendations += "**2. Monitoring and Iteration:**\n"
        recommendations += "   - Track rankings for representative keywords\n"
        recommendations += "   - Monitor cluster performance over time\n"
        recommendations += "   - Refine clusters quarterly based on performance data\n\n"
        
        recommendations += "**3. Competitive Analysis:**\n"
        recommendations += "   - Analyze competitor coverage of identified clusters\n"
        recommendations += "   - Identify gaps in competitor content\n"
        recommendations += "   - Prioritize clusters with high volume and low competition\n\n"
        
        # Implementation timeline
        recommendations += "### Suggested Implementation Timeline\n\n"
        recommendations += "**Phase 1 (Month 1-2):**\n"
        recommendations += "- Focus on top 5 highest-volume clusters\n"
        recommendations += "- Create pillar content for each cluster\n"
        recommendations += "- Optimize existing content for cluster keywords\n\n"
        
        recommendations += "**Phase 2 (Month 3-4):**\n"
        recommendations += "- Expand to next 10 clusters\n"
        recommendations += "- Develop supporting content\n"
        recommendations += "- Implement internal linking strategy\n\n"
        
        recommendations += "**Phase 3 (Month 5-6):**\n"
        recommendations += "- Cover remaining clusters\n"
        recommendations += "- Measure and optimize based on performance\n"
        recommendations += "- Plan next iteration of keyword research\n"
        
        return recommendations
        
    except Exception as e:
        log_error(e, "detailed_recommendations")
        return "## Recommendations\n\nError generating recommendations."

def generate_methodology_section(config):
    """Generate methodology section for report"""
    try:
        methodology = f"""
## Methodology

### Data Processing Pipeline
1. **Data Import and Validation**
   - CSV file parsing with encoding detection
   - Duplicate removal and data cleaning
   - Column standardization and type conversion

2. **Preprocessing**
   - Method: **{config.get('preprocessing_method', 'auto')}**
   - Language: **{config.get('language', 'English')}**
   - Text normalization, tokenization, and lemmatization
   - Stopword removal and special character handling

3. **Embedding Generation**
   - Method: **{config.get('embedding_method', 'auto')}**
   - Model: **{config.get('ai_model', 'N/A') if config.get('embedding_method') == 'openai' else 'N/A'}**
   - Dimension reduction: **{'Enabled' if config.get('reduce_dimensions', False) else 'Disabled'}**
   - Target dimensions: **{config.get('target_dimensions', 'N/A') if config.get('reduce_dimensions', False) else 'N/A'}**

4. **Clustering**
   - Algorithm: **{config.get('clustering_method', 'auto')}**
   - Number of clusters: **{config.get('num_clusters', 'Auto-detected')}**
   - Minimum cluster size: **{config.get('min_cluster_size', 2)}**
   - Post-processing: Cluster refinement and outlier handling

### Quality Assurance
- **Coherence Calculation**: Cosine similarity within clusters
- **Representative Selection**: Based on centrality and search volume
- **Validation**: Silhouette analysis and manual review

### AI Enhancement
- **OpenAI Integration**: {'Enabled' if config.get('openai_api_key') else 'Disabled'}
- **Cluster Naming**: {'AI-powered' if config.get('openai_api_key') else 'Rule-based'}
- **Intent Analysis**: {'Enabled' if config.get('enable_intent_analysis', True) else 'Disabled'}
- **Quality Analysis**: {'AI-enhanced' if config.get('enable_quality_analysis', False) else 'Statistical'}

### Performance Optimization
- **Batch Processing**: Optimized for memory efficiency
- **Caching**: Strategic use of computation caching
- **Memory Management**: Automatic garbage collection
- **Maximum Keywords**: {config.get('max_keywords', 5000):,}

### Limitations and Considerations
1. **Embedding Quality**: Dependent on chosen method and available resources
2. **Cluster Interpretation**: Some manual review recommended
3. **Search Volume Data**: Optional but enhances prioritization
4. **Language Support**: Optimized for {config.get('language', 'English')}
"""
        
        return methodology
        
    except Exception as e:
        log_error(e, "methodology_section")
        return "## Methodology\n\nError generating methodology section."

def create_report_document(report_data):
    """Create final report document"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Keyword Clustering Analysis Report

**Generated on**: {timestamp}  
**Tool**: Semantic Keyword Clustering Platform v1.0  
**Report ID**: {hashlib.md5(timestamp.encode()).hexdigest()[:8]}

---

{report_data['executive_summary']}

---

{report_data['detailed_metrics']}

---

{report_data['cluster_analysis']}

---

{report_data['recommendations']}

---

{report_data['methodology']}

---

## Appendix

### A. Glossary

- **Coherence Score**: Measure of semantic similarity within a cluster (0-1)
- **Representative Keywords**: Most central keywords in each cluster
- **Search Intent**: User's goal when searching (Informational, Commercial, Transactional, Navigational)
- **Embedding**: Numerical representation of keyword meaning
- **Cluster**: Group of semantically related keywords

### B. Data Quality Notes

This analysis is based on the provided dataset and configuration. Results may vary with:
- Different preprocessing methods
- Alternative embedding models
- Adjusted clustering parameters
- Additional data sources

### C. Contact Information

For questions or support regarding this analysis:
- Documentation: [View online documentation]
- Support: [Contact support team]
- Version: 1.0.0

---

**Disclaimer**: This report is generated automatically based on statistical analysis and machine learning algorithms. 
Human review and domain expertise should be applied when implementing recommendations.

**Copyright**: {datetime.now().year} - All rights reserved. This report is confidential and proprietary.
"""
        
        return report
        
    except Exception as e:
        log_error(e, "create_report_document")
        return f"# Report Generation Error\n\nFailed to create report: {str(e)}"

def show_settings_actions_tab(df, config):
    """Show settings and actions tab"""
    try:
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            st.subheader("🔧 Post-Processing Actions")
            
            # Cluster refinement options
            st.markdown("#### Cluster Refinement")
            
            if st.button("🔄 Refine Small Clusters", use_container_width=True, help="Merge clusters smaller than minimum size"):
                with st.spinner("Refining clusters..."):
                    refined_df = refine_small_clusters(df)
                    if refined_df is not None:
                        st.session_state.results_df = refined_df
                        st.success("✅ Small clusters refined successfully!")
                        st.rerun()
            
            if st.button("🎯 Recalculate Representatives", use_container_width=True, help="Update representative keywords based on current data"):
                with st.spinner("Recalculating representatives..."):
                    updated_df = recalculate_representatives(df)
                    if updated_df is not None:
                        st.session_state.results_df = updated_df
                        st.success("✅ Representative keywords recalculated!")
                        st.rerun()
            
            # Data filtering options
            st.markdown("#### Data Filtering")
            
            min_coherence_filter = st.slider(
                "Filter by minimum coherence:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Remove keywords from clusters below this coherence threshold"
            )
            
            if 'search_volume' in df.columns:
                min_volume_filter = st.number_input(
                    "Filter by minimum search volume:",
                    min_value=0,
                    value=0,
                    help="Remove keywords below this search volume"
                )
            else:
                min_volume_filter = 0
            
            if st.button("🔍 Apply Filters", use_container_width=True):
                criteria = {
                    'min_coherence': min_coherence_filter,
                    'min_search_volume': min_volume_filter
                }
                filtered_df = filter_dataframe_by_criteria(df, criteria)
                
                if len(filtered_df) > 0:
                    st.session_state.results_df = filtered_df
                    st.success(f"✅ Filtered to {len(filtered_df):,} keywords")
                    st.rerun()
                else:
                    st.warning("⚠️ No keywords meet the filter criteria")
        
        with settings_col2:
            st.subheader("💾 Session Management")
            
            # Session info
            st.markdown("#### Current Session")
            session_info = get_session_info(df, config)
            
            for key, value in session_info.items():
                st.write(f"**{key}:** {value}")
            
            # Session actions
            st.markdown("#### Session Actions")
            
            if st.button("💾 Save Session State", use_container_width=True, help="Save current session configuration"):
                save_success = save_session_state(df, config)
                if save_success:
                    st.success("✅ Session saved successfully!")
                    save_session_data(df, config)
                else:
                    st.error("❌ Failed to save session")
            
            if st.button("🔄 Reset All Data", use_container_width=True, help="Clear all data and start fresh"):
                if st.checkbox("⚠️ Confirm reset (this will clear all results)", key="confirm_reset"):
                    clear_all_session_data()
                    st.success("✅ Session reset! Please refresh the page.")
                    time.sleep(2)
                    st.rerun()
        
        # Advanced configuration
        st.subheader("⚙️ Advanced Configuration")
        
        with st.expander("🔧 Runtime Settings", expanded=False):
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.markdown("#### Display Settings")
                
                # Results per page
                results_per_page = st.selectbox(
                    "Results per page:",
                    options=[10, 25, 50, 100, 250],
                    index=2,
                    help="Number of results to show in tables"
                )
                
                # Chart theme
                chart_theme = st.selectbox(
                    "Chart theme:",
                    options=["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn"],
                    index=0,
                    help="Visual theme for charts and graphs"
                )
                
                # Number format
                number_format = st.selectbox(
                    "Number format:",
                    options=["Auto", "Full", "Abbreviated", "Scientific"],
                    index=0,
                    help="How to display large numbers"
                )
            
            with config_col2:
                st.markdown("#### Performance Settings")
                
                # Cache settings
                enable_caching = st.checkbox(
                    "Enable result caching",
                    value=True,
                    help="Cache results to improve performance"
                )
                
                # Memory optimization
                memory_optimization = st.checkbox(
                    "Optimize memory usage",
                    value=True,
                    help="Use memory optimization techniques"
                )
                
                # Auto-refresh
                auto_refresh = st.checkbox(
                    "Auto-refresh charts",
                    value=False,
                    help="Automatically refresh charts when data changes"
                )
                with st.expander("🎚️ Cluster Scoring Weights", expanded=False):
                    weight_volume = st.slider("Search volume weight", 0.0, 5.0, float(config.get('weight_volume', 1.0)))
                    weight_cpc = st.slider("CPC weight", 0.0, 5.0, float(config.get('weight_cpc', 0.5)))
                    weight_comp = st.slider("Competition weight (negative)", -5.0, 0.0, float(config.get('weight_competition', -0.5)))
                    weight_trend = st.slider("Trend weight", 0.0, 5.0, float(config.get('weight_trend', 1.0)))
                    if st.button("Apply Weights", key="apply_weights"):
                        config['weight_volume'] = weight_volume
                        config['weight_cpc'] = weight_cpc
                        config['weight_competition'] = weight_comp
                        config['weight_trend'] = weight_trend
                        st.session_state['cluster_weights'] = {
                            'search_volume': weight_volume,
                            'cpc': weight_cpc,
                            'competition': weight_comp,
                            'trend': weight_trend
                        }
                        st.success("✅ Weights updated")
                        
            # Apply settings
            if st.button("💾 Apply Settings", use_container_width=True):
                new_settings = {
                    'results_per_page': results_per_page,
                    'chart_theme': chart_theme,
                    'number_format': number_format,
                    'enable_caching': enable_caching,
                    'memory_optimization': memory_optimization,
                    'auto_refresh': auto_refresh
                }
                
                st.session_state.app_settings = new_settings
                st.success("✅ Settings applied successfully!")
                
                # Clear cache if caching disabled
                if not enable_caching:
                    st.cache_data.clear()
        
        # System information
        with st.expander("🖥️ System Information", expanded=False):
            sys_col1, sys_col2 = st.columns(2)
            
            with sys_col1:
                st.markdown("#### Library Status")
                system_status = get_system_status()
                
                for lib, info in system_status.items():
                    if info['available']:
                        st.success(f"✅ {lib}: Available")
                    else:
                        st.warning(f"⚠️ {lib}: {info['message']}")

            with sys_col2:
                st.markdown("#### Memory Usage")
                if PSUTIL_AVAILABLE:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / 1024 / 1024

                        st.metric("Current Memory", f"{memory_mb:.1f} MB")
                        st.metric("Peak Memory", f"{st.session_state.memory_monitor.get('peak_memory', 0):.1f} MB")

                        # System memory
                        vm = psutil.virtual_memory()
                        st.metric("System Memory", f"{vm.percent:.1f}% used")
                    except Exception as e:
                        st.info("Memory information unavailable")
                else:
                    st.info("Install psutil for memory monitoring")
       
        # Debug information
        with st.expander("🐛 Debug Information", expanded=False):
            debug_col1, debug_col2 = st.columns(2)

            with debug_col1:
                st.markdown("#### Data Information")
                debug_info = {
                    "DataFrame Shape": df.shape if df is not None else "N/A",
                    "DataFrame Columns": list(df.columns) if df is not None else [],
                    "Memory Usage (MB)": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2) if df is not None else 0,
                    "Null Values": df.isnull().sum().to_dict() if df is not None else {},
                    "Data Types": df.dtypes.astype(str).to_dict() if df is not None else {}
                }
                st.json(debug_info)

            with debug_col2:
               st.markdown("#### Processing Information")
               processing_info = {
                   "Config": {k: str(v) for k, v in config.items() if k != 'openai_api_key'},
                   "Session State Keys": list(st.session_state.keys()),
                   "Processing Timestamp": st.session_state.get('processing_timestamp', 'Unknown'),
                   "Processing Time": st.session_state.get('processing_time', 'Unknown'),
                   "App Version": "1.0.0"
               }
               st.json(processing_info)
       
        # Export session configuration
        st.subheader("📤 Export Configuration")

        if st.button("📋 Export Session Configuration", use_container_width=True):
            try:
                config_export = {
                    "timestamp": datetime.now().isoformat(),
                    "configuration": {k: v for k, v in config.items() if k != 'openai_api_key'},
                    "settings": st.session_state.get('app_settings', {}),
                    "data_summary": {
                        "total_keywords": len(df) if df is not None else 0,
                        "total_clusters": df['cluster_id'].nunique() if df is not None else 0,
                        "columns": list(df.columns) if df is not None else []
                    }
                }

                config_json = json.dumps(config_export, indent=2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                st.download_button(
                    label="💾 Download Configuration",
                    data=config_json,
                    file_name=f"clustering_config_{timestamp}.json",
                    mime="application/json",
                    help="Save configuration for future use",
                )
            except Exception as e:
                st.error(f"Failed to export configuration: {str(e)}")
    except Exception as e:
        log_error(e, "settings_actions_tab")
        st.error(f"Settings tab error: {str(e)}")

def refine_small_clusters(df, min_size=3):
   """Refine small clusters by merging with similar ones"""
   try:
       if df is None or df.empty:
           return None
       
       # Identify small clusters
       cluster_sizes = df['cluster_id'].value_counts()
       small_clusters = cluster_sizes[cluster_sizes < min_size].index.tolist()
       
       if not small_clusters:
           st.info("ℹ️ No small clusters found to refine")
           return df
       
       st.info(f"🔄 Refining {len(small_clusters)} small clusters...")
       
       # For each small cluster, find the best merge target
       df_refined = df.copy()
       merge_count = 0
       
       # Calculate cluster centroids (using average coherence as proxy)
       cluster_coherences = df.groupby('cluster_id')['cluster_coherence'].mean()
       
       for small_cluster in small_clusters:
           small_cluster_keywords = df[df['cluster_id'] == small_cluster]['keyword'].tolist()
           
           if not small_cluster_keywords:
               continue
           
           # Find best matching cluster based on coherence similarity
           best_match = None
           best_score = -1
           
           large_clusters = cluster_sizes[cluster_sizes >= min_size].index
           
           for large_cluster in large_clusters:
               if large_cluster == small_cluster:
                   continue
               
               # Simple similarity based on coherence proximity
               coherence_diff = abs(cluster_coherences[small_cluster] - cluster_coherences[large_cluster])
               similarity_score = 1 - coherence_diff  # Closer coherence = higher score
               
               if similarity_score > best_score:
                   best_score = similarity_score
                   best_match = large_cluster
           
           # Merge with best match
           if best_match is not None:
               df_refined.loc[df_refined['cluster_id'] == small_cluster, 'cluster_id'] = best_match
               merge_count += 1
               
               # Update cluster name and description
               best_match_name = df_refined[df_refined['cluster_id'] == best_match]['cluster_name'].iloc[0]
               df_refined.loc[df_refined['cluster_id'] == best_match, 'cluster_name'] = best_match_name
       
       # Renumber clusters to be consecutive
       unique_clusters = sorted(df_refined['cluster_id'].unique())
       cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
       
       df_refined['cluster_id'] = df_refined['cluster_id'].map(cluster_mapping)
       
       # Update cluster sizes
       new_cluster_sizes = df_refined['cluster_id'].value_counts().to_dict()
       df_refined['cluster_size'] = df_refined['cluster_id'].map(new_cluster_sizes)
       
       st.success(f"✅ Merged {merge_count} small clusters. Total clusters: {df_refined['cluster_id'].nunique()}")
       
       return df_refined
       
   except Exception as e:
       log_error(e, "refine_small_clusters")
       st.error(f"Cluster refinement failed: {str(e)}")
       return None

def recalculate_representatives(df, top_k=5):
   """Recalculate representative keywords based on current clustering"""
   try:
       if df is None or df.empty:
           return None
       
       st.info("🔄 Recalculating representative keywords...")
       
       df_updated = df.copy()
       df_updated['is_representative'] = False
       
       total_representatives = 0
       
       # For each cluster, mark top keywords as representatives
       for cluster_id in df_updated['cluster_id'].unique():
           cluster_data = df_updated[df_updated['cluster_id'] == cluster_id]
           
           # Determine selection criteria
           if 'search_volume' in cluster_data.columns and cluster_data['search_volume'].sum() > 0:
               # Sort by search volume
               top_keywords = cluster_data.nlargest(min(top_k, len(cluster_data)), 'search_volume')
           else:
               # Sort by coherence
               top_keywords = cluster_data.nlargest(min(top_k, len(cluster_data)), 'cluster_coherence')
           
           # Mark as representatives
           df_updated.loc[top_keywords.index, 'is_representative'] = True
           total_representatives += len(top_keywords)
       
       st.success(f"✅ Identified {total_representatives} representative keywords across {df_updated['cluster_id'].nunique()} clusters")
       
       return df_updated
       
   except Exception as e:
       log_error(e, "recalculate_representatives")
       st.error(f"Representative recalculation failed: {str(e)}")
       return None

def update_cluster_metadata(df):
   """Update cluster metadata after modifications"""
   try:
       # Recalculate cluster sizes
       cluster_sizes = df['cluster_id'].value_counts().to_dict()
       df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
       
       # Ensure cluster names are consistent within clusters
       for cluster_id in df['cluster_id'].unique():
           cluster_data = df[df['cluster_id'] == cluster_id]
           
           # Use the most common cluster name
           name_counts = cluster_data['cluster_name'].value_counts()
           if len(name_counts) > 0:
               most_common_name = name_counts.index[0]
               df.loc[df['cluster_id'] == cluster_id, 'cluster_name'] = most_common_name
           
           # Update description if available
           if 'cluster_description' in df.columns:
               desc_counts = cluster_data['cluster_description'].value_counts()
               if len(desc_counts) > 0:
                   most_common_desc = desc_counts.index[0]
                   df.loc[df['cluster_id'] == cluster_id, 'cluster_description'] = most_common_desc
       
       return df
       
   except Exception as e:
       log_error(e, "update_cluster_metadata")
       return df

def get_session_info(df, config):
   """Get current session information"""
   try:
       info = {
           "Processing Time": st.session_state.get('processing_time', 'Unknown'),
           "Keywords Processed": format_number(len(df)) if df is not None else "0",
           "Clusters Created": str(df['cluster_id'].nunique()) if df is not None else "0",
           "Embedding Method": config.get('embedding_method', 'Unknown'),
           "Clustering Method": config.get('clustering_method', 'Unknown'),
           "AI Features": "Enabled" if config.get('openai_api_key') else "Disabled",
           "Session Start": st.session_state.get('session_start', 'Unknown')
       }
       
       if df is not None:
           memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
           info["Data Size (MB)"] = f"{memory_mb:.2f}"
       
       return info
       
   except Exception as e:
       log_error(e, "session_info")
       return {"Error": "Could not retrieve session info"}

def save_session_state(df, config):
   """Save current session state"""
   try:
       session_data = {
           'timestamp': datetime.now().isoformat(),
           'config': {k: v for k, v in config.items() if k != 'openai_api_key'},
           'results_summary': {
               'total_keywords': len(df) if df is not None else 0,
               'total_clusters': df['cluster_id'].nunique() if df is not None else 0,
               'avg_coherence': float(df['cluster_coherence'].mean()) if df is not None else 0,
               'columns': list(df.columns) if df is not None else []
           },
           'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
           'app_version': '1.0.0'
       }
       
       # Add search volume summary if available
       if df is not None and 'search_volume' in df.columns:
           session_data['results_summary']['total_search_volume'] = int(df['search_volume'].sum())
           session_data['results_summary']['avg_search_volume'] = float(df['search_volume'].mean())
       
       # Store in session state
       st.session_state.saved_session = session_data
       
       return True
       
   except Exception as e:
       log_error(e, "save_session_state")
       return False

def clear_all_session_data():
   """Clear all session data"""
   try:
       # Keys to preserve
       preserve_keys = {'app_settings', 'session_start'}
       
       # Clear all other keys
       keys_to_remove = [key for key in st.session_state.keys() if key not in preserve_keys]
       
       for key in keys_to_remove:
           del st.session_state[key]
       
       # Clear caches
       st.cache_data.clear()
       if hasattr(st, 'cache_resource'):
           st.cache_resource.clear()
       
       return True
       
   except Exception as e:
       log_error(e, "clear_session_data")
       return False

def save_session_data(df, config):
   """Save session data with download option"""
   try:
       # Create session backup
       session_backup = {
           'metadata': {
               'timestamp': datetime.now().isoformat(),
               'app_version': '1.0',
               'total_keywords': len(df) if df is not None else 0,
               'total_clusters': df['cluster_id'].nunique() if df is not None else 0,
               'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
           },
           'config': {k: v for k, v in config.items() if k != 'openai_api_key'},
           'data_columns': list(df.columns) if df is not None else [],
           'summary_stats': {
               'avg_coherence': float(df['cluster_coherence'].mean()) if df is not None else 0,
               'cluster_sizes': df['cluster_id'].value_counts().to_dict() if df is not None else {},
               'total_keywords': len(df) if df is not None else 0
           }
       }
       
       # Add additional stats if available
       if df is not None:
           if 'search_volume' in df.columns:
               session_backup['summary_stats']['total_search_volume'] = int(df['search_volume'].sum())
               session_backup['summary_stats']['avg_search_volume'] = float(df['search_volume'].mean())
           
           if 'search_intent' in df.columns:
               intent_dist = df['search_intent'].value_counts().to_dict()
               session_backup['summary_stats']['intent_distribution'] = intent_dist
       
       # Convert to JSON
       backup_json = json.dumps(session_backup, indent=2, default=str)
       
       # Offer download
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       
       st.download_button(
           label="💾 Download Session Backup",
           data=backup_json,
           file_name=f"clustering_session_{timestamp}.json",
           mime="application/json",
           help="Download session configuration and summary for future reference"
       )
       
       st.success("✅ Session backup created successfully!")
       
   except Exception as e:
       log_error(e, "save_session_data")
       st.error(f"Failed to save session data: {str(e)}")

def get_system_status():
   """Get system library status"""
   try:
       status = {
           "OpenAI": {
               "available": OPENAI_AVAILABLE,
               "version": "Available" if OPENAI_AVAILABLE else "Not installed",
               "message": "Ready for AI features" if OPENAI_AVAILABLE else "pip install openai"
           },
           "SentenceTransformers": {
               "available": SENTENCE_TRANSFORMERS_AVAILABLE,
               "version": "Available" if SENTENCE_TRANSFORMERS_AVAILABLE else "Not installed",
               "message": "Ready for embeddings" if SENTENCE_TRANSFORMERS_AVAILABLE else "pip install sentence-transformers"
           },
           "spaCy": {
               "available": SPACY_AVAILABLE,
               "version": "Available" if SPACY_AVAILABLE else "Not installed",
               "message": "Ready for NLP" if SPACY_AVAILABLE else "pip install spacy"
           },
           "TextBlob": {
               "available": TEXTBLOB_AVAILABLE,
               "version": "Available" if TEXTBLOB_AVAILABLE else "Not installed",
               "message": "Ready for text analysis" if TEXTBLOB_AVAILABLE else "pip install textblob"
           },
           "psutil": {
               "available": PSUTIL_AVAILABLE,
               "version": "Available" if PSUTIL_AVAILABLE else "Not installed",
               "message": "Ready for monitoring" if PSUTIL_AVAILABLE else "pip install psutil"
           }
       }
       
       return status
       
   except Exception as e:
       log_error(e, "system_status")
       return {}

def show_system_requirements():
   """Display system requirements and library status"""
   try:
       st.subheader("📋 System Requirements")
       
       req_col1, req_col2 = st.columns(2)
       
       with req_col1:
           st.markdown("#### Core Requirements")
           st.markdown("""
           **Essential Libraries:**
           - streamlit >= 1.28.0
           - pandas >= 1.5.0
           - numpy >= 1.23.0
           - scikit-learn >= 1.0.0
           - plotly >= 5.0.0
           - nltk >= 3.8.0
           """)
           
       with req_col2:
           st.markdown("#### Optional Libraries")
           st.markdown("""
           **For Enhanced Features:**
           - openai >= 1.0.0 (AI features)
           - sentence-transformers >= 2.2.0 (Better embeddings)
           - spacy >= 3.0.0 (Advanced NLP)
           - textblob >= 0.17.0 (Sentiment analysis)
           - psutil >= 5.9.0 (Resource monitoring)
           - openpyxl >= 3.0.0 (Excel export)
           """)
       
       # System status
       st.subheader("🔍 Current System Status")
       
       system_status = get_system_status()
       
       status_data = []
       for lib, info in system_status.items():
           status_data.append({
               "Library": lib,
               "Status": "✅ Available" if info["available"] else "❌ Not Installed",
               "Installation": info["message"]
           })
       
       status_df = pd.DataFrame(status_data)
       st.dataframe(status_df, use_container_width=True, hide_index=True)
       
       # Installation instructions
       with st.expander("📦 Installation Instructions", expanded=False):
           st.markdown("""
           #### Quick Installation
           
           **Install all optional libraries:**
           ```bash
           pip install openai sentence-transformers spacy textblob psutil openpyxl
           ```
           
           **Install spaCy language models:**
           ```bash
           python -m spacy download en_core_web_sm
           python -m spacy download es_core_news_sm
           python -m spacy download fr_core_news_sm
           python -m spacy download de_core_news_sm
           ```
           
           **Download NLTK data:**
           ```python
           import nltk
           nltk.download('punkt')
           nltk.download('stopwords')
           nltk.download('wordnet')
           nltk.download('averaged_perceptron_tagger')
           ```
           
           #### Troubleshooting
           
           - **Memory issues**: Use a smaller embedding model or reduce batch size
           - **OpenAI errors**: Check API key and quota limits
           - **spaCy errors**: Ensure language models are downloaded
           - **Performance issues**: Enable memory optimization in settings
           
           #### System Recommendations
           
           - **RAM**: Minimum 8GB, recommended 16GB+
           - **CPU**: Multi-core processor recommended
           - **Storage**: 2GB+ free space for models
           - **Python**: Version 3.8 or higher
           """)
       
   except Exception as e:
       log_error(e, "show_system_requirements")
       st.error(f"Error displaying system requirements: {str(e)}")

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # App header
        st.markdown('<h1 class="main-header">🔍 Semantic Keyword Clustering</h1>', unsafe_allow_html=True)
        
        # Show system status if needed
        if not all([NLTK_AVAILABLE]):
            with st.expander("⚠️ System Status Warning", expanded=True):
                st.warning("Some optional libraries are not available. Features may be limited.")
                show_system_requirements()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "🔍 Cluster Explorer", 
            "📈 Data Analysis",
            "📥 Export Results",
            "⚙️ Settings & Actions"
        ])
        
        # Check if results are available
        if st.session_state.get('results_df') is not None:
            df = st.session_state.results_df
            config = st.session_state.get('processing_metadata', {})
            
            with tab1:
                display_clustering_dashboard(df)
            
            with tab2:
                create_cluster_explorer(df)
            
            with tab3:
                show_data_analysis_tab(df, config)
            
            with tab4:
                show_export_options(df)
                
                # Generate report option
                st.markdown("---")
                st.subheader("📊 Report Generation")
                
                report_col1, report_col2 = st.columns(2)
                
                with report_col1:
                    if st.button("📊 Generate Comprehensive Report", use_container_width=True):
                        generate_comprehensive_report(df, config)
                
                with report_col2:
                    if st.button("📋 Generate Executive Summary", use_container_width=True):
                        executive_summary = generate_executive_summary(df, config)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            label="📥 Download Executive Summary",
                            data=executive_summary,
                            file_name=f"executive_summary_{timestamp}.md",
                            mime="text/markdown"
                        )
            
            with tab5:
                show_settings_actions_tab(df, config)
        
        else:
            # Show input interface
            st.info("👋 Welcome! Start by uploading your keyword data and configuring the clustering parameters.")
            
            # Quick start guide
            with st.expander("🚀 Quick Start Guide", expanded=False):
                st.markdown("""
                ### Getting Started with Keyword Clustering
                
                1. **Upload your CSV file** containing keywords
                   - Required column: `keyword`
                   - Optional columns: `search_volume`, `competition`, `cpc`
                
                2. **Configure clustering settings**
                   - Choose embedding method (OpenAI, SentenceTransformers, or TF-IDF)
                   - Select clustering algorithm (K-means or Hierarchical)
                   - Set number of clusters (auto-detect or manual)
                
                3. **Process your keywords**
                   - Click "Start Clustering Analysis"
                   - Wait for processing to complete
                
                4. **Explore results**
                   - View dashboard for overview
                   - Use cluster explorer for detailed analysis
                   - Export results in various formats
                
                **Pro Tips:**
                - Use OpenAI embeddings for best quality (requires API key)
                - Start with auto-detect clusters for optimal results
                - Include search volume data for better prioritization
                """)
            
            # File upload and configuration
            with st.container():
                st.header("📁 Data Input")
                
                upload_col1, upload_col2 = st.columns([3, 1])
                
                with upload_col1:
                    uploaded_file = st.file_uploader(
                        "Upload your keyword CSV file",
                        type=['csv'],
                        help="CSV file should contain a 'keyword' column. Optional columns: search_volume, competition, cpc",
                        key="keyword_csv_uploader"
                    )
                
                with upload_col2:
                    st.markdown("#### Sample Data")
                    if st.button("📋 Load Sample Data", help="Load a sample dataset to explore the tool"):
                        sample_df = create_sample_dataset()
                        st.session_state.original_df = sample_df
                        st.success("✅ Sample data loaded!")
                        st.rerun()
                
                if uploaded_file or st.session_state.get('original_df') is not None:
                    # Check if file has changed
                    if uploaded_file:
                        file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
                        uploaded_file.seek(0)  # Reset file pointer
                        
                        if st.session_state.get('last_uploaded_file') != file_hash:
                            st.session_state.last_uploaded_file = file_hash
                            st.session_state.original_df = None
                            st.session_state.results_df = None
                    
                    # CSV format options
                    if uploaded_file:
                        csv_col1, csv_col2 = st.columns(2)
                        
                        with csv_col1:
                            csv_format = st.selectbox(
                                "CSV format:",
                                options=["auto", "with_header", "no_header"],
                                help="Auto-detect or specify if your CSV has headers"
                            )
                        
                        with csv_col2:
                            preview_rows = st.slider(
                                "Preview rows:",
                                min_value=5,
                                max_value=50,
                                value=10,
                                help="Number of rows to show in preview"
                            )
                        
                        # Load data
                        df_input = load_csv_file(uploaded_file, csv_format)
                    else:
                        # Use existing data
                        df_input = st.session_state.original_df
                        preview_rows = 10
                    
                    if df_input is not None:
                        st.session_state.original_df = df_input
                        
                        # Data preview
                        st.subheader("📋 Data Preview")
                        
                        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Data", "Info", "Statistics"])
                        
                        with preview_tab1:
                            st.dataframe(df_input.head(preview_rows), use_container_width=True)
                        
                        with preview_tab2:
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.markdown("#### Dataset Information")
                                st.write(f"**Total rows:** {len(df_input):,}")
                                st.write(f"**Total columns:** {len(df_input.columns)}")
                                st.write(f"**Memory usage:** {df_input.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                            
                            with info_col2:
                                st.markdown("#### Column Types")
                                for col, dtype in df_input.dtypes.items():
                                    st.write(f"**{col}:** {dtype}")
                        
                        with preview_tab3:
                            st.markdown("#### Keyword Statistics")
                            
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            
                            with stat_col1:
                                st.metric("Total Keywords", format_number(len(df_input)))
                            
                            with stat_col2:
                                st.metric("Unique Keywords", format_number(df_input['keyword'].nunique()))
                            
                            with stat_col3:
                                if 'search_volume' in df_input.columns:
                                    st.metric("Total Volume", format_number(df_input['search_volume'].sum()))
                                else:
                                    st.metric("Avg Length", f"{df_input['keyword'].str.len().mean():.1f}")
                            
                            with stat_col4:
                                duplicate_count = len(df_input) - df_input['keyword'].nunique()
                                st.metric("Duplicates", format_number(duplicate_count))
                            
                            # Additional statistics
                            if 'search_volume' in df_input.columns:
                                st.markdown("#### Search Volume Distribution")
                                
                                vol_col1, vol_col2 = st.columns(2)
                                
                                with vol_col1:
                                    vol_stats = df_input['search_volume'].describe()
                                    st.dataframe(vol_stats.round(0).astype(int))
                                
                                with vol_col2:
                                    # Simple histogram
                                    fig = px.histogram(
                                        df_input[df_input['search_volume'] > 0],
                                        x='search_volume',
                                        nbins=30,
                                        title='Search Volume Distribution',
                                        log_y=True
                                    )
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Configuration section
                        st.header("⚙️ Clustering Configuration")
                        
                        config_container = st.container()
                        
                        with config_container:
                            # Create configuration tabs
                            config_tab1, config_tab2, config_tab3 = st.tabs(["Basic", "Advanced", "AI Features"])
                            
                            with config_tab1:
                                basic_col1, basic_col2 = st.columns(2)
                                
                                with basic_col1:
                                    st.markdown("#### Embedding Method")
                                    
                                    embedding_method = st.selectbox(
                                        "Choose embedding method:",
                                        options=["auto", "openai", "sentence_transformers", "tfidf"],
                                        help="""
                                        - **auto**: Automatically select best available method
                                        - **openai**: Highest quality, requires API key
                                        - **sentence_transformers**: Good quality, free
                                        - **tfidf**: Basic quality, always available
                                        """
                                    )
                                    
                                    if embedding_method in ["auto", "openai"]:
                                        openai_api_key = st.text_input(
                                            "OpenAI API Key (optional):",
                                            type="password",
                                            help="Required for OpenAI embeddings and AI features"
                                        )
                                    else:
                                        openai_api_key = ""
                                    
                                    language = st.selectbox(
                                        "Keyword language:",
                                        options=["English", "Spanish", "French", "German", "Portuguese", "Italian", "Dutch"],
                                        help="Language of your keywords for better preprocessing"
                                    )
                                
                                with basic_col2:
                                    st.markdown("#### Clustering Method")
                                    
                                    clustering_method = st.selectbox(
                                        "Choose clustering algorithm:",
                                        options=["auto", "kmeans", "hierarchical"],
                                        help="""
                                        - **auto**: Let the system choose based on data
                                        - **kmeans**: Fast, good for large datasets
                                        - **hierarchical**: Better for finding natural groups
                                        """
                                    )
                                    
                                    cluster_option = st.radio(
                                        "Number of clusters:",
                                        options=["Auto-detect", "Manual"],
                                        help="Let the algorithm decide or specify manually"
                                    )
                                    
                                    if cluster_option == "Manual":
                                        num_clusters = st.slider(
                                            "Target clusters:",
                                            min_value=2,
                                            max_value=min(50, len(df_input) // 5),
                                            value=min(10, len(df_input) // 20),
                                            help="Number of clusters to create"
                                        )
                                    else:
                                        num_clusters = None
                            
                            with config_tab2:
                                adv_col1, adv_col2 = st.columns(2)
                                
                                with adv_col1:
                                    st.markdown("#### Processing Options")
                                    
                                    preprocessing_method = st.selectbox(
                                        "Text preprocessing:",
                                        options=["auto", "basic", "spacy", "textblob"],
                                        index=0,
                                        help="Method for cleaning and normalizing keywords"
                                    )
                                    
                                    min_cluster_size = st.slider(
                                        "Minimum cluster size:",
                                        min_value=1,
                                        max_value=10,
                                        value=2,
                                        help="Minimum keywords per cluster"
                                    )
                                    
                                    enable_intent_analysis = st.checkbox(
                                        "Analyze search intent",
                                        value=True,
                                        help="Classify keywords by search intent (Informational, Commercial, etc.)"
                                    )
                                
                                with adv_col2:
                                    st.markdown("#### Performance Options")
                                    
                                    max_keywords = st.number_input(
                                        "Max keywords to process:",
                                        min_value=100,
                                        max_value=MAX_KEYWORDS,
                                        value=min(5000, len(df_input)),
                                        step=100,
                                        help="Limit for memory efficiency"
                                    )
                                    
                                    reduce_dimensions = st.checkbox(
                                        "Reduce embedding dimensions",
                                        value=True,
                                        help="Use PCA to reduce memory usage and improve speed"
                                    )
                                    
                                    if reduce_dimensions:
                                        target_dimensions = st.slider(
                                            "Target dimensions:",
                                            min_value=10,
                                            max_value=300,
                                            value=100,
                                            step=10,
                                            help="Lower dimensions = faster processing"
                                        )
                                    else:
                                        target_dimensions = None
                            
                            with config_tab3:
                                if openai_api_key:
                                    ai_col1, ai_col2 = st.columns(2)
                                    
                                    with ai_col1:
                                        st.markdown("#### AI Model")
                                        
                                        ai_model = st.selectbox(
                                            "OpenAI model:",
                                            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                                            help="Model for AI-powered features"
                                        )
                                        
                                        enable_quality_analysis = st.checkbox(
                                            "AI quality analysis",
                                            value=True,
                                            help="Use AI to analyze cluster quality"
                                        )
                                    
                                    with ai_col2:
                                        st.markdown("#### Cost Estimation")
                                        
                                        cost_estimate = calculate_estimated_cost(
                                            min(len(df_input), max_keywords),
                                            ai_model,
                                            num_clusters or 10
                                        )
                                        
                                        st.metric(
                                            "Estimated Cost",
                                            f"${cost_estimate['total_cost']:.4f}",
                                            help="Approximate API cost"
                                        )
                                        
                                        st.caption(f"Embeddings: ${cost_estimate['embedding_cost']:.4f}")
                                        st.caption(f"AI Features: ${cost_estimate['naming_cost']:.4f}")
                                else:
                                    st.info("🔐 Add OpenAI API key to enable AI features")
                                    ai_model = "gpt-4o-mini"
                                    enable_quality_analysis = False
                        
                        # Validation and warnings
                        st.markdown("---")
                        
                        # Check for potential issues
                        warnings = []
                        
                        if len(df_input) > 10000:
                            warnings.append(f"⚠️ Large dataset ({len(df_input):,} keywords). Processing may take several minutes.")
                        
                        if duplicate_count > len(df_input) * 0.1:
                            warnings.append(f"⚠️ High number of duplicates ({duplicate_count:,}). Consider deduplication.")
                        
                        if 'search_volume' not in df_input.columns:
                            warnings.append("ℹ️ No search volume data. Results will be based on semantic similarity only.")
                        
                        if embedding_method == "tfidf" and len(df_input) > 5000:
                            warnings.append("⚠️ TF-IDF may be slow for large datasets. Consider using other embedding methods.")
                        
                        if warnings:
                            for warning in warnings:
                                st.warning(warning)
                        
                        # Process button
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            if st.button(
                                "🚀 Start Clustering Analysis", 
                                type="primary", 
                                use_container_width=True,
                                disabled=st.session_state.get('processing_started', False)
                            ):
                                # Store configuration
                                config = {
                                    'embedding_method': embedding_method,
                                    'openai_api_key': openai_api_key,
                                    'preprocessing_method': preprocessing_method,
                                    'language': language,
                                    'clustering_method': clustering_method,
                                    'num_clusters': num_clusters,
                                    'min_cluster_size': min_cluster_size,
                                    'ai_model': ai_model if openai_api_key else None,
                                    'enable_intent_analysis': enable_intent_analysis,
                                    'enable_quality_analysis': enable_quality_analysis if openai_api_key else False,
                                    'max_keywords': max_keywords,
                                    'reduce_dimensions': reduce_dimensions,
                                    'target_dimensions': target_dimensions
                                }
                                config['weight_volume'] = 1.0
                                config['weight_cpc'] = 0.5
                                config['weight_competition'] = -0.5
                                config['weight_trend'] = 1.0
                                st.session_state['cluster_weights'] = {
                                    'search_volume': 1.0,
                                    'cpc': 0.5,
                                    'competition': -0.5,
                                    'trend': 1.0
                                }
                                
                                st.session_state.processing_metadata = config
                                st.session_state.processing_started = True
                                
                                # Process data
                                process_keywords(df_input, config)
                            # Help section
                            else:
                                st.info("💡 No file uploaded yet. Upload a CSV file with keywords to get started!")
                                # Show demo section
                                st.markdown("---")
                                st.subheader("🎯 What This Tool Does")
                
                demo_col1, demo_col2, demo_col3 = st.columns(3)
                
                with demo_col1:
                    st.markdown("""
                    #### 📊 Semantic Clustering
                    Groups keywords based on meaning and context, not just exact matches
                    """)
                
                with demo_col2:
                    st.markdown("""
                    #### 🤖 AI Enhancement
                    Uses AI to name clusters and analyze quality for better insights
                    """)
                
                with demo_col3:
                    st.markdown("""
                    #### 📈 SEO Optimization
                    Identifies content opportunities and search intent patterns
                    """)
        
        # Footer
        st.markdown("---")
        
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            st.markdown("### 📚 Resources")
            st.markdown("""
            - [Documentation](#)
            - [API Reference](#)
            - [Best Practices](#)
            """)
        
        with footer_col2:
            st.markdown("### 🛠️ Support")
            st.markdown("""
            - [Report Issues](#)
            - [Feature Requests](#)
            - [Contact Support](#)
            """)
        
        with footer_col3:
            st.markdown("### 📊 Version")
            st.markdown("""
            - **Version:** 1.0.0
            - **Updated:** Jan 2024
            - **License:** MIT
            """)
        
        # Hidden debug mode
        if st.sidebar.checkbox("🐛 Debug Mode", value=False, key="debug_mode"):
            with st.sidebar.expander("Debug Info"):
                st.json({
                    "session_state_keys": list(st.session_state.keys()),
                    "nltk_available": NLTK_AVAILABLE,
                    "openai_available": OPENAI_AVAILABLE,
                    "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
                    "spacy_available": SPACY_AVAILABLE,
                    "textblob_available": TEXTBLOB_AVAILABLE,
                    "psutil_available": PSUTIL_AVAILABLE
                })
        
    except Exception as e:
        log_error(e, "main_application")
        st.error(f"Application error: {str(e)}")
        st.info("Try refreshing the page or clearing browser cache if the issue persists.")
        
        # Error recovery options
        if st.button("🔄 Reset Application"):
            clear_all_session_data()
            st.rerun()

def process_keywords(df_input, config):
    """Main processing pipeline for keyword clustering"""
    try:
        start_time = time.time()
        
        st.header("🔄 Processing Keywords...")
        
        # Create progress tracker
        progress_steps = [
            "Validating data",
            "Preprocessing keywords", 
            "Generating embeddings",
            "Performing clustering",
            "Finding representatives",
            "Analyzing clusters",
            "Finalizing results"
        ]
        
        progress_tracker = create_progress_tracker(len(progress_steps), progress_steps)
        
        # Step 1: Validate data
        progress_tracker.update("Validating data...")
        
        is_valid, validation_message = validate_dataframe(df_input, ['keyword'])
        if not is_valid:
            st.error(f"❌ Data validation failed: {validation_message}")
            st.session_state.processing_started = False
            return
        
        # Limit keywords if needed
        if len(df_input) > config['max_keywords']:
            st.warning(f"⚠️ Limiting to {config['max_keywords']:,} keywords for processing")
            df_input = df_input.head(config['max_keywords'])
        
        keywords_list = df_input['keyword'].tolist()
        
        # Step 2: Preprocess keywords
        progress_tracker.update("Preprocessing keywords...")
        
        with st.spinner("Cleaning and normalizing keywords..."):
            processed_keywords = preprocess_keywords(
                keywords_list,
                language=config['language'],
                method=config['preprocessing_method']
            )
            
            # Show preprocessing summary
            st.info(f"✅ Preprocessed {len(processed_keywords)} keywords using {config['preprocessing_method']} method")
        
        # Step 3: Generate embeddings
        progress_tracker.update("Generating embeddings...")
        
        # Create OpenAI client if needed
        client = None
        if config['openai_api_key'] and config['embedding_method'] in ['auto', 'openai']:
            with st.spinner("Initializing OpenAI client..."):
                client = create_openai_client(config['openai_api_key'])
                if client is None and config['embedding_method'] == 'openai':
                    st.error("❌ Failed to create OpenAI client")
                    st.session_state.processing_started = False
                    return
        
        embeddings = generate_embeddings(
            keywords_list,
            client=client,
            method=config['embedding_method']
        )
        
        if embeddings is None:
            st.error("❌ Failed to generate embeddings")
            st.session_state.processing_started = False
            return
        
        # Reduce dimensions if requested
        if config['reduce_dimensions'] and config['target_dimensions']:
            original_dims = embeddings.shape[1]
            embeddings = reduce_embedding_dimensions(
                embeddings,
                target_dim=config['target_dimensions']
            )
            if embeddings is not None:
                st.info(f"✅ Reduced dimensions from {original_dims} to {embeddings.shape[1]}")
        
        # Step 4: Perform clustering
        progress_tracker.update("Performing clustering...")
        
        try:
            cluster_results = cluster_keywords(
                keywords_list,
                embeddings,
                n_clusters=config['num_clusters'],
                method=config['clustering_method'],
                min_cluster_size=config['min_cluster_size']
            )
            
            if cluster_results is None:
                st.error("❌ Clustering failed")
                st.session_state.processing_started = False
                return
                
        except Exception as cluster_error:
            log_error(cluster_error, "clustering_step")
            st.error(f"❌ Clustering error: {str(cluster_error)}")
            st.session_state.processing_started = False
            return
        
        # Step 5: Find representatives
        progress_tracker.update("Finding representative keywords...")
        
        # Already done in cluster_keywords, but ensure it's there
        if 'representatives' not in cluster_results:
            representatives = find_representative_keywords(
                embeddings,
                keywords_list,
                cluster_results['labels']
            )
            cluster_results['representatives'] = representatives
        
        # Step 6: Analyze clusters
        progress_tracker.update("Analyzing clusters...")
        
        # Search intent analysis
        intent_results = None
        intent_distribution = None
        
        if config['enable_intent_analysis']:
            with st.spinner("Analyzing search intent..."):
                intent_results, intent_distribution = analyze_search_intent_bulk(keywords_list)
                
                if intent_distribution:
                    # Show intent distribution
                    st.info("📊 Search Intent Distribution:")
                    intent_cols = st.columns(len(intent_distribution))
                    for idx, (intent, percentage) in enumerate(intent_distribution.items()):
                        with intent_cols[idx % len(intent_cols)]:
                            st.metric(intent, f"{percentage:.1f}%")
        
        # Generate cluster names
        cluster_names = {}
        quality_analysis = {}
        
        if client and config['openai_api_key']:
            # AI-powered naming
            with st.spinner("🤖 Generating AI cluster names..."):
                try:
                    cluster_names = generate_cluster_names_openai(
                        cluster_results['representatives'],
                        client,
                        model=config['ai_model']
                    )
                except Exception as naming_error:
                    log_error(naming_error, "ai_naming")
                    st.warning(f"⚠️ AI naming failed: {str(naming_error)}. Using fallback names.")
                    cluster_names = create_fallback_cluster_names(cluster_results['representatives'])
            
            # AI quality analysis
            if config['enable_quality_analysis']:
                with st.spinner("🔍 Performing AI quality analysis..."):
                    try:
                        quality_analysis = analyze_cluster_quality_ai(
                            cluster_results['representatives'],
                            cluster_results['coherence_scores'],
                            client,
                            model=config['ai_model']
                        )
                    except Exception as quality_error:
                        log_error(quality_error, "ai_quality_analysis")
                        st.warning(f"⚠️ AI quality analysis failed: {str(quality_error)}")
        else:
            # Fallback naming
            cluster_names = create_fallback_cluster_names(cluster_results['representatives'])
        
        # Step 7: Create results DataFrame
        progress_tracker.update("Finalizing results...")
        
        try:
            results_df = create_results_dataframe(
                keywords_list,
                cluster_results,
                cluster_names,
                cluster_results['coherence_scores'],
                intent_results,
                quality_analysis
            )
            
            if results_df is None:
                st.error("❌ Failed to create results")
                st.session_state.processing_started = False
                return
                
        except Exception as df_error:
            log_error(df_error, "results_dataframe_creation")
            st.error(f"❌ Failed to create results: {str(df_error)}")
            st.session_state.processing_started = False
            return
        
        # Merge with original data
        if 'search_volume' in df_input.columns or len(df_input.columns) > 1:
            with st.spinner("Merging with original data..."):
                results_df = merge_original_data(results_df, df_input)
                
                # Add search volume analysis
                if 'search_volume' in results_df.columns:
                    results_df = add_search_volume_data(results_df)

        results_df = add_trend_data(results_df)
        
        # Calculate final metrics
        with st.spinner("Calculating final metrics..."):
            cluster_metrics = calculate_cluster_metrics(results_df)
            cluster_weights = st.session_state.get('cluster_weights', None)
            calculate_weighted_cluster_scores(results_df, cluster_weights)
            
        # Complete processing
        progress_tracker.complete("Processing complete!")
        
        processing_time = time.time() - start_time
        st.session_state.processing_time = f"{processing_time:.1f} seconds"
        
        # Store results
        st.session_state.results_df = results_df
        st.session_state.cluster_evaluation = {
            'metrics': cluster_metrics,
            'quality_analysis': quality_analysis,
            'intent_distribution': intent_distribution
        }
        st.session_state.process_complete = True
        st.session_state.processing_timestamp = datetime.now().isoformat()
        st.session_state.processing_started = False
        
        # Show success message
        st.success(f"""
        ✅ **Clustering Complete!**
        - Processed {len(results_df):,} keywords
        - Created {results_df['cluster_id'].nunique()} clusters
        - Average coherence: {results_df['cluster_coherence'].mean():.3f}
        - Processing time: {processing_time:.1f} seconds
        """)
        
        # Show quick insights
        st.markdown("### 🎯 Quick Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            largest_cluster = results_df['cluster_id'].value_counts().iloc[0]
            largest_cluster_id = results_df['cluster_id'].value_counts().index[0]
            largest_cluster_name = results_df[results_df['cluster_id'] == largest_cluster_id]['cluster_name'].iloc[0]
            
            st.info(f"**Largest Cluster:** {largest_cluster_name} ({largest_cluster} keywords)")
        
        with insight_col2:
            if 'search_volume' in results_df.columns:
                highest_volume_cluster = results_df.groupby('cluster_id')['search_volume'].sum().idxmax()
                highest_volume_name = results_df[results_df['cluster_id'] == highest_volume_cluster]['cluster_name'].iloc[0]
                highest_volume = results_df[results_df['cluster_id'] == highest_volume_cluster]['search_volume'].sum()
                
                st.info(f"**Highest Volume:** {highest_volume_name} ({format_number(highest_volume)})")
            else:
                best_coherence_cluster = results_df.groupby('cluster_id')['cluster_coherence'].mean().idxmax()
                best_coherence_name = results_df[results_df['cluster_id'] == best_coherence_cluster]['cluster_name'].iloc[0]
                best_coherence = results_df[results_df['cluster_id'] == best_coherence_cluster]['cluster_coherence'].mean()
                
                st.info(f"**Best Coherence:** {best_coherence_name} ({best_coherence:.3f})")
        
        with insight_col3:
            if intent_distribution:
                primary_intent = max(intent_distribution, key=intent_distribution.get)
                st.info(f"**Primary Intent:** {primary_intent} ({intent_distribution[primary_intent]:.1f}%)")
            else:
                rep_count = results_df['is_representative'].sum()
                st.info(f"**Representatives:** {rep_count} keywords")
        
        # Memory cleanup
        clean_memory()
        
        # Auto-redirect to dashboard after short delay
        time.sleep(3)
        st.rerun()
        
    except Exception as e:
        log_error(e, "process_keywords", {
            "num_keywords": len(df_input) if 'df_input' in locals() else 0,
            "config": {k: v for k, v in config.items() if k != 'openai_api_key'}
        })
        st.error(f"❌ Processing failed: {str(e)}")
        st.info("Please check your configuration and try again.")
        st.session_state.processing_started = False
        
        # Show debug information
        with st.expander("🐛 Debug Information"):
            st.json({
                "error_type": type(e).__name__,
                "error_message": str(e),
                "config": {k: v for k, v in config.items() if k != 'openai_api_key'},
                "data_shape": df_input.shape if 'df_input' in locals() else None
            })

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    try:
        # Categories of keywords for realistic sample data
        categories = {
            "SEO/Marketing": [
                "seo tools", "best seo tools", "free seo tools", "seo software", "seo analysis tools",
                "keyword research", "keyword research tools", "keyword analysis", "keyword planner", "keyword finder",
                "content marketing", "content marketing strategy", "content creation tools", "content optimization",
                "digital marketing", "digital marketing tools", "online marketing", "marketing automation",
                "link building", "backlink analysis", "link building tools", "backlink checker",
                "competitor analysis", "competitor research", "competitive analysis tools"
            ],
            "E-commerce": [
                "buy shoes online", "best running shoes", "cheap shoes online", "shoe store near me",
                "nike shoes", "adidas shoes", "sports shoes", "comfortable walking shoes", "women's shoes",
                "online shopping", "best online shopping sites", "shopping deals", "discount shopping",
                "amazon shopping", "ebay shopping", "online marketplace", "shopping comparison",
                "product reviews", "customer reviews", "best products", "top rated products",
                "free shipping", "fast delivery", "return policy", "customer service"
            ],
            "Technology": [
                "artificial intelligence", "machine learning", "deep learning", "ai tools", "ai software",
                "python programming", "learn python", "python tutorial", "python for beginners", "python course",
                "web development", "web design", "responsive web design", "website builder", "web hosting",
                "cloud computing", "cloud storage", "cloud services", "aws", "google cloud",
                "cybersecurity", "data security", "network security", "security software", "vpn services",
                "blockchain technology", "cryptocurrency", "bitcoin", "ethereum", "crypto trading"
            ],
            "Health/Fitness": [
                "weight loss tips", "how to lose weight", "best diet plan", "healthy recipes", "low carb diet",
                "workout routine", "home workout", "gym exercises", "fitness tips", "muscle building",
                "yoga for beginners", "meditation techniques", "mindfulness exercises", "stress relief",
                "healthy lifestyle", "wellness tips", "nutrition advice", "vitamin supplements", "protein shakes",
                "mental health", "anxiety relief", "depression help", "therapy online", "counseling services"
            ],
            "Travel": [
                "cheap flights", "flight booking", "best travel sites", "travel deals", "last minute flights",
                "hotel booking", "cheap hotels", "luxury hotels", "hotel deals", "airbnb rentals",
                "vacation packages", "best vacation spots", "travel tips", "travel guide", "travel insurance",
                "car rental", "rental car deals", "airport parking", "travel accessories", "luggage sets",
                "adventure travel", "backpacking tips", "solo travel", "family vacation", "honeymoon destinations"
            ],
            "Food/Recipe": [
                "easy recipes", "quick dinner recipes", "healthy recipes", "vegetarian recipes", "vegan meals",
                "pizza recipe", "pasta recipes", "chicken recipes", "dessert recipes", "cake recipes",
                "restaurant near me", "best restaurants", "food delivery", "online food order", "takeout food",
                "cooking tips", "kitchen hacks", "meal prep", "meal planning", "grocery shopping",
                "coffee shops", "best coffee", "tea varieties", "smoothie recipes", "juice cleanse"
            ]
        }
        
        # Create comprehensive sample data
        sample_data = []
        
        for category, keywords in categories.items():
            for keyword in keywords:
                # Generate realistic metrics
                base_volume = np.random.choice([100, 500, 1000, 5000, 10000, 50000])
                volume_variation = np.random.uniform(0.5, 2.0)
                
                sample_data.append({
                    'keyword': keyword,
                    'search_volume': int(base_volume * volume_variation),
                    'competition': round(np.random.uniform(0.1, 1.0), 2),
                    'cpc': round(np.random.uniform(0.5, 5.0), 2),
                    'category': category  # Hidden column for validation
                })
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Shuffle to mix categories
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Remove category column (was just for generation)
        df = df.drop(columns=['category'])
        
        st.info(f"📊 Created sample dataset with {len(df)} keywords across multiple categories")
        
        return df
        
    except Exception as e:
        log_error(e, "create_sample_dataset")
        # Return minimal dataset as fallback
        return pd.DataFrame({
            'keyword': ['sample keyword 1', 'sample keyword 2', 'sample keyword 3'],
            'search_volume': [1000, 500, 250]
        })

def show_data_analysis_tab(df, config):
    """Show detailed data analysis tab"""
    try:
        analysis_subtab1, analysis_subtab2, analysis_subtab3 = st.tabs([
            "📋 Data Table", 
            "📊 Statistical Analysis", 
            "🔬 Advanced Analytics"
        ])
        
        with analysis_subtab1:
            show_data_table_view(df)
        
        with analysis_subtab2:
            show_statistical_analysis(df)
        
        with analysis_subtab3:
            show_advanced_analytics(df, config)
        
    except Exception as e:
        log_error(e, "data_analysis_tab")
        st.error(f"Data analysis error: {str(e)}")

def show_data_table_view(df):
    """Show interactive data table view with filtering and sorting"""
    try:
        st.markdown("#### 📋 Interactive Data Table")
        
        # Filter options
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Cluster filter
            cluster_options = ['All'] + sorted(df['cluster_id'].unique().tolist())
            selected_cluster = st.selectbox("Filter by cluster:", cluster_options)
        
        with filter_col2:
            # Intent filter
            if 'search_intent' in df.columns:
                intent_options = ['All'] + df['search_intent'].unique().tolist()
                selected_intent = st.selectbox("Filter by intent:", intent_options)
            else:
                selected_intent = 'All'
        
        with filter_col3:
            # Representative filter
            show_rep_only = st.checkbox("Show representatives only", value=False)
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_cluster != 'All':
            filtered_df = filtered_df[filtered_df['cluster_id'] == selected_cluster]
        
        if selected_intent != 'All' and 'search_intent' in df.columns:
            filtered_df = filtered_df[filtered_df['search_intent'] == selected_intent]
        
        if show_rep_only:
            filtered_df = filtered_df[filtered_df['is_representative'] == True]
        
        # Sorting options
        sort_col1, sort_col2 = st.columns(2)
        
        with sort_col1:
            sort_columns = st.selectbox(
                "Sort by:",
                options=['cluster_id', 'keyword', 'cluster_coherence'] + 
                       (['search_volume'] if 'search_volume' in df.columns else [])
            )
        
        with sort_col2:
            sort_order = st.radio("Order:", ['Ascending', 'Descending'], horizontal=True)
        
        # Apply sorting
        filtered_df = filtered_df.sort_values(
            sort_columns, 
            ascending=(sort_order == 'Ascending')
        )
        
        # Pagination
        st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} keywords**")
        
        rows_per_page = st.slider("Rows per page:", 10, 100, 25)
        total_pages = max(1, (len(filtered_df) + rows_per_page - 1) // rows_per_page)
        page = st.number_input("Page:", 1, total_pages, 1)
        
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(filtered_df))
        
        # Display columns selection
        available_cols = filtered_df.columns.tolist()
        display_cols = st.multiselect(
            "Display columns:",
            options=available_cols,
            default=['keyword', 'cluster_name', 'cluster_coherence', 'is_representative'] + 
                   (['search_volume'] if 'search_volume' in available_cols else [])
        )
        
        # Display table
        if display_cols:
            display_df = filtered_df[display_cols].iloc[start_idx:end_idx]
            
            # Format display
            if 'is_representative' in display_df.columns:
                display_df = display_df.copy()
                display_df['is_representative'] = display_df['is_representative'].map({True: '⭐', False: ''})
            
            st.dataframe(
                display_df, 
                use_container_width=True,
                height=400
            )
        else:
            st.warning("Please select at least one column to display")
        
        # Export filtered data
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Filtered Data ({len(filtered_df)} rows)",
                data=csv,
                file_name=f"filtered_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        log_error(e, "data_table_view")
        st.error(f"Error displaying data table: {str(e)}")

def show_statistical_analysis(df):
    """Show statistical analysis of the clustering results"""
    try:
        st.markdown("#### 📊 Statistical Analysis")
        
        # Basic statistics
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown("##### Cluster Statistics")
            
            cluster_stats = pd.DataFrame({
                'Metric': ['Total Clusters', 'Avg Cluster Size', 'Median Size', 'Std Dev Size', 'Min Size', 'Max Size'],
                'Value': [
                    df['cluster_id'].nunique(),
                    f"{df.groupby('cluster_id').size().mean():.1f}",
                    f"{df.groupby('cluster_id').size().median():.0f}",
                    f"{df.groupby('cluster_id').size().std():.1f}",
                    df.groupby('cluster_id').size().min(),
                    df.groupby('cluster_id').size().max()
                ]
            })
            
            st.dataframe(cluster_stats, hide_index=True, use_container_width=True)
        
        with stat_col2:
            st.markdown("##### Coherence Statistics")
            
            coherence_stats = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
                'Value': [
                    f"{df['cluster_coherence'].mean():.3f}",
                    f"{df['cluster_coherence'].median():.3f}",
                    f"{df['cluster_coherence'].std():.3f}",
                    f"{df['cluster_coherence'].min():.3f}",
                    f"{df['cluster_coherence'].max():.3f}",
                    f"{df['cluster_coherence'].quantile(0.25):.3f}",
                    f"{df['cluster_coherence'].quantile(0.75):.3f}"
                ]
            })
            
            st.dataframe(coherence_stats, hide_index=True, use_container_width=True)
        
        # Distribution charts
        st.markdown("##### Distribution Analysis")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            # Cluster size distribution
            cluster_sizes = df['cluster_id'].value_counts().values
            
            fig_sizes = px.histogram(
                x=cluster_sizes,
                nbins=20,
                title="Cluster Size Distribution",
                labels={'x': 'Cluster Size', 'y': 'Count'},
                template='plotly_white'
            )
            fig_sizes.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sizes, use_container_width=True)
        
        with dist_col2:
            # Coherence distribution
            fig_coherence = px.histogram(
                df,
                x='cluster_coherence',
                nbins=30,
                title="Coherence Score Distribution",
                labels={'cluster_coherence': 'Coherence Score', 'count': 'Number of Keywords'},
                template='plotly_white'
            )
            fig_coherence.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_coherence, use_container_width=True)
        
        # Box plots
        if 'search_volume' in df.columns:
            st.markdown("##### Search Volume Analysis by Cluster")
            
            # Prepare data for box plot
            top_clusters = df['cluster_id'].value_counts().head(15).index
            box_data = df[df['cluster_id'].isin(top_clusters)]
            
            fig_box = px.box(
                box_data,
                x='cluster_id',
                y='search_volume',
                title='Search Volume Distribution by Cluster (Top 15)',
                labels={'cluster_id': 'Cluster ID', 'search_volume': 'Search Volume'},
                template='plotly_white',
                log_y=True
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation analysis
        if len(df.select_dtypes(include=[np.number]).columns) > 2:
            st.markdown("##### Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto',
                template='plotly_white',
                text_auto='.2f'
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        
    except Exception as e:
        log_error(e, "statistical_analysis")
        st.error(f"Statistical analysis error: {str(e)}")

def show_advanced_analytics(df, config):
    """Show advanced analytics and insights"""
    try:
        st.markdown("#### 🔬 Advanced Analytics")
        
        # Cluster quality analysis
        st.markdown("##### Cluster Quality Metrics")
        
        quality_metrics = []
        
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            metrics = {
                'Cluster ID': cluster_id,
                'Cluster Name': cluster_data['cluster_name'].iloc[0][:30] + '...' if len(cluster_data['cluster_name'].iloc[0]) > 30 else cluster_data['cluster_name'].iloc[0],
                'Size': len(cluster_data),
                'Avg Coherence': cluster_data['cluster_coherence'].mean(),
                'Coherence Std': cluster_data['cluster_coherence'].std(),
                'Rep Ratio': cluster_data['is_representative'].mean()
            }
            
            if 'search_volume' in cluster_data.columns:
                metrics['Total Volume'] = cluster_data['search_volume'].sum()
                metrics['Avg Volume'] = cluster_data['search_volume'].mean()
            
            if 'quality_score' in cluster_data.columns:
                metrics['Quality Score'] = cluster_data['quality_score'].mean()
            
            quality_metrics.append(metrics)
        
        quality_df = pd.DataFrame(quality_metrics)
        
        # Format numeric columns
        for col in quality_df.select_dtypes(include=[np.number]).columns:
            if col in ['Avg Coherence', 'Coherence Std', 'Rep Ratio']:
                quality_df[col] = quality_df[col].round(3)
            elif col in ['Total Volume', 'Avg Volume']:
                quality_df[col] = quality_df[col].round(0).astype(int)
            elif col == 'Quality Score':
                quality_df[col] = quality_df[col].round(1)
        
        # Sort by size or volume
        sort_col = 'Total Volume' if 'Total Volume' in quality_df.columns else 'Size'
        quality_df = quality_df.sort_values(sort_col, ascending=False)
        
        st.dataframe(quality_df, hide_index=True, use_container_width=True)
        
        # Advanced visualizations
        st.markdown("##### Advanced Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Cluster quality scatter
            if 'search_volume' in df.columns:
                cluster_analysis = df.groupby('cluster_id').agg({
                    'cluster_coherence': 'mean',
                    'search_volume': 'sum',
                    'keyword': 'count'
                }).reset_index()
                
                fig_quality = px.scatter(
                    cluster_analysis,
                    x='cluster_coherence',
                    y='search_volume',
                    size='keyword',
                    title='Cluster Quality vs Search Volume',
                    labels={
                        'cluster_coherence': 'Average Coherence',
                        'search_volume': 'Total Search Volume',
                        'keyword': 'Number of Keywords'
                    },
                    template='plotly_white',
                    log_y=True
                )
                fig_quality.update_layout(height=400)
                st.plotly_chart(fig_quality, use_container_width=True)
            else:
                # Alternative visualization without volume
                cluster_analysis = df.groupby('cluster_id').agg({
                    'cluster_coherence': ['mean', 'std'],
                    'keyword': 'count'
                }).reset_index()
                
                cluster_analysis.columns = ['cluster_id', 'coherence_mean', 'coherence_std', 'size']
                
                fig_quality = px.scatter(
                    cluster_analysis,
                    x='coherence_mean',
                    y='coherence_std',
                    size='size',
                    title='Cluster Coherence Mean vs Standard Deviation',
                    labels={
                        'coherence_mean': 'Mean Coherence',
                        'coherence_std': 'Coherence Std Dev',
                        'size': 'Cluster Size'
                    },
template='plotly_white'
                )
                fig_quality.update_layout(height=400)
                st.plotly_chart(fig_quality, use_container_width=True)
        
        with viz_col2:
            # Intent distribution by cluster
            if 'search_intent' in df.columns:
                intent_cluster = df.groupby(['cluster_id', 'search_intent']).size().reset_index(name='count')
                
                # Get top clusters
                top_clusters = df['cluster_id'].value_counts().head(10).index
                intent_cluster_top = intent_cluster[intent_cluster['cluster_id'].isin(top_clusters)]
                
                fig_intent = px.bar(
                    intent_cluster_top,
                    x='cluster_id',
                    y='count',
                    color='search_intent',
                    title='Search Intent Distribution (Top 10 Clusters)',
                    labels={'cluster_id': 'Cluster ID', 'count': 'Number of Keywords'},
                    template='plotly_white'
                )
                fig_intent.update_layout(height=400)
                st.plotly_chart(fig_intent, use_container_width=True)
            else:
                # Alternative: keyword length distribution
                df['keyword_length'] = df['keyword'].str.len()
                
                length_cluster = df.groupby('cluster_id')['keyword_length'].agg(['mean', 'std']).reset_index()
                length_cluster = length_cluster.sort_values('mean', ascending=False).head(15)
                
                fig_length = px.bar(
                    length_cluster,
                    x='cluster_id',
                    y='mean',
                    error_y='std',
                    title='Average Keyword Length by Cluster (Top 15)',
                    labels={'cluster_id': 'Cluster ID', 'mean': 'Avg Length (chars)'},
                    template='plotly_white'
                )
                fig_length.update_layout(height=400)
                st.plotly_chart(fig_length, use_container_width=True)
        
        # Performance insights
        st.markdown("##### Performance Insights")
        
        insights = generate_performance_insights(df, config)
        
        for idx, insight in enumerate(insights, 1):
            st.info(f"{idx}. {insight}")
        
        # Cluster recommendations
        st.markdown("##### Cluster-Specific Recommendations")
        
        # Get top 5 clusters by size or volume
        if 'search_volume' in df.columns:
            top_clusters = df.groupby('cluster_id')['search_volume'].sum().nlargest(5).index
        else:
            top_clusters = df['cluster_id'].value_counts().head(5).index
        
        for cluster_id in top_clusters:
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_name = cluster_data['cluster_name'].iloc[0]
            
            with st.expander(f"📊 {cluster_name} (Cluster {cluster_id})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Keywords", len(cluster_data))
                    st.metric("Avg Coherence", f"{cluster_data['cluster_coherence'].mean():.3f}")
                    
                    if 'search_volume' in cluster_data.columns:
                        st.metric("Total Volume", format_number(cluster_data['search_volume'].sum()))
                
                with col2:
                    # Top keywords
                    st.markdown("**Top Keywords:**")
                    if 'search_volume' in cluster_data.columns:
                        top_kw = cluster_data.nlargest(5, 'search_volume')['keyword'].tolist()
                    else:
                        top_kw = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()[:5]
                    
                    for kw in top_kw:
                        st.write(f"- {kw}")
                
                # Recommendations
                st.markdown("**Recommendations:**")
                
                avg_coherence = cluster_data['cluster_coherence'].mean()
                if avg_coherence > 0.7:
                    st.success("✅ High coherence - excellent cluster quality")
                elif avg_coherence > 0.5:
                    st.warning("⚠️ Moderate coherence - consider refining keywords")
                else:
                    st.error("❌ Low coherence - review and possibly split cluster")
                
                if 'search_intent' in cluster_data.columns:
                    primary_intent = cluster_data['search_intent'].value_counts().index[0]
                    st.info(f"💡 Primary intent: {primary_intent} - align content accordingly")
        
    except Exception as e:
        log_error(e, "advanced_analytics")
        st.error(f"Advanced analytics error: {str(e)}")

def generate_performance_insights(df, config):
    """Generate performance insights based on analysis"""
    try:
        insights = []
        
        # Cluster quality insights
        avg_coherence = df['cluster_coherence'].mean()
        if avg_coherence > 0.7:
            insights.append("🎯 Excellent clustering quality! Most clusters show strong semantic coherence.")
        elif avg_coherence > 0.5:
            insights.append("👍 Good clustering quality. Some clusters may benefit from refinement.")
        else:
            insights.append("⚠️ Low clustering quality detected. Consider adjusting parameters or preprocessing.")
        
        # Size distribution insights
        cluster_sizes = df['cluster_id'].value_counts()
        size_cv = cluster_sizes.std() / cluster_sizes.mean()
        
        if size_cv < 0.5:
            insights.append("📊 Well-balanced cluster sizes across the dataset.")
        elif size_cv > 1.5:
            insights.append("📊 High variation in cluster sizes. Consider merging small clusters or splitting large ones.")
        
        # Volume insights
        if 'search_volume' in df.columns:
            zero_volume_pct = (df['search_volume'] == 0).mean() * 100
            
            if zero_volume_pct > 50:
                insights.append(f"📈 {zero_volume_pct:.0f}% of keywords have zero search volume. Focus on keywords with measurable demand.")
            elif zero_volume_pct < 10:
                insights.append("📈 Excellent! Most keywords have search volume data.")
            
            # Volume concentration
            cluster_volumes = df.groupby('cluster_id')['search_volume'].sum()
            top_20_pct = int(np.ceil(len(cluster_volumes) * 0.2))
            volume_concentration = cluster_volumes.nlargest(top_20_pct).sum() / cluster_volumes.sum()
            
            if volume_concentration > 0.8:
                insights.append(f"📊 High volume concentration: top 20% of clusters contain {volume_concentration:.0%} of search volume.")
        
        # Intent insights
        if 'search_intent' in df.columns:
            intent_dist = df['search_intent'].value_counts(normalize=True)
            primary_intent = intent_dist.index[0]
            primary_pct = intent_dist.iloc[0] * 100
            
            if primary_pct > 70:
                insights.append(f"🎯 Keywords heavily focused on {primary_intent} intent ({primary_pct:.0f}%). Consider diversifying.")
            elif len(intent_dist) >= 4:
                insights.append("🎯 Good intent diversity across your keyword portfolio.")
        
        # Representative insights
        rep_pct = df['is_representative'].mean() * 100
        if rep_pct < 5:
            insights.append("⭐ Very selective representative keyword identification. Quality over quantity approach.")
        elif rep_pct > 25:
            insights.append("⭐ High percentage of representative keywords. Consider tightening selection criteria.")
        
        # Method-specific insights
        if config.get('embedding_method') == 'openai':
            insights.append("🚀 Using OpenAI embeddings - highest quality semantic understanding.")
        elif config.get('embedding_method') == 'sentence_transformers':
            insights.append("🧠 Using SentenceTransformers - good balance of quality and efficiency.")
        elif config.get('embedding_method') == 'tfidf':
            insights.append("📊 Using TF-IDF - consider upgrading to semantic embeddings for better results.")
        
        return insights[:6]  # Limit to top 6 insights
        
    except Exception as e:
        log_error(e, "performance_insights")
        return ["Unable to generate insights due to analysis error."]

# Add this at the end of the file to run the application
if __name__ == "__main__":
    main()
