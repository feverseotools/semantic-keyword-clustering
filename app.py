"""
Advanced Semantic Keyword Clustering Application
Block 1: Imports, Configuration, and Constants
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
from functools import lru_cache
from io import StringIO
from collections import Counter

# Core data processing
import numpy as np
import pandas as pd

# Streamlit for UI
import streamlit as st

# NLP and ML libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# NLTK imports and downloads
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None
except Exception as e:
    logger.warning(f"NLTK setup failed: {str(e)}")
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
"""
Block 2: Utility Functions and Resource Management
"""

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
        'results_df': None
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
        import nltk
        datasets = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger']
        
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
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
        import spacy
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
"""
Block 3: Text Preprocessing Functions
"""

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
"""
Block 4: Embedding Generation Functions
"""

@st.cache_data(ttl=3600, max_entries=5)
def generate_openai_embeddings(keywords_list, client, model="text-embedding-3-small", batch_size=100):
    """Generate embeddings using OpenAI API with batching and caching"""
    if not client or not keywords_list:
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
                        response = client.embeddings.create(
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

def propagate_embeddings_to_similar(embeddings, keywords_list, max_propagation=1000):
    """Propagate embeddings to similar keywords using cosine similarity"""
    if embeddings is None:
        return None
        
    if len(embeddings) >= len(keywords_list):
        return embeddings
    
    try:
        st.info("🔄 Propagating embeddings to remaining keywords...")
        
        # Keywords with embeddings
        embedded_keywords = keywords_list[:len(embeddings)]
        remaining_keywords = keywords_list[len(embeddings):]
        
        if len(remaining_keywords) > max_propagation:
            st.warning(f"⚠️ Limiting propagation to {max_propagation} keywords")
            remaining_keywords = remaining_keywords[:max_propagation]
            # Update keywords_list to match
            keywords_list = embedded_keywords + remaining_keywords
        
        # Find most similar embedded keyword for each remaining keyword
        propagated_embeddings = []
        
        for remaining_kw in remaining_keywords:
            if not isinstance(remaining_kw, str) or not remaining_kw.strip():
                # Use first embedding as fallback for invalid keywords
                propagated_embeddings.append(embeddings[0].copy())
                continue
                
            best_similarity = -1
            best_embedding = embeddings[0].copy()  # fallback
            
            remaining_words = set(remaining_kw.lower().split())
            
            for i, embedded_kw in enumerate(embedded_keywords):
                if not isinstance(embedded_kw, str) or not embedded_kw.strip():
                    continue
                    
                # Calculate word overlap similarity
                embedded_words = set(embedded_kw.lower().split())
                overlap = len(remaining_words & embedded_words)
                
                # Boost similarity for exact substring matches
                if remaining_kw.lower() in embedded_kw.lower() or embedded_kw.lower() in remaining_kw.lower():
                    overlap += 2
                
                if overlap > best_similarity:
                    best_similarity = overlap
                    best_embedding = embeddings[i].copy()
            
            # Add small random noise to avoid identical embeddings
            noise_scale = 0.01
            noise = np.random.normal(0, noise_scale, best_embedding.shape).astype(best_embedding.dtype)
            propagated_embedding = best_embedding + noise
            
            # Normalize to maintain unit vector property
            norm = np.linalg.norm(propagated_embedding)
            if norm > 0:
                propagated_embedding = propagated_embedding / norm
            
            propagated_embeddings.append(propagated_embedding)
        
        # Combine original and propagated embeddings
        all_embeddings = np.vstack([embeddings, np.array(propagated_embeddings)])
        
        st.success(f"✅ Propagated embeddings to {len(propagated_embeddings)} keywords")
        return all_embeddings.astype(embeddings.dtype)
        
    except Exception as e:
        log_error(e, "embedding_propagation")
        st.warning(f"⚠️ Embedding propagation failed: {str(e)}")
        return embeddings
"""
Block 5: Clustering Algorithms
"""

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
                    elbow_k = cluster_range[elbow_idx + 1]
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
            best_sil_k = cluster_range[best_sil_idx]
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
        raise e}
        
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
"""
Block 6: AI-Powered Analysis Functions (Corrected Version)
"""

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
that explains the search intent and content opportunity."""
        
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
"""
Block 7: Data Processing and DataFrame Management (Corrected Version)
"""

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
        df_clean = df_clean.drop_duplicates(subset=['keyword_lower'])
        df = df_clean.drop(columns=['keyword_lower'])
        
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
            
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        
        if summary_df.empty:
            return summary_df
        
        # Sort by business value (combination of size, volume, and quality)
        if 'total_search_volume' in summary_df.columns:
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

def validate_results_dataframe(df):
    """Comprehensive validation of the final results DataFrame"""
    try:
        # Check if DataFrame exists and is not empty
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check required columns
        required_columns = ['keyword', 'cluster_id', 'cluster_name', 'cluster_coherence']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Data quality checks
        issues = []
        
        # Check for empty keywords
        empty_keywords = df['keyword'].isna().sum() + (df['keyword'] == '').sum()
        if empty_keywords > 0:
            issues.append(f"{empty_keywords} empty keywords")
        
        # Check cluster ID validity
        if df['cluster_id'].isna().sum() > 0:
            issues.append("NaN values in cluster_id column")
            df['cluster_id'] = df['cluster_id'].fillna(-1)
        
        # Check coherence scores
        invalid_coherence = ((df['cluster_coherence'] < 0) | (df['cluster_coherence'] > 1)).sum()
        if invalid_coherence > 0:
            issues.append(f"{invalid_coherence} invalid coherence scores")
            df['cluster_coherence'] = df['cluster_coherence'].clip(0, 1)
        
        # Check for duplicate keywords
        duplicate_keywords = df['keyword'].duplicated().sum()
        if duplicate_keywords > 0:
            issues.append(f"{duplicate_keywords} duplicate keywords")
        
        # Check cluster size consistency
        actual_sizes = df['cluster_id'].value_counts()
        if 'cluster_size' in df.columns:
            reported_sizes = df.groupby('cluster_id')['cluster_size'].first()
            
            size_mismatches = (actual_sizes != reported_sizes).sum()
            if size_mismatches > 0:
                issues.append(f"{size_mismatches} cluster size inconsistencies")
                # Fix cluster sizes
                df['cluster_size'] = df['cluster_id'].map(actual_sizes)
        
        # Check representative keywords distribution
        if 'is_representative' in df.columns:
            clusters_without_reps = 0
            for cluster_id in df['cluster_id'].unique():
                cluster_data = df[df['cluster_id'] == cluster_id]
                if not cluster_data['is_representative'].any():
                    clusters_without_reps += 1
            
            if clusters_without_reps > 0:
                issues.append(f"{clusters_without_reps} clusters without representative keywords")
        
        # Check search volume data (if present)
        if 'search_volume' in df.columns:
            negative_volumes = (df['search_volume'] < 0).sum()
            if negative_volumes > 0:
                issues.append(f"{negative_volumes} negative search volumes")
                df['search_volume'] = df['search_volume'].clip(lower=0)
        
        # Check quality scores (if present)
        if 'quality_score' in df.columns:
            invalid_quality = ((df['quality_score'] < 0) | (df['quality_score'] > 10)).sum()
            if invalid_quality > 0:
                issues.append(f"{invalid_quality} invalid quality scores")
                df['quality_score'] = df['quality_score'].clip(0, 10)
        
        # Report issues
        if issues:
            st.warning(f"⚠️ Data quality issues found and fixed: {'; '.join(issues)}")
        
        # Final statistics
        n_clusters = df['cluster_id'].nunique()
        n_keywords = len(df)
        avg_cluster_size = n_keywords / n_clusters if n_clusters > 0 else 0
        avg_coherence = df['cluster_coherence'].mean()
        
        st.info(f"""📊 Validation Summary:
        - Keywords: {n_keywords:,}
        - Clusters: {n_clusters}
        - Average cluster size: {avg_cluster_size:.1f}
        - Average coherence: {avg_coherence:.3f}
        """)
        
        return True, df
        
    except Exception as e:
        log_error(e, "dataframe_validation")
        return False, f"Validation error: {str(e)}"

def prepare_download_data(df, format_type="csv"):
    """Prepare data for download in various formats with enhanced options"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "csv":
            data, filename = export_results_to_csv(df, f"keyword_clusters_{timestamp}.csv")
            mime_type = "text/csv"
            
        elif format_type.lower() == "excel":
            # Create Excel with multiple sheets
            from io import BytesIO
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

def detect_csv_delimiter(file_content, sample_size=1000):
    """Detect CSV delimiter from file content"""
    try:
        # Take a sample of the file
        sample = file_content[:sample_size]
        
        # Common delimiters to test
        delimiters = [',', ';', '\t', '|']
        delimiter_scores = {}
        
        for delimiter in delimiters:
            lines = sample.split('\n')[:10]  # Check first 10 lines
            
            if len(lines) < 2:
                continue
            
            # Count delimiter occurrences per line
            counts = []
            for line in lines:
                if line.strip():
                    counts.append(line.count(delimiter))
            
            if counts:
                # Good delimiter should have consistent count across lines
                avg_count = np.mean(counts)
                consistency = 1 - (np.std(counts) / (avg_count + 1))  # Normalize by mean
                
                delimiter_scores[delimiter] = avg_count * consistency
        
        if delimiter_scores:
            best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
            return best_delimiter
        else:
            return ','  # Default fallback
            
    except Exception as e:
        log_error(e, "csv_delimiter_detection")
        return ','  # Default fallback

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
    try:
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # Optimize string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if col in df.columns:
                # Convert to category if many repeated values
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
        
        # Optimize integer columns
        int_columns = df.select_dtypes(include=['int64']).columns
        for col in int_columns:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= 0 and col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_min >= 0 and col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_min >= -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        float_columns = df.select_dtypes(include=['float64']).columns
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        memory_saved = initial_memory - final_memory
        
        if memory_saved > 0.1:  # Only report if significant savings
            st.info(f"💾 Memory optimized: {memory_saved:.1f}MB saved ({memory_saved/initial_memory*100:.1f}%)")
        
        return df
        
    except Exception as e:
        log_error(e, "dataframe_memory_optimization")
        return df
"""
Block 8: Visualization Functions (Corrected Version)
"""

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
        from plotly.subplots import make_subplots
        
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

def create_clustering_summary_metrics(df):
    """Create comprehensive summary metrics with enhanced calculations"""
    try:
        if df is None or df.empty:
            return {}
        
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
                    "Over 50% of keywords have zero search volume. Focus on clusters with measurable search demand."
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
                    f"Keywords are heavily skewed toward {metrics.get('primary_intent', 'unknown')} intent. Consider diversifying content strategy."
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

def create_advanced_visualizations(df):
    """Create additional advanced visualizations"""
    try:
        if df is None or df.empty:
            return None
        
        st.subheader("🔬 Advanced Visualizations")
        
        # Visualization selection
        viz_options = [
            "Cluster Network Graph",
            "Keyword Length Distribution", 
            "Coherence Correlation Matrix",
            "Performance Bubble Chart"
        ]
        
        if 'search_volume' in df.columns:
            viz_options.extend(["Volume vs Quality Scatter", "Volume Distribution by Intent"])
        
        if 'search_intent' in df.columns:
            viz_options.append("Intent Flow Diagram")
        
        selected_viz = st.selectbox(
            "Select advanced visualization:",
            options=viz_options,
            help="Choose from advanced analysis visualizations"
        )
        
        if selected_viz == "Cluster Network Graph":
            return create_cluster_network_graph(df)
        elif selected_viz == "Keyword Length Distribution":
            return create_keyword_length_distribution(df)
        elif selected_viz == "Coherence Correlation Matrix":
            return create_coherence_correlation_matrix(df)
        elif selected_viz == "Performance Bubble Chart":
            return create_performance_bubble_chart(df)
        elif selected_viz == "Volume vs Quality Scatter" and 'search_volume' in df.columns:
            return create_volume_quality_scatter(df)
        elif selected_viz == "Volume Distribution by Intent" and 'search_volume' in df.columns:
            return create_volume_by_intent(df)
        elif selected_viz == "Intent Flow Diagram" and 'search_intent' in df.columns:
            return create_intent_flow_diagram(df)
        
        return None
        
    except Exception as e:
        log_error(e, "advanced_visualizations")
        st.error(f"Advanced visualization error: {str(e)}")
        return None

def create_keyword_length_distribution(df):
    """Create keyword length distribution analysis"""
    try:
        # Calculate keyword lengths
        df_viz = df.copy()
        df_viz['keyword_length'] = df_viz['keyword'].str.len()
        df_viz['word_count'] = df_viz['keyword'].str.split().str.len()
        
        # Create subplot
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Character Length Distribution', 'Word Count Distribution', 
                          'Length vs Coherence', 'Length by Intent'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Character length histogram
        fig.add_trace(
            go.Histogram(x=df_viz['keyword_length'], nbinsx=30, name='Char Length'),
            row=1, col=1
        )
        
        # Word count histogram  
        fig.add_trace(
            go.Histogram(x=df_viz['word_count'], nbinsx=10, name='Word Count'),
            row=1, col=2
        )
        
        # Length vs coherence scatter
        fig.add_trace(
            go.Scatter(
                x=df_viz['keyword_length'],
                y=df_viz['cluster_coherence'],
                mode='markers',
                name='Length vs Coherence',
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # Length by intent box plot
        if 'search_intent' in df_viz.columns:
            for intent in df_viz['search_intent'].unique():
                intent_data = df_viz[df_viz['search_intent'] == intent]
                fig.add_trace(
                    go.Box(y=intent_data['keyword_length'], name=intent, showlegend=False),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=600,
            title_text="Keyword Length Analysis",
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        log_error(e, "keyword_length_distribution")
        return None

def create_performance_bubble_chart(df):
    """Create performance bubble chart combining multiple metrics"""
    try:
        # Aggregate cluster-level data
        cluster_metrics = df.groupby(['cluster_id', 'cluster_name']).agg({
            'cluster_coherence': 'mean',
            'keyword': 'count',
            'is_representative': 'sum'
        }).reset_index()
        
        cluster_metrics.columns = ['cluster_id', 'cluster_name', 'coherence', 'size', 'representatives']
        
        # Add search volume if available
        if 'search_volume' in df.columns:
            volume_data = df.groupby('cluster_id')['search_volume'].sum()
            cluster_metrics['volume'] = cluster_metrics['cluster_id'].map(volume_data)
        else:
            cluster_metrics['volume'] = cluster_metrics['size']
        
        # Add quality score if available
        if 'quality_score' in df.columns:
            quality_data = df.groupby('cluster_id')['quality_score'].mean()
            cluster_metrics['quality'] = cluster_metrics['cluster_id'].map(quality_data)
        else:
            cluster_metrics['quality'] = cluster_metrics['coherence'] * 10
        
        # Create bubble chart
        fig = px.scatter(
            cluster_metrics,
            x='coherence',
            y='quality',
            size='size',
            color='volume',
            hover_name='cluster_name',
            hover_data={
                'coherence': ':.3f',
                'quality': ':.1f', 
                'size': ':,',
                'volume': ':,.0f',
                'representatives': ':,'
            },
            title='Cluster Performance Analysis (Bubble Chart)',
            labels={
                'coherence': 'Semantic Coherence',
                'quality': 'Quality Score',
                'volume': 'Search Volume'
            },
            template='plotly_white',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=500)
        
        return fig
        
    except Exception as e:
        log_error(e, "performance_bubble_chart")
        return None
Block 9: Streamlit User Interface (Reordered & Clean Version)
"""

# =============================================================================
# SECTION 1: CONFIGURACIÓN Y SETUP
# =============================================================================

def create_sidebar_configuration():
    """Create comprehensive sidebar with configuration options and validation"""
    try:
        st.sidebar.header("⚙️ Configuration")
        
        # File upload section
        st.sidebar.subheader("📂 Data Input")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File",
            type=['csv', 'txt'],
            help="Upload a CSV file containing keywords. Supports UTF-8 encoding.",
            accept_multiple_files=False
        )
        
        # Show file info if uploaded
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 100:
                st.sidebar.error(f"⚠️ File too large: {file_size_mb:.1f}MB (max: 100MB)")
            else:
                st.sidebar.success(f"📁 {uploaded_file.name} ({file_size_mb:.1f}MB)")
        
        # CSV format configuration
        csv_format = st.sidebar.selectbox(
            "📋 CSV Format",
            options=["auto", "no_header", "with_header"],
            index=0,
            help="Auto-detect or specify CSV format:\n"
                 "• Auto: Automatically detect format\n"
                 "• No Header: Simple keyword list\n"
                 "• With Header: Structured data with columns"
        )
        
        # Language selection
        st.sidebar.subheader("🌍 Language & Processing")
        language_options = ["Auto"] + list(SPACY_MODELS.keys())
        selected_language = st.sidebar.selectbox(
            "Content Language",
            options=language_options,
            index=0,
            help="Select the language of your keywords for optimal preprocessing"
        )
        
        # OpenAI API configuration
        st.sidebar.subheader("🤖 AI Enhancement")
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            help="Enter your OpenAI API key for:\n"
                 "• High-quality semantic embeddings\n"
                 "• AI-powered cluster naming\n"
                 "• Quality analysis and recommendations",
            placeholder="sk-..."
        )
        
        # Validate API key format
        api_key_valid = True
        if openai_api_key:
            if not openai_api_key.startswith('sk-'):
                st.sidebar.warning("⚠️ API key should start with 'sk-'")
                api_key_valid = False
            elif len(openai_api_key) < 20:
                st.sidebar.warning("⚠️ API key seems too short")
                api_key_valid = False
            else:
                st.sidebar.success("✅ API key format looks valid")
        
        # Advanced clustering settings
        with st.sidebar.expander("🔧 Clustering Settings"):
            st.markdown("#### Algorithm Configuration")
            
            clustering_method = st.selectbox(
                "Clustering Algorithm",
                options=["auto", "kmeans", "hierarchical"],
                index=0,
                help="• Auto: Automatically choose best algorithm\n"
                     "• K-means: Fast, good for large datasets\n"
                     "• Hierarchical: Better for smaller datasets"
            )
            
            num_clusters = st.slider(
                "Target Number of Clusters",
                min_value=0,
                max_value=50,
                value=0,
                help="0 = Auto-detect optimal number\n"
                     "Manual selection overrides auto-detection"
            )
            
            min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=1,
                max_value=20,
                value=2,
                help="Minimum keywords per cluster\n"
                     "Smaller clusters will be merged with similar ones"
            )
            
            st.markdown("#### Embedding Configuration")
            
            embedding_method = st.selectbox(
                "Embedding Method",
                options=["auto", "openai", "sentence_transformers", "tfidf"],
                index=0,
                help="• Auto: Use best available method\n"
                     "• OpenAI: Highest quality (requires API key)\n"
                     "• SentenceTransformers: Good quality, free\n"
                     "• TF-IDF: Basic quality, always available"
            )
            
            # Show embedding method availability
            if embedding_method == "openai" and not openai_api_key:
                st.warning("⚠️ OpenAI method requires API key")
            elif embedding_method == "sentence_transformers" and not SENTENCE_TRANSFORMERS_AVAILABLE:
                st.warning("⚠️ SentenceTransformers not available")
        
        # Performance settings
        with st.sidebar.expander("⚡ Performance Settings"):
            max_keywords = st.slider(
                "Max Keywords to Process",
                min_value=100,
                max_value=MAX_KEYWORDS,
                value=min(10000, MAX_KEYWORDS),
                step=500,
                help=f"Limit keywords for memory management\n"
                     f"Maximum supported: {MAX_KEYWORDS:,}"
            )
            
            # Memory usage estimation
            estimated_memory = estimate_memory_usage(max_keywords, embedding_method)
            if estimated_memory > 1000:  # > 1GB
                st.warning(f"⚠️ Estimated memory: {estimated_memory:.0f}MB")
            else:
                st.info(f"📊 Estimated memory: {estimated_memory:.0f}MB")
            
            preprocessing_method = st.selectbox(
                "Text Preprocessing",
                options=["auto", "spacy", "textblob", "basic"],
                index=0,
                help="• Auto: Use best available method\n"
                     "• spaCy: Advanced NLP preprocessing\n"
                     "• TextBlob: Simple NLP preprocessing\n"
                     "• Basic: Minimal preprocessing"
            )
        
        # AI analysis settings
        with st.sidebar.expander("🧠 AI Analysis Settings"):
            ai_model = st.selectbox(
                "AI Model for Analysis",
                options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                index=0,
                help="• gpt-4o-mini: Fast and cost-effective\n"
                     "• gpt-4o: Balanced performance\n"
                     "• gpt-4-turbo: Highest quality analysis"
            )
            
            enable_intent_analysis = st.checkbox(
                "Enable Search Intent Analysis",
                value=True,
                help="Classify keywords by search intent:\n"
                     "Informational, Commercial, Transactional, Navigational"
            )
            
            enable_quality_analysis = st.checkbox(
                "Enable AI Quality Analysis",
                value=bool(openai_api_key and api_key_valid),
                disabled=not (openai_api_key and api_key_valid),
                help="Use AI to analyze cluster quality and provide recommendations\n"
                     "(Requires valid OpenAI API key)"
            )
            
            enable_content_suggestions = st.checkbox(
                "Generate Content Suggestions",
                value=bool(openai_api_key and api_key_valid),
                disabled=not (openai_api_key and api_key_valid),
                help="Generate content strategy recommendations\n"
                     "(Requires valid OpenAI API key)"
            )
        
        # Cost calculator
        with st.sidebar.expander("💰 Cost Calculator"):
            if uploaded_file and openai_api_key:
                try:
                    # Quick file analysis for cost estimation
                    temp_df = load_csv_file(uploaded_file, csv_format)
                    if temp_df is not None:
                        num_keywords = min(len(temp_df), max_keywords)
                        target_clusters = num_clusters if num_clusters > 0 else min(20, max(5, num_keywords // 100))
                        
                        cost_info = calculate_estimated_cost(num_keywords, ai_model, target_clusters)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Keywords", format_number(cost_info['processed_keywords']))
                            st.metric("Embeddings", f"${cost_info['embedding_cost']:.4f}")
                        with col2:
                            st.metric("AI Analysis", f"${cost_info['naming_cost']:.4f}")
                            st.metric("Total Est.", f"${cost_info['total_cost']:.4f}")
                        
                        # Cost warnings
                        if cost_info['total_cost'] > 5.0:
                            st.error("💸 High cost! Consider reducing keywords or using free alternatives")
                        elif cost_info['total_cost'] > 1.0:
                            st.warning("💰 Moderate cost. Review settings if budget is a concern")
                        else:
                            st.success("💚 Low cost estimation")
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                
                except Exception as e:
                    st.error(f"Cost calculation error: {str(e)}")
            else:
                st.info("Upload file and add API key for cost estimation")
        
        # System status
        with st.sidebar.expander("📚 System Status"):
            status_data = get_system_status()
            
            for library, status in status_data.items():
                if status['available']:
                    st.success(f"✅ {library}: {status['version']}")
                else:
                    st.error(f"❌ {library}: {status['message']}")
            
            # Memory status
            if PSUTIL_AVAILABLE:
                memory_info = get_memory_status()
                if memory_info['available_gb'] < 1:
                    st.warning(f"⚠️ Low memory: {memory_info['available_gb']:.1f}GB available")
                else:
                    st.info(f"💾 Memory: {memory_info['available_gb']:.1f}GB available")
        
        # Export current configuration
        with st.sidebar.expander("💾 Configuration Management"):
            config_dict = {
                'csv_format': csv_format,
                'language': selected_language,
                'clustering_method': clustering_method,
                'num_clusters': num_clusters,
                'min_cluster_size': min_cluster_size,
                'embedding_method': embedding_method,
                'max_keywords': max_keywords,
                'ai_model': ai_model,
                'enable_intent_analysis': enable_intent_analysis,
                'enable_quality_analysis': enable_quality_analysis,
                'preprocessing_method': preprocessing_method
            }
            
            config_json = json.dumps(config_dict, indent=2)
            
            st.download_button(
                label="📄 Export Configuration",
                data=config_json,
                file_name=f"clustering_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download current configuration for future use"
            )
            
            # Configuration upload
            uploaded_config = st.file_uploader(
                "Upload Configuration",
                type=['json'],
                help="Upload a previously saved configuration file"
            )
            
            if uploaded_config:
                try:
                    config_data = json.loads(uploaded_config.read())
                    st.success("✅ Configuration loaded (restart to apply)")
                    with st.expander("Preview Configuration"):
                        st.json(config_data)
                except Exception as e:
                    st.error(f"Invalid configuration file: {str(e)}")
        
        return {
            'uploaded_file': uploaded_file,
            'csv_format': csv_format,
            'language': selected_language if selected_language != "Auto" else "English",
            'openai_api_key': openai_api_key if api_key_valid else None,
            'clustering_method': clustering_method,
            'num_clusters': num_clusters if num_clusters > 0 else None,
            'min_cluster_size': min_cluster_size,
            'embedding_method': embedding_method,
            'max_keywords': max_keywords,
            'ai_model': ai_model,
            'enable_intent_analysis': enable_intent_analysis,
            'enable_quality_analysis': enable_quality_analysis,
            'enable_content_suggestions': enable_content_suggestions,
            'preprocessing_method': preprocessing_method,
            'api_key_valid': api_key_valid
        }
        
    except Exception as e:
        log_error(e, "sidebar_configuration")
        st.sidebar.error(f"Configuration error: {str(e)}")
        return create_fallback_config()

def create_fallback_config():
    """Create fallback configuration when sidebar fails"""
    return {
        'uploaded_file': None,
        'csv_format': 'auto',
        'language': 'English',
        'openai_api_key': None,
        'clustering_method': 'auto',
        'num_clusters': None,
        'min_cluster_size': 2,
        'embedding_method': 'auto',
        'max_keywords': 10000,
        'ai_model': 'gpt-4o-mini',
        'enable_intent_analysis': True,
        'enable_quality_analysis': False,
        'enable_content_suggestions': False,
        'preprocessing_method': 'auto',
        'api_key_valid': False
    }

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

def get_system_status():
    """Get system library status"""
    try:
        status = {
            "OpenAI": {
                "available": OPENAI_AVAILABLE,
                "version": "Available" if OPENAI_AVAILABLE else "Not installed",
                "message": "" if OPENAI_AVAILABLE else "pip install openai"
            },
            "SentenceTransformers": {
                "available": SENTENCE_TRANSFORMERS_AVAILABLE,
                "version": "Available" if SENTENCE_TRANSFORMERS_AVAILABLE else "Not installed",
                "message": "" if SENTENCE_TRANSFORMERS_AVAILABLE else "pip install sentence-transformers"
            },
            "spaCy": {
                "available": SPACY_AVAILABLE,
                "version": "Available" if SPACY_AVAILABLE else "Not installed",
                "message": "" if SPACY_AVAILABLE else "pip install spacy"
            },
            "TextBlob": {
                "available": TEXTBLOB_AVAILABLE,
                "version": "Available" if TEXTBLOB_AVAILABLE else "Not installed",
                "message": "" if TEXTBLOB_AVAILABLE else "pip install textblob"
            },
            "psutil": {
                "available": PSUTIL_AVAILABLE,
                "version": "Available" if PSUTIL_AVAILABLE else "Not installed",
                "message": "" if PSUTIL_AVAILABLE else "pip install psutil"
            }
        }
        
        return status
        
    except Exception as e:
        log_error(e, "system_status")
        return {}

def get_memory_status():
    """Get current memory status"""
    try:
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent
            }
        else:
            return {"available_gb": 2.0, "total_gb": 4.0, "used_percent": 50.0}
    except Exception as e:
        log_error(e, "memory_status")
        return {"available_gb": 2.0, "total_gb": 4.0, "used_percent": 50.0}

# =============================================================================
# SECTION 2: PANTALLA DE BIENVENIDA
# =============================================================================

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
                <h4>🤖 AI-Powered Analysis</h4>
                <ul style='padding-left: 1.2rem;'>
                    <li>OpenAI semantic embeddings</li>
                    <li>Intelligent cluster naming</li>
                    <li>Search intent classification</li>
                    <li>Content strategy recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h4>📊 Advanced Analytics</h4>
                <ul style='padding-left: 1.2rem;'>
                    <li>Interactive dashboards</li>
                    <li>Quality scoring & metrics</li>
                    <li>Performance visualizations</li>
                    <li>Business value analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h4>⚡ Enterprise Features</h4>
                <ul style='padding-left: 1.2rem;'>
                    <li>Process up to 25K keywords</li>
                    <li>Multiple export formats</li>
                    <li>Configurable parameters</li>
                    <li>Batch processing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Getting started section
        st.markdown("---")
        st.subheader("🚀 Getting Started")
        
        start_col1, start_col2 = st.columns([1, 1])
        
        with start_col1:
            st.markdown("""
            #### 1️⃣ Prepare Your Data
            
            **Supported Formats:**
            - CSV files with keywords
            - UTF-8 encoding recommended
            - Optional: search volume, competition data
            - Max file size: 100MB
            
            **Data Examples:**
            ```
            Simple format:
            running shoes
            best sneakers
            athletic footwear
            ```
            
            ```
            Advanced format:
            keyword,search_volume,competition
            running shoes,5400,0.75
            best sneakers,3200,0.82
            ```
            """)
        
        with start_col2:
            st.markdown("""
            #### 2️⃣ Configure Settings
            
            **Basic Setup:**
            - Upload your CSV file
            - Select language (auto-detected)
            - Choose clustering method
            
            **Optional Enhancements:**
            - Add OpenAI API key for AI features
            - Adjust clustering parameters
            - Enable advanced analysis
            
            **Pro Tips:**
            - Start with default settings
            - Use OpenAI for best results
            - Review cost estimates first
            """)
        
        # Method comparison
        st.markdown("---")
        st.subheader("🔬 Analysis Methods Comparison")
        
        methods_data = {
            "Method": ["TF-IDF", "SentenceTransformers", "OpenAI Embeddings"],
            "Quality": ["⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
            "Speed": ["Fast", "Medium", "Medium"],
            "Cost": ["Free", "Free", "Paid"],
            "Best For": [
                "Quick analysis, large datasets",
                "Balanced quality & performance", 
                "Highest accuracy, production use"
            ]
        }
        
        methods_df = pd.DataFrame(methods_data)
        st.dataframe(methods_df, use_container_width=True, hide_index=True)
        
        # Sample data download
        st.markdown("---")
        st.subheader("📁 Sample Data")
        
        sample_col1, sample_col2 = st.columns([1, 1])
        
        with sample_col1:
            st.markdown("""
            **Download sample files to test the application:**
            
            Not sure about the format? Try our sample datasets to understand 
            how the clustering works and what results to expect.
            """)
        
        with sample_col2:
            # Create sample datasets
            create_sample_downloads()
        
        # Usage scenarios
        st.markdown("---")
        st.subheader("💼 Use Cases & Applications")
        
        use_case_tabs = st.tabs(["SEO Strategy", "Content Planning", "Market Research", "Competitive Analysis"])
        
        with use_case_tabs[0]:
            st.markdown("""
            #### 🎯 SEO Strategy Development
            
            **Keyword Grouping:**
            - Organize thousands of keywords into logical groups
            - Identify primary and supporting keywords
            - Discover keyword gaps and opportunities
            
            **Content Architecture:**
            - Plan topic clusters for better site structure
            - Create hub and spoke content strategies
            - Optimize for semantic search
            
            **Technical SEO:**
            - Group keywords by search intent
            - Plan internal linking strategies
            - Optimize page targeting
            """)
        
        with use_case_tabs[1]:
            st.markdown("""
            #### ✍️ Content Planning & Creation
            
            **Content Ideation:**
            - Generate content ideas from keyword clusters
            - Identify content gaps in your niche
            - Plan comprehensive topic coverage
            
            **Editorial Calendar:**
            - Prioritize content based on search volume
            - Plan content series and campaigns
            - Align content with user intent
            
            **Content Optimization:**
            - Identify related keywords for existing content
            - Plan content updates and expansions
            - Improve topical authority
            """)
        
        with use_case_tabs[2]:
            st.markdown("""
            #### 📈 Market Research & Analysis
            
            **Market Understanding:**
            - Discover market segments and niches
            - Understand customer language and terminology
            - Identify trending topics and interests
            
            **Opportunity Analysis:**
            - Find underserved market segments
            - Identify high-value, low-competition areas
            - Discover emerging trends
            
            **Audience Insights:**
            - Understand search behavior patterns
            - Segment audiences by intent
            - Plan targeted marketing campaigns
            """)
        
        with use_case_tabs[3]:
            st.markdown("""
            #### 🏆 Competitive Analysis & Intelligence
            
            **Competitor Research:**
            - Analyze competitor keyword strategies
            - Identify their content themes
            - Find their keyword gaps
            
            **Strategic Planning:**
            - Benchmark against industry standards
            - Identify differentiation opportunities
            - Plan competitive content strategies
            
            **Market Positioning:**
            - Understand competitive landscape
            - Find unique positioning opportunities
            - Plan market entry strategies
            """)
        
        # Call to action
        st.markdown("---")
        
        cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
        
        with cta_col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                <h3 style="margin-bottom: 1rem; color: white;">Ready to Get Started?</h3>
                <p style="margin-bottom: 1.5rem; color: #f0f0f0;">Upload your CSV file in the sidebar to begin your keyword clustering analysis!</p>
                <p style="font-size: 0.9rem; color: #e0e0e0;">💡 Tip: Start with our sample data if you want to explore the features first</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical specifications
        with st.expander("🔧 Technical Specifications & Limits", expanded=False):
            spec_col1, spec_col2 = st.columns(2)
            
            with spec_col1:
                st.markdown("""
                **Processing Limits:**
                - Maximum keywords: 25,000
                - Maximum file size: 100MB
                - Supported encodings: UTF-8, Latin-1
                - Supported separators: Comma, semicolon, tab
                
                **Performance:**
                - TF-IDF processing: ~1,000 keywords/second
                - SentenceTransformers: ~100 keywords/second
                - OpenAI embeddings: Rate limited by API
                """)
            
            with spec_col2:
                st.markdown("""
                **System Requirements:**
                - Minimum RAM: 2GB available
                - Recommended RAM: 4GB+ for large datasets
                - Browser: Modern browser with JavaScript
                - Internet: Required for OpenAI features
                
                **Data Privacy:**
                - Processing happens in your browser session
                - No data stored on our servers
                - OpenAI API calls follow their privacy policy
                """)
        
        return True
        
    except Exception as e:
        log_error(e, "welcome_screen")
        st.error(f"Welcome screen error: {str(e)}")
        return False

def create_sample_downloads():
    """Create sample dataset downloads"""
    try:
        # Sample dataset 1: E-commerce keywords
        ecommerce_data = {
            'keyword': [
                'running shoes', 'best running shoes', 'nike running shoes', 'cheap running shoes',
                'running shoes for women', 'trail running shoes', 'marathon running shoes',
                'wireless headphones', 'best wireless headphones', 'bluetooth headphones',
                'noise cancelling headphones', 'gaming headphones', 'sports headphones',
                'laptop', 'best laptop', 'gaming laptop', 'business laptop', 'laptop deals',
                'MacBook Pro', 'Dell laptop', 'HP laptop', 'laptop reviews'
            ],
            'search_volume': [
                5400, 3200, 2800, 1600, 4100, 1500, 890,
                8900, 4200, 7200, 3100, 2400, 1800,
                12000, 5600, 3900, 2100, 4800,
                2800, 1900, 1400, 2200
            ],
            'competition': [
                0.75, 0.82, 0.68, 0.58, 0.71, 0.55, 0.42,
                0.89, 0.91, 0.85, 0.76, 0.69, 0.62,
                0.95, 0.89, 0.78, 0.66, 0.73,
                0.84, 0.72, 0.69, 0.77
            ]
        }
        
        # Sample dataset 2: SEO keywords
        seo_data = {
            'keyword': [
                'SEO tips', 'how to do SEO', 'SEO best practices', 'SEO guide',
                'keyword research', 'keyword research tools', 'free keyword tools',
                'backlink building', 'how to build backlinks', 'link building strategies',
                'Google ranking factors', 'how to rank on Google', 'improve Google rankings',
                'content marketing', 'content strategy', 'blog content ideas',
                'technical SEO', 'page speed optimization', 'mobile SEO'
            ],
            'search_volume': [
                2200, 1800, 1400, 3100,
                4500, 2800, 1900,
                1600, 1200, 980,
                2100, 3400, 1700,
                5600, 2400, 1500,
                1100, 890, 760
            ]
        }
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            ecommerce_df = pd.DataFrame(ecommerce_data)
            ecommerce_csv = ecommerce_df.to_csv(index=False)
            
            st.download_button(
                label="📦 E-commerce Sample (22 keywords)",
                data=ecommerce_csv,
                file_name="sample_ecommerce_keywords.csv",
                mime="text/csv",
                help="Sample e-commerce keywords with search volume and competition data",
                use_container_width=True
            )
        
        with download_col2:
            seo_df = pd.DataFrame(seo_data)
            seo_csv = seo_df.to_csv(index=False)
            
            st.download_button(
                label="🎯 SEO Sample (19 keywords)",
                data=seo_csv,
                file_name="sample_seo_keywords.csv",
                mime="text/csv",
                help="Sample SEO-related keywords with search volume data",
                use_container_width=True
            )
        
        # Show preview
        with st.expander("👀 Preview Sample Data", expanded=False):
            preview_tab1, preview_tab2 = st.tabs(["E-commerce Sample", "SEO Sample"])
            
            with preview_tab1:
                st.dataframe(ecommerce_df.head(10), use_container_width=True)
            
            with preview_tab2:
                st.dataframe(seo_df.head(10), use_container_width=True)
        
    except Exception as e:
    except Exception as e:
        log_error(e, "sample_downloads")
        st.error(f"Sample download error: {str(e)}")

# =============================================================================
# SECTION 3: PANTALLA DE PROCESAMIENTO
# =============================================================================

def show_processing_screen(config):
    """Show enhanced processing screen with real-time updates"""
    try:
        # Header
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="color: #1f1f1f; font-size: 2.5rem; margin-bottom: 0.5rem;">
                🔄 Processing Your Keywords
            </h1>
            <p style="color: #666; font-size: 1.2rem;">
                Analyzing and clustering your keyword data...
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration summary in organized sections
        st.subheader("📋 Processing Configuration")
        
        config_tab1, config_tab2, config_tab3 = st.tabs(["📊 Data & Method", "🤖 AI Settings", "⚡ Performance"])
        
        with config_tab1:
            data_col1, data_col2 = st.columns(2)
            
            with data_col1:
                st.markdown("#### Data Configuration")
                st.info(f"""
                **Language:** {config['language']}  
                **CSV Format:** {config['csv_format']}  
                **Max Keywords:** {format_number(config['max_keywords'])}  
                **Preprocessing:** {config['preprocessing_method']}
                """)
            
            with data_col2:
                st.markdown("#### Clustering Method")
                st.info(f"""
                **Algorithm:** {config['clustering_method']}  
                **Target Clusters:** {config['num_clusters'] or 'Auto-detect'}  
                **Min Cluster Size:** {config['min_cluster_size']}  
                **Embedding Method:** {config['embedding_method']}
                """)
        
        with config_tab2:
            ai_col1, ai_col2 = st.columns(2)
            
            with ai_col1:
                st.markdown("#### AI Enhancement")
                openai_status = "✅ Enabled" if config['openai_api_key'] else "❌ Disabled"
                st.info(f"""
                **OpenAI API:** {openai_status}  
                **AI Model:** {config['ai_model']}  
                **Intent Analysis:** {'✅ Enabled' if config['enable_intent_analysis'] else '❌ Disabled'}
                """)
            
            with ai_col2:
                st.markdown("#### Advanced Features")
                quality_status = "✅ Enabled" if config['enable_quality_analysis'] else "❌ Disabled"
                content_status = "✅ Enabled" if config['enable_content_suggestions'] else "❌ Disabled"
                st.info(f"""
                **Quality Analysis:** {quality_status}  
                **Content Suggestions:** {content_status}  
                **API Key Valid:** {'✅ Yes' if config['api_key_valid'] else '❌ No'}
                """)
        
        with config_tab3:
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("#### Performance Settings")
                estimated_memory = estimate_memory_usage(config['max_keywords'], config['embedding_method'])
                st.info(f"""
                **Estimated Memory:** {estimated_memory:.0f} MB  
                **Batch Processing:** Enabled  
                **Memory Optimization:** Auto
                """)
            
            with perf_col2:
                st.markdown("#### Expected Timeline")
                timeline = estimate_processing_time(config)
                st.info(f"""
                **Estimated Time:** {timeline['total']}  
                **Preprocessing:** {timeline['preprocessing']}  
                **Clustering:** {timeline['clustering']}  
                **AI Analysis:** {timeline['ai_analysis']}
                """)
        
        # Processing steps overview
        st.subheader("🔄 Processing Pipeline")
        
        steps = get_processing_steps(config)
        step_cols = st.columns(len(steps))
        
        for i, (step_col, step) in enumerate(zip(step_cols, steps)):
            with step_col:
                status_icon = "⏳" if i == 0 else "⏸️"
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; border: 2px solid #e0e0e0; border-radius: 10px; margin: 0.5rem 0;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{status_icon}</div>
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">{step['name']}</div>
                    <div style="font-size: 0.9rem; color: #666;">{step['description']}</div>
                    <div style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">~{step['time']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Warnings and recommendations
        warnings = generate_processing_warnings(config)
        if warnings:
            st.subheader("⚠️ Important Notes")
            for warning in warnings:
                st.warning(warning)
        
        recommendations = generate_processing_recommendations(config)
        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.info(rec)
        
        # Ready to process indicator
        st.markdown("---")
        
        ready_col1, ready_col2, ready_col3 = st.columns([1, 2, 1])
        with ready_col2:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); border-radius: 10px; color: white;">
                <h3 style="margin-bottom: 1rem; color: white;">🚀 Ready to Process</h3>
                <p style="margin-bottom: 0; color: #f0f0f0;">Click the button below to start the analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        return True
        
    except Exception as e:
        log_error(e, "processing_screen")
        st.error(f"Processing screen error: {str(e)}")
        return False

def estimate_processing_time(config):
    """Estimate processing time based on configuration"""
    try:
        num_keywords = config['max_keywords']
        
        # Preprocessing time
        if config['preprocessing_method'] == 'spacy':
            preprocessing_time = max(10, num_keywords / 100)  # ~100 keywords/second
        elif config['preprocessing_method'] == 'textblob':
            preprocessing_time = max(5, num_keywords / 200)   # ~200 keywords/second
        else:
            preprocessing_time = max(2, num_keywords / 500)   # ~500 keywords/second
        
        # Embedding time
        if config['embedding_method'] == 'openai':
            embedding_time = max(30, num_keywords / 50)       # ~50 keywords/second (API limited)
        elif config['embedding_method'] == 'sentence_transformers':
            embedding_time = max(20, num_keywords / 100)      # ~100 keywords/second
        else:  # TF-IDF
            embedding_time = max(5, num_keywords / 1000)      # ~1000 keywords/second
        
        # Clustering time
        clustering_time = max(10, num_keywords / 500)         # ~500 keywords/second
        
        # AI analysis time
        if config['enable_quality_analysis'] and config['openai_api_key']:
            ai_time = max(30, (config['num_clusters'] or 10) * 3)  # ~3 seconds per cluster
        else:
            ai_time = 5  # Basic analysis
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f} seconds"
            elif seconds < 3600:
                return f"{seconds/60:.1f} minutes"
            else:
                return f"{seconds/3600:.1f} hours"
        
        total_time = preprocessing_time + embedding_time + clustering_time + ai_time
        
        return {
            'preprocessing': format_time(preprocessing_time),
            'clustering': format_time(embedding_time + clustering_time),
            'ai_analysis': format_time(ai_time),
            'total': format_time(total_time)
        }
        
    except Exception as e:
        log_error(e, "processing_time_estimation")
        return {
            'preprocessing': '~30 seconds',
            'clustering': '~1 minute', 
            'ai_analysis': '~30 seconds',
            'total': '~2 minutes'
        }

def get_processing_steps(config):
    """Get list of processing steps based on configuration"""
    try:
        steps = [
            {
                'name': 'Data Loading',
                'description': 'Parse and validate CSV data',
                'time': '10s'
            },
            {
                'name': 'Preprocessing',
                'description': f'{config["preprocessing_method"]} text processing',
                'time': '30s'
            },
            {
                'name': 'Embeddings',
                'description': f'{config["embedding_method"]} semantic vectors',
                'time': '1-3m'
            },
            {
                'name': 'Clustering', 
                'description': f'{config["clustering_method"]} algorithm',
                'time': '30s'
            }
        ]
        
        if config['enable_intent_analysis']:
            steps.append({
                'name': 'Intent Analysis',
                'description': 'Classify search intents',
                'time': '15s'
            })
        
        if config['enable_quality_analysis'] and config['openai_api_key']:
            steps.append({
                'name': 'AI Analysis',
                'description': 'Quality assessment & naming',
                'time': '1-2m'
            })
        
        steps.append({
            'name': 'Results',
            'description': 'Generate final output',
            'time': '10s'
        })
        
        return steps
        
    except Exception as e:
        log_error(e, "processing_steps")
        return [{'name': 'Processing', 'description': 'Analyzing data', 'time': '2-5m'}]

def generate_processing_warnings(config):
    """Generate warnings based on configuration"""
    try:
        warnings = []
        
        # High memory usage warning
        estimated_memory = estimate_memory_usage(config['max_keywords'], config['embedding_method'])
        if estimated_memory > 1000:
            warnings.append(
                f"⚠️ High memory usage expected ({estimated_memory:.0f}MB). "
                "Consider reducing the number of keywords if you experience issues."
            )
        
        # API key warnings
        if config['embedding_method'] == 'openai' and not config['openai_api_key']:
            warnings.append(
                "⚠️ OpenAI embedding method selected but no API key provided. "
                "Will fallback to SentenceTransformers or TF-IDF."
            )
        
        if config['enable_quality_analysis'] and not config['openai_api_key']:
            warnings.append(
                "⚠️ AI quality analysis enabled but no API key provided. "
                "Only basic quality metrics will be calculated."
            )
        
        # Large dataset warnings
        if config['max_keywords'] > 15000:
            warnings.append(
                f"⚠️ Large dataset ({format_number(config['max_keywords'])} keywords). "
                "Processing may take several minutes. Please be patient."
            )
        
        # Clustering parameter warnings
        if config['num_clusters'] and config['num_clusters'] > config['max_keywords'] / 3:
            warnings.append(
                "⚠️ Number of clusters is high relative to dataset size. "
                "This may result in very small clusters."
            )
        
        # Method availability warnings
        if config['embedding_method'] == 'sentence_transformers' and not SENTENCE_TRANSFORMERS_AVAILABLE:
            warnings.append(
                "⚠️ SentenceTransformers not available. Will fallback to TF-IDF embeddings."
            )
        
        return warnings
        
    except Exception as e:
        log_error(e, "processing_warnings")
        return []

def generate_processing_recommendations(config):
    """Generate recommendations based on configuration"""
    try:
        recommendations = []
        
        # Performance recommendations
        if config['max_keywords'] > 10000 and config['embedding_method'] == 'openai':
            recommendations.append(
                "💡 For large datasets with OpenAI embeddings, consider processing in smaller batches "
                "or using SentenceTransformers for faster results."
            )
        
        # Quality recommendations
        if not config['openai_api_key']:
            recommendations.append(
                "💡 Add an OpenAI API key for the highest quality clustering results and AI-powered insights."
            )
        
        # Cost optimization
        if config['openai_api_key'] and config['max_keywords'] > 5000:
            estimated_cost = calculate_estimated_cost(config['max_keywords'], config['ai_model'])
            if estimated_cost['total_cost'] > 2.0:
                recommendations.append(
                    f"💡 Estimated cost is ${estimated_cost['total_cost']:.2f}. "
                    "Consider using fewer keywords or SentenceTransformers for cost savings."
                )
        
        # Method recommendations
        if config['embedding_method'] == 'auto':
            if config['openai_api_key']:
                recommendations.append(
                    "💡 Auto mode will use OpenAI embeddings for best results since API key is provided."
                )
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                recommendations.append(
                    "💡 Auto mode will use SentenceTransformers for good quality, free embeddings."
                )
            else:
                recommendations.append(
                    "💡 Auto mode will use TF-IDF embeddings (basic quality but fast and free)."
                )
        
        # Dataset size recommendations
        if config['max_keywords'] < 100:
            recommendations.append(
                "💡 Small dataset detected. Results may be more meaningful with 100+ keywords."
            )
        
        return recommendations
        
    except Exception as e:
        log_error(e, "processing_recommendations")
        return []

# =============================================================================
# SECTION 4: PANTALLA DE RESULTADOS
# =============================================================================

def show_results_screen(df, config):
    """Show comprehensive results screen with enhanced navigation"""
    try:
        if df is None or df.empty:
            st.error("❌ No results data available")
            return False
        
        # Results header with key metrics
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="color: #1f1f1f; font-size: 2.5rem; margin-bottom: 0.5rem;">
                📊 Clustering Results
            </h1>
            <p style="color: #666; font-size: 1.2rem;">
                Your keyword analysis is complete!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats bar
        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
        
        with stats_col1:
            st.metric("Keywords", format_number(len(df)))
        with stats_col2:
            st.metric("Clusters", df['cluster_id'].nunique())
        with stats_col3:
            avg_coherence = df['cluster_coherence'].mean()
            st.metric("Avg Coherence", f"{avg_coherence:.3f}")
        with stats_col4:
            if 'search_volume' in df.columns:
                total_volume = df['search_volume'].sum()
                st.metric("Total Volume", format_number(total_volume))
            else:
                rep_count = df['is_representative'].sum()
                st.metric("Representatives", rep_count)
        with stats_col5:
            processing_time = st.session_state.get('processing_time', 'Unknown')
            st.metric("Processing Time", processing_time)
        
        # Main navigation tabs
        main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
            "📊 Dashboard", 
            "🔍 Cluster Explorer", 
            "📈 Data Analysis",
            "📥 Export & Download",
            "🔧 Settings & Actions"
        ])
        
        with main_tab1:
            st.markdown("### 📊 Clustering Dashboard")
            dashboard_success = display_clustering_dashboard(df)
            
            if not dashboard_success:
                st.error("❌ Dashboard could not be displayed")
        
        with main_tab2:
            st.markdown("### 🔍 Interactive Cluster Explorer")
            explorer_success = create_cluster_explorer(df)
            
            if not explorer_success:
                st.error("❌ Cluster explorer could not be displayed")
        
        with main_tab3:
            st.markdown("### 📈 Detailed Data Analysis")
            show_data_analysis_tab(df, config)
        
        with main_tab4:
            st.markdown("### 📥 Export & Download Options")
            export_success = show_export_options(df)
            
            if not export_success:
                st.error("❌ Export options could not be displayed")
        
        with main_tab5:
            st.markdown("### 🔧 Settings & Actions")
            show_settings_actions_tab(df, config)
        
        # Footer actions
        st.markdown("---")
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            if st.button("🔄 Process New Dataset", use_container_width=True, type="primary"):
                clear_session_and_restart()
        
        with footer_col2:
            if st.button("💾 Save Current Session", use_container_width=True):
                save_session_data(df, config)
        
        with footer_col3:
            if st.button("📊 Generate Report", use_container_width=True):
                generate_comprehensive_report(df, config)
        
        return True
        
    except Exception as e:
        log_error(e, "results_screen")
        st.error(f"Results screen error: {str(e)}")
        return False

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

# =============================================================================
# SECTION 5: CONFIGURACIONES Y ACCIONES
# =============================================================================

def show_settings_actions_tab(df, config):
    """Show settings and actions tab"""
    try:
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            st.subheader("🔧 Post-Processing Actions")
            
            # Cluster refinement options
            st.markdown("#### Cluster Refinement")
            
            if st.button("🔄 Refine Small Clusters", use_container_width=True):
                refined_df = refine_small_clusters(df)
                if refined_df is not None:
                    st.session_state.df_results = refined_df
                    st.success("✅ Small clusters refined successfully!")
                    st.rerun()
            
            if st.button("🎯 Recalculate Representatives", use_container_width=True):
                updated_df = recalculate_representatives(df)
                if updated_df is not None:
                    st.session_state.df_results = updated_df
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
            
            if st.button("🔍 Apply Coherence Filter", use_container_width=True):
                if min_coherence_filter > 0:
                    filtered_df = df[df['cluster_coherence'] >= min_coherence_filter]
                    if len(filtered_df) > 0:
                        st.session_state.df_results = filtered_df
                        st.success(f"✅ Filtered to {len(filtered_df):,} keywords with coherence ≥ {min_coherence_filter}")
                        st.rerun()
                    else:
                        st.warning("⚠️ No keywords meet the coherence criteria")
                else:
                    st.info("ℹ️ Set minimum coherence > 0 to apply filter")
        
        with settings_col2:
            st.subheader("💾 Session Management")
            
            # Session info
            st.markdown("#### Current Session")
            session_info = get_session_info(df, config)
            
            for key, value in session_info.items():
                st.write(f"**{key}:** {value}")
            
            # Session actions
            st.markdown("#### Session Actions")
            
            if st.button("💾 Save Session State", use_container_width=True):
                save_success = save_session_state(df, config)
                if save_success:
                    st.success("✅ Session saved successfully!")
                else:
                    st.error("❌ Failed to save session")
            
            if st.button("🔄 Reset All Data", use_container_width=True):
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
                    options=["plotly_white", "plotly", "plotly_dark", "ggplot2"],
                    index=0,
                    help="Visual theme for charts and graphs"
                )
                
                # Number format
                number_format = st.selectbox(
                    "Number format:",
                    options=["Auto", "Full", "Abbreviated"],
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
        
        # Debug information
        with st.expander("🐛 Debug Information", expanded=False):
            debug_col1, debug_col2 = st.columns(2)
            
            with debug_col1:
                st.markdown("#### Data Information")
                st.json({
                    "dataframe_shape": df.shape,
                    "dataframe_columns": list(df.columns),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                    "null_values": df.isnull().sum().to_dict(),
                    "data_types": df.dtypes.astype(str).to_dict()
                })
            
            with debug_col2:
                st.markdown("#### Processing Information")
                st.json({
                    "config": {k: str(v) for k, v in config.items() if k != 'openai_api_key'},
                    "session_state_keys": list(st.session_state.keys()),
                    "processing_timestamp": st.session_state.get('processing_timestamp', 'Unknown'),
                    "system_info": get_system_status()
                })
        
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
        
        # For simplicity, merge small clusters with the most coherent large cluster
        large_clusters = cluster_sizes[cluster_sizes >= min_size].index.tolist()
        
        if not large_clusters:
            st.warning("⚠️ No large clusters available for merging")
            return df
        
        # Find the most coherent large cluster
        large_cluster_coherence = df[df['cluster_id'].isin(large_clusters)].groupby('cluster_id')['cluster_coherence'].mean()
        target_cluster = large_cluster_coherence.idxmax()
        
        # Merge small clusters
        df_refined = df.copy()
        for small_cluster in small_clusters:
            df_refined.loc[df_refined['cluster_id'] == small_cluster, 'cluster_id'] = target_cluster
        
        # Update cluster names and sizes
        df_refined = update_cluster_metadata(df_refined)
        
        st.success(f"✅ Merged {len(small_clusters)} small clusters into cluster {target_cluster}")
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
        
        # For each cluster, mark top keywords as representatives
        for cluster_id in df_updated['cluster_id'].unique():
            cluster_data = df_updated[df_updated['cluster_id'] == cluster_id]
            
            # Sort by search volume if available, otherwise by coherence
            if 'search_volume' in cluster_data.columns:
                top_keywords = cluster_data.nlargest(top_k, 'search_volume')
            else:
                top_keywords = cluster_data.nlargest(top_k, 'cluster_coherence')
            
            # Mark as representatives
            df_updated.loc[top_keywords.index, 'is_representative'] = True
        
        total_representatives = df_updated['is_representative'].sum()
        st.success(f"✅ Identified {total_representatives} representative keywords")
        
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
        
        # Update cluster names for merged clusters
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            if len(cluster_data) > 0:
                # Use the most common cluster name, or generate a new one
                existing_names = cluster_data['cluster_name'].value_counts()
                if len(existing_names) > 0:
                    most_common_name = existing_names.index[0]
                    df.loc[df['cluster_id'] == cluster_id, 'cluster_name'] = most_common_name
        
        return df
        
    except Exception as e:
        log_error(e, "update_cluster_metadata")
        return df

# =============================================================================
# SECTION 6: GESTIÓN DE SESIÓN
# =============================================================================

def get_session_info(df, config):
    """Get current session information"""
    try:
        info = {
            "Processing Time": st.session_state.get('processing_time', 'Unknown'),
            "Keywords Processed": format_number(len(df)) if df is not None else "0",
            "Clusters Created": df['cluster_id'].nunique() if df is not None else "0",
            "Embedding Method": config.get('embedding_method', 'Unknown'),
            "AI Features": "Enabled" if config.get('openai_api_key') else "Disabled",
            "Session Start": st.session_state.get('session_start', 'Unknown'),
            "Data Size (MB)": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2) if df is not None else "0"
        }
        
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
                'avg_coherence': df['cluster_coherence'].mean() if df is not None else 0,
            },
            'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        }
        
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
        preserve_keys = {'app_settings'}
        
        # Clear all other keys
        keys_to_remove = [key for key in st.session_state.keys() if key not in preserve_keys]
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        return True
        
    except Exception as e:
        log_error(e, "clear_session_data")
        return False

def clear_session_and_restart():
    """Clear session and restart application"""
    try:
        clear_all_session_data()
        st.rerun()
        
    except Exception as e:
        log_error(e, "clear_session_restart")
        st.error(f"Failed to restart: {str(e)}")

def save_session_data(df, config):
    """Save session data with download option"""
    try:
        # Create session backup
        session_backup = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'app_version': '1.0',
                'total_keywords': len(df) if df is not None else 0,
                'total_clusters': df['cluster_id'].nunique() if df is not None else 0
            },
            'config': {k: v for k, v in config.items() if k != 'openai_api_key'},
            'data_columns': list(df.columns) if df is not None else [],
            'summary_stats': {
                'avg_coherence': float(df['cluster_coherence'].mean()) if df is not None else 0,
                'cluster_sizes': df['cluster_id'].value_counts().to_dict() if df is not None else {}
            }
        }
        
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

# =============================================================================
# SECTION 7: SISTEMA DE REPORTES
# =============================================================================

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
        summary = f"""
## Executive Summary

This keyword clustering analysis processed **{len(df):,} keywords** using {config['embedding_method']} embeddings 
and {config['clustering_method']} clustering algorithm, resulting in **{df['cluster_id'].nunique()} distinct clusters**.

### Key Findings:
- Average semantic coherence: **{df['cluster_coherence'].mean():.3f}**
- Representative keywords identified: **{df['is_representative'].sum():,}** ({df['is_representative'].mean()*100:.1f}%)
- Largest cluster contains: **{df['cluster_id'].value_counts().max()} keywords**
- Processing method: **{config['embedding_method']}** embeddings with **{config['clustering_method']}** clustering
"""
        
        if 'search_volume' in df.columns:
            total_volume = df['search_volume'].sum()
            summary += f"- Total search volume: **{format_number(total_volume)}**\n"
        
        if 'search_intent' in df.columns:
            primary_intent = df['search_intent'].value_counts().index[0]
            summary += f"- Primary search intent: **{primary_intent}**\n"
        
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
- Total clusters: {metrics.get('total_clusters', 'Unknown')}
- Average cluster size: {metrics.get('avg_cluster_size', 0):.1f}
- Largest cluster: {metrics.get('largest_cluster_size', 'Unknown')} keywords
- Smallest cluster: {metrics.get('smallest_cluster_size', 'Unknown')} keywords
- Size coefficient of variation: {metrics.get('size_cv', 0):.2f}

### Quality Metrics
- Average coherence: {metrics.get('avg_coherence', 0):.3f}
- Coherence standard deviation: {metrics.get('coherence_std', 0):.3f}
- High coherence clusters (>0.7): {metrics.get('high_coherence_clusters', 0)}
"""
        
        if 'total_search_volume' in metrics:
            formatted_metrics += f"""
### Search Volume Analysis
- Total search volume: {format_number(metrics['total_search_volume'])}
- Average search volume: {format_number(metrics['avg_search_volume'])}
- Volume concentration (top 20%): {metrics.get('volume_concentration_20', 0):.1f}%
"""
        
        return formatted_metrics
        
    except Exception as e:
        log_error(e, "detailed_metrics")
        return "## Detailed Metrics\n\nError calculating metrics."

def generate_cluster_analysis_report(df):
    """Generate cluster-by-cluster analysis"""
    try:
        analysis = "## Cluster Analysis\n\n"
        
        # Get top 10 clusters by size
        top_clusters = df['cluster_id'].value_counts().head(10)
        
        for cluster_id in top_clusters.index:
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_name = cluster_data['cluster_name'].iloc[0]
            
            analysis += f"### {cluster_name} (ID: {cluster_id})\n"
            analysis += f"- Keywords: {len(cluster_data)}\n"
            analysis += f"- Coherence: {cluster_data['cluster_coherence'].mean():.3f}\n"
            
            # Representative keywords
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
            if rep_keywords:
                analysis += f"- Representative keywords: {', '.join(rep_keywords[:5])}\n"
            
            if 'search_volume' in cluster_data.columns:
                total_volume = cluster_data['search_volume'].sum()
                analysis += f"- Total search volume: {format_number(total_volume)}\n"
            
            if 'search_intent' in cluster_data.columns:
                primary_intent = cluster_data['search_intent'].value_counts().index[0]
                analysis += f"- Primary intent: {primary_intent}\n"
            
            analysis += "\n"
        
        return analysis
        
    except Exception as e:
        log_error(e, "cluster_analysis_report")
        return "## Cluster Analysis\n\nError generating cluster analysis."

def generate_detailed_recommendations(df, config):
    """Generate detailed recommendations"""
    try:
        recommendations = "## Recommendations\n\n"
        
        # Get basic recommendations
        basic_recs = generate_dashboard_recommendations(create_clustering_summary_metrics(df), df)
        
        for i, rec in enumerate(basic_recs, 1):
            recommendations += f"{i}. {rec}\n\n"
        
        # Add strategic recommendations
        recommendations += "### Strategic Recommendations\n\n"
        
        if 'search_volume' in df.columns:
            high_volume_clusters = df.groupby('cluster_id')['search_volume'].sum().nlargest(5)
            recommendations += f"- Focus content strategy on top 5 volume clusters: {', '.join(map(str, high_volume_clusters.index))}\n"
        
        if 'search_intent' in df.columns:
            intent_dist = df['search_intent'].value_counts(normalize=True)
            recommendations += f"- Develop content for underrepresented intents beyond {intent_dist.index[0]}\n"
        
        recommendations += "- Regularly review and refine cluster assignments for optimal performance\n"
        recommendations += "- Consider A/B testing content strategies for high-value clusters\n"
        
        return recommendations
        
    except Exception as e:
        log_error(e, "detailed_recommendations")
        return "## Recommendations\n\nError generating recommendations."

def generate_methodology_section(config):
    """Generate methodology section for report"""
    try:
        methodology = f"""
## Methodology

### Data Processing
- **Preprocessing method**: {config['preprocessing_method']}
- **Language**: {config['language']}
- **Maximum keywords processed**: {format_number(config['max_keywords'])}

### Clustering Configuration
- **Embedding method**: {config['embedding_method']}
- **Clustering algorithm**: {config['clustering_method']}
- **Minimum cluster size**: {config['min_cluster_size']}
- **Target clusters**: {config['num_clusters'] or 'Auto-detected'}

### AI Enhancement
- **OpenAI integration**: {'Enabled' if config['openai_api_key'] else 'Disabled'}
- **AI model**: {config['ai_model']}
- **Search intent analysis**: {'Enabled' if config['enable_intent_analysis'] else 'Disabled'}
- **Quality analysis**: {'Enabled' if config['enable_quality_analysis'] else 'Disabled'}

### Quality Metrics
- **Coherence scoring**: Cosine similarity within clusters
- **Representative selection**: Top keywords by search volume or coherence
- **Intent classification**: Rule-based pattern matching with ML validation
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

This report was generated automatically by the Semantic Keyword Clustering Platform.
For questions or support, please refer to the application documentation.

**Disclaimer**: This analysis is based on the provided data and selected configuration.
Results may vary with different settings or datasets.
"""
        
        return report
        
    except Exception as e:
        log_error(e, "create_report_document")
        return f"# Report Generation Error\n\nFailed to create report: {str(e)}""""
"""
Block 10: Main Application Logic - CORRECTED VERSION
"""

def run_clustering_pipeline(df, config):
    """Run the complete clustering pipeline with improved error handling"""
    try:
        # Validate inputs
        if df is None or len(df) == 0:
            raise ValueError("No data provided for clustering")
        
        if 'keyword' not in df.columns:
            raise ValueError("Dataset must contain a 'keyword' column")
        
        # Initialize progress tracker
        steps = ["Loading Data", "Preprocessing", "Generating Embeddings", "Clustering", 
                "Finding Representatives", "Naming Clusters", "Final Processing"]
        
        if config['enable_intent_analysis']:
            steps.append("Analyzing Search Intent")
        if config['enable_quality_analysis'] and config['openai_api_key']:
            steps.append("AI Quality Analysis")
        
        progress_tracker = create_progress_tracker(len(steps), steps)
        
        # Step 1: Prepare data
        progress_tracker.update("Loading and validating data...")
        
        # Limit keywords if necessary
        if len(df) > config['max_keywords']:
            st.warning(f"⚠️ Limiting dataset to {config['max_keywords']:,} keywords for performance")
            df = df.head(config['max_keywords'])
        
        # Clean and validate keywords
        df = df.dropna(subset=['keyword'])
        df = df[df['keyword'].str.strip() != '']
        
        if len(df) < 2:
            raise ValueError("Need at least 2 valid keywords for clustering")
        
        keywords_list = df['keyword'].str.strip().tolist()
        original_df = df.copy()
        
        # Step 2: Preprocessing
        progress_tracker.update("Preprocessing keywords...")
        
        try:
            processed_keywords = preprocess_keywords(
                keywords_list, 
                language=config['language'], 
                method="auto"
            )
        except Exception as e:
            st.warning(f"Preprocessing warning: {str(e)}. Using basic preprocessing.")
            processed_keywords = [kw.lower().strip() for kw in keywords_list]
        
        # Step 3: Generate embeddings
        progress_tracker.update("Generating semantic embeddings...")
        
        # Create OpenAI client if needed
        client = None
        if config['openai_api_key']:
            try:
                client = create_openai_client(config['openai_api_key'])
                if client is None:
                    st.warning("⚠️ OpenAI client creation failed. Falling back to free alternatives.")
            except Exception as e:
                st.warning(f"⚠️ OpenAI setup failed: {str(e)}. Using free alternatives.")
        
        # Generate embeddings with fallbacks
        embeddings = None
        embedding_method_used = config['embedding_method']
        
        try:
            embeddings = generate_embeddings(
                keywords_list,
                client=client,
                method=config['embedding_method']
            )
        except Exception as e:
            st.warning(f"⚠️ {config['embedding_method']} embeddings failed: {str(e)}. Trying fallbacks...")
            
            # Try fallback methods
            for fallback_method in ['sentence_transformers', 'tfidf']:
                if embeddings is None:
                    try:
                        embeddings = generate_embeddings(
                            keywords_list,
                            client=None,
                            method=fallback_method
                        )
                        embedding_method_used = fallback_method
                        st.info(f"✅ Using {fallback_method} embeddings as fallback")
                        break
                    except Exception as fallback_error:
                        st.warning(f"⚠️ {fallback_method} also failed: {str(fallback_error)}")
        
        if embeddings is None:
            raise ValueError("All embedding methods failed. Please check your data and try again.")
        
        # Validate embeddings
        if len(embeddings) != len(keywords_list):
            raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(keywords_list)}")
        
        # Reduce dimensions if needed (for performance)
        if embeddings.shape[1] > 100:
            try:
                embeddings = reduce_embedding_dimensions(embeddings, target_dim=100)
            except Exception as e:
                st.warning(f"⚠️ Dimension reduction failed: {str(e)}. Using original embeddings.")
        
        # Step 4: Clustering
        progress_tracker.update("Performing semantic clustering...")
        
        try:
            cluster_results = cluster_keywords(
                keywords_list,
                embeddings,
                n_clusters=config['num_clusters'],
                method=config['clustering_method'],
                min_cluster_size=config['min_cluster_size']
            )
        except Exception as e:
            st.warning(f"⚠️ Advanced clustering failed: {str(e)}. Trying basic K-means...")
            # Fallback to basic clustering
            from sklearn.cluster import KMeans
            n_clusters = min(config['num_clusters'] or 8, len(keywords_list) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            cluster_results = {
                'labels': labels,
                'model': kmeans,
                'representatives': find_representative_keywords(embeddings, keywords_list, labels),
                'coherence_scores': calculate_cluster_coherence(embeddings, labels),
                'cluster_sizes': {label: np.sum(labels == label) for label in np.unique(labels)}
            }
        
        # Step 5: Find representatives
        progress_tracker.update("Identifying representative keywords...")
        
        representatives = cluster_results['representatives']
        coherence_scores = cluster_results['coherence_scores']
        
        # Validate cluster results
        if len(representatives) == 0:
            raise ValueError("No valid clusters were created")
        
        # Step 6: Generate cluster names
        progress_tracker.update("Generating cluster names...")
        
        cluster_names = {}
        try:
            if client and config['enable_quality_analysis']:
                cluster_names = generate_cluster_names_openai(
                    representatives, 
                    client, 
                    model=config['ai_model']
                )
            else:
                raise ValueError("Using fallback naming")
        except Exception as e:
            st.warning(f"⚠️ AI cluster naming failed: {str(e)}. Using fallback names.")
            cluster_names = create_fallback_cluster_names(representatives)
        
        # Step 7: Search intent analysis (if enabled)
        intent_results = None
        if config['enable_intent_analysis']:
            progress_tracker.update("Analyzing search intent...")
            try:
                intent_results, intent_distribution = analyze_search_intent_bulk(keywords_list)
                st.info(f"✅ Search intent analysis completed. Primary intent: {max(intent_distribution, key=intent_distribution.get)}")
            except Exception as e:
                st.warning(f"⚠️ Search intent analysis failed: {str(e)}")
                intent_results = ["Unknown"] * len(keywords_list)
        
        # Step 8: Quality analysis (if enabled and API available)
        quality_analysis = None
        if config['enable_quality_analysis'] and client:
            progress_tracker.update("Performing AI quality analysis...")
            try:
                quality_analysis = analyze_cluster_quality_ai(
                    representatives, 
                    coherence_scores, 
                    client, 
                    model=config['ai_model']
                )
                st.info("✅ AI quality analysis completed")
            except Exception as e:
                st.warning(f"⚠️ AI quality analysis failed: {str(e)}")
                quality_analysis = create_basic_quality_analysis(representatives, coherence_scores)
        
        # Step 9: Create final DataFrame
        progress_tracker.update("Creating final results...")
        
        try:
            results_df = create_results_dataframe(
                keywords_list,
                cluster_results,
                cluster_names,
                coherence_scores,
                intent_results,
                quality_analysis
            )
        except Exception as e:
            st.error(f"❌ Failed to create results DataFrame: {str(e)}")
            raise e
        
        # Merge with original data if it has additional columns
        if len(original_df.columns) > 1:
            try:
                results_df = merge_original_data(results_df, original_df)
            except Exception as e:
                st.warning(f"⚠️ Could not merge original data: {str(e)}")
        
        # Add search volume analysis if available
        if 'search_volume' in results_df.columns:
            try:
                results_df = add_search_volume_data(results_df)
            except Exception as e:
                st.warning(f"⚠️ Search volume analysis failed: {str(e)}")
        
        # Validate final results
        try:
            is_valid, validated_df = validate_results_dataframe(results_df)
            if not is_valid:
                st.warning("⚠️ Results validation found some issues, but processing continued")
            results_df = validated_df
        except Exception as e:
            st.warning(f"⚠️ Results validation failed: {str(e)}")
        
        progress_tracker.complete("Clustering completed successfully!")
        
        # Store processing metadata
        processing_metadata = {
            'total_keywords': len(keywords_list),
            'total_clusters': len(representatives),
            'embedding_method': embedding_method_used,
            'clustering_method': config['clustering_method'],
            'has_search_volume': 'search_volume' in results_df.columns,
            'has_intent_analysis': intent_results is not None,
            'has_quality_analysis': quality_analysis is not None,
            'avg_coherence': np.mean(list(coherence_scores.values())) if coherence_scores else 0.5
        }
        
        # Memory cleanup
        clean_memory()
        
        return results_df, processing_metadata
        
    except Exception as e:
        # Clean up partial results and re-raise with context
        clean_memory()
        log_error(e, "clustering_pipeline", {
            "config": config,
            "num_keywords": len(df) if df is not None else 0
        })
        raise e

def handle_file_upload_and_validation(uploaded_file, csv_format):
    """Handle file upload and validation with comprehensive error checking"""
    try:
        if uploaded_file is None:
            return None, "No file uploaded", None
        
        # Validate file
        if uploaded_file.size == 0:
            return None, "Uploaded file is empty", None
        
        max_file_size = 50 * 1024 * 1024  # 50MB limit
        if uploaded_file.size > max_file_size:
            return None, f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Maximum size is 50MB.", None
        
        # Show file info
        file_size = uploaded_file.size / (1024 * 1024)  # MB
        st.info(f"📁 File: {uploaded_file.name} ({file_size:.2f} MB)")
        
        # Load CSV with error handling
        try:
            df = load_csv_file(uploaded_file, csv_format)
        except UnicodeDecodeError:
            return None, "File encoding error. Please save your CSV with UTF-8 encoding.", None
        except pd.errors.EmptyDataError:
            return None, "CSV file is empty or contains no valid data.", None
        except pd.errors.ParserError as e:
            return None, f"CSV parsing error: {str(e)}. Please check your file format.", None
        except Exception as e:
            return None, f"Failed to load CSV: {str(e)}", None
        
        if df is None:
            return None, "Failed to load CSV file", None
        
        # Validate data content
        if len(df) == 0:
            return None, "CSV file contains no data rows", None
        
        if 'keyword' not in df.columns:
            available_cols = list(df.columns)
            return None, f"No 'keyword' column found. Available columns: {available_cols}", None
        
        # Check for valid keywords
        valid_keywords = df['keyword'].dropna().str.strip()
        valid_keywords = valid_keywords[valid_keywords != '']
        
        if len(valid_keywords) == 0:
            return None, "No valid keywords found in the file", None
        
        if len(valid_keywords) < 2:
            return None, "Need at least 2 valid keywords for clustering", None
        
        # Calculate statistics for preview
        total_keywords = len(df)
        unique_keywords = df['keyword'].nunique()
        duplicate_rate = (1 - unique_keywords / total_keywords) * 100 if total_keywords > 0 else 0
        
        # Show preview
        with st.expander("👀 Data Preview", expanded=False):
            # Show first 10 rows
            st.subheader("First 10 Rows")
            preview_df = df.head(10).copy()
            
            # Format columns for display
            if 'search_volume' in preview_df.columns:
                preview_df['search_volume'] = pd.to_numeric(preview_df['search_volume'], errors='coerce')
            
            st.dataframe(preview_df, use_container_width=True)
            
            # Show statistics
            st.subheader("Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Keywords", format_number(total_keywords))
            with col2:
                st.metric("Unique Keywords", format_number(unique_keywords))
            with col3:
                st.metric("Duplicate Rate", f"{duplicate_rate:.1f}%")
            with col4:
                if 'search_volume' in df.columns:
                    total_volume = pd.to_numeric(df['search_volume'], errors='coerce').sum()
                    st.metric("Total Search Volume", format_number(total_volume))
                else:
                    st.metric("Columns", len(df.columns))
            
            # Show column information
            st.subheader("Column Information")
            col_info = []
            for col in df.columns:
                non_null = df[col].count()
                data_type = str(df[col].dtype)
                col_info.append({
                    "Column": col,
                    "Non-null Count": non_null,
                    "Data Type": data_type,
                    "Sample Values": ", ".join(str(x) for x in df[col].dropna().head(3).tolist())
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
        
        # Warnings and recommendations
        warnings = []
        if duplicate_rate > 10:
            warnings.append(f"High duplicate rate ({duplicate_rate:.1f}%). Consider removing duplicates.")
        
        if total_keywords > MAX_KEYWORDS:
            warnings.append(f"Large dataset ({total_keywords:,} keywords). Processing will be limited to {MAX_KEYWORDS:,} for performance.")
        
        if len(df.columns) == 1:
            warnings.append("Only keyword column detected. Additional data (search volume, etc.) can provide richer analysis.")
        
        if warnings:
            with st.expander("⚠️ Recommendations", expanded=False):
                for warning in warnings:
                    st.warning(warning)
        
        # Create processing summary
        processing_summary = {
            'total_keywords': total_keywords,
            'unique_keywords': unique_keywords,
            'duplicate_rate': duplicate_rate,
            'has_search_volume': 'search_volume' in df.columns,
            'additional_columns': [col for col in df.columns if col != 'keyword'],
            'file_size_mb': file_size
        }
        
        return df, "Success", processing_summary
        
    except Exception as e:
        log_error(e, "file_upload_validation")
        return None, f"Unexpected error during file validation: {str(e)}", None

def initialize_session_state_enhanced():
    """Enhanced session state initialization"""
    # Core state variables
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    if 'processing_metadata' not in st.session_state:
        st.session_state.processing_metadata = None
    
    if 'error_state' not in st.session_state:
        st.session_state.error_state = None
    
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    # Memory monitoring
    if 'memory_monitor' not in st.session_state:
        st.session_state.memory_monitor = {
            'last_check': time.time(),
            'peak_memory': 0,
            'warnings_shown': 0
        }
    
    # UI state
    if 'show_advanced_config' not in st.session_state:
        st.session_state.show_advanced_config = False

def reset_processing_state():
    """Reset processing-related session state"""
    st.session_state.processing_started = False
    st.session_state.processing_complete = False
    st.session_state.results_df = None
    st.session_state.processing_metadata = None
    st.session_state.error_state = None

def check_file_change(uploaded_file):
    """Check if a different file was uploaded"""
    if uploaded_file is None:
        return False
    
    current_file_info = (uploaded_file.name, uploaded_file.size)
    
    if st.session_state.last_uploaded_file != current_file_info:
        st.session_state.last_uploaded_file = current_file_info
        reset_processing_state()
        return True
    
    return False

def main_application():
    """Main application function with improved state management"""
    try:
        # Initialize session state
        initialize_session_state_enhanced()
        
        # Monitor resources
        monitor_resources()
        
        # Create sidebar configuration
        config = create_sidebar_configuration()
        
        # Check for file changes
        if config['uploaded_file']:
            check_file_change(config['uploaded_file'])
        else:
            if st.session_state.last_uploaded_file is not None:
                # File was removed
                reset_processing_state()
                st.session_state.last_uploaded_file = None
        
        # Show help section
        show_help_section()
        
        # Create sample CSV download
        create_sample_csv_download()
        
        # Main application flow
        if config['uploaded_file'] is None:
            # Welcome screen
            show_welcome_screen()
            
        elif st.session_state.error_state:
            # Error screen
            show_error_screen(
                st.session_state.error_state['message'],
                st.session_state.error_state.get('suggestions', [])
            )
            
            # Clear error button
            if st.button("🔄 Clear Error and Try Again"):
                st.session_state.error_state = None
                reset_processing_state()
                st.rerun()
                
        elif st.session_state.processing_complete and st.session_state.results_df is not None:
            # Results screen
            show_results_screen(st.session_state.results_df, config)
            
        else:
            # File uploaded, handle processing flow
            
            # Handle file upload and validation
            df, upload_message, processing_summary = handle_file_upload_and_validation(
                config['uploaded_file'], 
                config['csv_format']
            )
            
            if df is None:
                # File validation failed
                error_suggestions = [
                    "Check your CSV file format and encoding (should be UTF-8)",
                    "Ensure the file contains a 'keyword' column (if using headers)",
                    "Try with a smaller file first (< 50MB)",
                    "Make sure the file is properly saved as CSV",
                    "Remove any empty rows or columns",
                    "Check for special characters that might cause parsing errors"
                ]
                
                st.session_state.error_state = {
                    'message': upload_message,
                    'suggestions': error_suggestions
                }
                st.rerun()
                return
            
            # File is valid, show processing options
            if not st.session_state.processing_started:
                # Show processing preview screen
                show_processing_screen(config)
                
                # Show processing summary if available
                if processing_summary:
                    with st.expander("📊 Processing Summary", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Keywords to Process", format_number(processing_summary['total_keywords']))
                            if processing_summary['has_search_volume']:
                                st.success("✅ Search volume data detected")
                        
                        with col2:
                            st.metric("Unique Keywords", format_number(processing_summary['unique_keywords']))
                            if processing_summary['duplicate_rate'] > 0:
                                st.info(f"📊 {processing_summary['duplicate_rate']:.1f}% duplicates")
                        
                        with col3:
                            st.metric("File Size", f"{processing_summary['file_size_mb']:.2f} MB")
                            if processing_summary['additional_columns']:
                                st.info(f"📝 {len(processing_summary['additional_columns'])} extra columns")
                
                # Cost estimation
                if config['openai_api_key']:
                    with st.expander("💰 Cost Estimate", expanded=False):
                        cost_info = calculate_estimated_cost(
                            len(df), 
                            config['ai_model'], 
                            config['num_clusters'] or 10
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Embedding Cost", f"${cost_info['embedding_cost']:.4f}")
                        with col2:
                            st.metric("Analysis Cost", f"${cost_info['naming_cost']:.4f}")
                        with col3:
                            st.metric("Total Estimated", f"${cost_info['total_cost']:.4f}")
                        
                        if cost_info['total_cost'] > 5.0:
                            st.error("⚠️ High cost estimated! Consider reducing keywords or disabling AI features.")
                        elif cost_info['total_cost'] > 1.0:
                            st.warning("⚠️ Moderate cost estimated. Review settings if needed.")
                
                # Start button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    start_button = st.button(
                        "🚀 Start Clustering Analysis", 
                        use_container_width=True, 
                        type="primary",
                        help="Begin the semantic clustering process"
                    )
                    
                    if start_button:
                        st.session_state.processing_started = True
                        st.rerun()
            
            else:
                # Processing is running
                try:
                    st.markdown("""
                    <div class='main-header'>🔄 Processing Your Keywords</div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("🔄 Analyzing your keywords... This may take a few minutes."):
                        results_df, metadata = run_clustering_pipeline(df, config)
                        
                        # Store results
                        st.session_state.results_df = results_df
                        st.session_state.processing_metadata = metadata
                        st.session_state.processing_complete = True
                        st.session_state.processing_started = False
                        
                        st.success("✅ Clustering completed successfully!")
                        
                        # Show quick summary
                        if metadata:
                            st.balloons()
                            st.info(f"🎉 Created {metadata['total_clusters']} clusters from {metadata['total_keywords']} keywords using {metadata['embedding_method']} embeddings!")
                        
                        time.sleep(2)  # Brief pause to show success
                        st.rerun()
                        
                except Exception as e:
                    # Processing failed
                    st.session_state.processing_started = False
                    
                    error_msg = str(e)
                    
                    # Categorize errors and provide specific suggestions
                    suggestions = ["Try again with the same settings", "Refresh the page and restart"]
                    
                    if "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
                        suggestions = [
                            "Reduce the number of keywords in your dataset",
                            "Try using TF-IDF embeddings instead of OpenAI",
                            "Close other browser tabs to free memory",
                            "Use a smaller batch size in advanced settings"
                        ]
                    elif "api" in error_msg.lower() or "openai" in error_msg.lower() or "timeout" in error_msg.lower():
                        suggestions = [
                            "Check your OpenAI API key is valid and has credits",
                            "Try disabling OpenAI features and use free alternatives",
                            "Check your internet connection",
                            "Wait a moment and try again (API rate limits)"
                        ]
                    elif "embedding" in error_msg.lower():
                        suggestions = [
                            "Try TF-IDF embeddings (always available)",
                            "Check if SentenceTransformers is properly installed",
                            "Reduce the number of keywords",
                            "Clean your keyword data for special characters"
                        ]
                    elif "cluster" in error_msg.lower():
                        suggestions = [
                            "Try reducing the number of target clusters",
                            "Increase minimum cluster size",
                            "Use K-means clustering method",
                            "Check if you have enough keywords for clustering"
                        ]
                    
                    st.session_state.error_state = {
                        'message': error_msg,
                        'suggestions': suggestions
                    }
                    
                    log_error(e, "main_processing", {"config": config})
                    st.rerun()
    
    except Exception as e:
        # Critical application error
        log_error(e, "main_application")
        
        st.session_state.error_state = {
            'message': f"Critical application error: {str(e)}",
            'suggestions': [
                "Refresh the page to restart the application",
                "Check your browser console for detailed error messages",
                "Try with a different browser or incognito mode",
                "Clear your browser cache and cookies",
                "Ensure you have a stable internet connection"
            ]
        }
        st.rerun()

def setup_error_handling():
    """Setup global error handling and configurations"""
    try:
        # Configure pandas to avoid warnings
        import pandas as pd
        pd.options.mode.chained_assignment = None
        pd.options.display.max_colwidth = 100
        
        # Configure numpy to avoid warnings  
        import numpy as np
        np.seterr(divide='ignore', invalid='ignore')
        
        # Configure matplotlib if available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except ImportError:
            pass
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        
        return True
    except Exception as e:
        logger.error(f"Error setting up error handling: {str(e)}")
        return False

def check_system_requirements():
    """Check if system meets minimum requirements"""
    try:
        requirements_met = True
        issues = []
        warnings = []
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            requirements_met = False
            issues.append("Python 3.8+ required")
        elif sys.version_info < (3, 9):
            warnings.append("Python 3.9+ recommended for best performance")
        
        # Check required packages
        required_packages = {
            'pandas': 'Data manipulation',
            'numpy': 'Numerical computing', 
            'scikit-learn': 'Machine learning',
            'plotly': 'Interactive visualizations',
            'streamlit': 'Web interface'
        }
        
        for package, description in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                requirements_met = False
                issues.append(f"Missing required package: {package} ({description})")
        
        # Check optional packages
        optional_packages = {
            'openai': 'AI-powered analysis',
            'sentence_transformers': 'Advanced embeddings',
            'spacy': 'Natural language processing',
            'textblob': 'Text analysis',
            'psutil': 'System monitoring'
        }
        
        missing_optional = []
        for package, description in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(f"{package} ({description})")
        
        if missing_optional:
            warnings.append(f"Optional packages not available: {', '.join(missing_optional[:3])}")
        
        # Check memory (if available)
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                if memory.available < 512 * 1024 * 1024:  # Less than 512MB available
                    issues.append("Very low available memory (< 512MB)")
                elif memory.available < 1024 * 1024 * 1024:  # Less than 1GB available
                    warnings.append("Low available memory (< 1GB)")
            except Exception:
                pass
        
        return requirements_met, issues, warnings
        
    except Exception as e:
        logger.error(f"Error checking system requirements: {str(e)}")
        return False, [f"Error checking requirements: {str(e)}"], []

def display_startup_info():
    """Display startup information and system status"""
    try:
        # Check requirements
        requirements_met, issues, warnings = check_system_requirements()
        
        if not requirements_met:
            st.error("❌ System Requirements Not Met")
            st.write("**Critical Issues:**")
            for issue in issues:
                st.write(f"- {issue}")
            st.write("**Please install the missing packages and refresh the page.**")
            st.stop()
        
        # Show warnings in sidebar if any
        if warnings:
            with st.sidebar:
                with st.expander("⚠️ System Warnings", expanded=False):
                    for warning in warnings:
                        st.warning(warning)
        
        return True
    
    except Exception as e:
        logger.error(f"Error displaying startup info: {str(e)}")
        st.error(f"Error checking system status: {str(e)}")
        return False

def validate_configuration(config):
    """Validate configuration parameters"""
    try:
        errors = []
        warnings = []
        
        # Validate clustering parameters
        if config['num_clusters'] is not None:
            if config['num_clusters'] < 2:
                errors.append("Number of clusters must be at least 2")
            elif config['num_clusters'] > 100:
                warnings.append("Very high number of clusters may not be useful")
        
        if config['min_cluster_size'] < 1:
            errors.append("Minimum cluster size must be at least 1")
        elif config['min_cluster_size'] > 20:
            warnings.append("High minimum cluster size may result in too few clusters")
        
        if config['max_keywords'] < 10:
            errors.append("Must process at least 10 keywords")
        elif config['max_keywords'] > MAX_KEYWORDS:
            warnings.append(f"Processing limited to {MAX_KEYWORDS:,} keywords")
            config['max_keywords'] = MAX_KEYWORDS
        
        # Validate OpenAI settings
        if config['openai_api_key']:
            if len(config['openai_api_key']) < 20:
                warnings.append("OpenAI API key seems too short")
            elif not config['openai_api_key'].startswith(('sk-', 'sk-proj-')):
                warnings.append("OpenAI API key should start with 'sk-'")
        
        # Show validation results
        if errors:
            for error in errors:
                st.error(f"❌ Configuration Error: {error}")
            return False
        
        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ Configuration Warning: {warning}")
        
        return True
        
    except Exception as e:
        log_error(e, "configuration_validation")
        st.error(f"Configuration validation error: {str(e)}")
        return False

def cleanup_session_on_exit():
    """Cleanup function to run when session ends"""
    try:
        # Clear large objects from session state
        large_objects = ['results_df', 'processing_metadata']
        for obj in large_objects:
            if obj in st.session_state:
                del st.session_state[obj]
        
        # Force garbage collection
        clean_memory()
        
    except Exception as e:
        logger.warning(f"Error during session cleanup: {str(e)}")

# Main execution block
if __name__ == "__main__":
    try:
        # Setup error handling and configuration
        setup_success = setup_error_handling()
        if not setup_success:
            st.error("❌ Failed to setup error handling")
            st.stop()
        
        # Display startup information and check requirements
        startup_success = display_startup_info()
        if not startup_success:
            st.error("❌ System startup checks failed")
            st.stop()
        
        # Run main application
        main_application()
        
    except KeyboardInterrupt:
        # Handle graceful shutdown
        st.info("👋 Application interrupted by user")
        cleanup_session_on_exit()
        
    except Exception as e:
        # Final fallback error handling
        st.error("🚨 Critical Application Error")
        st.write(f"**Error:** {str(e)}")
        st.write("**Solution:** Please refresh the page and try again.")
        
        # Log the error with full traceback
        logger.critical(f"Critical application error: {str(e)}", exc_info=True)
        
        # Show detailed error info in expandable section
        with st.expander("🔧 Technical Details", expanded=False):
            st.code(f"Error Type: {type(e).__name__}\nError Message: {str(e)}")
            st.write("**Troubleshooting Steps:**")
            st.write("1. Refresh the page (Ctrl+F5 or Cmd+Shift+R)")
            st.write("2. Clear browser cache and cookies")
            st.write("3. Try in incognito/private browsing mode")
            st.write("4. Check browser console for additional error details")
            st.write("5. Ensure stable internet connection")
        
        # Show restart button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Restart Application", use_container_width=True):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Cleanup
        cleanup_session_on_exit()

# Footer with version and library information
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em; margin-top: 1rem;">
        🔍 <strong>Semantic Keyword Clustering Tool</strong><br>
        Built with Streamlit • scikit-learn • OpenAI • SentenceTransformers<br>
        <em>Advanced NLP for SEO Strategy</em>
    </div>
    """, unsafe_allow_html=True)

# Display library status in footer (collapsed by default)
with st.expander("📚 Technical Information", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Core Libraries:**")
        st.write(f"- Streamlit: ✅")
        st.write(f"- Pandas: ✅")
        st.write(f"- NumPy: ✅")
        st.write(f"- Scikit-learn: ✅")
        st.write(f"- Plotly: ✅")
    
    with col2:
        st.write("**Optional Libraries:**")
        st.write(f"- OpenAI: {'✅' if OPENAI_AVAILABLE else '❌'}")
        st.write(f"- SentenceTransformers: {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'}")
        st.write(f"- spaCy: {'✅' if SPACY_AVAILABLE else '❌'}")
        st.write(f"- TextBlob: {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
        st.write(f"- psutil: {'✅' if PSUTIL_AVAILABLE else '❌'}")
    
    # Show memory usage if available
    if PSUTIL_AVAILABLE:
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.write(f"**System Memory:** {memory.available / (1024**3):.1f}GB available of {memory.total / (1024**3):.1f}GB total")
        except Exception:
            pass

# Cache management in footer
if st.button("🗑️ Clear Cache", help="Clear all cached data to free memory"):
    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        
        # Clear session state except UI preferences
        keys_to_keep = ['show_advanced_config']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        # Force garbage collection
        clean_memory()
        
        st.success("✅ Cache cleared successfully!")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error clearing cache: {str(e)}")

# Version and build information (hidden by default)
if st.sidebar.button("ℹ️ Version Info", help="Show version and build information"):
    with st.sidebar:
        st.write("**Application Version:** 1.0.0")
        st.write("**Build Date:** 2025-01-01")
        st.write("**Python Version:** " + ".".join(map(str, sys.version_info[:3])))
        
        # Show package versions if available
        try:
            import pandas as pd
            import numpy as np
            import sklearn
            st.write(f"**Pandas:** {pd.__version__}")
            st.write(f"**NumPy:** {np.__version__}")
            st.write(f"**Scikit-learn:** {sklearn.__version__}")
        except Exception:
            pass
