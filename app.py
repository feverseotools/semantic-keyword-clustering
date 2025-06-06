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

# Optional libraries detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

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
    if 'process_complete' not in st.session_state:
        st.session_state.process_complete = False
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None
    if 'cluster_evaluation' not in st.session_state:
        st.session_state.cluster_evaluation = {}
    if 'memory_monitor' not in st.session_state:
        st.session_state.memory_monitor = {
            'last_check': time.time(),
            'peak_memory': 0,
            'warnings_shown': 0
        }

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
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
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
        client.models.list()
        logger.info("OpenAI client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {str(e)}")
        st.error(f"OpenAI API error: {str(e)}")
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
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
    
    # Check for malicious content in keyword column
    if 'keyword' in df.columns:
        suspicious_patterns = [r'<script', r'javascript:', r'\.\./', r'file://']
        for pattern in suspicious_patterns:
            if df['keyword'].astype(str).str.contains(pattern, case=False, regex=True).any():
                return False, f"Suspicious content detected"
    
    return True, "Validation passed"

def clean_memory():
    """Force garbage collection and memory cleanup"""
    try:
        gc.collect()
        if hasattr(st, 'cache_data'):
            # Clear old cache entries if memory is high
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > MAX_MEMORY_WARNING:
                    st.cache_data.clear()
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
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read content
        content = uploaded_file.read()
        
        # Decode if bytes
        if isinstance(content, bytes):
            content = content.decode(encoding)
        
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
    if not download_nltk_data():
        logger.warning("NLTK data not available, using basic preprocessing")
        return [kw.lower().strip() for kw in keywords_list]
    
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        # Get stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        lemmatizer = WordNetLemmatizer()
        processed_keywords = []
        
        for keyword in keywords_list:
            if not isinstance(keyword, str) or not keyword.strip():
                processed_keywords.append("")
                continue
            
            # Basic cleaning
            keyword = keyword.lower().strip()
            
            # Tokenization
            try:
                tokens = word_tokenize(keyword)
            except:
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
                    except:
                        processed_tokens.append(token)
            
            # Join tokens
            processed_keyword = " ".join(processed_tokens)
            processed_keywords.append(processed_keyword if processed_keyword else keyword)
        
        return processed_keywords
        
    except Exception as e:
        logger.warning(f"Basic preprocessing failed: {str(e)}")
        return [kw.lower().strip() for kw in keywords_list]

def preprocess_keywords_advanced(keywords_list, spacy_nlp, language="English"):
    """Advanced preprocessing using spaCy"""
    if not spacy_nlp:
        return preprocess_keywords_basic(keywords_list, language)
    
    try:
        processed_keywords = []
        
        for keyword in keywords_list:
            if not isinstance(keyword, str) or not keyword.strip():
                processed_keywords.append("")
                continue
            
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
            
            # Combine all features
            all_features = tokens + entities + noun_phrases
            
            # Create processed keyword
            if all_features:
                processed_keyword = " ".join(all_features[:10])  # Limit features
            else:
                processed_keyword = keyword.lower()
            
            processed_keywords.append(processed_keyword)
        
        return processed_keywords
        
    except Exception as e:
        logger.warning(f"Advanced preprocessing failed: {str(e)}")
        return preprocess_keywords_basic(keywords_list, language)

def preprocess_keywords_textblob(keywords_list):
    """Preprocessing using TextBlob"""
    if not TEXTBLOB_AVAILABLE:
        return preprocess_keywords_basic(keywords_list)
    
    try:
        from textblob import TextBlob
        processed_keywords = []
        
        for keyword in keywords_list:
            if not isinstance(keyword, str) or not keyword.strip():
                processed_keywords.append("")
                continue
            
            # Create TextBlob
            blob = TextBlob(keyword.lower())
            
            # Get noun phrases
            noun_phrases = list(blob.noun_phrases)
            
            # Get words (filtered)
            words = [word for word in blob.words 
                    if len(word) > 1 and word.isalpha()]
            
            # Combine features
            all_features = words + noun_phrases
            
            if all_features:
                processed_keyword = " ".join(str(f) for f in all_features[:8])
            else:
                processed_keyword = keyword.lower()
            
            processed_keywords.append(processed_keyword)
        
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
    if not isinstance(keyword, str):
        return {}
    
    keyword_lower = keyword.lower()
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
    if features is None:
        features = extract_keyword_features(keyword)
    
    scores = {
        "Informational": 0,
        "Commercial": 0,
        "Transactional": 0,
        "Navigational": 0
    }
    
    keyword_lower = keyword.lower() if isinstance(keyword, str) else ""
    
    # Score based on patterns and keywords
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        # Check keywords
        for kw in patterns["keywords"]:
            if kw in keyword_lower:
                scores[intent_type] += patterns["weight"]
        
        # Check regex patterns
        for pattern in patterns["patterns"]:
            if re.search(pattern, keyword_lower):
                scores[intent_type] += patterns["weight"] * 1.5
    
    # Apply feature-based scoring
    if features["has_question_word"]:
        scores["Informational"] += 2
    if features["has_commercial_intent"]:
        scores["Commercial"] += 2
    if features["has_transactional_intent"]:
        scores["Transactional"] += 2
    if features["has_navigational_intent"]:
        scores["Navigational"] += 2
    if features["has_local_intent"]:
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
            
            # Memory cleanup every batch
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
                # Clean batch - remove empty strings
                clean_batch = [kw for kw in batch if isinstance(kw, str) and kw.strip()]
                
                if not clean_batch:
                    # Add zero embeddings for empty batch
                    all_embeddings.extend([np.zeros(1536)] * len(batch))
                    continue
                
                # Make API call
                response = client.embeddings.create(
                    input=clean_batch,
                    model=model
                )
                
                # Extract embeddings
                batch_embeddings = []
                for j, embedding_obj in enumerate(response.data):
                    embedding = np.array(embedding_obj.embedding)
                    batch_embeddings.append(embedding)
                
                # Handle size mismatch (for empty strings in original batch)
                if len(batch_embeddings) < len(batch):
                    # Add zero embeddings for empty strings
                    zero_embedding = np.zeros(len(batch_embeddings[0]) if batch_embeddings else 1536)
                    for _ in range(len(batch) - len(batch_embeddings)):
                        batch_embeddings.append(zero_embedding)
                
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                progress.progress(min(1.0, (i + batch_size) / len(keywords_list)))
                
                # Rate limiting - small delay between batches
                if batch_num < total_batches:
                    time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"OpenAI embedding error for batch {batch_num}: {str(e)}")
                # Add zero embeddings for failed batch
                zero_embedding = np.zeros(1536)
                all_embeddings.extend([zero_embedding] * len(batch))
        
        progress.progress(1.0)
        status.text("✅ OpenAI embeddings generated successfully")
        
        if len(all_embeddings) != len(keywords_list):
            logger.warning(f"Embedding count mismatch: {len(all_embeddings)} vs {len(keywords_list)}")
            # Pad or trim to match
            while len(all_embeddings) < len(keywords_list):
                all_embeddings.append(np.zeros(1536))
            all_embeddings = all_embeddings[:len(keywords_list)]
        
        return np.array(all_embeddings)
        
    except Exception as e:
        log_error(e, "openai_embeddings", {"num_keywords": len(keywords_list)})
        st.error(f"OpenAI embeddings failed: {str(e)}")
        return None

@st.cache_data(ttl=3600, max_entries=3)
def generate_sentence_transformer_embeddings(keywords_list, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings using SentenceTransformers"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not keywords_list:
        return None
    
    try:
        from sentence_transformers import SentenceTransformer
        
        st.info(f"🧠 Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
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
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            
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
    try:
        if processed_keywords is None:
            processed_keywords = preprocess_keywords(keywords_list)
        
        # Clean processed keywords
        clean_processed = []
        for kw in processed_keywords:
            if isinstance(kw, str) and kw.strip():
                clean_processed.append(kw.strip())
            else:
                clean_processed.append("empty")
        
        st.info("🔄 Generating TF-IDF embeddings...")
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(clean_processed)
        embeddings = tfidf_matrix.toarray()
        
        st.success("✅ TF-IDF embeddings generated successfully")
        
        return embeddings
        
    except Exception as e:
        log_error(e, "tfidf_embeddings", {"num_keywords": len(keywords_list)})
        st.error(f"TF-IDF embeddings failed: {str(e)}")
        return None

def generate_embeddings(keywords_list, client=None, method="auto", **kwargs):
    """Main embedding generation function with multiple methods"""
    try:
        if not keywords_list:
            return None
        
        # Monitor resources
        monitor_resources()
        
        st.subheader("🧠 Generating Semantic Embeddings")
        
        # Limit keywords for memory efficiency
        if len(keywords_list) > MAX_KEYWORDS:
            st.warning(f"⚠️ Limiting to {MAX_KEYWORDS:,} keywords for memory efficiency")
            keywords_list = keywords_list[:MAX_KEYWORDS]
        
        embeddings = None
        
        if method == "auto":
            # Try methods in order of preference
            if client and OPENAI_AVAILABLE:
                st.info("🚀 Using OpenAI embeddings (highest quality)")
                embeddings = generate_openai_embeddings(keywords_list, client)
            
            if embeddings is None and SENTENCE_TRANSFORMERS_AVAILABLE:
                st.info("🧠 Using SentenceTransformers (good quality, free)")
                embeddings = generate_sentence_transformer_embeddings(keywords_list)
            
            if embeddings is None:
                st.info("📊 Using TF-IDF embeddings (basic quality, always available)")
                processed_keywords = preprocess_keywords(keywords_list)
                embeddings = generate_tfidf_embeddings(keywords_list, processed_keywords)
        
        elif method == "openai" and client:
            embeddings = generate_openai_embeddings(keywords_list, client)
        
        elif method == "sentence_transformers":
            embeddings = generate_sentence_transformer_embeddings(keywords_list)
        
        elif method == "tfidf":
            processed_keywords = preprocess_keywords(keywords_list)
            embeddings = generate_tfidf_embeddings(keywords_list, processed_keywords)
        
        # Validate embeddings
        if embeddings is None:
            raise ValueError("All embedding methods failed")
        
        if len(embeddings) != len(keywords_list):
            raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(keywords_list)}")
        
        # Normalize embeddings
        embeddings = normalize(embeddings, norm='l2')
        
        st.success(f"✅ Generated embeddings: {embeddings.shape}")
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Memory cleanup
        clean_memory()
        
        return embeddings
        
    except Exception as e:
        log_error(e, "embedding_generation", {
            "method": method,
            "num_keywords": len(keywords_list) if keywords_list else 0
        })
        st.error(f"Embedding generation failed: {str(e)}")
        return None

def reduce_embedding_dimensions(embeddings, target_dim=100, variance_threshold=0.95):
    """Reduce embedding dimensions using PCA"""
    if embeddings is None or embeddings.shape[1] <= target_dim:
        return embeddings
    
    try:
        st.info(f"🔄 Reducing dimensions from {embeddings.shape[1]} to ~{target_dim}")
        
        # Use Incremental PCA for large datasets
        if len(embeddings) > 10000:
            pca = IncrementalPCA(n_components=target_dim)
            
            # Fit in batches
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                pca.partial_fit(batch)
            
            # Transform all data
            reduced_embeddings = pca.transform(embeddings)
        else:
            # Standard PCA for smaller datasets
            pca = PCA()
            pca.fit(embeddings)
            
            # Find number of components for target variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            n_components = min(n_components, target_dim)
            
            # Apply PCA with optimal components
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings)
        
        st.success(f"✅ Dimensions reduced to {reduced_embeddings.shape[1]}")
        return reduced_embeddings
        
    except Exception as e:
        log_error(e, "dimension_reduction")
        st.warning(f"⚠️ Dimension reduction failed: {str(e)}. Using original embeddings.")
        return embeddings

def propagate_embeddings_to_similar(embeddings, keywords_list, max_propagation=1000):
    """Propagate embeddings to similar keywords using cosine similarity"""
    if embeddings is None or len(embeddings) >= len(keywords_list):
        return embeddings
    
    try:
        st.info("🔄 Propagating embeddings to remaining keywords...")
        
        # Keywords with embeddings
        embedded_keywords = keywords_list[:len(embeddings)]
        remaining_keywords = keywords_list[len(embeddings):]
        
        if len(remaining_keywords) > max_propagation:
            st.warning(f"⚠️ Limiting propagation to {max_propagation} keywords")
            remaining_keywords = remaining_keywords[:max_propagation]
        
        # Find most similar embedded keyword for each remaining keyword
        propagated_embeddings = []
        
        for remaining_kw in remaining_keywords:
            best_similarity = -1
            best_embedding = embeddings[0]  # fallback
            
            for i, embedded_kw in enumerate(embedded_keywords):
                # Simple text similarity
                similarity = len(set(remaining_kw.lower().split()) & 
                               set(embedded_kw.lower().split()))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_embedding = embeddings[i]
            
            # Add small noise to avoid identical embeddings
            noise = np.random.normal(0, 0.01, best_embedding.shape)
            propagated_embedding = best_embedding + noise
            propagated_embeddings.append(propagated_embedding)
        
        # Combine original and propagated embeddings
        all_embeddings = np.vstack([embeddings, np.array(propagated_embeddings)])
        
        st.success(f"✅ Propagated embeddings to {len(propagated_embeddings)} keywords")
        return all_embeddings
        
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
        max_clusters = min(max_clusters, n_samples // 2)
        
        if max_clusters <= min_clusters:
            return min_clusters
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)
        
        st.info("🔄 Finding optimal number of clusters...")
        progress = st.progress(0)
        
        for i, k in enumerate(cluster_range):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score
                if len(set(labels)) > 1:
                    sil_score = silhouette_score(embeddings, labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
                
                progress.progress((i + 1) / len(cluster_range))
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for k={k}: {str(e)}")
                inertias.append(float('inf'))
                silhouette_scores.append(0)
        
        # Find elbow point
        if len(inertias) >= 3:
            # Calculate rate of change
            rates = []
            for i in range(1, len(inertias) - 1):
                if inertias[i-1] != float('inf') and inertias[i+1] != float('inf'):
                    rate = (inertias[i-1] - inertias[i+1]) / 2
                    rates.append(rate)
                else:
                    rates.append(0)
            
            # Find point where rate of change decreases significantly
            if rates:
                elbow_idx = np.argmax(rates)
                elbow_k = cluster_range[elbow_idx + 1]
            else:
                elbow_k = min_clusters
        else:
            elbow_k = min_clusters
        
        # Find best silhouette score
        if silhouette_scores:
            best_sil_idx = np.argmax(silhouette_scores)
            best_sil_k = cluster_range[best_sil_idx]
        else:
            best_sil_k = min_clusters
        
        # Choose optimal k (prefer silhouette score if reasonable)
        if silhouette_scores[best_sil_idx - min_clusters] > 0.3:
            optimal_k = best_sil_k
        else:
            optimal_k = elbow_k
        
        # Ensure reasonable bounds
        optimal_k = max(min_clusters, min(optimal_k, max_clusters))
        
        st.success(f"✅ Optimal clusters determined: {optimal_k}")
        return optimal_k
        
    except Exception as e:
        log_error(e, "optimal_cluster_determination")
        st.warning(f"⚠️ Could not determine optimal clusters: {str(e)}. Using default.")
        return min(8, max_clusters)

def perform_kmeans_clustering(embeddings, n_clusters, random_state=42):
    """Perform K-means clustering"""
    try:
        from sklearn.cluster import KMeans
        
        st.info(f"🔄 Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate cluster statistics
        unique_labels = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        
        st.success(f"✅ K-means clustering completed. Cluster sizes: {cluster_sizes}")
        
        return cluster_labels, kmeans
        
    except Exception as e:
        log_error(e, "kmeans_clustering")
        raise e

def perform_hierarchical_clustering(embeddings, n_clusters, method='ward'):
    """Perform hierarchical clustering"""
    try:
        from sklearn.cluster import AgglomerativeClustering
        
        st.info(f"🔄 Performing hierarchical clustering with {n_clusters} clusters...")
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=method
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Calculate cluster statistics
        unique_labels = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        
        st.success(f"✅ Hierarchical clustering completed. Cluster sizes: {cluster_sizes}")
        
        return cluster_labels, clustering
        
    except Exception as e:
        log_error(e, "hierarchical_clustering")
        raise e

def perform_advanced_clustering(embeddings, method="auto", n_clusters=None):
    """Perform advanced clustering with automatic method selection"""
    try:
        n_samples = len(embeddings)
        
        # Determine optimal clusters if not provided
        if n_clusters is None:
            n_clusters = determine_optimal_clusters(embeddings)
        
        # Ensure reasonable cluster number
        n_clusters = max(2, min(n_clusters, n_samples // 2))
        
        if method == "auto":
            # Choose method based on dataset size and dimensionality
            if n_samples > 10000:
                method = "kmeans"  # Better for large datasets
            elif embeddings.shape[1] > 100:
                method = "kmeans"  # Better for high dimensions
            else:
                method = "hierarchical"  # Better for smaller datasets
        
        # Perform clustering
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
        
        return labels, model
        
    except Exception as e:
        log_error(e, "advanced_clustering", {"method": method, "n_clusters": n_clusters})
        raise e

def refine_clusters(embeddings, initial_labels, min_cluster_size=2):
    """Refine clusters by merging small clusters and outlier detection"""
    try:
        st.info("🔄 Refining clusters...")
        
        refined_labels = initial_labels.copy()
        unique_labels = np.unique(refined_labels)
        
        # Find small clusters
        small_clusters = []
        for label in unique_labels:
            cluster_size = np.sum(refined_labels == label)
            if cluster_size < min_cluster_size:
                small_clusters.append(label)
        
        if not small_clusters:
            st.success("✅ No refinement needed")
            return refined_labels
        
        # Merge small clusters with nearest large clusters
        for small_label in small_clusters:
            small_cluster_indices = np.where(refined_labels == small_label)[0]
            small_cluster_embeddings = embeddings[small_cluster_indices]
            
            # Find the best cluster to merge with
            best_distance = float('inf')
            best_target_label = None
            
            for target_label in unique_labels:
                if target_label == small_label or target_label in small_clusters:
                    continue
                
                target_indices = np.where(refined_labels == target_label)[0]
                target_embeddings = embeddings[target_indices]
                
                # Calculate average distance
                distances = cosine_similarity(small_cluster_embeddings, target_embeddings)
                avg_distance = 1 - np.mean(distances)  # Convert similarity to distance
                
                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_target_label = target_label
            
            # Merge the small cluster
            if best_target_label is not None:
                refined_labels[refined_labels == small_label] = best_target_label
        
        # Relabel clusters to be consecutive
        unique_refined = np.unique(refined_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_refined)}
        
        final_labels = np.array([label_mapping[label] for label in refined_labels])
        
        n_original = len(unique_labels)
        n_refined = len(unique_refined)
        
        st.success(f"✅ Clusters refined: {n_original} → {n_refined}")
        
        return final_labels
        
    except Exception as e:
        log_error(e, "cluster_refinement")
        st.warning(f"⚠️ Cluster refinement failed: {str(e)}. Using original clusters.")
        return initial_labels

def find_representative_keywords(embeddings, keywords, cluster_labels, top_k=5):
    """Find representative keywords for each cluster"""
    try:
        st.info("🔄 Finding representative keywords...")
        
        unique_labels = np.unique(cluster_labels)
        representatives = {}
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_embeddings = embeddings[cluster_indices]
            cluster_keywords = [keywords[i] for i in cluster_indices]
            
            if len(cluster_embeddings) == 0:
                continue
            
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find keywords closest to centroid
            similarities = cosine_similarity([centroid], cluster_embeddings)[0]
            
            # Get top-k most representative
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            representative_keywords = [cluster_keywords[i] for i in top_indices]
            
            representatives[label] = representative_keywords
        
        st.success(f"✅ Found representatives for {len(representatives)} clusters")
        
        return representatives
        
    except Exception as e:
        log_error(e, "representative_keywords")
        # Fallback: return first few keywords of each cluster
        unique_labels = np.unique(cluster_labels)
        representatives = {}
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0][:5]
            representatives[label] = [keywords[i] for i in cluster_indices]
        
        return representatives

def calculate_cluster_coherence(embeddings, cluster_labels):
    """Calculate coherence score for each cluster"""
    try:
        unique_labels = np.unique(cluster_labels)
        coherence_scores = {}
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_embeddings = embeddings[cluster_indices]
            
            if len(cluster_embeddings) < 2:
                coherence_scores[label] = 1.0
                continue
            
            # Calculate average pairwise similarity within cluster
            similarities = cosine_similarity(cluster_embeddings)
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            
            if len(upper_triangle) > 0:
                coherence = np.mean(upper_triangle)
            else:
                coherence = 1.0
            
            coherence_scores[label] = max(0, min(1, coherence))
        
        return coherence_scores
        
    except Exception as e:
        log_error(e, "cluster_coherence")
        # Return default scores
        unique_labels = np.unique(cluster_labels)
        return {label: 0.5 for label in unique_labels}

def cluster_keywords(keywords_list, embeddings, n_clusters=None, method="auto", min_cluster_size=2):
    """Main clustering function that orchestrates the entire process"""
    try:
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings provided")
        
        if len(keywords_list) != len(embeddings):
            raise ValueError(f"Keyword count ({len(keywords_list)}) doesn't match embedding count ({len(embeddings)})")
        
        st.subheader("🔗 Performing Semantic Clustering")
        
        # Monitor resources
        monitor_resources()
        
        # Perform clustering
        cluster_labels, model = perform_advanced_clustering(embeddings, method, n_clusters)
        
        # Refine clusters
        refined_labels = refine_clusters(embeddings, cluster_labels, min_cluster_size)
        
        # Find representative keywords
        representatives = find_representative_keywords(embeddings, keywords_list, refined_labels)
        
        # Calculate coherence scores
        coherence_scores = calculate_cluster_coherence(embeddings, refined_labels)
        
        # Create results summary
        unique_labels = np.unique(refined_labels)
        cluster_sizes = {label: np.sum(refined_labels == label) for label in unique_labels}
        
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
"""
Block 6: AI-Powered Analysis Functions
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
                            "name": f"{keywords[0].title()} Related",
                            "description": f"Keywords related to {', '.join(keywords[:2])}"
                        }
            
            progress.progress((i + batch_size) / len(cluster_ids))
        
        # Ensure all clusters have names
        for cluster_id in representatives.keys():
            if cluster_id not in cluster_names:
                keywords = representatives[cluster_id][:2]
                cluster_names[cluster_id] = {
                    "name": f"Cluster {cluster_id}",
                    "description": f"Keywords related to {', '.join(keywords)}"
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
"""
Block 7: Data Processing and DataFrame Management
"""

def load_csv_file(uploaded_file, csv_format="auto"):
    """Load and validate CSV file"""
    try:
        # Read file content
        content = safe_file_read(uploaded_file)
        
        # Detect format if auto
        if csv_format == "auto":
            first_line = content.split('\n')[0].lower()
            if 'keyword' in first_line or 'search' in first_line:
                csv_format = "with_header"
            else:
                csv_format = "no_header"
        
        # Parse CSV based on format
        if csv_format == "no_header":
            df = pd.read_csv(StringIO(content), header=None, names=["keyword"])
        else:
            df = pd.read_csv(StringIO(content))
            
            # Standardize column names
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'keyword' in col_lower or col_lower in ['query', 'term', 'phrase']:
                    column_mapping[col] = 'keyword'
                elif 'volume' in col_lower or col_lower in ['searches', 'search_volume']:
                    column_mapping[col] = 'search_volume'
                elif 'competition' in col_lower or col_lower in ['comp', 'difficulty']:
                    column_mapping[col] = 'competition'
                elif 'cpc' in col_lower or 'cost' in col_lower:
                    column_mapping[col] = 'cpc'
            
            df = df.rename(columns=column_mapping)
            
            # Ensure keyword column exists
            if 'keyword' not in df.columns:
                if len(df.columns) > 0:
                    df = df.rename(columns={df.columns[0]: 'keyword'})
                else:
                    raise ValueError("No keyword column found")
        
        # Validate and clean data
        is_valid, message = validate_dataframe(df, ['keyword'])
        if not is_valid:
            raise ValueError(message)
        
        # Clean keywords
        df['keyword'] = df['keyword'].astype(str).str.strip()
        df = df[df['keyword'] != ''].reset_index(drop=True)
        
        # Handle search volume if present
        if 'search_volume' in df.columns:
            df['search_volume'] = pd.to_numeric(df['search_volume'], errors='coerce').fillna(0)
        
        # Limit size for memory management
        if len(df) > MAX_KEYWORDS:
            st.warning(f"⚠️ Dataset too large. Limiting to {MAX_KEYWORDS:,} keywords.")
            df = df.head(MAX_KEYWORDS)
        
        st.success(f"✅ Loaded {len(df):,} keywords successfully")
        
        return df
        
    except Exception as e:
        log_error(e, "csv_loading")
        st.error(f"CSV loading failed: {str(e)}")
        return None

def create_results_dataframe(keywords_list, cluster_results, cluster_names, 
                           coherence_scores, intent_results=None, quality_analysis=None):
    """Create comprehensive results DataFrame"""
    try:
        # Basic DataFrame
        df = pd.DataFrame({
            'keyword': keywords_list,
            'cluster_id': cluster_results['labels'],
        })
        
        # Add cluster names and descriptions
        df['cluster_name'] = df['cluster_id'].map(
            lambda x: cluster_names.get(x, {}).get('name', f'Cluster {x}')
        )
        df['cluster_description'] = df['cluster_id'].map(
            lambda x: cluster_names.get(x, {}).get('description', '')
        )
        
        # Add coherence scores
        df['cluster_coherence'] = df['cluster_id'].map(
            lambda x: coherence_scores.get(x, 0.5)
        )
        
        # Mark representative keywords
        df['is_representative'] = False
        representatives = cluster_results.get('representatives', {})
        for cluster_id, rep_keywords in representatives.items():
            mask = (df['cluster_id'] == cluster_id) & (df['keyword'].isin(rep_keywords))
            df.loc[mask, 'is_representative'] = True
        
        # Add search intent if available
        if intent_results:
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
        
        # Add cluster size
        cluster_sizes = df['cluster_id'].value_counts().to_dict()
        df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
        
        # Sort by cluster_id and then by representative status
        df = df.sort_values(['cluster_id', 'is_representative'], ascending=[True, False])
        df = df.reset_index(drop=True)
        
        st.success(f"✅ Results DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        log_error(e, "dataframe_creation", {
            "num_keywords": len(keywords_list),
            "has_cluster_results": cluster_results is not None,
            "has_cluster_names": cluster_names is not None
        })
        # Create minimal DataFrame as fallback
        return pd.DataFrame({
            'keyword': keywords_list,
            'cluster_id': range(len(keywords_list)),
            'cluster_name': [f'Cluster {i}' for i in range(len(keywords_list))],
            'cluster_description': ['Individual keyword' for _ in keywords_list],
            'cluster_coherence': [1.0 for _ in keywords_list],
            'is_representative': [True for _ in keywords_list]
        })

def add_search_volume_data(df, search_volume_col='search_volume'):
    """Add search volume analysis to DataFrame"""
    try:
        if search_volume_col not in df.columns:
            st.info("ℹ️ No search volume data available")
            return df
        
        # Clean search volume data
        df[search_volume_col] = pd.to_numeric(df[search_volume_col], errors='coerce').fillna(0)
        
        # Calculate cluster-level metrics
        cluster_volume_stats = df.groupby('cluster_id')[search_volume_col].agg([
            'sum', 'mean', 'max', 'count'
        ]).round(0).astype(int)
        
        cluster_volume_stats.columns = [
            'cluster_total_volume',
            'cluster_avg_volume', 
            'cluster_max_volume',
            'cluster_keyword_count'
        ]
        
        # Merge back to main DataFrame
        df = df.merge(cluster_volume_stats, left_on='cluster_id', right_index=True, how='left')
        
        # Calculate volume percentiles
        if df[search_volume_col].max() > 0:
            df['volume_percentile'] = df[search_volume_col].rank(pct=True) * 100
        else:
            df['volume_percentile'] = 50.0
        
        st.success("✅ Search volume analysis added")
        
        return df
        
    except Exception as e:
        log_error(e, "search_volume_analysis")
        st.warning(f"⚠️ Search volume analysis failed: {str(e)}")
        return df

def calculate_cluster_metrics(df):
    """Calculate comprehensive cluster metrics"""
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
                'representative_count': cluster_data['is_representative'].sum(),
            }
            
            # Search volume metrics (if available)
            if 'search_volume' in df.columns:
                cluster_metrics.update({
                    'total_search_volume': cluster_data['search_volume'].sum(),
                    'avg_search_volume': cluster_data['search_volume'].mean(),
                    'max_search_volume': cluster_data['search_volume'].max(),
                })
            
            # Intent distribution
            if 'search_intent' in df.columns:
                intent_counts = cluster_data['search_intent'].value_counts()
                primary_intent = intent_counts.index[0] if len(intent_counts) > 0 else 'Unknown'
                intent_diversity = len(intent_counts)
                
                cluster_metrics.update({
                    'primary_intent': primary_intent,
                    'intent_diversity': intent_diversity,
                    'intent_distribution': intent_counts.to_dict()
                })
            
            # Quality metrics (if available)
            if 'quality_score' in df.columns:
                cluster_metrics['avg_quality_score'] = cluster_data['quality_score'].mean()
            
            metrics[cluster_id] = cluster_metrics
        
        st.success(f"✅ Calculated metrics for {len(metrics)} clusters")
        
        return metrics
        
    except Exception as e:
        log_error(e, "cluster_metrics_calculation")
        return {}

def create_cluster_summary_dataframe(df, metrics=None):
    """Create a summary DataFrame for clusters"""
    try:
        # Basic cluster summary
        summary_data = []
        
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            # Get representative keywords
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
            if not rep_keywords:
                rep_keywords = cluster_data['keyword'].head(3).tolist()
            
            summary_row = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_data['cluster_name'].iloc[0],
                'keyword_count': len(cluster_data),
                'representative_keywords': ', '.join(rep_keywords[:5]),
                'avg_coherence': cluster_data['cluster_coherence'].mean(),
            }
            
            # Add search volume if available
            if 'search_volume' in df.columns:
                summary_row['total_search_volume'] = cluster_data['search_volume'].sum()
                summary_row['avg_search_volume'] = cluster_data['search_volume'].mean()
            
            # Add intent information
            if 'search_intent' in df.columns:
                intent_counts = cluster_data['search_intent'].value_counts()
                primary_intent = intent_counts.index[0] if len(intent_counts) > 0 else 'Unknown'
                summary_row['primary_intent'] = primary_intent
            
            # Add quality score if available
            if 'quality_score' in df.columns:
                summary_row['avg_quality'] = cluster_data['quality_score'].mean()
            
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by keyword count (largest clusters first)
        summary_df = summary_df.sort_values('keyword_count', ascending=False)
        
        return summary_df
        
    except Exception as e:
        log_error(e, "summary_dataframe_creation")
        return pd.DataFrame()

def export_results_to_csv(df, filename=None):
    """Export results DataFrame to CSV"""
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keyword_clusters_{timestamp}.csv"
        
        # Create clean export DataFrame
        export_df = df.copy()
        
        # Round numeric columns
        numeric_columns = export_df.select_dtypes(include=[np.number]).columns
        export_df[numeric_columns] = export_df[numeric_columns].round(3)
        
        # Convert boolean columns to Yes/No
        bool_columns = export_df.select_dtypes(include=[bool]).columns
        for col in bool_columns:
            export_df[col] = export_df[col].map({True: 'Yes', False: 'No'})
        
        # Generate CSV
        csv_data = export_df.to_csv(index=False)
        
        return csv_data, filename
        
    except Exception as e:
        log_error(e, "csv_export")
        raise e

def filter_dataframe_by_criteria(df, criteria):
    """Filter DataFrame based on various criteria"""
    try:
        filtered_df = df.copy()
        
        # Filter by cluster size
        if criteria.get('min_cluster_size'):
            cluster_sizes = filtered_df['cluster_id'].value_counts()
            valid_clusters = cluster_sizes[cluster_sizes >= criteria['min_cluster_size']].index
            filtered_df = filtered_df[filtered_df['cluster_id'].isin(valid_clusters)]
        
        # Filter by coherence
        if criteria.get('min_coherence'):
            filtered_df = filtered_df[filtered_df['cluster_coherence'] >= criteria['min_coherence']]
        
        # Filter by search volume
        if criteria.get('min_search_volume') and 'search_volume' in df.columns:
            filtered_df = filtered_df[filtered_df['search_volume'] >= criteria['min_search_volume']]
        
        # Filter by search intent
        if criteria.get('search_intents') and 'search_intent' in df.columns:
            filtered_df = filtered_df[filtered_df['search_intent'].isin(criteria['search_intents'])]
        
        # Filter by quality score
        if criteria.get('min_quality') and 'quality_score' in df.columns:
            filtered_df = filtered_df[filtered_df['quality_score'] >= criteria['min_quality']]
        
        return filtered_df
        
    except Exception as e:
        log_error(e, "dataframe_filtering")
        return df

def merge_original_data(results_df, original_df):
    """Merge clustering results with original CSV data"""
    try:
        # Identify common columns to avoid conflicts
        common_cols = set(results_df.columns) & set(original_df.columns)
        results_only_cols = [col for col in results_df.columns if col not in common_cols or col == 'keyword']
        
        # Prepare DataFrames for merge
        merge_df = results_df[results_only_cols].copy()
        
        # Add original data columns (except keyword which is already present)
        original_cols = [col for col in original_df.columns if col != 'keyword']
        
        if original_cols:
            # Merge on keyword
            merged_df = merge_df.merge(
                original_df[['keyword'] + original_cols], 
                on='keyword', 
                how='left'
            )
        else:
            merged_df = merge_df
        
        st.success("✅ Original data merged with clustering results")
        
        return merged_df
        
    except Exception as e:
        log_error(e, "data_merging")
        st.warning(f"⚠️ Could not merge original data: {str(e)}")
        return results_df

def validate_results_dataframe(df):
    """Validate the final results DataFrame"""
    try:
        required_columns = ['keyword', 'cluster_id', 'cluster_name', 'cluster_coherence']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty keywords
        empty_keywords = df['keyword'].isna().sum() + (df['keyword'] == '').sum()
        if empty_keywords > 0:
            st.warning(f"⚠️ Found {empty_keywords} empty keywords")
        
        # Check cluster ID validity
        if df['cluster_id'].isna().sum() > 0:
            raise ValueError("Found NaN values in cluster_id column")
        
        # Check coherence scores
        invalid_coherence = ((df['cluster_coherence'] < 0) | (df['cluster_coherence'] > 1)).sum()
        if invalid_coherence > 0:
            st.warning(f"⚠️ Found {invalid_coherence} invalid coherence scores")
            df['cluster_coherence'] = df['cluster_coherence'].clip(0, 1)
        
        # Basic statistics
        n_clusters = df['cluster_id'].nunique()
        n_keywords = len(df)
        avg_cluster_size = n_keywords / n_clusters if n_clusters > 0 else 0
        
        st.info(f"📊 Validation Summary: {n_keywords:,} keywords in {n_clusters} clusters (avg size: {avg_cluster_size:.1f})")
        
        return True, df
        
    except Exception as e:
        log_error(e, "dataframe_validation")
        return False, df

def prepare_download_data(df, format_type="csv"):
    """Prepare data for download in various formats"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "csv":
            data, filename = export_results_to_csv(df, f"keyword_clusters_{timestamp}.csv")
            mime_type = "text/csv"
            
        elif format_type == "excel":
            # Create Excel with multiple sheets
            from io import BytesIO
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main results
                df.to_excel(writer, sheet_name='Clustering Results', index=False)
                
                # Cluster summary
                summary_df = create_cluster_summary_dataframe(df)
                if not summary_df.empty:
                    summary_df.to_excel(writer, sheet_name='Cluster Summary', index=False)
                
                # Intent analysis
                if 'search_intent' in df.columns:
                    intent_summary = df.groupby(['cluster_id', 'cluster_name', 'search_intent']).size().reset_index(name='count')
                    intent_summary.to_excel(writer, sheet_name='Intent Analysis', index=False)
            
            data = output.getvalue()
            filename = f"keyword_clusters_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return data, filename, mime_type
        
    except Exception as e:
        log_error(e, "download_preparation", {"format": format_type})
        # Fallback to CSV
        data, filename = export_results_to_csv(df)
        return data, filename, "text/csv"
"""
Block 8: Visualization Functions
"""

def create_cluster_size_chart(df):
    """Create cluster size distribution chart"""
    try:
        cluster_sizes = df['cluster_id'].value_counts().reset_index()
        cluster_sizes.columns = ['cluster_id', 'keyword_count']
        
        # Add cluster names
        cluster_names = df.groupby('cluster_id')['cluster_name'].first().reset_index()
        cluster_sizes = cluster_sizes.merge(cluster_names, on='cluster_id')
        
        # Create short labels
        cluster_sizes['label'] = cluster_sizes.apply(
            lambda x: f"{x['cluster_name'][:20]}{'...' if len(x['cluster_name']) > 20 else ''} ({x['cluster_id']})",
            axis=1
        )
        
        # Sort by size
        cluster_sizes = cluster_sizes.sort_values('keyword_count', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            cluster_sizes.tail(20),  # Show top 20 clusters
            x='keyword_count',
            y='label',
            orientation='h',
            title='Cluster Size Distribution (Top 20)',
            labels={'keyword_count': 'Number of Keywords', 'label': 'Cluster'},
            color='keyword_count',
            color_continuous_scale='blues'
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        log_error(e, "cluster_size_chart")
        return None

def create_coherence_chart(df):
    """Create cluster coherence analysis chart"""
    try:
        coherence_data = df.groupby(['cluster_id', 'cluster_name'])['cluster_coherence'].mean().reset_index()
        coherence_data['keyword_count'] = df['cluster_id'].value_counts().reindex(coherence_data['cluster_id']).values
        
        # Create short labels
        coherence_data['label'] = coherence_data.apply(
            lambda x: f"{x['cluster_name'][:20]}{'...' if len(x['cluster_name']) > 20 else ''} ({x['cluster_id']})",
            axis=1
        )
        
        # Create scatter plot
        fig = px.scatter(
            coherence_data,
            x='cluster_coherence',
            y='keyword_count',
            size='keyword_count',
            hover_name='label',
            title='Cluster Coherence vs Size',
            labels={
                'cluster_coherence': 'Semantic Coherence Score',
                'keyword_count': 'Number of Keywords'
            },
            color='cluster_coherence',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=500)
        
        return fig
        
    except Exception as e:
        log_error(e, "coherence_chart")
        return None

def create_intent_distribution_chart(df):
    """Create search intent distribution chart"""
    try:
        if 'search_intent' not in df.columns:
            return None
        
        intent_counts = df['search_intent'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=intent_counts.values,
            names=intent_counts.index,
            title='Search Intent Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
        
    except Exception as e:
        log_error(e, "intent_distribution_chart")
        return None

def create_cluster_quality_heatmap(df):
    """Create cluster quality heatmap"""
    try:
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
            cluster_data['quality'] = cluster_data['coherence'] * 10  # Scale coherence as proxy
        
        # Create bins for visualization
        cluster_data['size_bin'] = pd.cut(cluster_data['size'], bins=5, labels=['XS', 'S', 'M', 'L', 'XL'])
        cluster_data['coherence_bin'] = pd.cut(cluster_data['coherence'], bins=5, labels=['Low', 'Below Avg', 'Average', 'Above Avg', 'High'])
        
        # Create pivot table for heatmap
        heatmap_data = cluster_data.groupby(['size_bin', 'coherence_bin'])['quality'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='size_bin', columns='coherence_bin', values='quality')
        
        # Create heatmap
        fig = px.imshow(
            heatmap_pivot,
            title='Cluster Quality Heatmap (Size vs Coherence)',
            labels={
                'x': 'Coherence Level',
                'y': 'Cluster Size',
                'color': 'Avg Quality Score'
            },
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=400)
        
        return fig
        
    except Exception as e:
        log_error(e, "quality_heatmap")
        return None

def create_search_volume_analysis(df):
    """Create search volume analysis charts"""
    try:
        if 'search_volume' not in df.columns:
            return None, None
        
        # Cluster-level volume analysis
        volume_data = df.groupby(['cluster_id', 'cluster_name']).agg({
            'search_volume': ['sum', 'mean', 'max'],
            'keyword': 'count'
        }).reset_index()
        
        volume_data.columns = ['cluster_id', 'cluster_name', 'total_volume', 'avg_volume', 'max_volume', 'keyword_count']
        
        # Create short labels
        volume_data['label'] = volume_data.apply(
            lambda x: f"{x['cluster_name'][:15]}{'...' if len(x['cluster_name']) > 15 else ''} ({x['cluster_id']})",
            axis=1
        )
        
        # Chart 1: Total volume by cluster
        fig1 = px.bar(
            volume_data.nlargest(15, 'total_volume'),
            x='label',
            y='total_volume',
            title='Total Search Volume by Cluster (Top 15)',
            labels={'total_volume': 'Total Search Volume', 'label': 'Cluster'},
            color='total_volume',
            color_continuous_scale='blues'
        )
        
        fig1.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        # Chart 2: Volume vs cluster size
        fig2 = px.scatter(
            volume_data,
            x='keyword_count',
            y='total_volume',
            size='avg_volume',
            hover_name='label',
            title='Search Volume vs Cluster Size',
            labels={
                'keyword_count': 'Number of Keywords',
                'total_volume': 'Total Search Volume',
                'avg_volume': 'Average Volume per Keyword'
            },
            color='avg_volume',
            color_continuous_scale='viridis'
        )
        
        fig2.update_layout(height=400)
        
        return fig1, fig2
        
    except Exception as e:
        log_error(e, "search_volume_analysis")
        return None, None

def create_representative_keywords_chart(df, top_clusters=10):
    """Create chart showing representative keywords for top clusters"""
    try:
        # Get top clusters by size
        top_cluster_ids = df['cluster_id'].value_counts().head(top_clusters).index
        
        rep_data = []
        for cluster_id in top_cluster_ids:
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_name = cluster_data['cluster_name'].iloc[0]
            
            # Get representative keywords
            rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
            if not rep_keywords:
                rep_keywords = cluster_data['keyword'].head(3).tolist()
            
            rep_data.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'representative_keywords': ', '.join(rep_keywords[:5]),
                'keyword_count': len(cluster_data)
            })
        
        rep_df = pd.DataFrame(rep_data)
        
        # Create horizontal bar chart with annotations
        fig = px.bar(
            rep_df,
            x='keyword_count',
            y='cluster_name',
            orientation='h',
            title=f'Top {top_clusters} Clusters with Representative Keywords',
            labels={'keyword_count': 'Number of Keywords', 'cluster_name': 'Cluster'},
            hover_data=['representative_keywords']
        )
        
        fig.update_layout(
            height=max(400, top_clusters * 40),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        log_error(e, "representative_keywords_chart")
        return None

def create_clustering_summary_metrics(df):
    """Create summary metrics display"""
    try:
        metrics = {}
        
        # Basic metrics
        metrics['total_keywords'] = len(df)
        metrics['total_clusters'] = df['cluster_id'].nunique()
        metrics['avg_cluster_size'] = metrics['total_keywords'] / metrics['total_clusters']
        metrics['avg_coherence'] = df['cluster_coherence'].mean()
        
        # Representative keywords
        metrics['representative_keywords'] = df['is_representative'].sum()
        metrics['rep_percentage'] = (metrics['representative_keywords'] / metrics['total_keywords']) * 100
        
        # Search volume metrics (if available)
        if 'search_volume' in df.columns:
            metrics['total_search_volume'] = df['search_volume'].sum()
            metrics['avg_search_volume'] = df['search_volume'].mean()
            metrics['max_search_volume'] = df['search_volume'].max()
        
        # Intent distribution (if available)
        if 'search_intent' in df.columns:
            intent_dist = df['search_intent'].value_counts(normalize=True) * 100
            metrics['primary_intent'] = intent_dist.index[0] if len(intent_dist) > 0 else 'Unknown'
            metrics['intent_distribution'] = intent_dist.to_dict()
        
        # Quality metrics (if available)
        if 'quality_score' in df.columns:
            metrics['avg_quality'] = df['quality_score'].mean()
            high_quality_clusters = (df.groupby('cluster_id')['quality_score'].mean() >= 7).sum()
            metrics['high_quality_clusters'] = high_quality_clusters
        
        return metrics
        
    except Exception as e:
        log_error(e, "summary_metrics")
        return {}

def display_clustering_dashboard(df):
    """Display comprehensive clustering dashboard"""
    try:
        st.header("📊 Clustering Analysis Dashboard")
        
        # Summary metrics
        metrics = create_clustering_summary_metrics(df)
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Keywords", 
                    format_number(metrics['total_keywords'])
                )
                
            with col2:
                st.metric(
                    "Clusters Created", 
                    metrics['total_clusters']
                )
                
            with col3:
                st.metric(
                    "Avg Cluster Size", 
                    f"{metrics['avg_cluster_size']:.1f}"
                )
                
            with col4:
                st.metric(
                    "Avg Coherence", 
                    f"{metrics['avg_coherence']:.3f}"
                )
            
            # Additional metrics row
            if 'total_search_volume' in metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Search Volume", 
                        format_number(metrics['total_search_volume'])
                    )
                    
                with col2:
                    st.metric(
                        "Avg Search Volume", 
                        format_number(metrics['avg_search_volume'])
                    )
                    
                with col3:
                    if 'primary_intent' in metrics:
                        st.metric("Primary Intent", metrics['primary_intent'])
                    
                with col4:
                    if 'high_quality_clusters' in metrics:
                        st.metric("High Quality Clusters", metrics['high_quality_clusters'])
        
        # Charts in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📏 Cluster Sizes", 
            "🎯 Quality Analysis", 
            "🔍 Search Intent", 
            "📈 Search Volume"
        ])
        
        with tab1:
            size_chart = create_cluster_size_chart(df)
            if size_chart:
                st.plotly_chart(size_chart, use_container_width=True)
            
            coherence_chart = create_coherence_chart(df)
            if coherence_chart:
                st.plotly_chart(coherence_chart, use_container_width=True)
        
        with tab2:
            quality_heatmap = create_cluster_quality_heatmap(df)
            if quality_heatmap:
                st.plotly_chart(quality_heatmap, use_container_width=True)
            
            rep_chart = create_representative_keywords_chart(df)
            if rep_chart:
                st.plotly_chart(rep_chart, use_container_width=True)
        
        with tab3:
            intent_chart = create_intent_distribution_chart(df)
            if intent_chart:
                st.plotly_chart(intent_chart, use_container_width=True)
            else:
                st.info("ℹ️ Search intent analysis not available")
        
        with tab4:
            vol_chart1, vol_chart2 = create_search_volume_analysis(df)
            if vol_chart1 and vol_chart2:
                st.plotly_chart(vol_chart1, use_container_width=True)
                st.plotly_chart(vol_chart2, use_container_width=True)
            else:
                st.info("ℹ️ Search volume data not available")
        
        return True
        
    except Exception as e:
        log_error(e, "clustering_dashboard")
        st.error(f"Dashboard error: {str(e)}")
        return False

def create_cluster_explorer(df):
    """Create interactive cluster explorer"""
    try:
        st.header("🔍 Cluster Explorer")
        
        # Cluster selection
        cluster_options = {}
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_name = cluster_data['cluster_name'].iloc[0]
            keyword_count = len(cluster_data)
            cluster_options[f"{cluster_name} (ID: {cluster_id}, {keyword_count} keywords)"] = cluster_id
        
        selected_cluster_key = st.selectbox(
            "Select a cluster to explore:",
            options=list(cluster_options.keys())
        )
        
        if selected_cluster_key:
            selected_cluster_id = cluster_options[selected_cluster_key]
            cluster_data = df[df['cluster_id'] == selected_cluster_id]
            
            # Cluster overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Cluster Overview")
                st.write(f"**Name:** {cluster_data['cluster_name'].iloc[0]}")
                st.write(f"**Description:** {cluster_data['cluster_description'].iloc[0]}")
                st.write(f"**Keywords:** {len(cluster_data)}")
                st.write(f"**Coherence:** {cluster_data['cluster_coherence'].iloc[0]:.3f}")
                
                if 'search_volume' in cluster_data.columns:
                    total_volume = cluster_data['search_volume'].sum()
                    avg_volume = cluster_data['search_volume'].mean()
                    st.write(f"**Total Search Volume:** {format_number(total_volume)}")
                    st.write(f"**Avg Search Volume:** {format_number(avg_volume)}")
            
            with col2:
                # Representative keywords
                rep_keywords = cluster_data[cluster_data['is_representative'] == True]['keyword'].tolist()
                if rep_keywords:
                    st.subheader("⭐ Representative Keywords")
                    for i, kw in enumerate(rep_keywords[:10], 1):
                        st.write(f"{i}. {kw}")
                
                # Search intent if available
                if 'search_intent' in cluster_data.columns:
                    intent_dist = cluster_data['search_intent'].value_counts()
                    if len(intent_dist) > 0:
                        st.subheader("🎯 Search Intent Distribution")
                        for intent, count in intent_dist.items():
                            percentage = (count / len(cluster_data)) * 100
                            st.write(f"**{intent}:** {percentage:.1f}% ({count} keywords)")
            
            # All keywords table
            st.subheader("📝 All Keywords in this Cluster")
            
            # Prepare display data
            display_cols = ['keyword', 'is_representative']
            if 'search_volume' in cluster_data.columns:
                display_cols.append('search_volume')
            if 'search_intent' in cluster_data.columns:
                display_cols.append('search_intent')
            
            display_data = cluster_data[display_cols].copy()
            display_data['is_representative'] = display_data['is_representative'].map({True: '✅', False: ''})
            
            # Sort by representative first, then by search volume if available
            if 'search_volume' in display_data.columns:
                display_data = display_data.sort_values(['is_representative', 'search_volume'], ascending=[False, False])
            else:
                display_data = display_data.sort_values('is_representative', ascending=False)
            
            st.dataframe(display_data, use_container_width=True, height=400)
        
        return True
        
    except Exception as e:
        log_error(e, "cluster_explorer")
        st.error(f"Cluster explorer error: {str(e)}")
        return False

def show_export_options(df):
    """Show export options with download buttons"""
    try:
        st.header("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Standard Export")
            
            # CSV export
            csv_data, csv_filename = export_results_to_csv(df)
            st.download_button(
                label="📄 Download Full Results (CSV)",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True
            )
            
            # Summary CSV
            summary_df = create_cluster_summary_dataframe(df)
            if not summary_df.empty:
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download Cluster Summary (CSV)",
                    data=summary_csv,
                    file_name=f"cluster_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.subheader("📈 Advanced Export")
            
            # Excel export (if available)
            try:
                excel_data, excel_filename, excel_mime = prepare_download_data(df, "excel")
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name=excel_filename,
                    mime=excel_mime,
                    use_container_width=True
                )
            except Exception as e:
                st.info("📊 Excel export requires openpyxl package")
            
            # Show preview of export data
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        return True
        
    except Exception as e:
        log_error(e, "export_options")
        st.error(f"Export options error: {str(e)}")
        return False
"""
Block 9: Streamlit User Interface
"""

def create_sidebar_configuration():
    """Create sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload CSV File",
        type=['csv'],
        help="Upload a CSV file containing keywords"
    )
    
    # CSV format selection
    csv_format = st.sidebar.selectbox(
        "📋 CSV Format",
        options=["auto", "no_header", "with_header"],
        index=0,
        help="Auto-detect or specify CSV format"
    )
    
    # Language selection
    language_options = list(SPACY_MODELS.keys()) + ["Auto"]
    selected_language = st.sidebar.selectbox(
        "🌍 Language",
        options=language_options,
        index=0,
        help="Select the language of your keywords"
    )
    
    # OpenAI API key
    openai_api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key (Optional)",
        type="password",
        help="Enter your OpenAI API key for enhanced embeddings and AI analysis"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        # Clustering parameters
        clustering_method = st.selectbox(
            "Clustering Method",
            options=["auto", "kmeans", "hierarchical"],
            index=0,
            help="Choose clustering algorithm"
        )
        
        num_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=50,
            value=10,
            help="Target number of clusters (use 0 for auto-detection)"
        )
        
        min_cluster_size = st.slider(
            "Minimum Cluster Size",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum keywords per cluster"
        )
        
        # Embedding parameters
        embedding_method = st.selectbox(
            "Embedding Method",
            options=["auto", "openai", "sentence_transformers", "tfidf"],
            index=0,
            help="Choose embedding generation method"
        )
        
        max_keywords = st.slider(
            "Max Keywords to Process",
            min_value=100,
            max_value=MAX_KEYWORDS,
            value=10000,
            step=500,
            help="Limit keywords for memory management"
        )
        
        # AI analysis parameters
        ai_model = st.selectbox(
            "AI Model for Analysis",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=0,
            help="Choose AI model for cluster naming and analysis"
        )
        
        enable_intent_analysis = st.checkbox(
            "Enable Search Intent Analysis",
            value=True,
            help="Analyze search intent for keywords"
        )
        
        enable_quality_analysis = st.checkbox(
            "Enable AI Quality Analysis",
            value=True,
            help="Use AI to analyze cluster quality (requires OpenAI API)"
        )
    
    # Cost calculator
    with st.sidebar.expander("💰 Cost Calculator"):
        if uploaded_file:
            try:
                temp_df = load_csv_file(uploaded_file, csv_format)
                if temp_df is not None:
                    num_keywords = len(temp_df)
                    cost_info = calculate_estimated_cost(num_keywords, ai_model, num_clusters)
                    
                    st.metric("Keywords to Process", format_number(cost_info['processed_keywords']))
                    st.metric("Estimated Total Cost", f"${cost_info['total_cost']:.4f}")
                    
                    if cost_info['total_cost'] > 1.0:
                        st.warning("⚠️ High cost estimated. Consider using fewer keywords or free alternatives.")
                
                # Reset file pointer
                uploaded_file.seek(0)
            except Exception as e:
                st.error(f"Cost calculation error: {str(e)}")
    
    # Library status
    with st.sidebar.expander("📚 Library Status"):
        st.write("**OpenAI:**", "✅ Available" if OPENAI_AVAILABLE else "❌ Not available")
        st.write("**SentenceTransformers:**", "✅ Available" if SENTENCE_TRANSFORMERS_AVAILABLE else "❌ Not available")
        st.write("**spaCy:**", "✅ Available" if SPACY_AVAILABLE else "❌ Not available")
        st.write("**TextBlob:**", "✅ Available" if TEXTBLOB_AVAILABLE else "❌ Not available")
    
    return {
        'uploaded_file': uploaded_file,
        'csv_format': csv_format,
        'language': selected_language,
        'openai_api_key': openai_api_key,
        'clustering_method': clustering_method,
        'num_clusters': num_clusters if num_clusters > 0 else None,
        'min_cluster_size': min_cluster_size,
        'embedding_method': embedding_method,
        'max_keywords': max_keywords,
        'ai_model': ai_model,
        'enable_intent_analysis': enable_intent_analysis,
        'enable_quality_analysis': enable_quality_analysis
    }

def show_welcome_screen():
    """Show welcome screen with instructions"""
    st.markdown("""
    <div class='main-header'>🔍 Semantic Keyword Clustering</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Advanced Semantic Keyword Clustering tool! This application uses state-of-the-art 
    NLP techniques to group semantically similar keywords, analyze search intent, and provide 
    actionable insights for your SEO strategy.
    
    ### ✨ Key Features
    
    🤖 **Multiple Embedding Methods**
    - OpenAI embeddings (highest quality)
    - SentenceTransformers (good quality, free)
    - TF-IDF embeddings (basic quality, always available)
    
    🎯 **Advanced Analysis**
    - Search intent classification (Informational, Commercial, Transactional, Navigational)
    - AI-powered cluster naming and descriptions
    - Quality assessment and improvement suggestions
    - Representative keyword identification
    
    📊 **Rich Visualizations**
    - Interactive cluster size distributions
    - Coherence analysis charts
    - Search intent breakdowns
    - Search volume analysis (if data available)
    
    📥 **Export Options**
    - CSV export with full results
    - Excel reports with multiple sheets
    - Cluster summaries and insights
    
    ### 🚀 Getting Started
    
    1. **Upload your CSV file** using the sidebar
    2. **Configure settings** (optional - defaults work well)
    3. **Add OpenAI API key** for enhanced features (optional)
    4. **Click "Start Clustering"** and wait for results
    5. **Explore your clusters** using the interactive dashboard
    
    ### 📁 Supported CSV Formats
    
    - **No Header:** Simple list of keywords, one per line
    - **With Header:** Keyword Planner format with columns like 'Keyword', 'search_volume', etc.
    - **Auto-detect:** The app will try to determine the format automatically
    
    ### 💡 Tips for Best Results
    
    - Clean your keywords beforehand (remove duplicates, fix typos)
    - For large datasets (>10k keywords), consider using batches
    - Use OpenAI API for highest quality clustering
    - Review and refine clusters manually after automated processing
    
    **Ready to get started? Upload your CSV file in the sidebar! 👈**
    """)

def show_processing_screen(config):
    """Show processing screen during clustering"""
    st.markdown("""
    <div class='main-header'>🔄 Processing Your Keywords</div>
    """, unsafe_allow_html=True)
    
    # Show configuration summary
    with st.expander("📋 Processing Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Language:** {config['language']}")
            st.write(f"**Clustering Method:** {config['clustering_method']}")
            st.write(f"**Target Clusters:** {config['num_clusters'] or 'Auto-detect'}")
            st.write(f"**Min Cluster Size:** {config['min_cluster_size']}")
        
        with col2:
            st.write(f"**Embedding Method:** {config['embedding_method']}")
            st.write(f"**Max Keywords:** {format_number(config['max_keywords'])}")
            st.write(f"**OpenAI API:** {'✅ Enabled' if config['openai_api_key'] else '❌ Disabled'}")
            st.write(f"**Intent Analysis:** {'✅ Enabled' if config['enable_intent_analysis'] else '❌ Disabled'}")
    
    st.info("🔄 Processing will begin automatically. Please wait while we analyze your keywords...")

def show_results_screen(df, config):
    """Show results screen with full analysis"""
    st.markdown("""
    <div class='main-header'>📊 Clustering Results</div>
    """, unsafe_allow_html=True)
    
    # Show dashboard
    dashboard_success = display_clustering_dashboard(df)
    
    if dashboard_success:
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🔍 Cluster Explorer", "📊 Data Table", "📥 Export Options"])
        
        with tab1:
            create_cluster_explorer(df)
        
        with tab2:
            show_data_table_view(df)
        
        with tab3:
            show_export_options(df)
        
        # Reset option
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Process New File", use_container_width=True):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

def show_data_table_view(df):
    """Show detailed data table view with filtering options"""
    try:
        st.header("📊 Detailed Data View")
        
        # Filtering options
        with st.expander("🔍 Filter Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Cluster filter
                cluster_options = ["All"] + [f"Cluster {cid}" for cid in sorted(df['cluster_id'].unique())]
                selected_cluster = st.selectbox("Filter by Cluster", cluster_options)
                
                # Representative keywords filter
                show_representative_only = st.checkbox("Show Representative Keywords Only")
            
            with col2:
                # Search volume filter (if available)
                if 'search_volume' in df.columns:
                    min_volume = st.number_input(
                        "Minimum Search Volume",
                        min_value=0,
                        value=0,
                        help="Filter keywords by minimum search volume"
                    )
                else:
                    min_volume = 0
                
                # Coherence filter
                min_coherence = st.slider(
                    "Minimum Coherence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help="Filter clusters by minimum coherence score"
                )
            
            with col3:
                # Intent filter (if available)
                if 'search_intent' in df.columns:
                    intent_options = ["All"] + list(df['search_intent'].unique())
                    selected_intent = st.selectbox("Filter by Search Intent", intent_options)
                else:
                    selected_intent = "All"
                
                # Keyword search
                keyword_search = st.text_input(
                    "Search Keywords",
                    placeholder="Enter keyword to search...",
                    help="Search for specific keywords"
                )
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_cluster != "All":
            cluster_id = int(selected_cluster.split(" ")[1])
            filtered_df = filtered_df[filtered_df['cluster_id'] == cluster_id]
        
        if show_representative_only:
            filtered_df = filtered_df[filtered_df['is_representative'] == True]
        
        if min_volume > 0 and 'search_volume' in df.columns:
            filtered_df = filtered_df[filtered_df['search_volume'] >= min_volume]
        
        if min_coherence > 0:
            filtered_df = filtered_df[filtered_df['cluster_coherence'] >= min_coherence]
        
        if selected_intent != "All" and 'search_intent' in df.columns:
            filtered_df = filtered_df[filtered_df['search_intent'] == selected_intent]
        
        if keyword_search:
            filtered_df = filtered_df[filtered_df['keyword'].str.contains(keyword_search, case=False, na=False)]
        
        # Show filter results
        if len(filtered_df) < len(df):
            st.info(f"Showing {len(filtered_df):,} of {len(df):,} keywords after filtering")
        
        # Display table
        if len(filtered_df) > 0:
            # Prepare display columns
            display_columns = ['keyword', 'cluster_id', 'cluster_name', 'is_representative', 'cluster_coherence']
            
            if 'search_volume' in filtered_df.columns:
                display_columns.append('search_volume')
            if 'search_intent' in filtered_df.columns:
                display_columns.append('search_intent')
            if 'quality_score' in filtered_df.columns:
                display_columns.append('quality_score')
            
            # Format display data
            display_df = filtered_df[display_columns].copy()
            display_df['is_representative'] = display_df['is_representative'].map({True: '⭐', False: ''})
            display_df['cluster_coherence'] = display_df['cluster_coherence'].round(3)
            
            if 'quality_score' in display_df.columns:
                display_df['quality_score'] = display_df['quality_score'].round(1)
            
            # Sort by cluster ID and representative status
            display_df = display_df.sort_values(['cluster_id', 'is_representative'], ascending=[True, False])
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500
            )
            
            # Download filtered data
            if len(filtered_df) < len(df):
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name=f"filtered_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No keywords match the current filters. Please adjust your filter criteria.")
        
    except Exception as e:
        log_error(e, "data_table_view")
        st.error(f"Data table error: {str(e)}")

def show_error_screen(error_message, suggestions=None):
    """Show error screen with helpful suggestions"""
    st.markdown("""
    <div class='error-box'>
    <h2>❌ Processing Error</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.error(f"**Error:** {error_message}")
    
    if suggestions:
        st.markdown("### 💡 Suggested Solutions:")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    
    st.markdown("""
    ### 🛠️ Common Solutions:
    
    - **File too large:** Try reducing the number of keywords (max recommended: 25,000)
    - **Invalid format:** Ensure your CSV is properly formatted with UTF-8 encoding
    - **Memory issues:** Refresh the page and try with a smaller dataset
    - **API errors:** Check your OpenAI API key or try without API features
    - **Connection issues:** Ensure stable internet connection for API calls
    
    ### 📞 Need Help?
    
    If the problem persists, try:
    1. Refreshing the page and starting over
    2. Using a smaller sample of your data first
    3. Checking the CSV format and encoding
    4. Disabling advanced features if memory is limited
    """)
    
    # Reset button
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def create_sample_csv_download():
    """Create sample CSV download section"""
    with st.expander("📁 Download Sample CSV", expanded=False):
        st.markdown("""
        Not sure about the CSV format? Download our sample file to see the expected structure.
        """)
        
        # Create sample data
        sample_data = {
            'keyword': [
                'running shoes',
                'best running shoes',
                'nike running shoes',
                'how to choose running shoes',
                'running shoes for women',
                'cheap running shoes',
                'running shoes review',
                'marathon running shoes',
                'trail running shoes',
                'running shoes near me'
            ],
            'search_volume': [5400, 3200, 2800, 1900, 4100, 1600, 1200, 890, 1500, 720],
            'competition': [0.75, 0.82, 0.68, 0.45, 0.71, 0.58, 0.63, 0.42, 0.55, 0.89],
            'cpc': [1.25, 1.78, 1.35, 0.89, 1.42, 0.95, 1.12, 0.76, 1.08, 1.95]
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_csv = sample_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Sample CSV",
            data=sample_csv,
            file_name="sample_keywords.csv",
            mime="text/csv",
            help="Download a sample CSV file with the correct format"
        )
        
        st.dataframe(sample_df, use_container_width=True)

def show_help_section():
    """Show help and documentation section"""
    with st.expander("❓ Help & Documentation", expanded=False):
        help_tab1, help_tab2, help_tab3 = st.tabs(["📖 Quick Start", "🔧 Advanced Features", "❓ FAQ"])
        
        with help_tab1:
            st.markdown("""
            ### 🚀 Quick Start Guide
            
            1. **Prepare your data:** Create a CSV file with keywords (one per row)
            2. **Upload file:** Use the file uploader in the sidebar
            3. **Choose settings:** Basic settings work for most use cases
            4. **Start processing:** Click the "Start Clustering" button
            5. **Explore results:** Use the dashboard and cluster explorer
            6. **Export data:** Download your results in CSV or Excel format
            
            ### 📋 CSV Format Examples
            
            **Simple format (no header):**
            ```
            running shoes
            best sneakers
            athletic footwear
            ```
            
            **Full format (with header):**
            ```
            keyword,search_volume,competition,cpc
            running shoes,5400,0.75,1.25
            best sneakers,3200,0.82,1.78
            ```
            """)
        
        with help_tab2:
            st.markdown("""
            ### 🔧 Advanced Features
            
            **Embedding Methods:**
            - **OpenAI:** Highest quality, requires API key
            - **SentenceTransformers:** Good quality, free
            - **TF-IDF:** Basic quality, always available
            
            **Clustering Algorithms:**
            - **K-means:** Fast, works well for most datasets
            - **Hierarchical:** Better for smaller datasets
            - **Auto:** Automatically chooses best method
            
            **Search Intent Classification:**
            - **Informational:** How-to, what is, guides
            - **Commercial:** Best, reviews, comparisons
            - **Transactional:** Buy, price, shop
            - **Navigational:** Brand names, login, contact
            
            **Quality Analysis:**
            - Coherence scores measure semantic similarity
            - Representative keywords are most central to clusters
            - AI analysis provides improvement suggestions
            """)
        
        with help_tab3:
            st.markdown("""
            ### ❓ Frequently Asked Questions
            
            **Q: How many keywords can I process?**
            A: Up to 25,000 keywords for optimal performance. Larger datasets may cause memory issues.
            
            **Q: Do I need an OpenAI API key?**
            A: No, but it significantly improves clustering quality and enables AI analysis features.
            
            **Q: What file formats are supported?**
            A: Only CSV files are supported. Ensure UTF-8 encoding for special characters.
            
            **Q: How long does processing take?**
            A: Depends on dataset size and method: TF-IDF (seconds), SentenceTransformers (minutes), OpenAI (depends on API limits).
            
            **Q: Can I edit clusters manually?**
            A: Not directly in the app, but you can export results and edit in Excel or other tools.
            
            **Q: Is my data stored anywhere?**
            A: No, all processing happens in your browser session. Data is not stored on servers.
            """)
"""
Block 10: Main Application Logic
"""

def run_clustering_pipeline(df, config):
    """Run the complete clustering pipeline"""
    try:
        # Initialize progress tracker
        total_steps = 7
        if config['enable_intent_analysis']:
            total_steps += 1
        if config['enable_quality_analysis'] and config['openai_api_key']:
            total_steps += 1
        
        progress_tracker = create_progress_tracker(
            total_steps,
            ["Loading Data", "Preprocessing", "Generating Embeddings", "Clustering", 
             "Finding Representatives", "Naming Clusters", "Final Processing"]
        )
        
        # Step 1: Prepare data
        progress_tracker.update("Loading and validating data...")
        
        # Limit keywords if necessary
        if len(df) > config['max_keywords']:
            st.warning(f"⚠️ Limiting dataset to {config['max_keywords']:,} keywords for performance")
            df = df.head(config['max_keywords'])
        
        keywords_list = df['keyword'].tolist()
        original_df = df.copy()  # Keep original data for merging later
        
        # Step 2: Preprocessing
        progress_tracker.update("Preprocessing keywords...")
        
        processed_keywords = preprocess_keywords(
            keywords_list, 
            language=config['language'], 
            method="auto"
        )
        
        # Step 3: Generate embeddings
        progress_tracker.update("Generating semantic embeddings...")
        
        # Create OpenAI client if needed
        client = None
        if config['openai_api_key']:
            client = create_openai_client(config['openai_api_key'])
        
        # Generate embeddings
        embeddings = generate_embeddings(
            keywords_list,
            client=client,
            method=config['embedding_method']
        )
        
        if embeddings is None:
            raise ValueError("Failed to generate embeddings")
        
        # Reduce dimensions if needed
        if embeddings.shape[1] > 100:
            embeddings = reduce_embedding_dimensions(embeddings, target_dim=100)
        
        # Step 4: Clustering
        progress_tracker.update("Performing semantic clustering...")
        
        cluster_results = cluster_keywords(
            keywords_list,
            embeddings,
            n_clusters=config['num_clusters'],
            method=config['clustering_method'],
            min_cluster_size=config['min_cluster_size']
        )
        
        # Step 5: Find representatives
        progress_tracker.update("Identifying representative keywords...")
        
        representatives = cluster_results['representatives']
        coherence_scores = cluster_results['coherence_scores']
        
        # Step 6: Generate cluster names
        progress_tracker.update("Generating cluster names...")
        
        if client:
            cluster_names = generate_cluster_names_openai(
                representatives, 
                client, 
                model=config['ai_model']
            )
        else:
            cluster_names = create_fallback_cluster_names(representatives)
        
        # Step 7: Search intent analysis (if enabled)
        intent_results = None
        if config['enable_intent_analysis']:
            progress_tracker.update("Analyzing search intent...")
            intent_results, _ = analyze_search_intent_bulk(keywords_list)
        
        # Step 8: Quality analysis (if enabled and API available)
        quality_analysis = None
        if config['enable_quality_analysis'] and client:
            progress_tracker.update("Performing AI quality analysis...")
            quality_analysis = analyze_cluster_quality_ai(
                representatives, 
                coherence_scores, 
                client, 
                model=config['ai_model']
            )
        
        # Step 9: Create final DataFrame
        progress_tracker.update("Creating final results...")
        
        results_df = create_results_dataframe(
            keywords_list,
            cluster_results,
            cluster_names,
            coherence_scores,
            intent_results,
            quality_analysis
        )
        
        # Merge with original data
        if len(original_df.columns) > 1:
            results_df = merge_original_data(results_df, original_df)
        
        # Add search volume analysis if available
        if 'search_volume' in results_df.columns:
            results_df = add_search_volume_data(results_df)
        
        # Validate final results
        is_valid, validated_df = validate_results_dataframe(results_df)
        if not is_valid:
            st.warning("⚠️ Results validation found some issues, but processing continued")
        
        progress_tracker.complete("Clustering completed successfully!")
        
        # Memory cleanup
        clean_memory()
        
        return validated_df
        
    except Exception as e:
        log_error(e, "clustering_pipeline", {
            "config": config,
            "num_keywords": len(df) if df is not None else 0
        })
        raise e

def handle_file_upload_and_validation(uploaded_file, csv_format):
    """Handle file upload and validation"""
    try:
        if uploaded_file is None:
            return None, "No file uploaded"
        
        # Show file info
        file_size = uploaded_file.size / (1024 * 1024)  # MB
        st.info(f"📁 File: {uploaded_file.name} ({file_size:.2f} MB)")
        
        # Load CSV
        df = load_csv_file(uploaded_file, csv_format)
        
        if df is None:
            return None, "Failed to load CSV file"
        
        # Show preview
        with st.expander("👀 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Keywords", len(df))
            with col2:
                if 'search_volume' in df.columns:
                    total_volume = df['search_volume'].sum()
                    st.metric("Total Search Volume", format_number(total_volume))
            with col3:
                unique_keywords = df['keyword'].nunique()
                duplicate_rate = (1 - unique_keywords / len(df)) * 100
                st.metric("Duplicate Rate", f"{duplicate_rate:.1f}%")
        
        return df, "Success"
        
    except Exception as e:
        log_error(e, "file_upload_validation")
        return None, str(e)

def main_application():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Monitor resources
        monitor_resources()
        
        # Create sidebar configuration
        config = create_sidebar_configuration()
        
        # Show help section
        show_help_section()
        
        # Create sample CSV download
        create_sample_csv_download()
        
        # Main application logic
        if config['uploaded_file'] is None:
            # Show welcome screen
            show_welcome_screen()
            
        else:
            # Handle file upload
            df, upload_message = handle_file_upload_and_validation(
                config['uploaded_file'], 
                config['csv_format']
            )
            
            if df is None:
                # Show error
                show_error_screen(
                    upload_message,
                    [
                        "Check your CSV file format and encoding",
                        "Ensure the file contains a 'keyword' column (if using headers)",
                        "Try with a smaller file first",
                        "Make sure the file is properly saved as CSV"
                    ]
                )
                return
            
            # Check if we should start processing
            if 'processing_started' not in st.session_state:
                st.session_state.processing_started = False
            
            if 'results_df' not in st.session_state:
                st.session_state.results_df = None
            
            # Show start button or processing screen
            if not st.session_state.processing_started and st.session_state.results_df is None:
                # Show processing preview screen
                show_processing_screen(config)
                
                # Start button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Start Clustering Analysis", use_container_width=True, type="primary"):
                        st.session_state.processing_started = True
                        st.rerun()
            
            elif st.session_state.processing_started and st.session_state.results_df is None:
                # Run processing
                try:
                    with st.spinner("🔄 Processing your keywords... This may take a few minutes."):
                        results_df = run_clustering_pipeline(df, config)
                        st.session_state.results_df = results_df
                        st.session_state.processing_started = False
                        st.success("✅ Clustering completed successfully!")
                        time.sleep(2)  # Brief pause to show success message
                        st.rerun()
                        
                except Exception as e:
                    # Reset processing state on error
                    st.session_state.processing_started = False
                    
                    error_msg = str(e)
                    suggestions = [
                        "Try with a smaller dataset",
                        "Disable AI features if you're having API issues",
                        "Check your internet connection",
                        "Refresh the page and try again"
                    ]
                    
                    # Specific error suggestions
                    if "memory" in error_msg.lower():
                        suggestions.insert(0, "Reduce the number of keywords in your dataset")
                        suggestions.insert(1, "Try using TF-IDF embeddings instead of OpenAI")
                    elif "api" in error_msg.lower() or "openai" in error_msg.lower():
                        suggestions.insert(0, "Check your OpenAI API key")
                        suggestions.insert(1, "Try without OpenAI features")
                    elif "timeout" in error_msg.lower():
                        suggestions.insert(0, "Check your internet connection")
                        suggestions.insert(1, "Try again - sometimes API calls timeout")
                    
                    show_error_screen(error_msg, suggestions)
                    return
            
            else:
                # Show results
                if st.session_state.results_df is not None:
                    show_results_screen(st.session_state.results_df, config)
                else:
                    st.error("❌ No results available. Please try processing again.")
    
    except Exception as e:
        log_error(e, "main_application")
        show_error_screen(
            f"Critical application error: {str(e)}",
            [
                "Refresh the page to restart the application",
                "Check your browser console for detailed error messages",
                "Try with a different browser or incognito mode",
                "Clear your browser cache and cookies"
            ]
        )

def setup_error_handling():
    """Setup global error handling"""
    try:
        # Configure pandas to avoid warnings
        import pandas as pd
        pd.options.mode.chained_assignment = None
        
        # Configure numpy to avoid warnings  
        import numpy as np
        np.seterr(divide='ignore', invalid='ignore')
        
        return True
    except Exception as e:
        logger.error(f"Error setting up error handling: {str(e)}")
        return False

def check_system_requirements():
    """Check if system meets minimum requirements"""
    try:
        requirements_met = True
        issues = []
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            requirements_met = False
            issues.append("Python 3.8+ required")
        
        # Check required packages
        required_packages = ['pandas', 'numpy', 'scikit-learn', 'plotly', 'streamlit']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                requirements_met = False
                issues.append(f"Missing required package: {package}")
        
        # Check memory (if available)
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            if memory.available < 1024 * 1024 * 1024:  # Less than 1GB available
                issues.append("Low available memory (< 1GB)")
        
        return requirements_met, issues
        
    except Exception as e:
        logger.error(f"Error checking system requirements: {str(e)}")
        return False, [f"Error checking requirements: {str(e)}"]

def display_startup_info():
    """Display startup information and system status"""
    try:
        # Check requirements
        requirements_met, issues = check_system_requirements()
        
        if not requirements_met:
            st.error("❌ System Requirements Not Met")
            for issue in issues:
                st.write(f"- {issue}")
            st.stop()
        
        # Show library status in sidebar (brief)
        with st.sidebar:
            if issues:
                with st.expander("⚠️ System Warnings", expanded=False):
                    for issue in issues:
                        st.warning(issue)
    
    except Exception as e:
        logger.error(f"Error displaying startup info: {str(e)}")

# Main execution
if __name__ == "__main__":
    try:
        # Setup
        setup_error_handling()
        
        # Display startup info
        display_startup_info()
        
        # Run main application
        main_application()
        
    except Exception as e:
        # Final fallback error handling
        st.error("🚨 Critical Application Error")
        st.write(f"**Error:** {str(e)}")
        st.write("**Solution:** Please refresh the page and try again.")
        
        # Log the error
        logger.critical(f"Critical application error: {str(e)}", exc_info=True)
        
        # Show restart button
        if st.button("🔄 Restart Application"):
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;">
    🔍 Semantic Keyword Clustering Tool | Built with Streamlit, scikit-learn, and OpenAI
</div>
""", unsafe_allow_html=True)
