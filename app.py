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
# Note: scipy.cluster.hierarchy and sklearn.metrics.pairwise.cosine_similarity are imported but not used in the provided clustering logic (KMeans).
# They might be remnants of previous approaches (e.g., Agglomerative Clustering). I'll keep them for now.
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
        # Attempt basic formatting
        formatted_value = format_str.format(value)
        # Ensure it's not just empty or invalid after format attempt
        if not formatted_value.strip() or "nan" in formatted_value.lower():
             return "N/A"
        return formatted_value
    except (ValueError, TypeError):
        return "N/A"
    except Exception:
        # Catch any other unexpected errors during formatting
        return "N/A"


# --- Library Availability Checks ---
# Attempt to import OpenAI
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False
    st.warning("OpenAI library not installed. OpenAI features will be disabled.")

# Try to import advanced libraries
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    st.warning("SentenceTransformers library not installed. Will rely on TF-IDF or OpenAI embeddings.")

# We will load spaCy models dynamically based on language
try:
    import spacy
    spacy_base_available = True
except ImportError:
    spacy_base_available = False
    st.warning("spaCy library not installed. Advanced preprocessing for specific languages will be limited.")

try:
    from textblob import TextBlob
    textblob_available = True
except ImportError:
    textblob_available = False
    st.warning("TextBlob library not installed. Fallback preprocessing might be less effective.")


# HDBSCAN is imported but not used in clustering logic. Keep availability check just in case.
try:
    import hdbscan
    hdbscan_available = True
except ImportError:
    hdbscan_available = False
    # st.info("HDBSCAN library not installed. Advanced density-based clustering is disabled.") # Suppress this message if HDBSCAN isn't actively used.

# --- NLTK Downloads ---
# Download NLTK resources at startup
@st.cache_resource # Cache NLTK resources to avoid repeated downloads
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True) # Often needed for WordNet
        return True
    except Exception:
        st.error("Failed to download NLTK resources. Basic preprocessing might not work correctly.")
        return False

nltk_download_successful = download_nltk_resources()

################################################################
#         SEARCH INTENT CLASSIFICATION PATTERNS
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
        "suffixes": ["definition", "meaning", "examples", "ideas", "guide", "tutorial", "info", "information"],
        "exact_matches": [
            "guide to", "how-to", "tutorial", "resources", "information", "knowledge",
            "examples of", "definition of", "explanation", "steps to", "learn about",
            "facts about", "history of", "benefits of", "causes of", "types of",
            "what is", "how to", "why is" # Added common phrases
        ],
        "keyword_patterns": [
            r'\bhow\s+to\b', r'\bwhat\s+is\b', r'\bwhy\s+is\b', r'\bwhen\s+to\b',
            r'\bwhere\s+to\b', r'\bwho\s+is\b', r'\bwhich\b.*\bbest\b',
            r'\bdefinition\b', r'\bmeaning\b', r'\bexamples?\b', r'\btips\b',
            r'\btutorials?\b', r'\bguide\b', r'\blearn\b', r'\bsteps?\b',
            r'\bversus\b', r'\bvs\b', r'\bcompared?\b', r'\bdifference\b',
            r'\bhow\s+can\b', r'\bwhat\s+are\b', r'\bwhy\s+do\b' # Added more question patterns
        ],
        "weight": 1.0 # Base weight
    },

    "Navigational": {
        "prefixes": ["go to", "visit", "website", "homepage", "home page", "sign in", "login", "access"],
        "suffixes": ["login", "website", "homepage", "official", "online", "app", "portal"],
        "exact_matches": [
            "login", "sign in", "register", "create account", "download", "official website",
            "official site", "homepage", "contact", "support", "customer service", "app",
            "my account", "dashboard", "web portal" # Added common phrases
        ],
        "keyword_patterns": [
            r'\blogin\b', r'\bsign\s+in\b', r'\bwebsite\b', r'\bhomepage\b', r'\bportal\b',
            r'\baccount\b', r'\bofficial\b', r'\bdashboard\b', r'\bdownload\b', # removed .*\\bfrom\\b restriction
            r'\bcontact\b', r'\baddress\b', r'\blocation\b', r'\bdirections?\b',
            r'\bmap\b', r'\btrack\b.*\border\b', r'\bmy\s+\w+(\s+\w+)?\s+account\b', # Improved account pattern
            r'\bapp\s+download\b', r'\bget\s+\w+\s+app\b' # Added app patterns
        ],
        "brand_indicators": True,  # Presence of brand names indicates navigational intent - logic needs to be added to use this
        "weight": 1.2  # Navigational intent is often more clear-cut
    },
    "Transactional": {
        "prefixes": ["buy", "purchase", "order", "shop", "get", "subscribe", "download", "install"],
        "suffixes": [
            "for sale", "discount", "deal", "coupon", "price", "cost", "cheap", "online",
            "free", "download", "subscription", "trial", "buy", "shop", "order", "purchase"
        ],
        "exact_matches": [
            "buy", "purchase", "order", "shop", "subscribe", "download", "free trial",
            "coupon code", "discount", "deal", "sale", "cheap", "best price", "near me",
            "shipping", "delivery", "in stock", "available", "pay", "checkout",
            "pricing", "cost of", "where to buy", "get a quote", "sign up" # Added common phrases
        ],
        "keyword_patterns": [
            r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bshop\b', r'\bstores?\b',
            r'\bprice\b', r'\bcost\b', r'\bcheap\b', r'\bdiscount\b', r'\bdeal\b',
            r'\bsale\b', r'\bcoupon\b', r'\bpromo\b', r'\bfree\s+shipping\b',
            r'\bnear\s+me\b', r'\bshipping\b', r'\bdelivery\b', r'\bcheck\s*out\b',
            r'\bin\s+stock\b', r'\bavailable\b', r'\bsubscribe\b', r'\bdownload\b',
            r'\binstall\b', r'\bfor\s+sale\b', r'\bhire\b', r'\brent\b',
            r'\bget\s+a\s+quote\b', r'\bsign\s+up\b', r'\bpurchase\s+price\b' # Added more patterns
        ],
        "weight": 1.5  # Strong transactional signals are highly valuable
    },

    "Commercial": {
        "prefixes": ["best", "top", "review", "compare", "vs", "versus", "alternative", "find", "choose"],
        "suffixes": [
            "review", "reviews", "comparison", "vs", "versus", "alternative", "alternatives",
            "recommendation", "recommendations", "comparison", "guide", "list", "ranking", "best", "top"
        ],
        "exact_matches": [
            "best", "top", "vs", "versus", "comparison", "compare", "review", "reviews",
            "rating", "ratings", "ranked", "recommended", "alternative", "alternatives",
            "pros and cons", "features", "worth it", "should i buy", "is it good",
            "which is best", "compare prices", "product review", "service review" # Added common phrases
        ],
        "keyword_patterns": [
            r'\bbest\b', r'\btop\b', r'\breview\b', r'\bcompare\b', r'\bcompari(son|ng)\b',
            r'\bvs\b', r'\bversus\b', r'\balternatives?\b', r'\brated\b', r'\branking\b',
            r'\bworth\s+it\b', r'\bshould\s+I\s+buy\b', r'\bis\s+it\s+good\b',
            r'\bpros\s+and\s+cons\b', r'\badvantages?\b', r'\bdisadvantages?\b',
            r'\bfeatures\b', r'\bspecifications?\b', r'\bwhich\s+(is\s+)?(the\s+)?best\b',
            r'\btop\s+\d+\b', r'\blist\s+of\b', r'\bfind\s+(the\s+)?best\b' # Added more patterns
        ],
        "weight": 1.2  # Commercial intent signals future transactions
    }
}

################################################################
#         LANGUAGE MODEL MANAGEMENT
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
    "Korean": None, # Korea doesn't have a standard spacy model, only contributed ones
    "Japanese": None, # Japan doesn't have a standard spacy model, only contributed ones
    "Icelandic": None, # Iceland doesn't have a standard spacy model, only contributed ones
    "Lithuanian": None # Lithuania doesn't have a standard spacy model, only contributed ones
}

@st.cache_resource # Cache the loaded spaCy model
def load_spacy_model_by_language(selected_language):
    """
    Try to load a spaCy model for the given language. If it fails or doesn't exist, returns None.
    """
    if not spacy_base_available:
        st.warning("spaCy is not installed. Cannot load language models.")
        return None

    model_name = SPACY_LANGUAGE_MODELS.get(selected_language, None)
    if model_name is None:
        st.info(f"No spaCy model specified or available for '{selected_language}'.")
        return None

    try:
        st.info(f"Attempting to load spaCy model '{model_name}' for {selected_language}...")
        nlp = spacy.load(model_name)
        st.success(f"✅ spaCy model '{model_name}' loaded successfully.")
        return nlp
    except OSError:
        st.error(f"spaCy model '{model_name}' not found. Please install it, e.g., `pip install {model_name}`.")
        return None
    except Exception as e:
        st.error(f"Error loading spaCy model '{model_name}': {str(e)}")
        return None
################################################################
#         COST CALCULATION AND SUPPORT FUNCTIONS
################################################################

def calculate_api_cost(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=None):
    """
    Calculates the estimated cost of using the OpenAI API based on the number of keywords
    and approximate number of clusters.
    num_clusters is an estimate used for naming cost calculation.
    """
    # Updated prices (April 2025) - Always check OpenAI's official pricing page for the latest info.
    # Pricing examples: https://openai.com/pricing
    EMBEDDING_COST_PER_1K = 0.00002 # text-embedding-3-small per 1K tokens ($0.00002 / 1K tokens)

    # GPT-3.5-Turbo costs (approximate based on current pricing)
    GPT35_INPUT_COST_PER_1M = 0.50 # $0.50 / 1M tokens input (approx)
    GPT35_OUTPUT_COST_PER_1M = 1.50 # $1.50 / 1M tokens output (approx)

    # GPT-4-Turbo costs (approximate based on current pricing)
    GPT4T_INPUT_COST_PER_1M = 10.00 # $10.00 / 1M tokens input (approx)
    GPT4T_OUTPUT_COST_PER_1M = 30.00 # $30.00 / 1M tokens output (approx)

    # Handle potential None num_clusters if not provided
    estimated_num_clusters = num_clusters if num_clusters is not None and num_clusters > 0 else max(1, min(50, num_keywords // 100)) # Estimate if not provided

    results = {
        "embedding_cost": 0,
        "naming_cost": 0,
        "total_cost": 0,
        "processed_keywords": 0
    }

    # 1. Embedding cost (limited to 5000 keywords for direct OpenAI, remaining are propagated)
    keywords_for_embeddings = min(num_keywords, 5000) # Limit based on the propagation logic
    results["processed_keywords"] = keywords_for_embeddings

    # Estimate tokens per keyword - A simple average like 2 might be low for longer keywords.
    # A safer estimate could be based on average characters or words * a factor.
    # Let's assume an average keyword length of 3 words, roughly 5-7 tokens. Use 6 as average.
    avg_tokens_per_keyword = 6
    estimated_embedding_tokens = keywords_for_embeddings * avg_tokens_per_keyword

    # Note: text-embedding-3-small is very cheap. The cost is for 1M tokens, not 1k.
    results["embedding_cost"] = (estimated_embedding_tokens / 1_000_000) * (EMBEDDING_COST_PER_1K * 1000) # Convert 1K price to 1M

    # 2. Naming cost (per cluster)
    # This is a rough estimate based on the prompt structure and expected response length.
    # It's difficult to be precise without knowing the exact prompt and cluster characteristics.
    # Avg tokens per cluster prompt: 200 (prompt + sample keywords)
    # Avg output tokens per cluster: 80 (name + description)
    avg_tokens_per_cluster_call_input = 200
    avg_output_tokens_per_cluster_call_output = 80

    estimated_input_tokens_naming = estimated_num_clusters * avg_tokens_per_cluster_call_input
    estimated_output_tokens_naming = estimated_num_clusters * avg_output_tokens_per_cluster_call_output

    if selected_model == "gpt-3.5-turbo":
        input_cost_naming = (estimated_input_tokens_naming / 1_000_000) * GPT35_INPUT_COST_PER_1M
        output_cost_naming = (estimated_output_tokens_naming / 1_000_000) * GPT35_OUTPUT_COST_PER_1M
    else: # Assuming "gpt-4" implies GPT-4-Turbo or similar
        input_cost_naming = (estimated_input_tokens_naming / 1_000_000) * GPT4T_INPUT_COST_PER_1M
        output_cost_naming = (estimated_output_tokens_naming / 1_000_000) * GPT4T_OUTPUT_COST_PER_1M

    results["naming_cost"] = input_cost_naming + output_cost_naming
    results["total_cost"] = results["embedding_cost"] + results["naming_cost"]

    return results

def add_cost_calculator():
    st.sidebar.markdown("---")
    with st.sidebar.expander("💰 API Cost Calculator", expanded=False):
        st.markdown("""
        ### API Cost Calculator

        Estimate OpenAI usage costs for a given number of keywords and clusters.
        Based on April 2025 pricing for `text-embedding-3-small` and GPT models.
        """)

        calc_num_keywords = st.number_input(
            "Number of keywords",
            min_value=100,
            max_value=1000000, # Increased max for larger datasets
            value=1000,
            step=500
        )
        calc_num_clusters = st.number_input(
            "Approx. number of clusters",
            min_value=2,
            max_value=100, # Increased max clusters
            value=max(10, min(50, calc_num_keywords // 100)), # Auto-suggest based on keyword count
            step=1
        )
        calc_model = st.radio(
            "Model for naming clusters",
            options=["gpt-3.5-turbo", "gpt-4-turbo"], # Use gpt-4-turbo for clarity
            index=0,
            horizontal=True
        )

        if st.button("Calculate Estimated Cost", key="calc_cost_button", use_container_width=True):
            cost_results = calculate_api_cost(calc_num_keywords, calc_model, calc_num_clusters)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Keywords for direct OpenAI Embeddings",
                    f"{cost_results['processed_keywords']:,}",
                    help="Up to 5,000 keywords processed directly by OpenAI. The rest use similarity propagation."
                )
                st.metric(
                    "Estimated Embedding Cost",
                    f"${cost_results['embedding_cost']:.6f}", # Show more decimals for small costs
                    help="Cost using text-embedding-3-small"
                )
            with col2:
                st.metric(
                    "Estimated Cluster Naming Cost",
                    f"${cost_results['naming_cost']:.6f}", # Show more decimals
                    help=f"Cost using {calc_model} to name and describe clusters"
                )
                st.metric(
                    "ESTIMATED TOTAL COST",
                    f"${cost_results['total_cost']:.6f}", # Show more decimals
                    help="Approximate total cost for OpenAI services"
                )

            st.info("""
            **Note:** This is an estimate only. Actual costs may vary based on keyword length, prompt details,
            and OpenAI's exact tokenization/pricing at the time of use.
            Using SentenceTransformers instead of OpenAI embeddings is $0.
            """)

def show_csv_cost_estimate(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=None):
    """
    Displays the estimated cost for the current uploaded CSV file.
    """
    if num_keywords is None or num_keywords <= 0:
        return # Don't show estimate for empty data

    # Estimate number of clusters if not already determined from previous steps
    # A common heuristic is sqrt(N/2) or similar, or just a fixed max
    estimated_num_clusters_for_csv = num_clusters
    if estimated_num_clusters_for_csv is None or estimated_num_clusters_for_csv <= 0:
         estimated_num_clusters_for_csv = max(2, min(50, num_keywords // 100))


    cost_results = calculate_api_cost(num_keywords, selected_model, estimated_num_clusters_for_csv)

    # Check if the cost has already been displayed for this CSV in the sidebar to avoid duplicates
    # This is a simple check, might need more robust handling for state changes
    if 'last_csv_cost_estimate' not in st.session_state or st.session_state['last_csv_cost_estimate'] != (num_keywords, selected_model, estimated_num_clusters_for_csv):
         st.session_state['last_csv_cost_estimate'] = (num_keywords, selected_model, estimated_num_clusters_for_csv)
         with st.sidebar.expander("💰 Estimated Cost (Current CSV)", expanded=True):
             st.markdown(f"### Estimate for {num_keywords:,} Keywords")

             st.markdown(f"""
             - **Keywords for direct OpenAI Embeddings**: {cost_results['processed_keywords']:,}
             - **Estimated Embedding Cost**: ${cost_results['embedding_cost']:.6f}
             - **Estimated Cluster Naming Cost**: ${cost_results['naming_cost']:.6f}
             - **ESTIMATED TOTAL COST**: ${cost_results['total_cost']:.6f}
             """)

             if cost_results['processed_keywords'] < num_keywords:
                 st.info(f"""
                 {cost_results['processed_keywords']:,} keywords will be processed by OpenAI directly.
                 The remaining {num_keywords - cost_results['processed_keywords']:,} will use
                 similarity propagation to infer embeddings from the sample.
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
    Returns a sample CSV header row and data.
    """
    header = ["Keyword", "search_volume", "competition", "cpc"]
    months = [f"month{i:02d}" for i in range(1, 13)] # Use 0-padding for months
    header += months

    # Sample data for download - Added a few more for variety
    data = """Keyword,search_volume,competition,cpc,month01,month02,month03,month04,month05,month06,month07,month08,month09,month10,month11,month12
running shoes,5400,0.75,1.25,450,460,470,480,490,500,510,520,530,540,550,560
nike shoes,8900,0.82,1.78,700,720,740,760,780,800,820,840,860,880,900,920
adidas sneakers,3200,0.65,1.12,260,270,280,290,300,310,320,330,340,350,360,370
hiking boots,2800,0.45,0.89,230,240,250,260,270,280,290,300,310,320,330,340
women's running shoes,4100,0.68,1.35,340,350,360,370,380,390,400,410,420,430,440,450
best running shoes 2025,3100,0.78,1.52,280,290,300,310,320,330,340,350,360,370,380,390
how to choose running shoes,2500,0.42,0.95,220,230,240,250,260,270,280,290,300,310,320,330
running shoes for flat feet,1900,0.56,1.28,170,180,190,200,210,220,230,240,250,260,270,280
trail running shoes reviews,1700,0.64,1.42,150,160,170,180,190,200,210,220,230,240,250,260
buy nike air zoom,1500,0.87,1.95,130,140,150,160,170,180,190,200,210,220,230,240
waterproof trail running shoes,1200,0.58,1.15,100,110,120,130,140,150,160,170,180,190,200,210
running shoe brands,900,0.55,1.05,80,85,90,95,100,105,110,115,120,125,130,135
cheap running shoes online,1100,0.70,1.45,90,95,100,105,110,115,120,125,130,135,140,145
asics running shoes sale,800,0.79,1.60,70,75,80,85,90,95,100,105,110,115,120,125
where to buy saucony running shoes,700,0.85,1.85,60,65,70,75,80,85,90,95,100,105,110,115
"""

    return ",".join(header) + "\n" + data

################################################################
#         SEMANTIC PREPROCESSING
################################################################

def enhanced_preprocessing(text, use_lemmatization, spacy_nlp):
    """
    Enhanced preprocessing using spaCy or fallback with TextBlob/NLTK.
    Handles None/NaN inputs gracefully.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()

    try:
        if spacy_nlp is not None:  # We have a loaded spaCy model
            doc = spacy_nlp(text) # spaCy handles lowercasing internally if needed

            # Extract entities (e.g., brand names, specific product types if patterns are in model)
            entities = [ent.text for ent in doc.ents] # Keep original text of entity

            tokens = []
            for token in doc:
                # Filter out punctuation, spaces, numbers, and short tokens, and stop words
                if not token.is_stop and token.is_alpha and len(token.text) > 1:
                    tokens.append(token.lemma_) # Use lemma for base form

            # Extract Bigrams (sequences of 2 tokens)
            bigrams = []
            for i in range(len(doc) - 1):
                 # Consider bigrams if both tokens are alphabetic and not stop words
                if (doc[i].is_alpha and not doc[i].is_stop) and \
                   (doc[i+1].is_alpha and not doc[i+1].is_stop):
                   bigrams.append(f"{doc[i].lemma_}_{doc[i+1].lemma_}") # Lemma bigrams

            processed_parts = tokens + bigrams + entities # Combine lemmas, bigrams, entities
            return " ".join(processed_parts)

        elif textblob_available:
            # Fallback to TextBlob
            from textblob import TextBlob # Import locally in case it's not available
            blob = TextBlob(text)

            # Get noun phrases as potential important terms
            noun_phrases = list(blob.noun_phrases) # Already lowercased by TextBlob

            words = []
            try:
                # Use NLTK stopwords if available and downloaded
                stop_words = set(stopwords.words('english')) if nltk_download_successful else set()
            except LookupError:
                 # Fallback to hardcoded if NLTK data not found
                 stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for', 'is', 'are'} # Added common stopwords
                 st.warning("NLTK stopwords not found. Using basic hardcoded list.")
            except Exception:
                 stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for', 'is', 'are'}
                 st.warning("Error loading NLTK stopwords. Using basic hardcoded list.")

            words = [w.lower() for w in blob.words if w.isalpha() and len(w) > 1 and w.lower() not in stop_words]

            if use_lemmatization and nltk_download_successful:
                try:
                    lemmatizer = WordNetLemmatizer()
                    lemmas = [lemmatizer.lemmatize(w) for w in words]
                    processed_parts = lemmas + noun_phrases
                except LookupError:
                     st.warning("NLTK WordNetLemmatizer not found. Skipping lemmatization.")
                     processed_parts = words + noun_phrases
                except Exception:
                     st.warning("Error with NLTK Lemmatizer. Skipping lemmatization.")
                     processed_parts = words + noun_phrases
            else:
                processed_parts = words + noun_phrases

            return " ".join(processed_parts)

        else:
# fallback to standard nltk if TextBlob and spaCy are not available
            st.info("Using standard NLTK preprocessing (spaCy/TextBlob not available).")
            return preprocess_text(text, use_lemmatization)

    except Exception as e:
        # Catch any other errors during advanced processing
        st.error(f"Error during enhanced preprocessing: {str(e)}. Falling back to basic lowercasing.")
        return text # Return lowercased original text as last resort


def preprocess_text(text, use_lemmatization=True):
    """
    Basic NLTK-based text preprocessing as a fallback.
    Handles None/NaN inputs gracefully.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        try:
            # Use NLTK stopwords if available and downloaded
            stop_words = set(stopwords.words('english')) if nltk_download_successful else set()
        except LookupError:
             # Fallback to hardcoded if NLTK data not found
             stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for', 'is', 'are'}
             st.warning("NLTK stopwords not found. Using basic hardcoded list.")
        except Exception:
            stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for', 'is', 'are'}
            st.warning("Error loading NLTK stopwords. Using basic hardcoded list.")

        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

        if use_lemmatization and nltk_download_successful:
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            except LookupError:
                 st.warning("NLTK WordNetLemmatizer not found. Skipping lemmatization.")
            except Exception:
                st.warning("Error with NLTK Lemmatizer. Skipping lemmatization.")

        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error during basic preprocessing: {str(e)}. Returning lowercased original text.")
        return text.lower() if isinstance(text, str) else "" # Return lowercased original text as last resort


def preprocess_keywords(keywords, use_advanced, spacy_nlp=None):
    """
    Main keyword preprocessing loop.
    Applies enhanced_preprocessing or preprocess_text based on options and availability.
    """
    processed_keywords = []
    progress_bar = st.progress(0)
    total = len(keywords)

    if use_advanced and (spacy_nlp is not None or textblob_available):
        st.info("Using advanced preprocessing (spaCy or TextBlob fallback).")
        preprocessing_func = lambda k: enhanced_preprocessing(k, True, spacy_nlp)
    else:
        st.info("Using standard NLTK preprocessing (advanced disabled or libraries not available).")
        preprocessing_func = lambda k: preprocess_text(k, True)

    # Use list comprehension for efficiency
    processed_keywords = [preprocessing_func(keyword) for i, keyword in enumerate(keywords)]

    # Update progress bar after processing, not inside the loop for better performance with Streamlit
    progress_bar.progress(1.0)
    st.success("✅ Preprocessing complete.")

    return processed_keywords
################################################################
#         EMBEDDING GENERATION
################################################################

@st.cache_data(show_spinner=False) # Cache embeddings based on dataframe content and API key/model
def generate_embeddings(df, openai_available, openai_api_key=None):
    """
    Generates embeddings using OpenAI, SentenceTransformers, or TF-IDF fallback.
    Includes logic to handle large datasets with OpenAI by processing a sample
    and propagating embeddings to the rest via TF-IDF similarity.
    """
    st.info("Generating embeddings for keywords...")

    embeddings = None # Initialize embeddings as None

    # Attempt OpenAI embeddings
    if openai_available and openai_api_key:
        try:
            st.info("Attempting to use OpenAI embeddings (high semantic precision)...")
            # Use the key directly, avoid os.environ if possible in Streamlit for security/state management
            client = OpenAI(api_key=openai_api_key)

            # Prepare keywords, ensuring no None/NaN values which can break APIs
            keywords = df['keyword_processed'].fillna('').tolist()
            if not any(keywords): # Check if all processed keywords are empty
                 st.warning("Processed keywords are all empty. Cannot generate embeddings.")
                 return generate_tfidf_embeddings(df['keyword_processed'], fallback_only=True) # Fallback immediately

            all_embeddings = []

            # If more than 5000 keywords, use partial approach
            # Note: The 5000 limit is arbitrary based on a common heuristic,
            #       the actual limit depends on the effectiveness of propagation.
            if len(keywords) > 5000:
                st.warning(f"Dataset size ({len(keywords)}) exceeds direct OpenAI processing limit (5000).")
                st.info("Processing a sample and propagating embeddings via TF-IDF similarity.")

                # Determine sample indices - take a distributed sample
                step = max(1, len(keywords) // 5000)
                sample_indices = list(range(0, len(keywords), step))[:5000]
                sample_keywords = [keywords[i] for i in sample_indices]

                progress_bar = st.progress(0)
                st.info("Requesting embeddings from OpenAI for sample...")

                # Ensure sample keywords are not empty strings before sending to API
                sample_keywords_clean = [kw for kw in sample_keywords if isinstance(kw, str) and kw.strip()]
                if not sample_keywords_clean:
                     st.error("Sample keywords for OpenAI are empty. Cannot proceed with OpenAI embeddings.")
                     return generate_tfidf_embeddings(df['keyword_processed'], fallback_only=True)


                # OpenAI Embedding API call
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=sample_keywords_clean
                    )
                    # Map embeddings back to original sample indices
                    sample_embeddings_dict = {sample_keywords_clean[i]: item.embedding for i, item in enumerate(response.data)}

                    sample_embeddings = np.array([sample_embeddings_dict.get(kw, np.zeros(len(response.data[0].embedding))) for kw in sample_keywords]) # Handle potential missing keys

                    progress_bar.progress(0.4)

                    st.info("Propagating embeddings to remaining keywords via TF-IDF similarity...")
                    # Use original keywords for TF-IDF, including those in the sample
                    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') # Limit features for performance
                    tfidf_matrix = vectorizer.fit_transform(keywords) # Fit on ALL keywords

                    all_embeddings = np.zeros((len(keywords), sample_embeddings.shape[1])) # Initialize full embedding matrix

                    # Assign direct embeddings to sample indices
                    for i, original_idx in enumerate(sample_indices):
                         if i < len(sample_embeddings): # Safety check
                            all_embeddings[original_idx] = sample_embeddings[i]

                    # Propagate embeddings for remaining indices
                    remaining_indices = [i for i in range(len(keywords)) if i not in sample_indices]

                    if remaining_indices:
                        # Train Nearest Neighbors on the TF-IDF vectors of the sample
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=min(5, len(sample_indices)), algorithm='auto') # Use auto algorithm
                        nn.fit(tfidf_matrix[sample_indices])

                        for i, original_idx in enumerate(remaining_indices):
                            # Find nearest neighbors in the sample TF-IDF space
                            # Handle potential empty strings that might result in zero vectors in tfidf
                            if keywords[original_idx].strip():
                                distances, neighbors = nn.kneighbors(tfidf_matrix[original_idx].reshape(1, -1))

                                # Use inverse distance weighting for propagation
                                # Add a small epsilon to avoid division by zero if distance is 0 (shouldn't happen with unique vectors but defensive)
                                weights = 1.0 / (1.0 + distances[0] + 1e-6)
                                weights = weights / weights.sum() # Normalize weights to sum to 1

                                # Calculate weighted average of neighbor embeddings
                                weighted_embedding = np.zeros_like(all_embeddings[original_idx])
                                for j, weight in zip(neighbors[0], weights):
                                     if j < len(sample_indices): # Safety check
                                         neighbor_original_idx = sample_indices[j]
                                         weighted_embedding += weight * all_embeddings[neighbor_original_idx]

                                all_embeddings[original_idx] = weighted_embedding
                            else:
                                # Assign a zero vector or handle appropriately if original keyword was empty
                                all_embeddings[original_idx] = np.zeros_like(all_embeddings[original_idx])


                            # Update progress bar - remaining indices portion
                            if i % 500 == 0: # Update less frequently for large datasets
                                prog_val = 0.4 + min(0.6, (i / len(remaining_indices)) * 0.6)
                                progress_bar.progress(prog_val)

                        progress_bar.progress(1.0)
                        st.success(f"✅ Propagated embeddings for {len(remaining_indices):,} keywords.")
                    else:
                        # All keywords were in the sample (dataset <= 5000)
                         progress_bar.progress(1.0)


                except Exception as api_call_error:
                     st.error(f"OpenAI API call failed during embedding generation: {str(api_call_error)}")
                     st.info("Falling back to SentenceTransformers or TF-IDF.")
                     embeddings = None # Ensure embeddings is None to trigger fallback


            else:
                # If under 5000 keywords, direct approach (batching for API limits/efficiency)
                progress_bar = st.progress(0)
                st.info(f"Requesting embeddings for all {len(keywords)} keywords from OpenAI...")
                batch_size = 1000 # Max input size for embedding model API call (approx tokens, but 1000 keywords is safe)
                all_embeddings = []

                # Ensure keywords are not empty strings
                keywords_clean = [kw for kw in keywords if isinstance(kw, str) and kw.strip()]
                if len(keywords_clean) != len(keywords):
                     st.warning(f"{len(keywords) - len(keywords_clean)} keywords were empty after preprocessing and will be skipped for embeddings.")

                if not keywords_clean:
                     st.error("All keywords are empty after preprocessing. Cannot generate OpenAI embeddings.")
                     return generate_tfidf_embeddings(df['keyword_processed'], fallback_only=True)


                # Collect embeddings with batching and progress
                embeddings_dict = {} # Use a dictionary to map keyword back to embedding robustly
                total_clean_keywords = len(keywords_clean)

                for i in range(0, total_clean_keywords, batch_size):
                    batch_end = min(i + batch_size, total_clean_keywords)
                    batch = keywords_clean[i:batch_end]

                    try:
                        response = client.embeddings.create(
                            model="text-embedding-3-small",
                            input=batch
                        )
                        for j, item in enumerate(response.data):
                             if item.index < len(batch): # Safety check
                                 keyword = batch[item.index]
                                 embeddings_dict[keyword] = item.embedding

                        progress_bar.progress(min(1.0, batch_end / total_clean_keywords))
                        time.sleep(0.1) # Small delay to be polite to the API and not spam Streamlit updates

                    except Exception as batch_error:
                        st.error(f"OpenAI API batch call failed at index {i}: {str(batch_error)}")
                        # Attempt to continue with next batches or break? For now, break and fallback.
                        embeddings = None # Trigger fallback
                        break

                if embeddings_dict: # If we have any embeddings
                     # Build the final embedding matrix, adding zero vectors for skipped or failed keywords
                     embedding_dimension = len(next(iter(embeddings_dict.values()))) if embeddings_dict else 1536 # Default dim for text-embedding-3-small
                     final_embeddings = np.zeros((len(keywords), embedding_dimension))
                     for i, keyword in enumerate(keywords):
                         if keyword in embeddings_dict:
                             final_embeddings[i] = embeddings_dict[keyword]
                         # else: it remains a zero vector

                     embeddings = np.array(final_embeddings) # Convert list of embeddings to numpy array
                     progress_bar.progress(1.0)
                     st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (OpenAI).")


        except Exception as e:
            st.error(f"An error occurred during OpenAI embedding process: {str(e)}")
            st.info("Falling back to SentenceTransformers or TF-IDF.")
            embeddings = None # Ensure embeddings is None to trigger fallback

    # Attempt SentenceTransformers if available and OpenAI embeddings failed or not attempted
    if embeddings is None and sentence_transformers_available:
        try:
            st.info("Using SentenceTransformer (free fallback)...")
            # Try to use locally cached models first
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
                st.success("Loaded SentenceTransformer from local cache.")
            except Exception as e_cache:
                st.warning(f"Could not load models from cache: {str(e_cache)[:100]}. Trying to download...")
                try:
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e_remote:
                    st.error(f"Error loading SentenceTransformer model: {str(e_remote)}")
                    st.info("Falling back to TF-IDF as embeddings.")
                    return generate_tfidf_embeddings(df['keyword_processed'].fillna(''))

            progress_bar = st.progress(0)
            keywords = df['keyword_processed'].fillna('').tolist()
            # Ensure keywords are not just empty strings
            keywords_clean = [kw for kw in keywords if isinstance(kw, str) and kw.strip()]

            if not keywords_clean:
                 st.error("All keywords are empty after preprocessing. Cannot generate SentenceTransformer embeddings.")
                 return generate_tfidf_embeddings(df['keyword_processed'], fallback_only=True)

            batch_size = 512 # Batch size for SentenceTransformers
            all_embeddings = []
            total_clean_keywords = len(keywords_clean)

            # Need to handle mapping embeddings back to original index if some were empty
            clean_keyword_to_original_index = {kw: [] for kw in keywords_clean}
            for i, kw in enumerate(keywords):
                 if isinstance(kw, str) and kw.strip():
                     clean_keyword_to_original_index[kw].append(i)

            try:
                # Encoding with progress bar feedback
                for i in range(0, total_clean_keywords, batch_size):
                    batch = keywords_clean[i:min(i+batch_size, total_clean_keywords)]
                    batch_embeddings = model.encode(batch, show_progress_bar=False)
                    all_embeddings.extend(batch_embeddings)
                    progress_bar.progress(min(1.0, (i + len(batch)) / total_clean_keywords))
                    time.sleep(0.1) # Small delay


                # Build the final embedding matrix, inserting embeddings at original indices
                embedding_dimension = all_embeddings[0].shape[0] if all_embeddings else 384 # Default dim for MiniLM models
                final_embeddings = np.zeros((len(keywords), embedding_dimension))
                current_embedding_idx = 0
                for keyword in keywords_clean:
                     if keyword in clean_keyword_to_original_index:
                         # There might be duplicate keywords in the original list, they will get the same embedding
                         original_indices = clean_keyword_to_original_index[keyword]
                         if original_indices and current_embedding_idx < len(all_embeddings):
                              for original_idx in original_indices:
                                   final_embeddings[original_idx] = all_embeddings[current_embedding_idx]
                         # Only advance the embedding index once per unique clean keyword
                         current_embedding_idx += 1


                embeddings = np.array(final_embeddings)

                progress_bar.progress(1.0)
                st.success(f"✅ Generated embeddings with {embeddings.shape[1]} dimensions (SentenceTransformers).")

            except Exception as encode_error:
                 st.error(f"Error encoding with SentenceTransformer: {str(encode_error)}")
                 st.info("Falling back to TF-IDF.")
                 embeddings = None # Trigger fallback


        except Exception as e:
            st.error(f"An error occurred during SentenceTransformer process: {str(e)}")
            st.info("Falling back to TF-IDF.")
            embeddings = None # Ensure embeddings is None to trigger fallback

    # Fallback to TF-IDF if no other embeddings were generated
    if embeddings is None:
         embeddings = generate_tfidf_embeddings(df['keyword_processed'])

    return embeddings


@st.cache_data(show_spinner=False) # Cache TF-IDF vectors based on processed text and params
def generate_tfidf_embeddings(texts, min_df=1, max_df=0.95, max_features=500, fallback_only=False):
    """
    Generates TF-IDF vectors as a fallback.
    Returns embeddings (dense numpy array).
    """
    if not fallback_only:
        st.info("Using TF-IDF as a fallback (less semantic precision than embeddings)...")

    progress_bar = st.progress(0)
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features, # Increased max features slightly
            min_df=min_df,
            max_df=max_df,
            stop_words='english' # Using standard English stopwords
        )
        # Ensure texts are strings, replace None/NaN with empty string
        clean_texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]

        if not any(clean_texts):
             st.error("TF-IDF input text is empty after cleaning. Cannot generate TF-IDF embeddings.")
             return np.random.rand(len(texts), 100) # Final random vector fallback

        progress_bar.progress(0.3)
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
        progress_bar.progress(0.8)

        embeddings = tfidf_matrix.toarray() # Convert sparse matrix to dense numpy array
        progress_bar.progress(1.0)

        st.success(f"✅ Generated {embeddings.shape[1]} TF-IDF vectors.")
        return embeddings
    except Exception as e:
        st.error(f"Error generating TF-IDF embeddings: {str(e)}")
        st.warning("Generating random vectors as a last resort.")
        random_embeddings = np.random.rand(len(texts), 100) # Generate 100 random dimensions
        return random_embeddings

################################################################
#         CLUSTERING ALGORITHMS
################################################################

@st.cache_data(show_spinner=False) # Cache clustering results based on embeddings and num_clusters
def improved_clustering(embeddings, num_clusters=None):
    """
    Applies KMeans clustering.
    num_clusters: The desired number of clusters.
    Returns cluster labels (1 to num_clusters).
    """
    st.info(f"Applying KMeans clustering to create {num_clusters} clusters...")
    try:
        from sklearn.cluster import KMeans
        # Ensure num_clusters is valid
        if num_clusters is None or num_clusters < 2:
             st.warning("Invalid number of clusters specified. Defaulting to 10.")
             num_clusters = 10
        if num_clusters > embeddings.shape[0]:
             st.warning(f"Number of clusters ({num_clusters}) is greater than the number of data points ({embeddings.shape[0]}). Setting clusters to number of data points.")
             num_clusters = embeddings.shape[0]
        if num_clusters < 2 and embeddings.shape[0] >= 2:
              # If num_clusters became < 2 due to data point count, set to 2 if possible
              num_clusters = 2
        if num_clusters < 2: # Still less than 2, cannot cluster
             st.warning("Less than 2 data points available for clustering.")
             return np.ones(embeddings.shape[0], dtype=int) # Assign all to cluster 1

        # Use more iterations for robustness
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20, max_iter=300) # Increased n_init
        labels = kmeans.fit_predict(embeddings) + 1 # Add 1 to make cluster IDs 1-based
        st.success(f"✅ KMeans clustering complete. Found {len(np.unique(labels))} clusters.")
        return labels
    except Exception as e:
        st.error(f"Error during KMeans clustering: {str(e)}")
        st.warning("Assigning all keywords to a single cluster.")
        # Fallback: assign all to cluster 1
        return np.ones(embeddings.shape[0], dtype=int)

def refine_clusters(df, embeddings, original_cluster_column='cluster_id'):
    """
    Placeholder function for potential cluster refinement (e.g., merging small clusters).
    Currently, it just returns the dataframe unchanged.
    """
    # st.info("Skipping cluster refinement (function is a placeholder)...") # Suppress if not implemented
    return df

################################################################
#         GENERATE CLUSTER NAMES
################################################################

# This function is NOT cached because it depends on the user's custom prompt
# and involves external API calls that shouldn't be cached across sessions/prompt changes.
def generate_cluster_names(
    clusters_with_representatives,
    client,
    model="gpt-3.5-turbo",
    custom_prompt=None
):
    """
    Generate SEO-friendly names and descriptions for clusters using OpenAI.
    Robustly handles JSON parsing and error recovery with retries.
    """
    if not clusters_with_representatives:
        st.warning("No clusters found to name.")
        return {}

    if not client:
        st.warning("OpenAI client not available. Cannot generate cluster names.")
        return {c_id: (f"Cluster {c_id}", f"Keywords group {c_id}") for c_id in clusters_with_representatives.keys()} # Return generic names

    results = {}
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Generating SEO-friendly cluster names/descriptions...")

    # Ensure default prompt provides clear instructions for JSON output
    default_prompt = (
        "You are an expert in SEO and content marketing. Below you'll see several clusters "
        "with a list of representative keywords. Your task is to assign each cluster a short, "
        "clear name (3-6 words) and write a concise SEO meta description (1 or 2 sentences), "
        "briefly explaining the topic and likely search intent."
    )
    # Use custom prompt if provided, otherwise use default
    effective_prompt_base = custom_prompt if custom_prompt and custom_prompt.strip() else default_prompt

    # Process clusters in smaller batches to avoid context limitations
    cluster_ids = list(clusters_with_representatives.keys())
    batch_size = 5  # Process 5 clusters at a time (can adjust based on token limits/model)

    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(cluster_ids))
        batch_cluster_ids = cluster_ids[batch_start:batch_end]

        # Create a simplified prompt for just this batch, explicitly asking for JSON
        batch_prompt_content = effective_prompt_base.strip() + "\n\n"
        batch_prompt_content += (
            "FOR EACH CLUSTER, provide:\n"
            "1. A clear, concise name (3-6 words)\n"
            "2. A brief description (1-2 sentences)\n\n"
            "FORMAT YOUR RESPONSE AS A JSON OBJECT WITH A TOP-LEVEL KEY 'clusters'. "
            "The value of 'clusters' should be an array of objects, each with keys 'cluster_id', 'cluster_name', and 'cluster_description'. "
            "Include ONLY the JSON in your response. Do not include any other text.\n\n"
            "Here are the clusters:\n"
        )

        for cluster_id in batch_cluster_ids:
            sample_kws = clusters_with_representatives.get(cluster_id, [])[:15] # Limit to 15 keywords
            # Ensure sample keywords are strings and not empty
            sample_kws_clean = [str(kw) for kw in sample_kws if kw is not None] # Convert all to strings
            sample_kws_clean = [kw for kw in sample_kws_clean if kw.strip()] # Filter empty strings
            batch_prompt_content += f"- Cluster {cluster_id}: {', '.join(sample_kws_clean)}\n"

        num_retries = 3
        batch_results = {} # Store results for the current batch

        for attempt in range(num_retries):
            try:
                progress_text.text(f"Generating names for clusters {batch_start+1}-{batch_end} (attempt {attempt+1}/{num_retries})...")

                # Try API call with error handling and JSON response format preference
                response = None
                content = ""
                
                try:
                    # Attempt with response_format={"type": "json_object"} if the model supports it
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": batch_prompt_content}],
                        temperature=0.3,
                        response_format={"type": "json_object"}, # Prefer JSON object output
                        max_tokens=1500 # Sufficient tokens for the expected JSON
                    )
                    content = response.choices[0].message.content.strip()
                except Exception as e_json_format:
                    # Fallback without response_format if the model doesn't support it or other API error
                    st.warning(f"Model {model} might not support JSON response format or API error: {str(e_json_format)[:100]}. Falling back to text response with JSON request.")
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": batch_prompt_content + "\nRespond strictly with valid JSON only."}],
                            temperature=0.3,
                            max_tokens=1500
                        )
                        content = response.choices[0].message.content.strip()
                    except Exception as e_fallback:
                        st.error(f"Fallback API call also failed: {str(e_fallback)[:100]}")
                        if attempt == num_retries - 1:
                            # On last retry, use generic names
                            for c_id in batch_cluster_ids:
                                batch_results[c_id] = (f"Cluster {c_id}", f"Keywords group {c_id}")
                        continue # Skip to next retry attempt if API call failed

                # Try to extract JSON from markdown code blocks if present
                json_pattern = r'```json\s*([\s\S]*?)\s*```' # Look specifically for ```json ... ```
                json_matches = re.findall(json_pattern, content)

                if json_matches:
                    content = json_matches[0]  # Take the first JSON code block

                # Try to parse JSON
                try:
                    json_data = json.loads(content)

                    if "clusters" in json_data and isinstance(json_data["clusters"], list):
                        for item in json_data["clusters"]:
                            # Robustly extract cluster_id, name, and description
                            c_id_raw = item.get("cluster_id")
                            if c_id_raw is not None:
                                try:
                                    # Clean and convert cluster_id to integer
                                    c_id_clean_str = ''.join(filter(str.isdigit, str(c_id_raw).strip()))
                                    if c_id_clean_str:
                                        c_id = int(c_id_clean_str)
                                        # Check if this cluster ID was in the batch we requested names for
                                        if c_id in batch_cluster_ids:
                                            c_name = str(item.get("cluster_name", f"Cluster {c_id}")) # Ensure string
                                            c_desc = str(item.get("cluster_description", "No description provided")) # Ensure string
                                            batch_results[c_id] = (c_name, c_desc)
                                        else:
                                            st.warning(f"Received cluster ID {c_id} in API response but it was not in the requested batch. Skipping.")
                                    else:
                                        st.warning(f"API returned invalid/non-numeric cluster_id: '{c_id_raw}'. Skipping.")
                                except (ValueError, TypeError) as e_id_conv:
                                    st.warning(f"Error converting cluster_id '{c_id_raw}' to int: {str(e_id_conv)}. Skipping this item.")
                                except Exception as e_item:
                                     st.warning(f"Error processing API response item for cluster ID {c_id_raw}: {str(e_item)}. Skipping this item.")

                        # If we got *some* good results from the batch, consider it successful and break the retry loop
                        if batch_results:
                            break # Break out of retry loop
                        elif attempt == num_retries - 1:
                             st.warning(f"Last attempt failed to extract valid cluster data from JSON for clusters {batch_start+1}-{batch_end}. Using generic names for this batch.")
                             # On last attempt failure, generate generic names for this batch
                             for c_id in batch_cluster_ids:
                                  if c_id not in batch_results: # Only add generic if it wasn't successfully parsed
                                       batch_results[c_id] = (f"Cluster {c_id}", f"Keywords group {c_id}")
                                       
                except json.JSONDecodeError as e_json:
                    st.warning(f"API response is not valid JSON (Attempt {attempt+1}): {str(e_json)}. Content: {content[:200]}... Trying regex fallback.")
                    # Fallback to regex extraction if JSON parsing fails
                    for cluster_id in batch_cluster_ids:
                        # Look for patterns like "cluster_id": 1, "cluster_name": "..."
                        name_match = re.search(rf'"cluster_id"\s*:\s*{cluster_id},\s*"cluster_name"\s*:\s*"([^"]+)"', content, re.DOTALL)
                        desc_match = re.search(rf'"cluster_id"\s*:\s*{cluster_id},\s*.*?"cluster_description"\s*:\s*"([^"]+)"', content, re.DOTALL)

                        if name_match:
                            c_name = name_match.group(1).strip() 
                            c_desc = desc_match.group(1).strip() if desc_match else f"Group of related keywords (cluster {cluster_id})"
                            batch_results[cluster_id] = (c_name, c_desc)
                        elif attempt == num_retries - 1:
                            # On last attempt, if regex also failed, add generic names for this cluster
                             batch_results[cluster_id] = (f"Cluster {cluster_id}", f"Keywords group {cluster_id}")

                    if batch_results and len(batch_results) == len(batch_cluster_ids): # If regex got all of them for this batch
                         break # Break retry loop if regex extraction seems successful for the whole batch
                    elif attempt == num_retries - 1:
                         st.warning(f"Regex fallback also failed for some clusters in batch {batch_start+1}-{batch_end}. Using generic names for remaining.")
                         for c_id in batch_cluster_ids:
                               if c_id not in batch_results:
                                    batch_results[c_id] = (f"Cluster {c_id}", f"Keywords group {c_id}")

                except Exception as api_error:
                    st.error(f"An unexpected error occurred during API processing (Attempt {attempt+1}): {str(api_error)[:100]}...")
                    if attempt == num_retries - 1:
                         st.warning(f"Final attempt failed for clusters {batch_start+1}-{batch_end}. Using generic names.")
                         # On last attempt, generate generic names for this batch
                         for c_id in batch_cluster_ids:
                              if c_id not in batch_results:
                                   batch_results[c_id] = (f"Cluster {c_id}", f"Keywords group {c_id}")

            except Exception as outer_error:
                st.error(f"An outer error occurred during batch processing (Attempt {attempt+1}): {str(outer_error)[:100]}...")
                if attempt < num_retries - 1:
                    time.sleep(2) # Wait a bit before retrying
                else:
                     st.warning(f"All attempts failed for clusters {batch_start+1}-{batch_end}. Using generic names.")
                     # On last attempt, generate generic names for this batch if no results were obtained
                     if not batch_results:
                          for c_id in batch_cluster_ids:
                               batch_results[c_id] = (f"Cluster {c_id}", f"Keywords group {c_id}")

        # Add batch results to overall results
        results.update(batch_results)

        # Update progress
        progress_bar.progress(min(1.0, (batch_end) / len(cluster_ids)))

    # Final check: ensure all requested cluster IDs have a name, even if generic
    for c_id in cluster_ids:
         if c_id not in results:
              st.warning(f"Cluster ID {c_id} did not receive a name from the API after retries. Assigning generic name.")
              results[c_id] = (f"Cluster {c_id}", f"Keywords group {c_id}")

    progress_bar.progress(1.0)
    progress_text.text("✅ Cluster naming completed.")
    return results
################################################################
#         SEARCH INTENT CLASSIFICATION
################################################################

def extract_features_for_intent(keyword, search_intent_description=""):
    """
    Extracts features for search intent classification based on keyword patterns
    and potentially a description.
    Returns a dictionary of boolean and count features.
    """
    if not isinstance(keyword, str) or not keyword.strip():
         return {f: False for f in [
             "has_informational_prefix", "has_navigational_prefix", "has_transactional_prefix", "has_commercial_prefix",
             "has_informational_suffix", "has_navigational_suffix", "has_transactional_suffix", "has_commercial_suffix",
             "is_informational_exact_match", "is_navigational_exact_match", "is_transactional_exact_match", "is_commercial_exact_match",
             "modal_verbs", "local_intent", "includes_price_modifier", "includes_product_modifier", "includes_brand"
         ]} | {f: 0 for f in [
             "informational_pattern_matches", "navigational_pattern_matches", "transactional_pattern_matches", "commercial_pattern_matches"
         ]} # Return zero/False features for empty keywords

    keyword_lower = keyword.lower()
    words = keyword_lower.split()

    features = {
        # Initialize all relevant features to False or 0
        "keyword_length": len(words),
        "keyword_lower": keyword_lower, # Keep lower for reference
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
        "includes_brand": False, # Requires a brand list lookup, omitted for now but kept as feature placeholder
        "includes_product_modifier": False,
        "includes_price_modifier": False,
        "local_intent": False,
        "modal_verbs": False
    }

    # Check prefixes
    if words:
        first_word = words[0]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if "prefixes" in patterns and any(first_word == prefix.lower() for prefix in patterns["prefixes"]):
                features[f"has_{intent_type.lower()}_prefix"] = True

    # Check suffixes
    if words and len(words) > 1:
        last_word = words[-1]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
             if "suffixes" in patterns and any(last_word == suffix.lower() for suffix in patterns["suffixes"]):
                 features[f"has_{intent_type.lower()}_suffix"] = True

    # Check exact matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        if "exact_matches" in patterns:
            for exact_match in patterns["exact_matches"]:
                if exact_match.lower() in keyword_lower:
                    features[f"is_{intent_type.lower()}_exact_match"] = True
                    # Optimization: if an exact match is found, no need to check other exact matches for this intent
                    break

    # Check pattern matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        if "keyword_patterns" in patterns:
            match_count = 0
            for pattern in patterns["keyword_patterns"]:
                # Use re.search for pattern matching anywhere in the string
                if re.search(pattern, keyword_lower):
                    match_count += 1
            features[f"{intent_type.lower()}_pattern_matches"] = match_count

    # Additional features
    # Use regex for more robust matching of phrases
    features["local_intent"] = bool(re.search(r'\bnear me\b|\bnearby\b|\bin my area\b|\bclose to me\b|\bclosest\b|\blocal\b', keyword_lower))
    features["modal_verbs"] = any(modal in words for modal in ["can", "could", "should", "would", "will", "may", "might"])
    features["includes_price_modifier"] = bool(re.search(r'\bprice\b|\bcost\b|\bcheap\b|\bexpensive\b|\baffordable\b|\bdiscount\b|\boffer\b|\bdeal\b|\bcoupon\b|\bpricing\b', keyword_lower))
    features["includes_product_modifier"] = bool(re.search(r'\bbest\b|\btop\b|\bcheap\b|\bpremium\b|\bquality\b|\bnew\b|\bused\b|\brefurbished\b|\balternative\b', keyword_lower))

    # Brand detection would require a list of brands to check against the keyword.
    # This feature is currently not implemented but kept as a placeholder.
    # features["includes_brand"] = check_for_brand(keyword_lower, list_of_known_brands) # Placeholder

    return features

def classify_search_intent_ml(keywords, search_intent_description="", cluster_name=""):
    """
    Enhanced search intent classification using a weighted feature scoring approach
    based on patterns and optional AI-generated descriptions/names.
    """
    # Ensure keywords are all strings
    keywords = [str(kw) if kw is not None else "" for kw in keywords]
    
    # Use a sample of keywords for intent classification if the cluster is very large
    sample_keywords = keywords[:min(len(keywords), 50)] # Increased sample size slightly

    if not sample_keywords:
        return {
            "primary_intent": "Unknown",
            "scores": {
                "Informational": 25.0,
                "Navigational": 25.0,
                "Transactional": 25.0,
                "Commercial": 25.0
            },
            "evidence": {}
        }

    # Aggregate features across the sample keywords
    aggregated_features = {
        "Informational": set(),
        "Navigational": set(),
        "Transactional": set(),
        "Commercial": set()
    }

    for keyword in sample_keywords:
        features = extract_features_for_intent(keyword, search_intent_description)

        # Map features to intent types
        if features.get("has_informational_prefix"): aggregated_features["Informational"].add("Informational Prefix")
        if features.get("has_informational_suffix"): aggregated_features["Informational"].add("Informational Suffix")
        if features.get("is_informational_exact_match"): aggregated_features["Informational"].add("Informational Exact Match")
        if features.get("informational_pattern_matches") > 0: aggregated_features["Informational"].add(f"Matches {features['informational_pattern_matches']} Info Patterns")
        if features.get("modal_verbs"): aggregated_features["Informational"].add("Includes Modal Verb (Question)")

        if features.get("has_navigational_prefix"): aggregated_features["Navigational"].add("Navigational Prefix")
        if features.get("has_navigational_suffix"): aggregated_features["Navigational"].add("Navigational Suffix")
        if features.get("is_navigational_exact_match"): aggregated_features["Navigational"].add("Navigational Exact Match")
        if features.get("navigational_pattern_matches") > 0: aggregated_features["Navigational"].add(f"Matches {features['navigational_pattern_matches']} Nav Patterns")
        if features.get("includes_brand"): aggregated_features["Navigational"].add("Includes Brand Name") # Placeholder

        if features.get("has_transactional_prefix"): aggregated_features["Transactional"].add("Transactional Prefix")
        if features.get("has_transactional_suffix"): aggregated_features["Transactional"].add("Transactional Suffix")
        if features.get("is_transactional_exact_match"): aggregated_features["Transactional"].add("Transactional Exact Match")
        if features.get("transactional_pattern_matches") > 0: aggregated_features["Transactional"].add(f"Matches {features['transactional_pattern_matches']} Trans Patterns")
        if features.get("includes_price_modifier"): aggregated_features["Transactional"].add("Includes Price Modifier")
        if features.get("local_intent"): aggregated_features["Transactional"].add("Local Intent")

        if features.get("has_commercial_prefix"): aggregated_features["Commercial"].add("Commercial Prefix")
        if features.get("has_commercial_suffix"): aggregated_features["Commercial"].add("Commercial Suffix")
        if features.get("is_commercial_exact_match"): aggregated_features["Commercial"].add("Commercial Exact Match")
        if features.get("commercial_pattern_matches") > 0: aggregated_features["Commercial"].add(f"Matches {features['commercial_pattern_matches']} Comm Patterns")
        if features.get("includes_product_modifier"): aggregated_features["Commercial"].add("Includes Product Modifier")


    # Calculate raw scores based on unique aggregated signals and weights
    info_score = len(aggregated_features["Informational"]) * SEARCH_INTENT_PATTERNS["Informational"]["weight"]
    nav_score = len(aggregated_features["Navigational"]) * SEARCH_INTENT_PATTERNS["Navigational"]["weight"]
    trans_score = len(aggregated_features["Transactional"]) * SEARCH_INTENT_PATTERNS["Transactional"]["weight"]
    comm_score = len(aggregated_features["Commercial"]) * SEARCH_INTENT_PATTERNS["Commercial"]["weight"]

    # Boost scores based on signals in the AI-generated description or cluster name
    if search_intent_description and isinstance(search_intent_description, str):
        desc_lower = search_intent_description.lower()
        if re.search(r'\binformational\b|\binformation\s+intent\b|\binformation\s+search\b|\blearning\b|\bquestion\b', desc_lower):
             info_score += 5
             aggregated_features["Informational"].add("Description Hint")
        if re.search(r'\bnavigational\b|\bnavigate\b|\bfind\s+\w+\s+website\b|\bfind\s+\w+\s+page\b|\baccess\b', desc_lower):
             nav_score += 5
             aggregated_features["Navigational"].add("Description Hint")
        if re.search(r'\btransactional\b|\bbuy\b|\bpurchase\b|\bshopping\b|\bsale\b|\btransaction\b|\bget\s+quote\b|\bsign\s+up\b', desc_lower):
             trans_score += 5
             aggregated_features["Transactional"].add("Description Hint")
        if re.search(r'\bcommercial\b|\bcompar(e|ing|ison)\b|\breview\b|\balternative\b|\bbest\b|\btop\b|\bproduct\s+review\b', desc_lower):
             comm_score += 5
             aggregated_features["Commercial"].add("Description Hint")

    if cluster_name and isinstance(cluster_name, str):
        name_lower = cluster_name.lower()
        if re.search(r'\bhow\b|\bwhat\b|\bwhy\b|\bwhen\b|\bguide\b|\btutorial\b|\binfo\b', name_lower):
            info_score += 3
            aggregated_features["Informational"].add("Name Hint")
        if re.search(r'\bwebsite\b|\bofficial\b|\blogin\b|\bportal\b|\bdownload\b|\baccount\b', name_lower):
            nav_score += 3
            aggregated_features["Navigational"].add("Name Hint")
        if re.search(r'\bbuy\b|\bshop\b|\bpurchase\b|\bsale\b|\bdiscount\b|\bcost\b|\bprice\b|\bpricing\b', name_lower):
            trans_score += 3
            aggregated_features["Transactional"].add("Name Hint")
        if re.search(r'\bbest\b|\btop\b|\breview\b|\bcompare\b|\bvs\b|\balternative\b', name_lower):
            comm_score += 3
            aggregated_features["Commercial"].add("Name Hint")


    # Normalize scores to percentages
    total_score = max(1.0, info_score + nav_score + trans_score + comm_score) # Avoid division by zero
    info_pct = (info_score / total_score) * 100
    nav_pct = (nav_score / total_score) * 100
    trans_pct = (trans_score / total_score) * 100
    comm_pct = (comm_score / total_score) * 100

    # Prepare scores dictionary, rounded for display
    scores = {
        "Informational": round(info_pct, 2),
        "Navigational": round(nav_pct, 2),
        "Transactional": round(trans_pct, 2),
        "Commercial": round(comm_pct, 2)
    }

    # Determine primary intent
    primary_intent = max(scores, key=scores.get)

    # If the highest score is below a threshold (e.g., 40%) or multiple scores are very close,
    # consider it mixed intent.
    max_score = max(scores.values())
    # Check for multiple intents close to the max score
    close_intents = [intent for intent, score in scores.items() if max_score - score < 15] # Within 15 points of the max
    if max_score < 40 or len(close_intents) > 1:
        primary_intent = "Mixed Intent"

    # Collect evidence (unique signals)
    evidence = {intent: list(signals) for intent, signals in aggregated_features.items()}

    return {
        "primary_intent": primary_intent,
        "scores": scores,
        "evidence": evidence
    }

def analyze_cluster_for_intent_flow(df, cluster_id):
    """
    Analyzes the intent distribution within a cluster to suggest customer journey phase.
    """
    # Get keywords for this cluster
    cluster_keywords_df = df[df['cluster_id'] == cluster_id]

    if cluster_keywords_df.empty:
        return None

    # Classify each keyword individually
    keyword_intents = []
    # Limit processing to a sample for performance in large clusters
    keyword_sample_for_analysis = cluster_keywords_df['keyword'].sample(min(len(cluster_keywords_df), 50), random_state=42).tolist() if len(cluster_keywords_df) > 50 else cluster_keywords_df['keyword'].tolist()

    # Ensure all keywords are strings
    keyword_sample_for_analysis = [str(kw) if kw is not None else "" for kw in keyword_sample_for_analysis]

    for keyword in keyword_sample_for_analysis:
         # Use a simplified classification here if classifying every keyword is too slow,
         # or rely on the main classify_search_intent_ml which already takes a sample.
         # Let's reuse the main classifier on individual keywords, which samples features internally.
         intent_data = classify_search_intent_ml([keyword]) # Pass as list to match expected input
         keyword_intents.append({
             "keyword": keyword,
             "primary_intent": intent_data["primary_intent"],
             "scores": intent_data["scores"]
         })

    if not keyword_intents:
         return None # Should not happen if cluster_keywords_df is not empty, but defensive

    # Calculate distribution of primary intents within the sample
    intent_counts = Counter([item["primary_intent"] for item in keyword_intents])
    total_sample = len(keyword_intents)

    # Calculate average scores across the sample keywords
    avg_scores = {
        "Informational": sum(item["scores"].get("Informational", 0) for item in keyword_intents) / total_sample if total_sample > 0 else 0,
        "Navigational": sum(item["scores"].get("Navigational", 0) for item in keyword_intents) / total_sample if total_sample > 0 else 0,
        "Transactional": sum(item["scores"].get("Transactional", 0) for item in keyword_intents) / total_sample if total_sample > 0 else 0,
        "Commercial": sum(item["scores"].get("Commercial", 0) for item in keyword_intents) / total_sample if total_sample > 0 else 0
    }


    # Analyze if this represents a customer journey phase based on percentage distribution
    # Typically, customer journey: Info -> Commercial -> Transactional
    # Thresholds are heuristic and can be adjusted
    info_pct = (intent_counts.get("Informational", 0) / total_sample) * 100 if total_sample > 0 else 0
    comm_pct = (intent_counts.get("Commercial", 0) / total_sample) * 100 if total_sample > 0 else 0
    trans_pct = (intent_counts.get("Transactional", 0) / total_sample) * 100 if total_sample > 0 else 0
    nav_pct = (intent_counts.get("Navigational", 0) / total_sample) * 100 if total_sample > 0 else 0 # Include nav for completeness

    journey_phase = "Unknown"
    if info_pct > max(comm_pct, trans_pct, nav_pct) and info_pct > 40: # Predominantly Informational
        journey_phase = "Early (Awareness/Research)"
    elif comm_pct > max(info_pct, trans_pct, nav_pct) and comm_pct > 40: # Predominantly Commercial
        journey_phase = "Middle (Consideration)"
    elif trans_pct > max(info_pct, comm_pct, nav_pct) and trans_pct > 40: # Predominantly Transactional
        journey_phase = "Late (Decision/Purchase)"
    elif nav_pct > max(info_pct, comm_pct, trans_pct) and nav_pct > 40: # Predominantly Navigational
         journey_phase = "Specific Destination Seeking" # Navigational is less about journey stage, more direct access

    # Handle mixed phases
    elif info_pct > 20 and comm_pct > 20:
        journey_phase = "Awareness/Research to Consideration Transition"
    elif comm_pct > 20 and trans_pct > 20:
        journey_phase = "Consideration to Decision/Purchase Transition"
    elif info_pct > 20 and trans_pct > 20:
         journey_phase = "Mixed (Research and Purchase Interest)" # Less common transition directly
    else:
        journey_phase = "Mixed/Unclear Journey Stage" # Default if no clear pattern

    return {
        "intent_distribution": {intent: round((count / total_sample) * 100, 2) for intent, count in intent_counts.items()} if total_sample > 0 else {},
        "avg_scores": {intent: round(score, 2) for intent, score in avg_scores.items()},
        "journey_phase": journey_phase,
        "keyword_sample": [{"keyword": k["keyword"], "intent": k["primary_intent"]} for k in keyword_intents[:10]] # Limit example keywords
    }

################################################################
#         EVALUATION FUNCTIONS
################################################################

# This function does not need caching as it operates on the provided df and embeddings
def evaluate_cluster_quality(df, embeddings, cluster_column='cluster_id'):
    """
    Assigns a 'cluster_coherence' score based on distances within clusters.
    """
    st.subheader("Cluster Quality Evaluation")

    if embeddings is None or embeddings.shape[0] == 0:
         st.warning("No embeddings available for coherence calculation.")
         df['cluster_coherence'] = 0.0
         return df

    # Create a copy to avoid modifying the original DataFrame in place unexpectedly if df is from cache
    df_evaluated = df.copy()
    df_evaluated['cluster_coherence'] = 0.0  # Default value for clusters with < 2 items

    # Get unique clusters
    unique_clusters = df_evaluated[cluster_column].unique()

    if len(unique_clusters) <= 1:
         st.info("Only one cluster or no clusters found. Skipping coherence calculation.")
         return df_evaluated # Return with default 0.0 coherence

    st.info(f"Calculating cluster coherence scores for {len(unique_clusters)} clusters...")
    try:
        progress_bar = st.progress(0)

        for i, cluster_id in enumerate(unique_clusters):
            # Get indices for this cluster in the original DataFrame
            cluster_indices_in_df = df_evaluated[df_evaluated[cluster_column] == cluster_id].index.tolist()

            if len(cluster_indices_in_df) > 1:  # Need at least 2 points for coherence
                # Get embeddings for this cluster using the original indices
                cluster_embeddings = embeddings[cluster_indices_in_df]

                # Calculate coherence (using cosine similarity to centroid)
                coherence = calculate_cluster_coherence(cluster_embeddings)

                # Assign to all rows in this cluster using .loc for reliable assignment
                df_evaluated.loc[cluster_indices_in_df, 'cluster_coherence'] = coherence

            progress_bar.progress(min(1.0, (i + 1) / len(unique_clusters)))

        progress_bar.progress(1.0)
        st.success(f"✅ Coherence scores calculated for {len(unique_clusters)} clusters.")
    except Exception as e:
        st.error(f"Error calculating coherence: {str(e)}")
        st.warning("Using default coherence value of 0.0")
        df_evaluated['cluster_coherence'] = 0.0

    return df_evaluated

def calculate_cluster_coherence(cluster_embeddings):
    """
    Calculate coherence score based on average cosine similarity to the cluster centroid.
    Returns a score between 0 and 1.
    """
    if cluster_embeddings.shape[0] < 2:
        return 0.0 # Coherence is not well-defined for a single point

    try:
        # Calculate mean embedding (centroid)
        centroid = np.mean(cluster_embeddings, axis=0)

        # Handle potential zero vector centroid (e.g., if all embeddings were zero vectors)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-6: # Use a small threshold
            return 0.0 # Cannot calculate similarity with a zero vector centroid

        centroid_normalized = centroid / centroid_norm

        # Calculate cosine similarity between each point and the centroid
        similarities = []
        for embedding in cluster_embeddings:
            # Normalize the embedding
            emb_norm = np.linalg.norm(embedding)
            if emb_norm < 1e-6:
                 # Assign 0 similarity if embedding is a zero vector
                 similarity = 0.0
            else:
                embedding_normalized = embedding / emb_norm
                # Calculate similarity - np.dot is efficient for normalized vectors
                similarity = np.dot(embedding_normalized, centroid_normalized)

            similarities.append(similarity)

        # Return average similarity. Cosine similarity is in [-1, 1].
        # An average near 1 means high coherence. Average near 0 or negative means low coherence.
        average_similarity = np.mean(similarities)

        # Scale to a 0-1 range. A simple scaling (sim + 1) / 2 maps [-1, 1] to [0, 1].
        coherence = (average_similarity + 1) / 2.0

        # Ensure it's within the 0-1 range due to potential floating point inaccuracies
        coherence = max(0.0, min(1.0, coherence))

        return coherence
    except Exception as e:
        st.error(f"Error calculating individual cluster coherence: {str(e)}. Returning 0.0.")
        return 0.0

################################################################
#         CLUSTER SEMANTIC ANALYSIS (AI-POWERED)
################################################################

# This function is NOT cached because it depends on the user's custom prompt
# and involves external API calls that shouldn't be cached across sessions/prompt changes.
def generate_semantic_analysis(
    clusters_with_representatives,
    client,
    model="gpt-3.5-turbo",
    custom_prompt=None # Allow adding to the prompt for analysis focus
):
    """
    Calls OpenAI to analyze each cluster for:
      1) Main search intent
      2) Suggestion of internal splitting with specific subclusters
      3) Additional SEO-focused insights
      4) Coherence score (based on AI assessment, not calculated metric)
    Returns a dictionary of analysis results by cluster ID.
    """
    results = {}
    if not clusters_with_representatives:
        st.warning("No clusters found for semantic analysis.")
        return results

    if not client:
        st.info("No OpenAI client provided. Skipping AI-based semantic analysis.")
        return results

    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Performing semantic analysis on clusters using OpenAI...")

    # Base prompt instructions
    base_prompt_instructions = (
        "You are an expert in SEO and keyword research analysis. Analyze each keyword cluster below and provide insights relevant for SEO strategy. For EACH cluster, provide the following analysis in the specified JSON format:\n"
        "1.  **Search Intent**: Describe the primary user motivation behind these searches (e.g., Informational, Commercial, Transactional, Navigational, or Mixed). Explain *why* based on the keywords.\n"
        "2.  **Split Suggestion**: Indicate if the cluster seems broad enough to be split into smaller, distinct subclusters ('Yes' or 'No'). If 'Yes', briefly suggest 2-3 logical subcluster themes based on prominent keyword groups within the cluster.\n"
        "3.  **SEO Insights**: Provide brief SEO-focused commentary, such as potential content types (blog post, landing page, product page, category page), keyword difficulty considerations (high, medium, low, based on keyword patterns), or other strategic notes.\n"
        "4.  **Coherence Score**: Assign a score from 0-10 assessing how well the keywords in the cluster relate to each other semantically, where 10 is perfectly coherent.\n\n"
        "FORMAT YOUR RESPONSE AS A JSON OBJECT WITH A TOP-LEVEL KEY 'clusters'. The value of 'clusters' should be an array of objects. Each object in the array should have these keys: 'cluster_id' (integer), 'search_intent' (string), 'split_suggestion' (string, 'Yes' or 'No'), 'additional_info' (string), 'coherence_score' (integer 0-10), and optionally 'subclusters' (array of objects with 'name' and 'keywords' - include this ONLY if 'split_suggestion' is 'Yes').\n"
        "Include ONLY the JSON in your response. Do not include any other text.\n\n"
        "Here are the clusters to analyze:\n"
    )

    # Append custom prompt instructions if provided
    effective_prompt_base = base_prompt_instructions
    if custom_prompt and custom_prompt.strip():
        effective_prompt_base += f"Additional instructions: {custom_prompt.strip()}\n\n"

    # Process clusters in smaller batches
    cluster_ids = list(clusters_with_representatives.keys())
    batch_size = 5  # Process 5 clusters at a time

    # Initialize a dictionary to hold results, starting with default/fallback values
    # This ensures we have an entry for every cluster even if API calls fail
    analysis_results = {c_id: {
        "search_intent_api": "API analysis failed or not attempted", # Changed default message
        "split_suggestion": "API analysis failed or not attempted",
        "additional_info": "API analysis failed or not attempted",
        "coherence_score_api": 5, # Neutral score
        "subclusters": [],
        "intent_classification_ml": classify_search_intent_ml(
             # Ensure we convert all items to strings to avoid float.split() error
             [str(kw) if kw is not None else "" for kw in clusters_with_representatives.get(c_id, [])], 
             "API analysis failed or not attempted", # Pass failure message as context
             f"Cluster {c_id}"
         ) # Default ML classification fallback
    } for c_id in cluster_ids}

    # Batch processing for API calls
    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(cluster_ids))
        batch_cluster_ids = cluster_ids[batch_start:batch_end]

        # Process this batch
        batch_analysis_results_from_api = {}
        num_retries = 3
        
        for attempt in range(num_retries):
            try:
                progress_text.text(f"Analyzing clusters {batch_start+1}-{batch_end} (attempt {attempt+1}/{num_retries})...")

                # Create prompt for this batch
                batch_prompt_content = effective_prompt_base.strip() + "\n\n"
                for cluster_id in batch_cluster_ids:
                    # Get sample keywords, ensuring they are strings and not empty
                    sample_kws = clusters_with_representatives.get(cluster_id, [])[:15] # Limit sample keywords
                    sample_kws_clean = [str(kw) for kw in sample_kws if kw is not None] # Convert all to strings
                    sample_kws_clean = [kw for kw in sample_kws_clean if kw.strip()] # Filter empty strings
                    batch_prompt_content += f"Cluster {cluster_id}: {', '.join(sample_kws_clean)}\n"

                # Try API call with error handling
                try:
                    # Try with response_format parameter
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": batch_prompt_content}],
                            temperature=0.3, # Use lower temperature for more consistent output
                            response_format={"type": "json_object"},
                            max_tokens=2500 # Allow more tokens for analysis details
                        )
                        content = response.choices[0].message.content.strip()
                    except Exception as e_json_format:
                         # Fallback without response_format
                         st.warning(f"Model {model} might not support JSON response format or API error: {str(e_json_format)[:100]}. Falling back to text response with JSON request.")
                         response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": batch_prompt_content + "\nRespond strictly with valid JSON only."}],
                            temperature=0.3,
                            max_tokens=2500
                        )
                        content = response.choices[0].message.content.strip()

                    # Try to extract JSON from markdown code blocks
                    json_pattern = r'```json\s*([\s\S]*?)\s*```'
                    json_matches = re.findall(json_pattern, content)

                    if json_matches:
                        content = json_matches[0] # Take the first JSON block

                    # Try to parse JSON
                    try:
                        json_data = json.loads(content)

                        if "clusters" in json_data and isinstance(json_data["clusters"], list):
                            for item in json_data["clusters"]:
                                c_id_raw = item.get("cluster_id")
                                if c_id_raw is not None:
                                    try:
                                        # Clean and convert cluster_id to integer
                                        c_id_clean_str = ''.join(filter(str.isdigit, str(c_id_raw).strip()))
                                        if c_id_clean_str:
                                             c_id = int(c_id_clean_str)
                                             # Check if this ID was part of the requested batch
                                             if c_id in batch_cluster_ids:
                                                 # Extract analysis details, providing default empty strings/values
                                                 search_intent = str(item.get("search_intent", "No intent analysis provided by API")) 
                                                 split_suggestion = str(item.get("split_suggestion", "No split suggestion provided by API")) 
                                                 additional_info = str(item.get("additional_info", "No SEO insights provided by API")) 
                                                 
                                                 # Safely convert coherence score, default to 5 if conversion fails or out of range
                                                 coherence_score_raw = item.get("coherence_score")
                                                 try:
                                                      coherence_score = int(coherence_score_raw) if coherence_score_raw is not None else 5
                                                      coherence_score = max(0, min(10, coherence_score)) # Clamp to 0-10 range
                                                 except (ValueError, TypeError):
                                                      st.warning(f"API returned invalid coherence_score '{coherence_score_raw}' for cluster {c_id}. Defaulting to 5.")
                                                      coherence_score = 5

                                                 subclusters = item.get("subclusters", []) # Expecting a list
                                                 
                                                 # Ensure subclusters are properly formatted and have string keywords
                                                 clean_subclusters = []
                                                 for sub in subclusters:
                                                     if isinstance(sub, dict):
                                                         sub_name = str(sub.get("name", "Unnamed Subcluster"))
                                                         sub_keywords = [str(kw) for kw in sub.get("keywords", []) if kw is not None]
                                                         clean_subclusters.append({"name": sub_name, "keywords": sub_keywords})
                                                 
                                                 # Run our ML-based classifier with proper string conversion
                                                 ml_intent_classification = classify_search_intent_ml(
                                                     [str(kw) if kw is not None else "" for kw in clusters_with_representatives.get(c_id, [])],
                                                     search_intent, # Pass API's intent description to ML classifier as context
                                                     f"Cluster {c_id}" # Pass cluster name as context
                                                 )

                                                 # Store results for this specific cluster
                                                 batch_analysis_results_from_api[c_id] = {
                                                     "search_intent_api": search_intent,
                                                     "split_suggestion": split_suggestion,
                                                     "additional_info": additional_info,
                                                     "coherence_score_api": coherence_score,
                                                     "subclusters": clean_subclusters,
                                                     "intent_classification_ml": ml_intent_classification # Include ML analysis result
                                                 }
                                             else:
                                                  st.warning(f"Received cluster ID {c_id} in API response but it was not in the requested batch. Skipping.")

                                    except (ValueError, TypeError) as e_id_conv:
                                         st.warning(f"Error converting cluster_id '{c_id_raw}' to int: {str(e_id_conv)}. Skipping this item from API response.")
                                    except Exception as e_item:
                                         st.warning(f"Error processing API response item for cluster ID {c_id_raw}: {str(e_item)}. Skipping this item.")

                            # If we successfully parsed *any* clusters, consider the attempt successful
                            if batch_analysis_results_from_api:
                                break # Exit retry loop
                    
                    except json.JSONDecodeError as e_json:
                        st.warning(f"API response is not valid JSON (Attempt {attempt+1}): {str(e_json)}. Content: {content[:200]}...")
                        # On last attempt, try regex extraction
                        if attempt == num_retries - 1:
                            for cluster_id in batch_cluster_ids:
                                # Look for cluster ID patterns
                                intent_pattern = rf'cluster_id["\s:]+{cluster_id}["\s,}}]+.*?search_intent["\s:]+([^"]+)["\s,}}]+'
                                split_pattern = rf'cluster_id["\s:]+{cluster_id}["\s,}}]+.*?split_suggestion["\s:]+([^"]+)["\s,}}]+'
                                
                                intent_matches = re.findall(intent_pattern, content, re.DOTALL)
                                split_matches = re.findall(split_pattern, content, re.DOTALL)
                                
                                if intent_matches:
                                    search_intent = intent_matches[0].strip()
                                    split_suggestion = split_matches[0].strip() if split_matches else "No split suggestion extracted"
                                    
                                    # Run ML classifier with string conversion
                                    ml_intent_classification = classify_search_intent_ml(
                                        [str(kw) if kw is not None else "" for kw in clusters_with_representatives.get(cluster_id, [])],
                                        search_intent,
                                        f"Cluster {cluster_id}"
                                    )
                                    
                                    batch_analysis_results_from_api[cluster_id] = {
                                        "search_intent_api": search_intent,
                                        "split_suggestion": split_suggestion,
                                        "additional_info": "Unable to extract from API response",
                                        "coherence_score_api": 5, # Default
                                        "subclusters": [],
                                        "intent_classification_ml": ml_intent_classification
                                    }

                except Exception as api_error:
                    st.error(f"An unexpected error occurred during API processing (Attempt {attempt+1}): {str(api_error)[:100]}...")
                    
                    # On last attempt, try a simplified prompt for each cluster individually
                    if attempt == num_retries - 1:
                        for cluster_id in batch_cluster_ids:
                            if cluster_id not in batch_analysis_results_from_api:
                                try:
                                    kws = [str(kw) for kw in clusters_with_representatives.get(cluster_id, [])[:5] if kw is not None]
                                    if kws:
                                        simple_prompt = f"What is the search intent for these keywords: {', '.join(kws)}? Answer in one sentence."
                                        
                                        simple_response = client.chat.completions.create(
                                            model=model,
                                            messages=[{"role": "user", "content": simple_prompt}],
                                            temperature=0.3,
                                            max_tokens=100
                                        )
                                        
                                        simple_intent = simple_response.choices[0].message.content.strip()
                                        
                                        # Run ML classifier
                                        ml_intent_classification = classify_search_intent_ml(
                                            [str(kw) if kw is not None else "" for kw in clusters_with_representatives.get(cluster_id, [])],
                                            simple_intent,
                                            f"Cluster {cluster_id}"
                                        )
                                        
                                        batch_analysis_results_from_api[cluster_id] = {
                                            "search_intent_api": simple_intent,
                                            "split_suggestion": "No analysis available for split recommendation",
                                            "additional_info": "No SEO insights available from fallback",
                                            "coherence_score_api": 5,
                                            "subclusters": [],
                                            "intent_classification_ml": ml_intent_classification
                                        }
                                except Exception:
                                    # Skip this cluster if even the simple request fails
                                    pass
            
            except Exception as outer_error:
                st.error(f"An error occurred during batch processing (Attempt {attempt+1}): {str(outer_error)[:100]}...")
                if attempt < num_retries - 1:
                    time.sleep(2) # Wait before retrying
                else:
                    st.warning(f"All attempts failed for batch {batch_start+1}-{batch_end}.")
        
        # Update our results with this batch's API results
        results.update(batch_analysis_results_from_api)
        
        # If any clusters in this batch had no API results, use the default ML results already stored in analysis_results
        for c_id in batch_cluster_ids:
            if c_id not in batch_analysis_results_from_api:
                results[c_id] = analysis_results[c_id]
        
        # Update progress
        progress_bar.progress(min(1.0, (batch_end) / len(cluster_ids)))

    # For any clusters that didn't get processed at all, include their default results
    for c_id in cluster_ids:
        if c_id not in results:
            results[c_id] = analysis_results[c_id]

    progress_bar.progress(1.0)
    progress_text.text("✅ Semantic analysis completed.")
    return results


# This function orchestrates the AI-driven analysis and intent flow analysis
# It is NOT cached because it relies on non-cached inputs (client, user prompts)
def evaluate_and_refine_clusters(df, client, model="gpt-3.5-turbo", user_prompt_for_analysis=None):
    """
    Performs AI-powered semantic analysis and intent flow analysis for clusters.
    Returns a dictionary of analysis results by cluster ID.
    """
    st.subheader("AI-Powered Cluster Analysis")

    # Build a dict of cluster -> representative keywords needed for both AI and ML analysis
    clusters_with_representatives = {}
    for c_id in df['cluster_id'].unique():
        # First try to get marked representative keywords (if any were set in naming)
        reps = df[(df['cluster_id'] == c_id) & (df['representative'] == True)]['keyword'].tolist()

        # If none found or not enough, take the first 20 keywords from this cluster
        if not reps or len(reps) < 5: # Use at least 5 reps if available, fallback to first 20
            cluster_kws = df[df['cluster_id'] == c_id]['keyword'].tolist()
            reps = cluster_kws[:min(20, len(cluster_kws))]

        # Convert all cluster representative items to strings to avoid float.split() errors
        clusters_with_representatives[c_id] = [str(kw) if kw is not None else "" for kw in reps]


    if not client:
        st.info("No OpenAI client available. Running local ML intent analysis and intent flow analysis.")
        # Return analysis results with just ML intent classification and default values for API parts
        analysis_results = {}
        for c_id in clusters_with_representatives.keys():
             ml_intent_classification = classify_search_intent_ml(
                 clusters_with_representatives.get(c_id, []),
                 "No API analysis available",
                 f"Cluster {c_id}" # Pass cluster ID as name context
             )
             # Always run local intent flow analysis if data exists
             intent_flow = analyze_cluster_for_intent_flow(df, c_id)

             analysis_results[c_id] = {
                 "search_intent_api": "No API analysis available",
                 "split_suggestion": "No API analysis available",
                 "additional_info": "No API analysis available",
                 "coherence_score_api": 5, # Default score
                 "subclusters": [],
                 "intent_classification_ml": ml_intent_classification,
                 "intent_flow": intent_flow # Include local intent flow analysis
             }
        st.success("✅ Local ML intent classification and intent flow analysis completed.")
        return analysis_results


    try:
        # Call GPT-based semantic analysis
        # This function now returns results that *include* the ML intent classification calculated internally per cluster per batch.
        semantic_analysis_results = generate_semantic_analysis(
            clusters_with_representatives=clusters_with_representatives,
            client=client,
            model=model,
            custom_prompt=user_prompt_for_analysis # Pass user's prompt for analysis
        )

        # Process intent flow (customer journey) for each cluster using our local ML classifier
        # This is separate from the AI's general search intent description, but we add it to the results
        st.info("Running local Intent Flow (Customer Journey) analysis...")
        # Iterate through the clusters for which we have any analysis results (API or ML default)
        for c_id in semantic_analysis_results:
            # Ensure the cluster ID exists in the DataFrame before analyzing intent flow
             if c_id in df['cluster_id'].values:
                 intent_flow = analyze_cluster_for_intent_flow(df, c_id)
                 if intent_flow:
                     semantic_analysis_results[c_id]['intent_flow'] = intent_flow # Add intent flow analysis to results
                 else:
                     st.warning(f"Could not perform local intent flow analysis for cluster {c_id}.")
                     semantic_analysis_results[c_id]['intent_flow'] = None # Ensure key exists even if analysis failed
             else:
                 st.warning(f"Cluster ID {c_id} from analysis results not found in DataFrame. Skipping intent flow analysis.")
                 semantic_analysis_results[c_id]['intent_flow'] = None # Ensure key exists


        # Check if we got any useful results from either API or ML fallback
        # We should always have ML results now due to initialization within generate_semantic_analysis
        if semantic_analysis_results:
             st.success(f"✅ Cluster analysis completed for {len(semantic_analysis_results)} clusters.")
        else:
            st.warning("No cluster analysis results were generated.")

        return semantic_analysis_results

    except Exception as e:
        st.error(f"An unexpected error occurred in the AI-powered cluster analysis orchestration: {str(e)}")
        st.warning("Returning empty analysis results.")
        return {}
################################################################
#         MAIN CLUSTERING PIPELINE
################################################################

# This function is NOT cached because it orchestrates the entire process,
# including API calls and state updates.
def run_clustering(
    uploaded_file,
    openai_api_key,
    num_clusters,
    pca_variance,
    max_pca_components,
    min_df,
    max_df,
    gpt_model,
    user_prompt_for_naming, # Renamed for clarity
    user_prompt_for_analysis, # New parameter for analysis prompt
    csv_format,
    selected_language,
    use_advanced_preprocessing_option # Added option
):
    """
    Executes the full clustering pipeline.
    Handles file loading, preprocessing, embeddings, dimensionality reduction,
    clustering, naming, evaluation, and AI analysis.
    Returns success status and the final DataFrame.
    """
    if uploaded_file is None:
        st.warning("Please upload a CSV file with keywords.")
        return False, None

    st.info("Starting advanced semantic clustering pipeline...")

    # Attempt to create OpenAI client if key provided and library is available
    client = None
    if openai_api_key and openai_available:
        try:
            # Use the key directly with the client
            client = OpenAI(api_key=openai_api_key)
            # Basic check - removed explicit check here, will handle errors during API calls
            st.success("Attempting to use OpenAI for embeddings/naming/analysis.")
        except Exception as e:
            st.error(f"Error configuring OpenAI client: {str(e)}")
            client = None
            st.warning("OpenAI client could not be configured. OpenAI features will be disabled.")
    elif openai_available:
         st.info("No OpenAI API Key provided. OpenAI features disabled.")
    else:
        st.warning("OpenAI library not available. OpenAI features disabled.")


    # Attempt to load spaCy model for selected language if advanced preprocessing is enabled
    spacy_nlp = None
    if use_advanced_preprocessing_option:
         spacy_nlp = load_spacy_model_by_language(selected_language)
         if spacy_nlp is None:
             # If spaCy loading failed or not available, inform the user about fallback
             if spacy_base_available:
                 st.warning(f"Could not load spaCy model for {selected_language}. Falling back to TextBlob/NLTK.")
             elif textblob_available:
                 st.warning("spaCy not available. Using TextBlob fallback for preprocessing.")
             else:
                 st.warning("spaCy and TextBlob not available. Using basic NLTK fallback for preprocessing.")
    else:
         st.info("Advanced preprocessing disabled by user.")


    try:
        # Load CSV according to user's choice
        st.subheader("Loading Data")
        if csv_format == "no_header":
            # No header, one column assumed to be 'keyword'
            try:
                 df = pd.read_csv(uploaded_file, header=None, names=["keyword"], keep_default_na=False, na_values=[''], encoding='utf-8', on_bad_lines='skip') # Treat empty strings as non-NaN
                 df = df.dropna(subset=['keyword']) # Drop rows where keyword is missing after potential NA conversion
                 df['keyword'] = df['keyword'].astype(str).str.strip() # Ensure keyword is string and strip whitespace
                 df = df[df['keyword'] != ""] # Remove rows with empty keywords after stripping
                 st.success(f"✅ Loaded {len(df)} keywords from CSV (no header).")
                 # Initialize other potential columns with default values if they don't exist
                 for col in ["search_volume", "competition", "cpc"]:
                      if col not in df.columns:
                           df[col] = np.nan # Use NaN for missing numeric data
                 # Initialize month columns if they don't exist
                 for i in range(1, 13):
                      month_col = f"month{i:02d}"
                      if month_col not in df.columns:
                           df[month_col] = np.nan


            except Exception as e:
                 st.error(f"Error loading CSV with no header: {str(e)}")
                 return False, None

        else: # With header
            try:
                 df = pd.read_csv(uploaded_file, header=0, keep_default_na=False, na_values=[''], encoding='utf-8', on_bad_lines='skip')
                 # Standardize 'Keyword' column name to lowercase 'keyword'
                 if "Keyword" in df.columns:
                     df.rename(columns={"Keyword": "keyword"}, inplace=True)
                 if "keyword" not in df.columns:
                     st.error("No 'Keyword' column found in the CSV. Please check your file.")
                     return False, None

                 df = df.dropna(subset=['keyword']) # Drop rows where keyword is missing
                 df['keyword'] = df['keyword'].astype(str).str.strip() # Ensure keyword is string and strip whitespace
                 df = df[df['keyword'] != ""] # Remove rows with empty keywords after stripping

                 # Convert standard numeric columns, coercing errors
                 for col in ["search_volume", "competition", "cpc"]:
                      if col in df.columns:
                           df[col] = pd.to_numeric(df[col], errors='coerce')
                      else:
                           df[col] = np.nan # Add if missing

                 # Convert month columns, coercing errors
                 for i in range(1, 13):
                      month_col = f"month{i:02d}" # Use 0-padding to match sample
                      if month_col in df.columns:
                           df[month_col] = pd.to_numeric(df[month_col], errors='coerce')
                      else:
                           df[month_col] = np.nan # Add if missing

                 # Ensure other potential string columns like 'cluster_name', 'cluster_description' exist as empty strings
                 for col in ['cluster_name', 'cluster_description', 'representative']:
                      if col not in df.columns:
                           if col == 'representative':
                               df[col] = False
                           else:
                               df[col] = ""


                 st.success(f"✅ Loaded {len(df)} rows from CSV (with header).")

            except Exception as e:
                 st.error(f"Error loading CSV with header: {str(e)}")
                 return False, None


        if df.empty:
             st.error("No valid keywords found in the CSV file after loading and cleaning.")
             return False, None

        num_keywords = len(df)
        # Show cost estimate based on loaded data and parameters
        show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)


        # --- Preprocessing ---
        st.subheader("Keyword Preprocessing")
        # Pass the use_advanced_preprocessing_option directly
        keywords_processed = preprocess_keywords(
            df["keyword"].tolist(),
            use_advanced=use_advanced_preprocessing_option,
            spacy_nlp=spacy_nlp
        )
        df['keyword_processed'] = keywords_processed


        # --- Generate embeddings ---
        st.subheader("Generating Semantic Vectors (Embeddings)")
        keyword_embeddings = generate_embeddings(df, openai_available, openai_api_key)

        if keyword_embeddings is None or keyword_embeddings.shape[0] != len(df):
             st.error("Failed to generate embeddings. Cannot proceed with clustering.")
             return False, df # Return df with processed keywords if available


        # --- Dimensionality reduction (PCA) ---
        keyword_embeddings_reduced = keyword_embeddings
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
                # Find number of components to reach target variance, or up to max_pca_components
                n_components_auto = np.argmax(cum_var >= target_var) + 1
                # Limit by max_pca_components and ensure at least 1 component if possible
                n_components = max(1, min(n_components_auto, max_pca_components, keyword_embeddings.shape[0]))

                pca_text.text(f"Optimal components for ~{pca_variance}% variance: {n_components_auto}. Using {n_components} components (limited by max allowed).")
                pca_progress.progress(0.6)

                if n_components > 0 and n_components <= keyword_embeddings.shape[0]:
                    pca = PCA(n_components=n_components)
                    keyword_embeddings_reduced = pca.fit_transform(keyword_embeddings)
                    st.success(f"✅ PCA applied: reduced to {keyword_embeddings_reduced.shape[1]} dimensions.")
                else:
                     st.warning("PCA resulted in an invalid number of components. Skipping PCA.")
                     keyword_embeddings_reduced = keyword_embeddings # Use original embeddings

                pca_progress.progress(1.0)
                pca_text.text("PCA analysis complete.")

            except Exception as e:
                st.error(f"Error applying PCA: {str(e)}")
                st.info("Proceeding without PCA.")
                keyword_embeddings_reduced = keyword_embeddings
        else:
            st.info(f"No PCA needed. Embedding dimension ({keyword_embeddings.shape[1]}) is within the max allowed ({max_pca_components}).")


        # --- Clustering ---
        st.subheader("Advanced Semantic Clustering")
        # Ensure clustering is attempted only if there are enough data points and clusters requested
        if keyword_embeddings_reduced is None or keyword_embeddings_reduced.shape[0] < max(2, num_clusters if num_clusters is not None else 2):
            st.error("Not enough data points or invalid cluster number to perform clustering.")
            df["cluster_id"] = 1 # Assign all to a single cluster if clustering fails
            st.warning("Assigning all keywords to a single cluster.")
        else:
            cluster_labels = improved_clustering(keyword_embeddings_reduced, num_clusters=num_clusters)
            df["cluster_id"] = cluster_labels
            final_clusters_count = len(df['cluster_id'].unique())
            st.success(f"✅ Clustering complete: {final_clusters_count} clusters created.")


        # --- Refinement (Placeholder) ---
        # st.subheader("Cluster Refinement") # Keep subheader commented if function does nothing
        # df = refine_clusters(df, keyword_embeddings_reduced) # Currently a placeholder


# --- Representative keywords ---
        st.subheader("Representative Keywords")
        rep_progress = st.progress(0)
        rep_text = st.empty()
        rep_text.text("Finding representative keywords...")
        clusters_with_representatives = {}

        try:
            unique_cluster_ids = df['cluster_id'].unique()
            df['representative'] = False # Reset representative flag

            for i, cnum in enumerate(unique_cluster_ids):
                cluster_df = df[df['cluster_id'] == cnum]
                csize = len(cluster_df)
                n_rep = min(20, csize) # Get up to 20 representatives

                if csize > 0:
                    # Get embeddings for keywords in this cluster
                    # Need to map the dataframe index to the embedding index correctly
                    indices_in_original_df = cluster_df.index.tolist()
                    c_embs = keyword_embeddings_reduced[indices_in_original_df] # Get embeddings using original df indices

                    # Calculate centroid and distances
                    centroid = np.mean(c_embs, axis=0)
                    # Calculate distance for each keyword in the cluster to the cluster centroid
                    distances = np.linalg.norm(c_embs - centroid, axis=1)

                    # Get indices within the *cluster_df* sorted by distance
                    sorted_cluster_indices = np.argsort(distances)[:n_rep]

                    # Get the original dataframe indices for the representatives
                    rep_indices_in_original_df = [indices_in_original_df[j] for j in sorted_cluster_indices]

                    # Get the representative keywords
                    rep_kws = df.loc[rep_indices_in_original_df, 'keyword'].tolist()
                    clusters_with_representatives[cnum] = rep_kws

                    # Mark representatives in the DataFrame
                    df.loc[rep_indices_in_original_df, 'representative'] = True


                rep_progress.progress(min(1.0, (i+1) / len(unique_cluster_ids)))

            rep_progress.progress(1.0)
            rep_text.text(f"✅ Representative keywords identified for {len(clusters_with_representatives)} clusters.")
        except Exception as e:
            st.error(f"Error finding representative keywords: {str(e)}")
            st.warning("Could not identify representatives automatically.")
            # Fallback: use the first keywords in the cluster as representatives
            clusters_with_representatives = {}
            df['representative'] = False
            try:
                for cnum in df['cluster_id'].unique():
                    cluster_kws = df[df['cluster_id'] == cnum]['keyword'].tolist()
                    reps = cluster_kws[:min(20, len(cluster_kws))]
                    clusters_with_representatives[cnum] = reps
                    # Mark the first few keywords as representatives (less accurate)
                    first_few_indices = df[df['cluster_id'] == cnum].head(min(20, len(cluster_kws))).index
                    df.loc[first_few_indices, 'representative'] = True
            except Exception as fallback_e:
                st.error(f"Error in representative keyword fallback: {str(fallback_e)}")


        # --- Generate cluster names & descriptions ---
        if client:
            st.subheader("Generating Cluster Names & Descriptions (SEO-focused)")
            try:
                cluster_names_and_descriptions = generate_cluster_names(
                    clusters_with_representatives=clusters_with_representatives,
                    client=client,
                    model=gpt_model,
                    custom_prompt=user_prompt_for_naming # Pass user's prompt for naming
                )
                if not cluster_names_and_descriptions:
                    st.warning("Cluster naming function returned empty results. Using fallback names.")
                    cluster_names_and_descriptions = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
            except Exception as e:
                st.error(f"Error during cluster naming: {str(e)}")
                st.info("Using fallback generic cluster names.")
                cluster_names_and_descriptions = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}
        else:
            st.warning("No OpenAI client available. Using generic cluster names.")
            cluster_names_and_descriptions = {k: (f"Cluster {k}", f"Keywords group {k}") for k in df['cluster_id'].unique()}

        # Apply names and descriptions to the DataFrame
        df['cluster_name'] = ''
        df['cluster_description'] = ''
        try:
            for cnum, (name, desc) in cluster_names_and_descriptions.items():
                # Safety check - ensure cluster exists in dataframe
                if cnum in df['cluster_id'].values:
                    df.loc[df['cluster_id'] == cnum, 'cluster_name'] = name
                    df.loc[df['cluster_id'] == cnum, 'cluster_description'] = desc
        except Exception as e:
             st.error(f"Error applying cluster names/descriptions to DataFrame: {str(e)}")
             st.warning("Cluster names might not be assigned correctly.")


        # --- Evaluate cluster quality (coherence) ---
        df = evaluate_cluster_quality(df, keyword_embeddings_reduced)


        # --- AI-based semantic analysis ---
        # Store evaluation results in session state
        # Pass the representative keywords explicitly for analysis function
        st.session_state.cluster_evaluation = evaluate_and_refine_clusters(
             df=df, # Pass the full dataframe to allow intent flow analysis per cluster
             client=client,
             model=gpt_model,
             user_prompt_for_analysis=user_prompt_for_analysis # Pass user's prompt for analysis
         )


        st.success("🥳 Keyword clustering and analysis complete!")

        return True, df

    except Exception as e:
        st.error(f"An unexpected error occurred in the main clustering pipeline: {str(e)}")
        return False, None


################################################################
#         MAIN STREAMLIT APP
################################################################

st.set_page_config(
    page_title="Advanced Semantic Keyword Clustering",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/your_github_repo', # Replace with your repo URL if applicable
        'Report a bug': 'https://github.com/your_github_repo/issues', # Replace with your repo URL if applicable
        'About': """
            # Advanced Semantic Keyword Clustering

            This tool performs semantic clustering of keywords using various NLP techniques
            and optionally leverages OpenAI for enhanced analysis, naming, and evaluation.

            Developed by [Your Name/Organization - Optional].
            Source code: [Link to your GitHub repo - Optional]
            """
    }
)

st.markdown("""
<style>
    /* Inject custom CSS for better styling */
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
     .warning-box {
        background-color: #fff3cd;
        color: #664d03;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
     .error-box {
        background-color: #f8d7da;
        color: #58151c;
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
        border-left: 5px solid #43a047; /* Green */
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-middle {
        background-color: #e3f2fd;
        border-left: 5px solid #1e88e5; /* Blue */
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-late {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800; /* Orange */
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-transition {
        background-color: #f3e5f5;
        border-left: 5px solid #8e24aa; /* Purple */
        padding: 10px;
        margin-bottom: 10px;
    }
    .journey-mixed {
        background-color: #f5f5f5;
        border-left: 5px solid #9e9e9e; /* Grey */
        padding: 10px;
        margin-bottom: 10px;
    }
    .evidence-list {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
        margin-left: 20px;
        list-style-type: disc;
    }

    .keyword-example {
        display: inline-block;
        background-color: #f5f5f5;
        border-radius: 3px;
        padding: 3px 6px;
        margin: 2px;
        font-size: 0.85em;
        border: 1px solid #e0e0e0;
    }

     .info-tag { /* Styles for Intent tags */
        background-color: #e3f2fd; /* Light Blue */
        color: #0d47a1; /* Dark Blue */
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
        font-weight: bold;
    }
    .commercial-tag {
        background-color: #f3e5f5; /* Light Purple */
        color: #4a148c; /* Dark Purple */
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
         font-weight: bold;
    }

    .transactional-tag {
        background-color: #fff3e0; /* Light Orange */
        color: #e65100; /* Dark Orange */
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
         font-weight: bold;
    }
     .navigational-tag {
        background-color: #e8f5e9; /* Light Green */
        color: #1b5e20; /* Dark Green */
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
         font-weight: bold;
    }
    .mixed-tag, .unknown-tag {
         background-color: #f5f5f5; /* Light Grey */
        color: #212121; /* Dark Grey */
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-right: 5px;
         font-weight: bold;
    }
    .streamlit-expanderContent {
        overflow-x: auto; /* Add horizontal scroll to expander content */
    }
    .stDataFrame {
        width: 100% !important; /* Ensure DataFrame uses full width */
    }

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Advanced Semantic Keyword Clustering</div>", unsafe_allow_html=True)
st.markdown("""
This application clusters semantically similar keywords using advanced NLP and clustering methods.
It can optionally leverage OpenAI for enhanced analysis, naming, and evaluation.

You can upload:
- A **simple CSV** with no header (just one keyword per line), or
- A **Keyword Planner-like CSV** with a header (must contain a 'Keyword' column). Additional columns like search volume, competition, cpc, and monthly data will be preserved.
""")

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'clustering_successful' not in st.session_state:
    st.session_state['clustering_successful'] = False
if 'clustered_df' not in st.session_state:
    st.session_state['clustered_df'] = None
if 'cluster_evaluation' not in st.session_state:
    st.session_state['cluster_evaluation'] = {}
# state for cost estimate display in sidebar
if 'last_csv_cost_estimate' not in st.session_state:
     st.session_state['last_csv_cost_estimate'] = None


# --- Sidebar Options ---
st.sidebar.markdown("## 📁 File Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

csv_format = st.sidebar.radio(
    "CSV File Format",
    options=["with_header", "no_header"],
    index=0,
    help="Select 'no_header' if your file is just a list of keywords, one per line.",
    horizontal=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Clustering Parameters")

num_clusters = st.sidebar.slider(
    "Number of Clusters (KMeans)",
    min_value=2,
    max_value=100, # Increased max clusters
    value=20,
    step=1,
    help="The desired number of clusters to create using the KMeans algorithm."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ✨ Embedding & Processing Options")

# Option to use advanced preprocessing (spaCy/TextBlob)
use_advanced_preprocessing_option = st.sidebar.checkbox(
    "Use Advanced Preprocessing (SpaCy/TextBlob)",
    value=True, # Default to True if libraries are available
    help="Enables preprocessing using spaCy (if available for language) or TextBlob, which can improve semantic understanding.",
    disabled=not (spacy_base_available or textblob_available) # Disable if libraries not installed
)
if use_advanced_preprocessing_option and not (spacy_base_available or textblob_available):
    st.sidebar.warning("Advanced preprocessing libraries (spaCy/TextBlob) not available. Using NLTK fallback.")


# Language selection for spaCy
language_options = list(SPACY_LANGUAGE_MODELS.keys())
# Add an option for "Auto-detect" or "English (default fallback)" if needed
selected_language = st.sidebar.selectbox(
    "Language for Preprocessing (if using SpaCy)",
    options=language_options,
    index=language_options.index("English") if "English" in language_options else 0,
    help="Select the language of your keywords. Requires spaCy model installation for advanced processing. Ignored if advanced preprocessing is disabled.",
    disabled=not (spacy_base_available and use_advanced_preprocessing_option) # Disable if spaCy not available or advanced is off
)
if use_advanced_preprocessing_option and spacy_base_available and SPACY_LANGUAGE_MODELS.get(selected_language) is None:
     st.sidebar.info(f"No specific spaCy model available for {selected_language}. Using TextBlob/NLTK fallback for this language.")


st.sidebar.markdown("---")
st.sidebar.markdown("## 🧠 OpenAI Integration (Optional)")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key to enable semantic embeddings, cluster naming, and AI analysis. Leave blank to use free alternatives (SentenceTransformers, TF-IDF)."
)
if not openai_available:
    st.sidebar.warning("OpenAI library not installed. API key ignored.")

gpt_model = st.sidebar.radio(
    "GPT Model for Naming & Analysis",
    options=["gpt-3.5-turbo", "gpt-4-turbo"], # Offer recent models
    index=0,
    horizontal=True,
    help="Choose the GPT model for generating cluster names, descriptions, and performing analysis. GPT-4-turbo is more capable but more expensive.",
    disabled=not (openai_available and bool(openai_api_key)) # Disable if no key or library
)

user_prompt_for_naming = st.sidebar.text_area(
    "Custom Prompt for Cluster Naming",
    value="",
    placeholder="e.g., Focus on including transactional terms where relevant...",
    help="Add specific instructions for the AI when generating cluster names and descriptions. Leave blank for default SEO-focused prompt.",
     disabled=not (openai_available and bool(openai_api_key)) # Disable if no key or library
)

user_prompt_for_analysis = st.sidebar.text_area(
    "Custom Prompt for Cluster Analysis",
    value="",
    placeholder="e.g., Pay attention to identifying long-tail keywords...",
    help="Add specific instructions for the AI when performing cluster analysis (intent, split suggestions, SEO insights). Leave blank for default prompt.",
    disabled=not (openai_available and bool(openai_api_key)) # Disable if no key or library
)


st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 Advanced Options")

# PCA parameters
with st.sidebar.expander("Dimensionality Reduction (PCA)"):
    st.markdown("Reduce the number of dimensions in embeddings.")
    pca_variance = st.slider(
        "Explained Variance (%)",
        min_value=70,
        max_value=100,
        value=95,
        step=1,
        help="The percentage of variance the PCA components should retain."
    )
    max_pca_components = st.number_input(
        "Maximum PCA Components",
        min_value=2,
        max_value=500, # Limit max components
        value=300, # Default max components
        step=1,
         help="Hard limit on the number of PCA components."
    )

# TF-IDF parameters (if used as fallback)
with st.sidebar.expander("TF-IDF Fallback Options"):
    st.markdown("Parameters for TF-IDF if embeddings fail or are not used.")
    tfidf_min_df = st.slider(
        "Min document frequency (min_df)",
        min_value=1,
        max_value=50,
        value=1,
        step=1,
        help="Ignore terms that appear in fewer than min_df documents."
    )
    tfidf_max_df = st.slider(
        "Max document frequency (max_df)",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.01,
        help="Ignore terms that appear in more than max_df proportion of documents."
    )
    tfidf_max_features = st.number_input(
        "Max TF-IDF Features",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
         help="Maximum number of features (terms) to be used in TF-IDF."
    )


# --- Cost Calculator in Sidebar ---
add_cost_calculator()


# --- Sample CSV Download ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📥 Sample Data")
sample_csv_content = generate_sample_csv()
st.sidebar.download_button(
    label="Download Sample CSV",
    data=sample_csv_content,
    file_name="sample_keywords.csv",
    mime="text/csv",
    use_container_width=True
)


# --- Run Button ---
st.markdown("---")
if st.button("🚀 Run Advanced Clustering"):
    if uploaded_file is None:
        st.warning("Please upload a CSV file to begin.")
    else:
        # Clear previous results from session state
        st.session_state['clustering_successful'] = False
        st.session_state['clustered_df'] = None
        st.session_state['cluster_evaluation'] = {}

        with st.spinner("Running clustering pipeline..."):
            clustering_successful, clustered_df = run_clustering(
                uploaded_file=uploaded_file,
                openai_api_key=openai_api_key,
                num_clusters=num_clusters,
                pca_variance=pca_variance,
                max_pca_components=max_pca_components,
                min_df=tfidf_min_df,
                max_df=tfidf_max_df,
                gpt_model=gpt_model,
                user_prompt_for_naming=user_prompt_for_naming,
                user_prompt_for_analysis=user_prompt_for_analysis,
                csv_format=csv_format,
                selected_language=selected_language,
                use_advanced_preprocessing_option=use_advanced_preprocessing_option # Pass the option
            )

            st.session_state['clustering_successful'] = clustering_successful
            st.session_state['clustered_df'] = clustered_df
            # cluster_evaluation is already saved in session state by evaluate_and_refine_clusters

# --- Display Results ---
if st.session_state['clustering_successful'] and st.session_state['clustered_df'] is not None:
    st.markdown("<div class='main-header'>📊 Clustering Results</div>", unsafe_allow_html=True)

    final_df = st.session_state['clustered_df']
    cluster_evaluation_results = st.session_state['cluster_evaluation']

    st.markdown("<div class='sub-header'>Clustered Data Table</div>", unsafe_allow_html=True)
    st.info("Below is the table with keywords assigned to clusters, including their names, descriptions, and coherence scores.")

    # Define columns to display and their order
    display_columns = [
        'cluster_id',
        'cluster_name',
        'cluster_description',
        'cluster_coherence',
        'representative',
        'keyword'
    ]
    # Add other columns from the original CSV if they exist and are not the processed keyword column
    other_cols = [col for col in final_df.columns if col not in display_columns and col != 'keyword_processed']
    display_columns.extend(other_cols)

    # Ensure all display columns exist in the DataFrame before selecting
    display_columns = [col for col in display_columns if col in final_df.columns]


    # Use safe_numeric_formatting for relevant columns in the DataFrame display
    # Although st.dataframe is good, explicit formatting can ensure consistency
    # For simplicity and allowing interactive sort in st.dataframe, we won't apply string formatting here,
    # but rely on st.dataframe's default rendering. If specific formatting is needed,
    # a copy of the DataFrame with formatted strings would be necessary before display.
    st.dataframe(final_df[display_columns])

    # Download button for the results
    @st.cache_data # Cache the conversion to CSV
    def convert_df_to_csv(df):
        # Use the dataframe with all columns for download
        return df.to_csv(index=False).encode('utf-8')

    csv_download = convert_df_to_csv(final_df)

    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv_download,
        file_name='clustered_keywords.csv',
        mime='text/csv',
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("<div class='sub-header'>Cluster Details and Analysis</div>", unsafe_allow_html=True)

    unique_cluster_ids = sorted(final_df['cluster_id'].unique()) # Sort clusters by ID

    if not cluster_evaluation_results and openai_available and openai_api_key:
         st.warning("AI analysis was requested but no results were returned. Check API calls and prompt.")
         # Fallback to just showing basic cluster info if AI analysis failed entirely
         for c_id in unique_cluster_ids:
              cluster_df = final_df[final_df['cluster_id'] == c_id]
              with st.expander(f"📦 Cluster {c_id} ({len(cluster_df)} keywords)"):
                   st.markdown(f"**Name:** Not available")
                   st.markdown(f"**Description:** Not available")
                   st.markdown(f"**Calculated Coherence Score:** {cluster_df['cluster_coherence'].iloc[0]:.2f}")
                   st.markdown(f"**AI Search Intent:** Not available")
                   st.markdown(f"**ML Search Intent:** Not available")
                   st.markdown(f"**Split Suggestion:** Not available")
                   st.markdown(f"**Additional SEO Info:** Not available")
                   st.markdown(f"**Customer Journey Phase:** Not available")
                   st.markdown("**Representative Keywords:**")
                   reps = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                   if reps:
                        st.write(", ".join(reps))
                   else:
                        st.write("None identified automatically.")
                   st.markdown("**All Keywords in Cluster:**")
                   st.write(", ".join(cluster_df['keyword'].tolist()))


    elif not cluster_evaluation_results and (not openai_available or not openai_api_key):
         st.info("OpenAI not configured or available. Displaying basic cluster information based on local analysis.")
         # Display basic cluster info based on local analysis if no OpenAI
         for c_id in unique_cluster_ids:
             cluster_df = final_df[final_df['cluster_id'] == c_id]
             cluster_name = cluster_df['cluster_name'].iloc[0] if not cluster_df['cluster_name'].empty else f"Cluster {c_id}"
             cluster_desc = cluster_df['cluster_description'].iloc[0] if not cluster_df['cluster_description'].empty else f"Group of related keywords (cluster {c_id})"
             coherence_score = cluster_df['cluster_coherence'].iloc[0] if not cluster_df['cluster_coherence'].empty else 0.0


             # Run local ML intent classification for display here if not already done/stored
             # Check if 'intent_classification_ml' and 'intent_flow' exist from the fallback in evaluate_and_refine_clusters
             analysis_data = cluster_evaluation_results.get(c_id, {})
             ml_intent_data = analysis_data.get("intent_classification_ml", classify_search_intent_ml(cluster_df['keyword'].tolist())) # Run if not available
             intent_flow_data = analysis_data.get("intent_flow", analyze_cluster_for_intent_flow(cluster_df, c_id)) # Run if not available


             with st.expander(f"📦 {cluster_name} ({len(cluster_df)} keywords)"):
                  st.markdown(f"**Cluster ID:** {c_id}")
                  st.markdown(f"**Description:** {cluster_desc}")
                  st.markdown(f"**Calculated Coherence Score:** {coherence_score:.2f}")

                  st.markdown("---")
                  st.markdown("**Local ML Search Intent Analysis:**")
                  if ml_intent_data:
                      intent_class = ml_intent_data.get("primary_intent", "Unknown")
                      scores = ml_intent_data.get("scores", {})
                      st.markdown(f"Primary Intent: <span class='{'intent-' + intent_class.lower().replace(' ', '-')} intent-box'>**{intent_class}**</span>", unsafe_allow_html=True)
                      st.markdown(f"Scores: Info: {scores.get('Informational', 0)}%, Nav: {scores.get('Navigational', 0)}%, Trans: {scores.get('Transactional', 0)}%, Comm: {scores.get('Commercial', 0)}%")
                      # Display evidence if needed
                      # st.markdown("Evidence:")
                      # st.json(ml_intent_data.get("evidence", {}))
                  else:
                      st.info("ML Intent analysis not available.")

                  st.markdown("---")
                  st.markdown("**Local Intent Flow (Customer Journey) Analysis:**")
                  if intent_flow_data:
                      journey_phase = intent_flow_data.get("journey_phase", "Unknown")
                      intent_dist = intent_flow_data.get("intent_distribution", {})
                      journey_css_class = 'journey-' + journey_phase.lower().replace(' ', '-').replace('/', '-').replace('(', '').replace(')', '')
                      st.markdown(f"Detected Journey Phase: <span class='{journey_css_class}'>**{journey_phase}**</span>", unsafe_allow_html=True)
                      st.markdown("Intent Distribution within Cluster Sample:")
                      if intent_dist:
                           dist_text = ", ".join([f"{intent}: {pct:.2f}%" for intent, pct in intent_dist.items()])
                           st.markdown(dist_text)
                           # Display keyword sample if needed
                           # st.markdown("Sample Keywords for Analysis:")
                           # sample_kws_display = intent_flow_data.get("keyword_sample", [])
                           # st.write(", ".join([k['keyword'] for k in sample_kws_display]))
                      else:
                           st.info("Intent distribution data not available.")

                  st.markdown("---")
                  st.markdown("**Representative Keywords:**")
                  reps = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                  if reps:
                       st.write(", ".join(reps))
                  else:
                       st.write("None identified automatically.")
                  st.markdown("**All Keywords in Cluster:**")
                  st.write(", ".join(cluster_df['keyword'].tolist()))


else: # OpenAI analysis results are available
        for c_id in unique_cluster_ids:
            cluster_df = final_df[final_df['cluster_id'] == c_id]
            # Get data from evaluation results dictionary
            analysis_data = cluster_evaluation_results.get(c_id, {}) # Get analysis data for this cluster ID

            cluster_name = cluster_df['cluster_name'].iloc[0] if not cluster_df['cluster_name'].empty else f"Cluster {c_id}"
            cluster_desc = cluster_df['cluster_description'].iloc[0] if not cluster_df['cluster_description'].empty else f"Group of related keywords (cluster {c_id})"
            coherence_score_calculated = cluster_df['cluster_coherence'].iloc[0] if not cluster_df['cluster_coherence'].empty else 0.0


            with st.expander(f"📦 {cluster_name} ({len(cluster_df)} keywords)"):
                st.markdown(f"**Cluster ID:** {c_id}")
                st.markdown(f"**Description:** {cluster_desc}")
                st.markdown(f"**Calculated Coherence Score:** {coherence_score_calculated:.2f}") # Show calculated score

                # Display AI Analysis results
                st.markdown("---")
                st.markdown("**OpenAI Analysis:**")
                search_intent_api = analysis_data.get("search_intent_api", "N/A")
                coherence_score_api = analysis_data.get("coherence_score_api", "N/A")
                split_suggestion = analysis_data.get("split_suggestion", "N/A")
                additional_info = analysis_data.get("additional_info", "N/A")
                subclusters_api = analysis_data.get("subclusters", [])

                st.markdown(f"AI Search Intent: {search_intent_api}")
                st.markdown(f"AI Coherence Score (0-10): {coherence_score_api}")
                st.markdown(f"Split Suggestion: {split_suggestion}")
                st.markdown(f"Additional SEO Info: {additional_info}")

                if subclusters_api and split_suggestion.lower() == 'yes':
                    st.markdown("**Suggested Subclusters:**")
                    for sub in subclusters_api:
                        sub_name = sub.get("name", "Unnamed Subcluster")
                        sub_keywords = sub.get("keywords", [])
                        st.markdown(f"- **{sub_name}**: {', '.join(sub_keywords)}")


                st.markdown("---")
                # Display Local ML Intent Analysis
                st.markdown("**Local ML Search Intent Analysis:**")
                ml_intent_data = analysis_data.get("intent_classification_ml", {}) # Get ML analysis data

                if ml_intent_data:
                     intent_class = ml_intent_data.get("primary_intent", "Unknown")
                     scores = ml_intent_data.get("scores", {})
                     intent_css_class = 'intent-' + intent_class.lower().replace(' ', '-').replace('/', '-')
                     st.markdown(f"Primary Intent: <span class='{intent_css_class} intent-box'>**{intent_class}**</span>", unsafe_allow_html=True)
                     st.markdown(f"Scores: Info: {scores.get('Informational', 0)}%, Nav: {scores.get('Navigational', 0)}%, Trans: {scores.get('Transactional', 0)}%, Comm: {scores.get('Commercial', 0)}%")
                     # Display evidence
                     # with st.expander("Show Intent Evidence"):
                     #     evidence_dict = ml_intent_data.get("evidence", {})
                     #     for intent, signals in evidence_dict.items():
                     #          if signals:
                     #               st.markdown(f"**{intent}:**")
                     #               st.markdown("".join([f"- {signal}<br>" for signal in signals]), unsafe_allow_html=True)


                else:
                     st.info("ML Intent analysis data not available for this cluster.")

                st.markdown("---")
                # Display Intent Flow Analysis
                st.markdown("**Local Intent Flow (Customer Journey) Analysis:**")
                intent_flow_data = analysis_data.get("intent_flow", {}) # Get intent flow data

                if intent_flow_data:
                    journey_phase = intent_flow_data.get("journey_phase", "Unknown")
                    intent_dist = intent_flow_data.get("intent_distribution", {})
                    journey_css_class = 'journey-' + journey_phase.lower().replace(' ', '-').replace('/', '-').replace('(', '').replace(')', '')
                    st.markdown(f"Detected Journey Phase: <span class='{journey_css_class}'>**{journey_phase}**</span>", unsafe_allow_html=True)
                    st.markdown("Intent Distribution within Cluster Sample:")
                    if intent_dist:
                        dist_text = ", ".join([f"<span class='{intent.lower()}-tag'>{intent}: {pct:.2f}%</span>" for intent, pct in intent_dist.items()])
                        st.markdown(dist_text, unsafe_allow_html=True)
                        # Display keyword sample for intent analysis
                        # st.markdown("Sample Keywords for Intent Analysis:")
                        # sample_kws_display = intent_flow_data.get("keyword_sample", [])
                        # st.write(", ".join([f"{k['keyword']} ({k['intent']})" for k in sample_kws_display])) # Show keyword and its individual intent


                    else:
                        st.info("Intent distribution data not available.")

                st.markdown("---")
                st.markdown("**Representative Keywords:**")
                reps = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                if reps:
                     st.write(", ".join(reps))
                else:
                     st.write("None identified automatically.")
                st.markdown("**All Keywords in Cluster:**")
                st.write(", ".join(cluster_df['keyword'].tolist()))
    
    with st.expander("Download Results"):
        csv_data = final_df.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_data,
            file_name="semantic_clustered_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.subheader("Clusters Summary")
        summary_df = final_df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
        summary_df.columns = ['ID', 'Name', 'Description', 'Number of Keywords']
        
        # Add search volume if it exists
        if 'search_volume' in final_df.columns:
            # Convert to numeric and handle errors before aggregation
            final_df['search_volume'] = pd.to_numeric(final_df['search_volume'], errors='coerce')
            volume_df = final_df.groupby('cluster_id')['search_volume'].sum().reset_index()
            summary_df = summary_df.merge(volume_df, left_on='ID', right_on='cluster_id')
            summary_df.drop('cluster_id', axis=1, inplace=True)
            summary_df.rename(columns={'search_volume': 'Total Search Volume'}, inplace=True)
        
        # Merge coherence
        coherence_df = final_df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id')
        summary_df.drop('cluster_id', axis=1, inplace=True)
        summary_df.rename(columns={'cluster_coherence': 'Coherence'}, inplace=True)
        
        # Representative keywords
        def get_rep_keywords(cid):
            reps = final_df[(final_df['cluster_id'] == cid) & (final_df['representative'] == True)]['keyword'].tolist()
            return ', '.join(reps[:5])
        summary_df['Representative Keywords'] = summary_df['ID'].apply(get_rep_keywords)
        
        # AI evaluation info
        if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
            evaluated_ids = list(st.session_state.cluster_evaluation.keys())
            summary_df['AI Evaluation?'] = summary_df['ID'].apply(lambda x: "Yes" if x in evaluated_ids else "No")
            
            # Add primary search intent
            def get_search_intent(cid):
                if cid in st.session_state.cluster_evaluation:
                    intent_data = st.session_state.cluster_evaluation[cid].get('intent_classification_ml', {})
                    return intent_data.get('primary_intent', 'Unknown')
                return 'Unknown'
            
            summary_df['Primary Intent'] = summary_df['ID'].apply(get_search_intent)
            
            # Add journey phase if available
            def get_journey_phase(cid):
                if cid in st.session_state.cluster_evaluation and 'intent_flow' in st.session_state.cluster_evaluation[cid]:
                    if st.session_state.cluster_evaluation[cid]['intent_flow']:
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
