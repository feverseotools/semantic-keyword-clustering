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

# For OpenAI, import with error handling
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

try:
    import spacy
    try:
        # Load English model if your keywords are in English
        nlp = spacy.load("en_core_web_sm")
        spacy_available = True
    except:
        spacy_available = False
except ImportError:
    spacy_available = False

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
        
        Calculate approximate OpenAI costs for your keywords.
        """)
        
        calc_num_keywords = st.number_input(
            "Number of keywords to process", 
            min_value=100, 
            max_value=100000, 
            value=1000,
            step=500
        )
        calc_num_clusters = st.number_input(
            "Approximate number of clusters",
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
                    help="OpenAI processes up to 5,000 keywords; the rest use similarity propagation."
                )
                st.metric(
                    "Embeddings cost", 
                    f"${cost_results['embedding_cost']:.4f}",
                    help="Cost for text-embedding-3-small"
                )
            with col2:
                st.metric(
                    "Cluster naming cost", 
                    f"${cost_results['naming_cost']:.4f}",
                    help=f"Cost using {calc_model} for naming/describing clusters"
                )
                st.metric(
                    "TOTAL COST", 
                    f"${cost_results['total_cost']:.4f}",
                    help="Approximate total cost"
                )
            
            st.info("""
            **Note:** This is only an estimate. Actual costs may vary 
            based on the length of keywords and complexity.
            Using SentenceTransformers is $0.
            """)

def show_csv_cost_estimate(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    """
    Shows cost estimate in the sidebar for the currently uploaded CSV.
    """
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
                {cost_results['processed_keywords']:,} keywords will be processed directly with OpenAI.
                The remaining {num_keywords - cost_results['processed_keywords']:,} 
                will use similarity propagation.
                """)
            
            st.markdown("""
            **Cost Savings**: If you prefer not to use OpenAI, 
            you can use SentenceTransformers or TF-IDF at no cost.
            """)


################################################################
#          SAMPLE CSV GENERATION
################################################################

def generate_sample_csv():
    """
    Returns a sample CSV header row:
    Keyword,search_volume,competition,cpc,month1..month12
    """
    header = ["Keyword","search_volume","competition","cpc"]
    months = [f"month{i}" for i in range(1,13)]
    header += months
    return ",".join(header) + "\n"


################################################################
#          SEMANTIC PREPROCESSING
################################################################

def enhanced_preprocessing(text, use_lemmatization=True):
    """Enhanced preprocessing using spaCy or TextBlob if available."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Option 1: spaCy
        if spacy_available:
            doc = nlp(text.lower())
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
        
        # Option 2: TextBlob
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
        
        # Fallback: standard NLTK
        else:
            return preprocess_text(text, use_lemmatization)
    
    except Exception:
        return text.lower() if isinstance(text, str) else ""

def preprocess_text(text, use_lemmatization=True):
    """Basic NLTK-based preprocessing."""
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

def preprocess_keywords(keywords, use_advanced=True):
    processed_keywords = []
    progress_bar = st.progress(0)
    total = len(keywords)
    
    if use_advanced:
        if spacy_available:
            st.success("Using advanced preprocessing with spaCy")
        elif textblob_available:
            st.success("Using alternative preprocessing with TextBlob")
        else:
            st.info("Using standard NLP with NLTK")
    else:
        st.info("Using standard NLP with NLTK")
    
    for i, kw in enumerate(keywords):
        if use_advanced and (spacy_available or textblob_available):
            processed_keywords.append(enhanced_preprocessing(kw))
        else:
            processed_keywords.append(preprocess_text(kw))
        
        if i % 100 == 0:
            progress_bar.progress(min(i/total,1.0))
    
    progress_bar.progress(1.0)
    return processed_keywords


################################################################
#          EMBEDDING GENERATION
################################################################

def generate_embeddings(df, openai_available, openai_api_key=None):
    """
    Generate embeddings using OpenAI if possible, else SentenceTransformers, else TF-IDF.
    """
    st.info("Generating embeddings for keywords...")
    keywords = df['keyword_processed'].fillna('').tolist()
    
    # 1) Attempt OpenAI
    if openai_available and openai_api_key:
        try:
            st.info("Using OpenAI embeddings (high semantic precision)")
            os.environ["OPENAI_API_KEY"] = openai_api_key
            client = OpenAI()
            
            all_embeddings = []
            if len(keywords) > 5000:
                st.warning(f"Limiting to 5000 of {len(keywords)} keywords for direct embeddings.")
                step = max(1,len(keywords)//5000)
                sample_indices = list(range(0,len(keywords),step))[:5000]
                sample_keywords = [keywords[i] for i in sample_indices]
                
                progress_bar = st.progress(0)
                st.info("Requesting embeddings from OpenAI in a single call for sample subset...")
                response = client.embeddings.create(model="text-embedding-3-small", input=sample_keywords)
                progress_bar.progress(0.5)
                
                sample_embs = np.array([item.embedding for item in response.data])
                
                # TF-IDF to propagate
                st.info("Propagating to remaining keywords with similarity approach...")
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(keywords)
                
                all_embeddings = np.zeros((len(keywords), len(sample_embs[0])))
                for i, idx in enumerate(sample_indices):
                    all_embeddings[idx] = sample_embs[i]
                
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(3,len(sample_indices)))
                nn.fit(tfidf_matrix[sample_indices])
                
                remaining = [i for i in range(len(keywords)) if i not in sample_indices]
                for i, idx in enumerate(remaining):
                    distances, neighbors = nn.kneighbors(tfidf_matrix[idx:idx+1])
                    weights = 1.0 / (1.0 + distances[0])
                    weights /= weights.sum()
                    
                    weighted_emb = np.zeros_like(sample_embs[0])
                    for j, weight in zip(neighbors[0],weights):
                        sample_idx = sample_indices[j]
                        weighted_emb += weight * all_embeddings[sample_idx]
                    
                    all_embeddings[idx] = weighted_emb
                    
                    if i % 100 == 0:
                        val = 0.5 + (0.5* i/ len(remaining))
                        progress_bar.progress(min(val,1.0))
                
                progress_bar.progress(1.0)
            
            else:
                progress_bar = st.progress(0)
                st.info(f"Requesting embeddings for all {len(keywords)} keywords from OpenAI (batch approach).")
                batch_size=1000
                for i in range(0, len(keywords), batch_size):
                    batch_end = min(i+batch_size, len(keywords))
                    sublist = keywords[i:batch_end]
                    response = client.embeddings.create(model="text-embedding-3-small", input=sublist)
                    batch_embs = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embs)
                    progress_bar.progress(min(float(batch_end/len(keywords)),1.0))
                progress_bar.progress(1.0)
            
            embeddings = np.array(all_embeddings) if isinstance(all_embeddings,list) else all_embeddings
            st.success(f"✅ OpenAI embeddings done. Dim={embeddings.shape[1]}")
            return embeddings
        except Exception as e:
            st.error(f"OpenAI embeddings error: {str(e)}")
            st.info("Falling back to SentenceTransformers next...")

    # 2) Attempt SentenceTransformers
    if sentence_transformers_available:
        try:
            st.success("Using SentenceTransformers (free fallback)")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            progress_bar= st.progress(0)
            batch_size=512
            all_embeddings=[]
            for i in range(0,len(keywords),batch_size):
                batch=keywords[i:i+batch_size]
                emb=model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(emb)
                progress_bar.progress(float(i+batch_size)/len(keywords))
            progress_bar.progress(1.0)
            embeddings=np.array(all_embeddings)
            st.success(f"✅ SentenceTransformers embeddings done. Dim={embeddings.shape[1]}")
            return embeddings
        except Exception as e:
            st.error(f"SentenceTransformers error: {str(e)}")

    # 3) Fallback to TF-IDF
    st.warning("Using TF-IDF as last resort.")
    return generate_tfidf_embeddings(df['keyword_processed'].fillna(''))

def generate_tfidf_embeddings(texts, min_df=1, max_df=0.95):
    st.info("Generating TF-IDF vectors for keywords...")
    progress_bar = st.progress(0)
    try:
        vectorizer=TfidfVectorizer(
            max_features=300,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        progress_bar.progress(0.3)
        tfidf_matrix=vectorizer.fit_transform(texts)
        progress_bar.progress(0.8)
        embeddings=tfidf_matrix.toarray()
        progress_bar.progress(1.0)
        st.success(f"✅ TF-IDF vectorization done. Dim={embeddings.shape[1]}")
        return embeddings
    except Exception as e:
        st.error(f"TF-IDF error: {str(e)}. Using random fallback.")
        return np.random.rand(len(texts),100)


################################################################
#          CLUSTERING ALGORITHMS
################################################################

def improved_clustering(embeddings, num_clusters=None, min_cluster_size=5):
    st.info("Applying advanced clustering algorithms...")

    # Auto-determine or use user input
    if num_clusters is None:
        # Try silhouette approach or fallback
        st.warning("No num_clusters provided; defaulting to 10.")
        num_clusters=10

    # Attempt HDBSCAN if available
    if hdbscan_available:
        try:
            st.info("Trying HDBSCAN for natural cluster detection...")
            clusterer= hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            cluster_labels= clusterer.fit_predict(embeddings)
            unique= np.unique(cluster_labels)
            non_noise= [c for c in unique if c!=-1]
            if (len(non_noise)>1) and (len(non_noise)<=num_clusters*2):
                st.success(f"HDBSCAN found {len(non_noise)} natural clusters.")
                # Reassign noise
                if -1 in unique:
                    noise_ids= np.where(cluster_labels==-1)[0]
                    for idx in noise_ids:
                        min_dist= 999999
                        best_c= non_noise[0]
                        for c in non_noise:
                            points= embeddings[cluster_labels==c]
                            if len(points)>0:
                                cent= np.mean(points,axis=0)
                                dist= np.linalg.norm(embeddings[idx]- cent)
                                if dist< min_dist:
                                    min_dist= dist
                                    best_c= c
                        cluster_labels[idx]= best_c
                # Remap
                old2new= {old_id:(i+1) for i,old_id in enumerate(np.unique(cluster_labels))}
                cluster_labels= np.array([old2new[x] for x in cluster_labels])
                return cluster_labels
        except Exception as e:
            st.warning(f"HDBSCAN error: {str(e)}. Trying hierarchical next...")

    # Attempt hierarchical
    try:
        st.info("Trying hierarchical clustering next...")
        from sklearn.cluster import KMeans
        methods= ['ward','complete','average']
        best_method='ward'
        Z= linkage(embeddings, method=best_method)
        labels= fcluster(Z, t=num_clusters, criterion="maxclust")
        return labels
    except Exception as e:
        st.error(f"Hierarchical clustering error: {str(e)}. Fallback to K-Means.")
        try:
            from sklearn.cluster import KMeans
            kmeans= KMeans(n_clusters=num_clusters, random_state=42,n_init=10)
            return kmeans.fit_predict(embeddings)+1
        except:
            st.warning("K-Means fallback also failed. Assigning random clusters.")
            return np.random.randint(1, num_clusters+1,size=len(embeddings))


def refine_clusters(df, embeddings, original_cluster_column='cluster_id'):
    """
    Refine clusters by outlier detection & merging. 
    (Already in your original code.)
    """
    st.info("Refining clusters to improve coherence...")
    df['original_cluster']= df[original_cluster_column]

    # 1. Outliers
    outliers=[]
    for cid in df[original_cluster_column].unique():
        c_idx= df[df[original_cluster_column]== cid].index.tolist()
        if len(c_idx)<=3:
            continue
        c_embs= np.array([embeddings[i] for i in c_idx])
        centroid= np.mean(c_embs, axis=0)
        dist= [np.linalg.norm(embeddings[i]- centroid) for i in c_idx]
        md= np.mean(dist)
        sd= np.std(dist)
        if sd==0: 
            continue
        z= [(d-md)/sd for d in dist]
        for i, val in enumerate(z):
            if val>2.0:
                outliers.append((c_idx[i],cid,val))

    # Reassign outliers
    reassigned=0
    for idx, ocid, _ in outliers:
        emb= embeddings[idx]
        min_distance= 1e9
        best_c= ocid
        for c_id in df[original_cluster_column].unique():
            if c_id==ocid:
                continue
            c_idx= df[df[original_cluster_column]==c_id].index.tolist()
            c_embs= np.array([embeddings[i] for i in c_idx])
            cent= np.mean(c_embs, axis=0)
            dist= np.linalg.norm(emb- cent)
            if dist<min_distance:
                min_distance=dist
                best_c= c_id
        if best_c != ocid:
            df.loc[idx, original_cluster_column]= best_c
            reassigned+=1

    # 2. Merge highly similar clusters
    sim_pairs=[]
    c_list= df[original_cluster_column].unique()
    for i, c1 in enumerate(c_list):
        for c2 in c_list[i+1:]:
            idx1= df[df[original_cluster_column]== c1].index.tolist()
            idx2= df[df[original_cluster_column]== c2].index.tolist()
            if len(idx1)<3 or len(idx2)<3:
                continue
            emb1= np.array([embeddings[x] for x in idx1])
            emb2= np.array([embeddings[x] for x in idx2])
            cent1= np.mean(emb1, axis=0)
            cent2= np.mean(emb2, axis=0)
            sim= np.dot(cent1, cent2)/ (np.linalg.norm(cent1)* np.linalg.norm(cent2))
            if sim>0.8:
                sim_pairs.append((c1,c2,sim))

    sim_pairs.sort(key=lambda x:x[2], reverse=True)
    merges=0
    used= set()
    for (c1,c2,_) in sim_pairs:
        if c1 in used or c2 in used:
            continue
        keep= min(c1,c2)
        remove= max(c1,c2)
        df.loc[df[original_cluster_column]== remove, original_cluster_column]= keep
        used.add(remove)
        merges+=1
        if merges>= len(c_list)//4:
            break

    st.success(f"Refinement done: {reassigned} outliers reassigned, {merges} merges." )
    return df


################################################################
#          GENERATE CLUSTER NAMES (with search intent)
################################################################

def generate_cluster_names(
    clusters_with_representatives,
    client,
    model="gpt-3.5-turbo",
    custom_prompt=None,
    selected_language="English"
):
    """
    We'll also include "search_intent" among 
    (informational, navigational, transactional, commercial).
    """
    if not clusters_with_representatives:
        return {}

    results={}
    
    progress_text= st.empty()
    progress_bar= st.progress(0)
    progress_text.text("Generating SEO-friendly cluster names/descriptions...")

    if not custom_prompt:
        custom_prompt= f"""
You are an SEO expert. The user is working in {selected_language}.
For each cluster:
1) Provide a short, clear name (3-6 words).
2) Provide a concise SEO meta description (1-2 sentences).
3) Determine the search intent (informational, navigational, transactional, commercial).

Respond ONLY with a JSON called "clusters", e.g.:

[
  {{
    "cluster_id": 1,
    "cluster_name": "Some name",
    "cluster_description": "some short description",
    "search_intent": "informational"
  }},
  ...
]
"""

    naming_prompt= custom_prompt.strip()+"\n\nHere are the clusters:\n"
    for cid, kws in clusters_with_representatives.items():
        subset= kws[:15]
        naming_prompt+= f"- Cluster {cid}: {', '.join(subset)}\n"

    naming_prompt+= "\nReturn ONLY the JSON array 'clusters'. Nothing else."

    try:
        response= client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":naming_prompt}],
            temperature=0.3,
            max_tokens=800
        )
        content= response.choices[0].message.content.strip()
        
        json_data= None
        try:
            json_data= json.loads(content)
        except json.JSONDecodeError:
            # Attempt to locate JSON
            match= re.search(r'(\{.*\"clusters\".*\})', content, re.DOTALL)
            if match:
                piece= match.group(1)
                piece= piece.replace("'",'"')
                piece= re.sub(r',\s*}','}',piece)
                piece= re.sub(r',\s*\]','}',piece)
                try:
                    json_data= json.loads(piece)
                except:
                    pass

        if not json_data or "clusters" not in json_data:
            st.warning("Could not parse GPT JSON. Using fallback.")
            st.text_area("GPT raw response",content, height=300)
            for c_id in clusters_with_representatives.keys():
                results[c_id]= (f"Cluster {c_id}",f"Description {c_id}","informational")
            return results

        cluster_array= json_data["clusters"]
        for item in cluster_array:
            cid= item.get("cluster_id")
            cname= item.get("cluster_name", f"Cluster {cid}")
            cdesc= item.get("cluster_description", "No SEO desc")
            cintent= item.get("search_intent","informational").lower()
            if cid is not None:
                results[cid]= (cname,cdesc,cintent)
    except Exception as e:
        st.error(f"Error in generate_cluster_names: {str(e)}")
        for c_id in clusters_with_representatives.keys():
            results[c_id]=(f"Cluster {c_id}", f"Fallback desc {c_id}", "informational")

    progress_bar.progress(1.0)
    progress_text.text("✅ SEO cluster naming + search intent done.")
    return results


################################################################
#          EVALUATION FUNCTIONS
################################################################

def evaluate_cluster_quality(df, embeddings, cluster_column='cluster_id'):
    """
    Evaluate cluster with silhouette/density/separation/coherence 
    and plot them. 
    """
    st.subheader("Advanced Cluster Quality Evaluation")
    
    metrics= {
        'silhouette': [],
        'density': [],
        'separation': [],
        'coherence': []
    }
    
    centroids={}
    for cid in df[cluster_column].unique():
        idxs= df[df[cluster_column]==cid].index.tolist()
        centroids[cid]= np.mean(np.array([embeddings[i] for i in idxs]), axis=0)

    cluster_progress= st.progress(0)
    unique_c= df[cluster_column].unique()
    
    for i, cid in enumerate(unique_c):
        cluster_idxs= df[df[cluster_column]==cid].index.tolist()
        c_embs= np.array([embeddings[j] for j in cluster_idxs])
        centroid= centroids[cid]
        
        # density
        dist= [np.linalg.norm(emb - centroid) for emb in c_embs]
        density= 1./(1.+ np.mean(dist)) if dist else 0
        metrics['density'].append((cid, density))
        
        # coherence
        coh= calculate_cluster_coherence(c_embs)
        metrics['coherence'].append((cid, coh))
        
        # separation
        min_sep= 99999
        for other_cid,other_cent in centroids.items():
            if other_cid!= cid:
                dist2= np.linalg.norm(centroid-other_cent)
                if dist2<min_sep:
                    min_sep= dist2
        if min_sep<99999:
            metrics['separation'].append((cid,min_sep))
        
        cluster_progress.progress(float(i+1)/ len(unique_c))

    # Plot
    col1, col2= st.columns(2)
    with col1:
        st.subheader("Coherence vs. Size")
        c_data= pd.DataFrame(metrics['coherence'], columns=['cluster_id','score'])
        c_size= df.groupby(cluster_column)['keyword'].count().reset_index()
        c_data= c_data.merge(c_size, on='cluster_id')
        c_data= c_data.merge(df.drop_duplicates(cluster_column)[['cluster_id','cluster_name']], on='cluster_id')
        fig= px.scatter(
            c_data,
            x='score',
            y='keyword',
            color='score',
            size='keyword',
            hover_data=['cluster_name'],
            labels={'score':'Semantic Coherence','keyword':'Cluster Size'},
            title='Relationship between Coherence & Size',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Separation vs. Density")
        if metrics['separation']:
            s_data= pd.DataFrame(metrics['separation'], columns=['cluster_id','separation'])
            d_data= pd.DataFrame(metrics['density'], columns=['cluster_id','density'])
            combo= s_data.merge(d_data, on='cluster_id')
            combo= combo.merge(df.drop_duplicates(cluster_column)[['cluster_id','cluster_name']], on='cluster_id')
            fig2= px.scatter(
                combo,
                x='separation',
                y='density',
                color='density',
                hover_data=['cluster_name'],
                labels={'separation':'Inter-cluster Separation','density':'Cluster Density'},
                title='Separation vs. Density',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cluster Diagnostics")
    cthresh= np.percentile([x[1] for x in metrics['coherence']],25)
    problem= [x[0] for x in metrics['coherence'] if x[1]< cthresh]

    # Assign final coherence
    for cid, cval in metrics['coherence']:
        df.loc[df[cluster_column]== cid, 'cluster_coherence']= cval
    
    if problem:
        st.warning(f"Clusters with low semantic coherence: {problem}")
        st.info("""
        Recommendations:
        - Increase number of clusters
        - Check keywords in these clusters
        - Use better embeddings
        - Possibly split them
        """)
    else:
        st.success("All clusters have good coherence!")
    
    return df

def calculate_cluster_coherence(cluster_embeddings):
    """
    Average cosine similarity to centroid
    """
    if len(cluster_embeddings)<=1:
        return 1.0
    try:
        cent= np.mean(cluster_embeddings, axis=0)
        sims=[]
        for emb in cluster_embeddings:
            n1= np.linalg.norm(emb)
            n2= np.linalg.norm(cent)
            if n1>0 and n2>0:
                sim= np.dot(emb, cent)/(n1*n2)
                sims.append(sim)
            else:
                sims.append(0.0)
        return np.mean(sims) if sims else 0.0
    except Exception as e:
        st.warning(f"coherence calc error: {str(e)}")
        return 0.5


################################################################
#          AI-BASED CLUSTER EVALUATION (SPLITTING)
################################################################

def evaluate_and_refine_clusters(df, client, model="gpt-3.5-turbo"):
    """
    Ask GPT if each cluster should be splitted or has subtopics
    """
    st.subheader("AI-Powered Cluster Splitting & Insights")
    unique_c= df['cluster_id'].unique()
    if len(unique_c)==0:
        st.warning("No clusters to evaluate.")
        return {}
    
    # We'll sample up to 5 clusters
    import random
    sample_c= random.sample(list(unique_c), min(5, len(unique_c)))
    eval_progress= st.progress(0)
    eval_results= {}

    for i, cid in enumerate(sample_c):
        c_df= df[df['cluster_id']== cid]
        c_name= c_df['cluster_name'].iloc[0]
        c_kws= c_df['keyword'].tolist()
        
        prompt= f"""
Evaluate the following cluster for possible internal splitting:

Cluster ID: {cid}
Name: {c_name}
Keywords: {', '.join(c_kws[:50])}

Return JSON:
{{
  "should_split": true/false,
  "split_recommendations": ["subtopic or reason 1", "subtopic or reason 2", ...],
  "other_insights": ["any additional info or suggestions"]
}}
"""
        try:
            response= client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=400
            )
            content= response.choices[0].message.content.strip()
            data={}
            try:
                data= json.loads(content)
            except:
                data= {"raw_response": content, "should_split":False, "split_recommendations":[],"other_insights":[]}
            
            eval_results[cid]= {
                "name": c_name,
                "evaluation": data
            }
        except Exception as e:
            st.error(f"Error evaluating cluster {cid} with AI: {str(e)}")
        
        eval_progress.progress(float(i+1)/ len(sample_c))

    return eval_results


################################################################
#          POST-CLUSTER ACTIONS: Architecture & Content
################################################################

def propose_information_architecture(df, selected_model="gpt-3.5-turbo"):
    """
    Generate a web architecture plan from the clusters. 
    We do a quick cost estimate & placeholders.
    """
    st.subheader("Information Architecture Proposal")
    c_count= len(df['cluster_id'].unique())
    arch_cost= 0.0001* c_count  # placeholder cost
    st.info(f"Estimated cost for architecture proposal: ${arch_cost:.4f}")

    st.write("**Sample Outline** (Placeholder):")
    st.write("- Top-level pages matching each cluster, sub-pages for subtopics...")


def propose_contents_by_intent(df, selected_model="gpt-3.5-turbo"):
    """
    Proposed content plan by search intent, with user specifying # of topics/cluster
    plus cost estimate.
    """
    st.subheader("Content Proposal by Search Intent")
    cluster_count= len(df['cluster_id'].unique())
    topics= st.number_input("Number of content topics per cluster", min_value=1, max_value=20, value=3)
    cont_cost= 0.0002*(cluster_count* topics)
    st.info(f"Estimated cost for content proposal: ${cont_cost:.4f}")

    if st.button("Generate Content Proposal"):
        st.success("**Placeholder**: recommended articles for each cluster, grouped by search intent, etc.")


################################################################
#          MAIN PIPELINE
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
    csv_format
):
    if uploaded_file is None:
        st.warning("Please upload a CSV.")
        return False, None
    
    st.info("Starting advanced semantic clustering pipeline...")

    # Setup client
    client=None
    if openai_api_key and openai_available:
        try:
            os.environ["OPENAI_API_KEY"]= openai_api_key
            client= OpenAI()
            # Quick test
            try:
                _= client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":"Test"}],
                    max_tokens=5
                )
                st.success("✅ OpenAI connected.")
            except Exception as e:
                st.error(f"OpenAI check error: {str(e)}")
                client=None
        except Exception as e:
            st.error(f"Error configuring OpenAI client: {str(e)}")
            client=None
    elif not openai_available:
        st.warning("OpenAI library is not installed, no GPT usage possible.")
    else:
        st.info("No OpenAI key, fallback methods only.")
    
    try:
        # Read CSV differently
        if csv_format=="no_header":
            df= pd.read_csv(uploaded_file, header=None, names=["keyword"])
            st.success(f"Loaded {len(df)} keywords (no header).")
        else:
            df= pd.read_csv(uploaded_file, header=0)
            if "Keyword" in df.columns:
                df.rename(columns={"Keyword":"keyword"}, inplace=True)
            if "keyword" not in df.columns:
                st.error("No 'Keyword' column found, please check your CSV.")
                return False, None
            st.success(f"Loaded {len(df)} rows with a header (possibly advanced columns).")
        
        num_keywords= len(df)
        show_csv_cost_estimate(num_keywords, gpt_model, num_clusters)
        
        st.subheader("Keyword Preprocessing")
        use_advanced= spacy_available
        df['keyword_processed']= preprocess_keywords(df["keyword"].tolist(), use_advanced=use_advanced)
        st.success("Preprocessing done.")
        
        # Embeddings
        st.subheader("Generating Embeddings")
        embeddings= generate_embeddings(df, openai_available, openai_api_key)
        
        # PCA
        if embeddings.shape[1]> max_pca_components:
            st.subheader("Dimensionality Reduction (PCA)")
            try:
                pca= PCA()
                pca.fit(embeddings)
                cum_var= np.cumsum(pca.explained_variance_ratio_)
                target_var= pca_variance/100.
                n_comp= np.argmax(cum_var>= target_var)+1
                if n_comp==1 and len(cum_var)>1:
                    n_comp= min(max_pca_components,len(cum_var))
                st.info(f"Components for {pca_variance}% variance: {n_comp}")
                
                max_comp= min(n_comp, max_pca_components)
                pca= PCA(n_components=max_comp)
                embeddings_reduced= pca.fit_transform(embeddings)
                st.success(f"Applied PCA: {max_comp} dims (~{pca_variance}% variance).")
            except Exception as e:
                st.error(f"PCA error: {str(e)}")
                embeddings_reduced= embeddings
        else:
            embeddings_reduced= embeddings
        
        # Clustering
        st.subheader("Advanced Clustering")
        labels= improved_clustering(embeddings_reduced, num_clusters= num_clusters)
        df["cluster_id"]= labels
        st.success(f"{len(df['cluster_id'].unique())} clusters created.")
        
        # Refinement
        st.subheader("Refinement")
        df= refine_clusters(df, embeddings_reduced)
        
        # Representative
        st.subheader("Representative Keywords")
        clusters_with_reps= {}
        unique_clusters= df['cluster_id'].unique()
        for cid in unique_clusters:
            idxs= df[df['cluster_id']==cid].index.tolist()
            n_rep= min(20,len(idxs))
            # We'll do centroid approach for reps:
            c_embs= np.array([embeddings_reduced[i] for i in idxs])
            centroid= np.mean(c_embs,axis=0)
            dist= [np.linalg.norm(embeddings_reduced[i]- centroid) for i in idxs]
            sorted_idx= np.argsort(dist)[:n_rep]
            rep_ids= [idxs[x] for x in sorted_idx]
            reps= df.loc[rep_ids,'keyword'].tolist()
            clusters_with_reps[cid]= reps
        
        # GPT naming + intent
        if client:
            st.subheader("Generating Cluster Names & Search Intent")
            cluster_names= generate_cluster_names(
                clusters_with_representatives=clusters_with_reps,
                client= client,
                model=gpt_model,
                custom_prompt=user_prompt,
                selected_language= st.session_state.get("language_choice","English")
            )
        else:
            # fallback
            cluster_names= {cid:(f"Cluster {cid}",f"Keywords group {cid}","informational") for cid in unique_clusters}

        df['cluster_name']=''
        df['cluster_description']=''
        df['search_intent']=''
        df['representative']=False
        for cid, (cname,cdesc,cintent) in cluster_names.items():
            df.loc[df['cluster_id']==cid,'cluster_name']= cname
            df.loc[df['cluster_id']==cid,'cluster_description']= cdesc
            df.loc[df['cluster_id']==cid,'search_intent']= cintent
            for kw in clusters_with_reps[cid]:
                match_idx= df[(df['cluster_id']==cid)&(df['keyword']==kw)].index
                if not match_idx.empty:
                    df.loc[match_idx,'representative']= True
        
        # Evaluate
        df= evaluate_cluster_quality(df, embeddings_reduced)
        
        # AI-based cluster evaluation
        if client:
            st.subheader("AI-based Cluster Splitting & Insights")
            try:
                eval_res= evaluate_and_refine_clusters(df, client, model=gpt_model)
                st.session_state.cluster_evaluation= eval_res
            except Exception as e:
                st.error(f"AI-based evaluation error: {str(e)}")
        
        # Show total cost 
        total_cost= calculate_api_cost(num_keywords, gpt_model, num_clusters)['total_cost']
        st.info(f"**Total cost to process** this CSV: approx ${total_cost:.4f}")
        
        return True, df

    except Exception as e:
        st.error(f"Error in pipeline: {str(e)}")
        return False, None


################################################################
#   ACTIONS AFTER CLUSTERING: ARCHITECTURE & CONTENT
################################################################

def propose_information_architecture(df, selected_model="gpt-3.5-turbo"):
    st.subheader("Information Architecture Proposal")
    c_count= len(df['cluster_id'].unique())
    arch_cost= 0.0001* c_count
    st.info(f"Estimated cost for architecture action: ${arch_cost:.4f}")
    st.write("**Proposed Outline** (placeholder). Each cluster becomes a top-level page, etc.")


def propose_contents_by_intent(df, selected_model="gpt-3.5-turbo"):
    st.subheader("Content Proposal by Search Intent")
    cluster_count= len(df['cluster_id'].unique())
    topics= st.number_input("Number of content topics per cluster", min_value=1, max_value=20, value=3)
    c_cost= 0.0002*(cluster_count* topics)
    st.info(f"Estimated cost for content proposal: ${c_cost:.4f}")

    if st.button("Generate Content Proposal"):
        st.success("**Placeholder**: recommended content, each cluster mapped to topics by intent.")


################################################################
#          STREAMLIT APP
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
Upload your CSV of keywords, either no_header or with_header. 
Configure parameters, get SEO naming, search intent, 
and optionally propose site architecture or content by cluster.
""")

with st.expander("CSV Format Info", expanded=False):
    st.markdown("""
**No Header**: each line is just one keyword.
**With Header**: first line has columns like 'Keyword', 'search_volume', etc.
""")

# Download sample template
if st.sidebar.button("Download Sample CSV Template"):
    templ= generate_sample_csv()
    st.sidebar.download_button(
        label="Download CSV Header",
        data=templ,
        file_name="sample_keyword_planner_template.csv",
        mime="text/csv",
        use_container_width=True
    )

csv_format= st.sidebar.selectbox("CSV Format", ["no_header","with_header"], index=0)
uploaded_file= st.sidebar.file_uploader("Upload your CSV", type=["csv"])

openai_api_key= st.sidebar.text_input("OpenAI API Key (optional)", type="password", help="Use for best embeddings/naming. Omit for free fallback.")
language_option= st.sidebar.selectbox("Language of project/keywords", ["English","Spanish","French","German","Portuguese"], index=0)
st.session_state["language_choice"]= language_option

st.sidebar.markdown("<div class='sub-header'>Parameters</div>", unsafe_allow_html=True)

with st.sidebar.expander("ℹ️ Parameters Guide", expanded=False):
    st.markdown("""
    - Number of clusters
    - PCA variance 
    - etc.
    """)

num_clusters= st.sidebar.slider("Number of clusters",2,50,10)
pca_variance= st.sidebar.slider("PCA explained variance (%)",50,99,95)
max_pca_components= st.sidebar.slider("Max PCA components",10,300,100)
min_df= st.sidebar.slider("Minimum term frequency",1,10,1)
max_df= st.sidebar.slider("Maximum term frequency (%)",50,100,95)
gpt_model= st.sidebar.selectbox("Model for naming clusters", ["gpt-3.5-turbo","gpt-4"],index=0)

st.sidebar.markdown("### Custom Prompt for Cluster Naming")
default_prompt=(
    "You are an SEO expert. Below you'll see several clusters with representative keywords. "
    "Name each cluster (3-6 words), give a concise SEO meta description (1-2 sentences), "
    "and label search intent as informational/navigational/transactional/commercial. "
    "Return only JSON."
)
user_prompt= st.sidebar.text_area("Prompt", value=default_prompt, height=150)

add_cost_calculator()

if 'process_complete' not in st.session_state:
    st.session_state.process_complete= False
if 'df_results' not in st.session_state:
    st.session_state.df_results= None

if uploaded_file and not st.session_state.process_complete:
    colA, colB, colC= st.columns([1,2,1])
    with colB:
        if st.button("Start Clustering",type="primary",use_container_width=True):
            success, df_res= run_clustering(
                uploaded_file=uploaded_file,
                openai_api_key=openai_api_key,
                num_clusters=num_clusters,
                pca_variance=pca_variance,
                max_pca_components=max_pca_components,
                min_df=min_df,
                max_df=max_df,
                gpt_model=gpt_model,
                user_prompt=user_prompt,
                csv_format=csv_format
            )
            if success and df_res is not None:
                st.session_state.df_results= df_res
                st.session_state.process_complete= True
                st.markdown("<div class='success-box'>✅ Clustering completed successfully!</div>",unsafe_allow_html=True)

if st.session_state.process_complete and st.session_state.df_results is not None:
    st.markdown("<div class='main-header'>Clustering Results</div>", unsafe_allow_html=True)
    df= st.session_state.df_results
    with st.expander("Visualizations", expanded=True):
        st.subheader("Cluster Distribution")
        cluster_sizes= df.groupby(['cluster_id','cluster_name']).size().reset_index(name='count')
        cluster_sizes['label']= cluster_sizes.apply(lambda x: f"{x['cluster_name']} (ID:{x['cluster_id']})", axis=1)
        
        fig= px.bar(
            cluster_sizes, x='label', y='count', color='count',
            labels={'count':'Number of Keywords','label':'Cluster'},
            title='Cluster Size',
            color_continuous_scale= px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Semantic Coherence")
        coh_data= df.groupby(['cluster_id','cluster_name'])['cluster_coherence'].mean().reset_index()
        coh_data['label']= coh_data.apply(lambda x: f"{x['cluster_name']} (ID:{x['cluster_id']})", axis=1)
        fig2= px.bar(
            coh_data, x='label', y='cluster_coherence',
            color='cluster_coherence',
            labels={'cluster_coherence':'Coherence','label':'Cluster'},
            title='Coherence by Cluster',
            color_continuous_scale=px.colors.sequential.Greens
        )
        st.plotly_chart(fig2,use_container_width=True)

    with st.expander("Explore Clusters", expanded=True):
        st.subheader("Cluster Details")
        c_opts= [
            f"{row['cluster_name']} (ID:{row['cluster_id']})"
            for _, row in df.drop_duplicates(['cluster_id','cluster_name'])[['cluster_id','cluster_name']].iterrows()
        ]
        sel= st.selectbox("Select a cluster", c_opts)
        if sel:
            cid= int(sel.split("ID:")[1].split(")")[0])
            cdf= df[df['cluster_id']== cid].copy()
            st.write(f"### {cdf['cluster_name'].iloc[0]}")
            st.write(f"**Description**: {cdf['cluster_description'].iloc[0]}")
            st.write(f"**Search Intent**: {cdf['search_intent'].iloc[0]}")
            st.write(f"**Coherence**: {cdf['cluster_coherence'].iloc[0]:.3f}")
            st.write(f"**Keywords in cluster**: {len(cdf)}")
            reps= cdf[cdf['representative']==True]['keyword'].tolist()
            if reps:
                st.write("**Representative keywords**:")
                st.write(", ".join(reps[:15]))
            
            # If AI evaluation
            if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
                e_dict= st.session_state.cluster_evaluation
                if cid in e_dict:
                    st.subheader("AI-based Splitting Suggestions / Insights")
                    st.json(e_dict[cid]['evaluation'])
            
            # Show all
            st.write("### All keywords")
            st.dataframe(cdf[['keyword']], use_container_width=True)

    with st.expander("Download Results", expanded=True):
        csv_data= df.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_data,
            file_name="semantic_clustered_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.subheader("Clusters Summary")
        summary_df= df.groupby(['cluster_id','cluster_name','cluster_description','search_intent'])['keyword'].count().reset_index()
        summary_df.columns= ['ID','Name','Description','Search Intent','Number of Keywords']
        cohs= df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary_df= summary_df.merge(cohs, left_on='ID', right_on='cluster_id')
        summary_df.drop('cluster_id', axis=1,inplace=True)
        summary_df.rename(columns={'cluster_coherence':'Coherence'}, inplace=True)

        def reps(c):
            r= df[(df['cluster_id']==c)&(df['representative']==True)]['keyword'].tolist()
            return ', '.join(r[:5])
        summary_df['Representative Keywords']= summary_df['ID'].apply(reps)

        if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
            e_ids= st.session_state.cluster_evaluation.keys()
            summary_df['AI Evaluation?']= summary_df['ID'].apply(lambda x:"Yes" if x in e_ids else "No")
        else:
            summary_df['AI Evaluation?']= "No"
        
        st.dataframe(summary_df, use_container_width=True)
        
        sum_csv= summary_df.to_csv(index=False)
        st.download_button(
            label="Download Clusters Summary",
            data=sum_csv,
            file_name="semantic_clusters_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Additional Actions")
    st.markdown("""
    - **Propose Information Architecture** based on clusters
    - **Propose Content by Search Intent** 
    """)
    c1,c2= st.columns(2)
    with c1:
        if st.button("Propose Information Architecture", use_container_width=True):
            propose_information_architecture(df, selected_model=gpt_model)
    with c2:
        if st.button("Propose Content by Intent", use_container_width=True):
            propose_contents_by_intent(df, selected_model=gpt_model)

if 'process_complete' in st.session_state and st.session_state.process_complete:
    if st.button("Reset", use_container_width=True):
        st.session_state.process_complete= False
        st.session_state.df_results= None
        st.experimental_rerun()

with st.expander("About Search Intent Types"):
    st.markdown("""
- **Informational**: user wants knowledge or answers ("how to..., what is...")
- **Navigational**: user wants a specific site/page ("facebook login", "gmail inbox")
- **Transactional**: user is ready to buy or take direct action ("buy iphone 13", "purchase domain")
- **Commercial**: user is researching or comparing ("best laptops 2023", "reviews for DSLR cameras")
""")

st.markdown("""
---
<div style="text-align:center; color:#888;">
Advanced semantic keyword clustering with CSV format options, search intent, architecture & content proposals
</div>
""", unsafe_allow_html=True)
