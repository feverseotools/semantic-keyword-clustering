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

# Attempt advanced libraries
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False

try:
    import spacy
    try:
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

# Download NLTK resources on startup
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass  # proceed even if downloads fail

################################################################
#   COST CALCULATION & SUPPORT
################################################################

def calculate_api_cost(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    """
    Estimate cost of using OpenAI for embeddings & naming.
    """
    # Pricing references (March 2025, hypothetical)
    EMBEDDING_COST_PER_1K = 0.02
    GPT35_INPUT_COST_PER_1K = 0.0005
    GPT35_OUTPUT_COST_PER_1K = 0.0015
    GPT4_INPUT_COST_PER_1K = 0.03
    GPT4_OUTPUT_COST_PER_1K = 0.06

    results = {
        "embedding_cost": 0.0,
        "naming_cost": 0.0,
        "total_cost": 0.0,
        "processed_keywords": 0
    }

    # 1) Embedding cost
    keywords_for_embeddings = min(num_keywords, 5000)
    results["processed_keywords"] = keywords_for_embeddings
    estimated_tokens = keywords_for_embeddings * 2
    results["embedding_cost"] = (estimated_tokens/1000.) * EMBEDDING_COST_PER_1K

    # 2) Naming cost
    avg_tokens_per_cluster = 200
    avg_output_tokens_per_cluster = 80
    est_input= num_clusters* avg_tokens_per_cluster
    est_output= num_clusters* avg_output_tokens_per_cluster

    if selected_model=="gpt-3.5-turbo":
        in_cost= (est_input/1000.)* GPT35_INPUT_COST_PER_1K
        out_cost= (est_output/1000.)* GPT35_OUTPUT_COST_PER_1K
    else:
        in_cost= (est_input/1000.)* GPT4_INPUT_COST_PER_1K
        out_cost= (est_output/1000.)* GPT4_OUTPUT_COST_PER_1K

    results["naming_cost"] = in_cost + out_cost
    results["total_cost"]= results["embedding_cost"] + results["naming_cost"]
    return results

def add_cost_calculator():
    st.sidebar.markdown("---")
    with st.sidebar.expander("💰 API Cost Calculator", expanded=False):
        st.markdown("### API Cost Calculator")
        calc_num_keywords= st.number_input("Number of keywords to process", min_value=100, max_value=100000, value=1000, step=500)
        calc_num_clusters= st.number_input("Approximate number of clusters", min_value=2, max_value=50, value=10, step=1)
        calc_model= st.radio("Model for naming clusters", options=["gpt-3.5-turbo","gpt-4"], index=0, horizontal=True)
        if st.button("Calculate Estimated Cost", use_container_width=True):
            cost= calculate_api_cost(calc_num_keywords, calc_model, calc_num_clusters)
            colA, colB= st.columns(2)
            with colA:
                st.metric("Keywords processed with OpenAI", f"{cost['processed_keywords']:,}",
                          help="Max 5,000 direct embeddings, rest via similarity.")
                st.metric("Embeddings cost", f"${cost['embedding_cost']:.4f}",
                          help="Cost with text-embedding-3-small")
            with colB:
                st.metric("Cluster naming cost", f"${cost['naming_cost']:.4f}",
                          help=f"Using {calc_model}")
                st.metric("TOTAL COST", f"${cost['total_cost']:.4f}",
                          help="Approx total cost")
            st.info("""
            **Note**: This is an estimate. Real cost can vary by tokens used.
            SentenceTransformers or TF-IDF is free ($0).
            """)

def show_csv_cost_estimate(num_keywords, selected_model="gpt-3.5-turbo", num_clusters=10):
    if num_keywords>0:
        cost_res= calculate_api_cost(num_keywords, selected_model, num_clusters)
        with st.sidebar.expander("💰 Estimated Cost (Current CSV)", expanded=True):
            st.markdown(f"### For {num_keywords:,} Keywords")
            st.markdown(f"""
- **Keywords processed with OpenAI**: {cost_res['processed_keywords']:,}
- **Embeddings cost**: ${cost_res['embedding_cost']:.4f}
- **Cluster naming cost**: ${cost_res['naming_cost']:.4f}
- **TOTAL COST**: ${cost_res['total_cost']:.4f}
""")
            if cost_res['processed_keywords']< num_keywords:
                st.info(f"{cost_res['processed_keywords']:,} direct embeddings, the rest via similarity.")
            st.markdown("""
            **If not using OpenAI**: fallback to SentenceTransformers or TF-IDF is free.
            """)

################################################################
#     SAMPLE CSV GENERATION
################################################################

def generate_sample_csv():
    header= ["Keyword","search_volume","competition","cpc"]
    months= [f"month{i}" for i in range(1,13)]
    header += months
    return ",".join(header)+"\n"


################################################################
#  PREPROCESSING
################################################################

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

def enhanced_preprocessing(text, use_lemmatization=True):
    if not isinstance(text,str) or not text.strip():
        return ""
    try:
        if spacy_available:
            doc= nlp(text.lower())
            entities= [ent.text for ent in doc.ents]
            tokens=[]
            for token in doc:
                if not token.is_stop and token.is_alpha and len(token.text)>1:
                    tokens.append(token.lemma_)
            bigrams=[]
            for i in range(len(doc)-1):
                if (not doc[i].is_stop and not doc[i+1].is_stop
                    and doc[i].is_alpha and doc[i+1].is_alpha):
                    bigrams.append(f"{doc[i].lemma_}_{doc[i+1].lemma_}")
            processed= tokens+ bigrams+ entities
            return " ".join(processed)
        elif textblob_available:
            from textblob import TextBlob
            blob= TextBlob(text.lower())
            try:
                stop_words= set(stopwords.words('english'))
            except:
                stop_words= {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
            noun_phrases= list(blob.noun_phrases)
            words= [w for w in blob.words if len(w)>1 and w.lower() not in stop_words]
            if use_lemmatization:
                lemmatizer= WordNetLemmatizer()
                lemmas= [lemmatizer.lemmatize(w) for w in words]
                processed= lemmas+ noun_phrases
            else:
                processed= words+ noun_phrases
            return " ".join(processed)
        else:
            return preprocess_text(text, use_lemmatization)
    except:
        return text.lower() if isinstance(text,str) else ""

def preprocess_text(text, use_lemmatization=True):
    if not isinstance(text,str) or not text.strip():
        return ""
    try:
        text= text.lower()
        tokens= word_tokenize(text)
        try:
            sw= set(stopwords.words('english'))
        except:
            sw= {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
        tokens= [t for t in tokens if t.isalpha() and t not in sw]
        if use_lemmatization:
            try:
                lem= WordNetLemmatizer()
                tokens= [lem.lemmatize(x) for x in tokens]
            except:
                pass
        return " ".join(tokens)
    except:
        return text.lower() if isinstance(text,str) else ""

def preprocess_keywords(keywords, use_advanced=True):
    processed=[]
    pb= st.progress(0)
    total= len(keywords)
    if use_advanced:
        if spacy_available:
            st.success("Using advanced spaCy preprocessing")
        elif textblob_available:
            st.success("Using alternative TextBlob preprocessing")
        else:
            st.info("Using standard NLTK preprocessing")
    else:
        st.info("Using standard NLTK preprocessing")

    for i, kw in enumerate(keywords):
        if use_advanced and (spacy_available or textblob_available):
            processed.append(enhanced_preprocessing(kw))
        else:
            processed.append(preprocess_text(kw))
        if i%100==0:
            pb.progress(min(float(i)/float(total),1.0))
    pb.progress(1.0)
    return processed

################################################################
# EMBEDDING GENERATION
################################################################

def generate_embeddings(df, openai_available, openai_api_key=None):
    st.info("Generating embeddings for keywords...")
    keys= df['keyword_processed'].fillna('').tolist()

    # 1) OpenAI
    if openai_available and openai_api_key:
        try:
            st.info("Using OpenAI embeddings (semantic).")
            os.environ["OPENAI_API_KEY"]= openai_api_key
            cli= OpenAI()
            all_embeds=[]
            if len(keys)>5000:
                st.warning(f"Limiting to 5000 of {len(keys)} for direct embedding.")
                step= max(1, len(keys)//5000)
                sample_idx= list(range(0,len(keys),step))[:5000]
                sample_kw= [keys[i] for i in sample_idx]
                pb= st.progress(0)
                st.info("Requesting sample subset in one call.")
                resp= cli.embeddings.create(model="text-embedding-3-small", input=sample_kw)
                pb.progress(0.5)
                sample_embs= np.array([x.embedding for x in resp.data])
                st.info("Propagating rest with TF-IDF similarity.")
                vec= TfidfVectorizer()
                tfidf_mat= vec.fit_transform(keys)
                all_embeds= np.zeros((len(keys), sample_embs.shape[1]))
                for i, idx in enumerate(sample_idx):
                    all_embeds[idx]= sample_embs[i]
                from sklearn.neighbors import NearestNeighbors
                nn= NearestNeighbors(n_neighbors=min(3,len(sample_idx)))
                nn.fit(tfidf_mat[sample_idx])
                remain= [i for i in range(len(keys)) if i not in sample_idx]
                for i, rid in enumerate(remain):
                    dist, neighs= nn.kneighbors(tfidf_mat[rid:rid+1])
                    w= 1./(1.+ dist[0])
                    w/= w.sum()
                    weighted= np.zeros_like(sample_embs[0])
                    for j, wt in zip(neighs[0], w):
                        real_idx= sample_idx[j]
                        weighted+= wt* all_embeds[real_idx]
                    all_embeds[rid]= weighted
                    if i%100==0:
                        val= 0.5 + float(i)/ len(remain)*0.5
                        pb.progress(min(val,1.0))
                pb.progress(1.0)
            else:
                pb= st.progress(0)
                st.info(f"Embedding all {len(keys)} with OpenAI, in batches.")
                bs= 1000
                for i in range(0, len(keys), bs):
                    batch_end= min(i+bs, len(keys))
                    sub= keys[i:batch_end]
                    r= cli.embeddings.create(model="text-embedding-3-small", input=sub)
                    all_embeds.extend([xx.embedding for xx in r.data])
                    pb.progress(min(float(batch_end)/ len(keys),1.0))
                pb.progress(1.0)

            emb= np.array(all_embeds) if isinstance(all_embeds,list) else all_embeds
            st.success(f"✅ OpenAI embeddings done. Dim={emb.shape[1]}")
            return emb
        except Exception as e:
            st.error(f"OpenAI error: {str(e)}")
            st.info("Falling back to SentenceTransformers...")

    # 2) SentenceTransformers
    if sentence_transformers_available:
        try:
            st.success("Using SentenceTransformers (free fallback).")
            mod= SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            pb= st.progress(0)
            bs=512
            emb_all=[]
            for i in range(0, len(keys), bs):
                batch= keys[i:i+bs]
                e= mod.encode(batch, show_progress_bar=False)
                emb_all.extend(e)
                pb.progress(float(i+bs)/ len(keys))
            pb.progress(1.0)
            emb= np.array(emb_all)
            st.success(f"✅ ST embeddings. Dim={emb.shape[1]}")
            return emb
        except Exception as e:
            st.error(f"SentenceTransformers error: {str(e)}")

    # 3) TF-IDF fallback
    st.warning("Using TF-IDF fallback.")
    return generate_tfidf_embeddings(keys)

def generate_tfidf_embeddings(texts):
    st.info("Generating TF-IDF vectors (fallback).")
    pb= st.progress(0)
    try:
        vectorizer= TfidfVectorizer(max_features=300, stop_words='english')
        pb.progress(0.3)
        tfmat= vectorizer.fit_transform(texts)
        pb.progress(0.8)
        arr= tfmat.toarray()
        pb.progress(1.0)
        st.success(f"TF-IDF done. Dim={arr.shape[1]}")
        return arr
    except Exception as e:
        st.error(f"TF-IDF error: {str(e)}. Using random fallback.")
        return np.random.rand(len(texts), 100)

################################################################
#          CLUSTERING
################################################################

def improved_clustering(embeddings, num_clusters=None, min_cluster_size=5):
    st.info("Applying advanced clustering algorithms...")
    if num_clusters is None:
        st.warning("No num_clusters provided; default=10.")
        num_clusters=10

    # Attempt HDBSCAN
    if hdbscan_available:
        try:
            st.info("Trying HDBSCAN first...")
            clusterer= hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels= clusterer.fit_predict(embeddings)
            unique= np.unique(labels)
            non_noise= [c for c in unique if c!=-1]
            if len(non_noise)>1 and len(non_noise)<= (num_clusters*2):
                st.success(f"HDBSCAN found {len(non_noise)} clusters.")
                if -1 in unique:
                    noise_ids= np.where(labels==-1)[0]
                    for idx in noise_ids:
                        min_d= 1e9
                        best_cluster= non_noise[0]
                        for c_id in non_noise:
                            c_points= embeddings[labels== c_id]
                            if len(c_points)>0:
                                cent= np.mean(c_points,axis=0)
                                dist= np.linalg.norm(embeddings[idx]- cent)
                                if dist< min_d:
                                    min_d= dist
                                    best_cluster= c_id
                        labels[idx]= best_cluster
                old2new= {old:(i+1) for i,old in enumerate(np.unique(labels))}
                final= np.array([old2new[x] for x in labels])
                return final
        except Exception as e:
            st.warning(f"HDBSCAN error: {str(e)}. Trying hierarchical next.")

    # Attempt hierarchical
    try:
        st.info("Trying hierarchical clustering next (Ward).")
        from sklearn.cluster import KMeans
        Z= linkage(embeddings, method='ward')
        labs= fcluster(Z, t=num_clusters, criterion="maxclust")
        return labs
    except Exception as e:
        st.error(f"Hierarchical error: {str(e)}. Fallback to K-Means.")
        try:
            from sklearn.cluster import KMeans
            km= KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            return km.fit_predict(embeddings)+1
        except:
            st.warning("K-Means also failed. Assigning random clusters.")
            return np.random.randint(1, num_clusters+1, size=len(embeddings))


def refine_clusters(df, embeddings, original_cluster_column='cluster_id'):
    st.info("Refining clusters by outlier detection & merges...")
    df['original_cluster']= df[original_cluster_column]

    outliers=[]
    for cid in df[original_cluster_column].unique():
        idxs= df[df[original_cluster_column]==cid].index.tolist()
        if len(idxs)<=3:
            continue
        c_embs= np.array([embeddings[i] for i in idxs])
        cent= np.mean(c_embs, axis=0)
        dist= [np.linalg.norm(embeddings[i]- cent) for i in idxs]
        md= np.mean(dist)
        sd= np.std(dist)
        if sd==0: 
            continue
        zscores= [(d-md)/sd for d in dist]
        for i, val in enumerate(zscores):
            if val>2.0:
                outliers.append((idxs[i],cid,val))

    # reassign outliers
    reassign_count=0
    for idx, ocid, _ in outliers:
        e= embeddings[idx]
        min_d= 1e9
        best_c= ocid
        for c2 in df[original_cluster_column].unique():
            if c2== ocid:
                continue
            c_ix= df[df[original_cluster_column]== c2].index.tolist()
            c_arr= np.array([embeddings[x] for x in c_ix])
            c_cent= np.mean(c_arr, axis=0)
            d= np.linalg.norm(e- c_cent)
            if d< min_d:
                min_d= d
                best_c= c2
        if best_c!= ocid:
            df.loc[idx, original_cluster_column]= best_c
            reassign_count+=1

    # merging
    merges=0
    c_list= df[original_cluster_column].unique()
    sim_pairs=[]
    for i, c1 in enumerate(c_list):
        for c2 in c_list[i+1:]:
            i1= df[df[original_cluster_column]== c1].index.tolist()
            i2= df[df[original_cluster_column]== c2].index.tolist()
            if len(i1)<3 or len(i2)<3:
                continue
            emb1= np.array([embeddings[x] for x in i1])
            emb2= np.array([embeddings[x] for x in i2])
            cent1= np.mean(emb1, axis=0)
            cent2= np.mean(emb2, axis=0)
            sim= np.dot(cent1,cent2)/(np.linalg.norm(cent1)* np.linalg.norm(cent2))
            if sim>0.8:
                sim_pairs.append((c1,c2,sim))
    sim_pairs.sort(key=lambda x:x[2], reverse=True)
    used= set()
    for (c1,c2,_) in sim_pairs:
        if c1 in used or c2 in used:
            continue
        keep= min(c1,c2)
        remove= max(c1,c2)
        df.loc[df[original_cluster_column]== remove, original_cluster_column]= keep
        merges+=1
        used.add(remove)
        if merges>= len(c_list)//4:
            break

    st.success(f"Refinement done: {reassign_count} outliers reassigned, {merges} merges.")
    return df

################################################################
#          GENERATE CLUSTER NAMES (GPT)
################################################################

def generate_cluster_names(
    clusters_with_reps,
    client,
    model="gpt-3.5-turbo",
    custom_prompt=None,
    selected_language="English"
):
    if not clusters_with_reps:
        return {}

    st.info("Generating GPT-based cluster names & intent.")
    results={}
    if not custom_prompt:
        custom_prompt= f"""
You are an SEO expert. The user is working in {selected_language}.
For each cluster:
1) Provide a short name (3-6 words).
2) Provide a concise SEO meta description (1-2 sentences).
3) Search intent: informational/navigational/transactional/commercial

Return only a JSON array named 'clusters'.
"""
    prompt= custom_prompt.strip()+"\n\nHere are the clusters:\n"
    for cid, kws in clusters_with_reps.items():
        some= kws[:15]
        prompt+= f"- Cluster {cid}: {', '.join(some)}\n"
    prompt+= "\nReturn ONLY the JSON array 'clusters'. Nothing else."

    try:
        resp= client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=800
        )
        txt= resp.choices[0].message.content.strip()
        try:
            jd= json.loads(txt)
        except json.JSONDecodeError:
            # Attempt partial parse
            match= re.search(r'(\{.*\"clusters\".*\})', txt, re.DOTALL)
            if match:
                piece= match.group(1)
                piece= piece.replace("'",'"')
                piece= re.sub(r',\s*}','}',piece)
                piece= re.sub(r',\s*\]','}',piece)
                try:
                    jd= json.loads(piece)
                except:
                    jd= None
            else:
                jd= None
        if not jd or "clusters" not in jd:
            st.warning("Couldn't parse GPT JSON. Fallback names used.")
            st.text_area("GPT raw response", txt, height=300)
            for c_id in clusters_with_reps.keys():
                results[c_id]= (f"Cluster {c_id}", f"Description {c_id}", "informational")
            return results
        
        cluster_array= jd["clusters"]
        for item in cluster_array:
            cid= item.get("cluster_id")
            cname= item.get("cluster_name", f"Cluster {cid}")
            cdesc= item.get("cluster_description","No SEO desc")
            cintent= item.get("search_intent","informational").lower()
            if cid is not None:
                results[cid]= (cname,cdesc,cintent)
    except Exception as e:
        st.error(f"Error naming clusters with GPT: {str(e)}")
        for c_id in clusters_with_reps.keys():
            results[c_id]= (f"Cluster {c_id}",f"Desc fallback {c_id}","informational")

    return results

################################################################
#          EVALUATION
################################################################

def calculate_cluster_coherence(cluster_embeds):
    if len(cluster_embeds)<=1:
        return 1.0
    try:
        cent= np.mean(cluster_embeds, axis=0)
        sims=[]
        for emb in cluster_embeds:
            n1= np.linalg.norm(emb)
            n2= np.linalg.norm(cent)
            if n1>0 and n2>0:
                sims.append(np.dot(emb,cent)/(n1*n2))
            else:
                sims.append(0.)
        return np.mean(sims) if sims else 0.
    except:
        return 0.5

def evaluate_cluster_quality(df, embeddings, cluster_col='cluster_id'):
    st.subheader("Cluster Quality Evaluation")
    centroids={}
    for cid in df[cluster_col].unique():
        idx= df[df[cluster_col]==cid].index.tolist()
        centroids[cid]= np.mean(np.array([embeddings[i] for i in idx]), axis=0)
    progress= st.progress(0)
    unique_cluster_ids= df[cluster_col].unique()

    metrics= {
        'density': [],
        'separation': [],
        'coherence': []
    }

    for i, cid in enumerate(unique_cluster_ids):
        cluster_indices= df[df[cluster_col]==cid].index.tolist()
        c_arr= np.array([embeddings[x] for x in cluster_indices])
        centroid= centroids[cid]
        dist= [np.linalg.norm(e- centroid) for e in c_arr]
        density= 1./(1.+ np.mean(dist)) if dist else 0.
        metrics['density'].append((cid,density))

        # coherence
        c_val= calculate_cluster_coherence(c_arr)
        metrics['coherence'].append((cid,c_val))

        # separation
        min_sep= 99999
        for ocid, ocen in centroids.items():
            if ocid!= cid:
                dd= np.linalg.norm(centroid- ocen)
                if dd< min_sep:
                    min_sep= dd
        metrics['separation'].append((cid, min_sep))

        progress.progress(float(i+1)/ len(unique_cluster_ids))

    # Plot
    c_df= pd.DataFrame(metrics['coherence'], columns=['cluster_id','coherence'])
    c_size= df.groupby(cluster_col)['keyword'].count().reset_index()
    c_df= c_df.merge(c_size, on='cluster_id')
    c_df= c_df.merge(df.drop_duplicates(cluster_col)[['cluster_id','cluster_name']], on='cluster_id')

    st.subheader("Coherence vs. Size")
    fig1= px.scatter(
        c_df, x='coherence', y='keyword', color='coherence',
        size='keyword', hover_data=['cluster_name'],
        labels={'coherence':'Coherence','keyword':'Size'},
        title='Coherence vs. Cluster Size'
    )
    st.plotly_chart(fig1, use_container_width=True)

    se_df= pd.DataFrame(metrics['separation'], columns=['cluster_id','separation'])
    de_df= pd.DataFrame(metrics['density'], columns=['cluster_id','density'])
    combo= se_df.merge(de_df,on='cluster_id')
    combo= combo.merge(df.drop_duplicates(cluster_col)[['cluster_id','cluster_name']], on='cluster_id')
    st.subheader("Separation vs. Density")
    fig2= px.scatter(
        combo, x='separation', y='density', color='density',
        hover_data=['cluster_name'],
        labels={'separation':'Separation','density':'Density'},
        title='Separation vs. Density'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Assign coherence
    for cid, val in metrics['coherence']:
        df.loc[df[cluster_col]== cid,'cluster_coherence']= val

    st.subheader("Coherence Diagnostics")
    c_vals= [x[1] for x in metrics['coherence']]
    c_threshold= np.percentile(c_vals, 25)
    bad= [x[0] for x in metrics['coherence'] if x[1]< c_threshold]
    if bad:
        st.warning(f"Clusters with low coherence: {bad}")
        st.info("Try more clusters, better embeddings, or manual splits.")
    else:
        st.success("All clusters have decent coherence.")
    overall= df['cluster_coherence'].mean()
    st.metric("Overall Average Coherence", f"{overall:.3f}")
    return df

################################################################
#          AI EVALUATION
################################################################

def evaluate_and_refine_clusters(df, client, model="gpt-3.5-turbo"):
    st.subheader("AI-Powered Cluster Splitting & Insights")
    unique_clusters= df['cluster_id'].unique()
    if len(unique_clusters)==0:
        st.warning("No clusters to evaluate.")
        return {}
    import random
    sample= random.sample(list(unique_clusters), min(5,len(unique_clusters)))
    progress= st.progress(0)
    results={}

    for i, cid in enumerate(sample):
        c_df= df[df['cluster_id']== cid]
        c_name= c_df['cluster_name'].iloc[0]
        kws= c_df['keyword'].tolist()
        prompt= f"""
Evaluate if cluster can be split or improved:

Cluster ID: {cid}
Name: {c_name}
Keywords: {', '.join(kws[:50])}

Return JSON:
{{
  "should_split": true/false,
  "reason": "Why or why not",
  "suggestions": ["suggestion1", "suggestion2"]
}}
"""
        try:
            resp= client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=400
            )
            txt= resp.choices[0].message.content.strip()
            try:
                js= json.loads(txt)
            except:
                js= {"raw_response": txt, "should_split": False, "reason":"", "suggestions":[]}
            results[cid]= {
                "name": c_name,
                "evaluation": js
            }
        except Exception as e:
            st.error(f"AI cluster eval error for cluster {cid}: {str(e)}")
        progress.progress(float(i+1)/ len(sample))
    return results

################################################################
#          ACTIONS: ARCHITECTURE & CONTENT
################################################################

def propose_information_architecture(df, selected_model="gpt-3.5-turbo"):
    st.subheader("Information Architecture Proposal")
    c_count= len(df['cluster_id'].unique())
    cost= 0.0001* c_count
    st.info(f"Estimated cost for architecture: ${cost:.4f}")

    if 'client' in st.session_state and st.session_state['client']:
        cli= st.session_state.client
        prompt= "Propose an SEO site architecture based on these clusters:\n"
        unique= df.drop_duplicates('cluster_id')[['cluster_id','cluster_name']].values.tolist()
        for (cid, cname) in unique:
            prompt+= f"- Cluster {cid}: {cname}\n"
        prompt+= "Return JSON with 'architecture' listing recommended pages/sub-pages.\n"
        try:
            resp= cli.chat.completions.create(
                model= selected_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=700
            )
            txt= resp.choices[0].message.content
            st.json(txt)
        except Exception as e:
            st.error(f"Error generating architecture: {str(e)}")
    else:
        st.warning("No GPT client, fallback suggestion: make pages for each cluster, subpages for subtopics.")


def propose_contents_by_intent(df, selected_model="gpt-3.5-turbo"):
    st.subheader("Content Proposal by Search Intent")
    c_count= len(df['cluster_id'].unique())
    num_topics= st.number_input("Number of content topics per cluster", min_value=1, max_value=20, value=3)
    cost= 0.0002*(c_count* num_topics)
    st.info(f"Estimated cost for content proposal: ${cost:.4f}")

    if st.button("Generate Content Proposal"):
        if 'client' in st.session_state and st.session_state['client']:
            cli= st.session_state.client
            prompt= f"Generate {num_topics} content ideas for each cluster with search intent:\n"
            for cid in df['cluster_id'].unique():
                row= df[df['cluster_id']== cid].iloc[0]
                c_name= row['cluster_name']
                s_intent= row['search_intent']
                prompt+= f"Cluster {cid}: {c_name} => intent: {s_intent}\n"
            prompt+= "Return JSON array 'content_ideas' with cluster_id & list of topics.\n"
            try:
                resp= cli.chat.completions.create(model=selected_model, 
                                                  messages=[{"role":"user","content":prompt}],
                                                  temperature=0.3,
                                                  max_tokens=700)
                txt= resp.choices[0].message.content
                st.json(txt)
            except Exception as e:
                st.error(f"Error generating content proposal: {str(e)}")
        else:
            st.warning("No GPT client. Basic fallback: create a few articles per cluster, matching its intent.")


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

    # If we have key & openai installed, store client
    client= None
    if openai_api_key and openai_available and openai_api_key.strip()!="":
        try:
            os.environ["OPENAI_API_KEY"]= openai_api_key
            check= OpenAI()
            # quick test
            check.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":"test"}], max_tokens=5)
            st.success("✅ OpenAI connected successfully.")
            client= check
            st.session_state.client= check
        except Exception as e:
            st.error(f"OpenAI config error: {str(e)}")
            client= None
    elif not openai_available:
        st.warning("OpenAI library not installed. No GPT usage.")
    else:
        st.info("No valid OpenAI key provided. Using free fallback.")
    
    try:
        # CSV reading
        if csv_format=="no_header":
            df= pd.read_csv(uploaded_file, header=None, names=["keyword"])
            st.success(f"Loaded {len(df)} keywords (no header).")
        else:
            df= pd.read_csv(uploaded_file, header=0)
            if "Keyword" in df.columns:
                df.rename(columns={"Keyword":"keyword"}, inplace=True)
            if "keyword" not in df.columns:
                st.error("No 'Keyword' column found. Check your CSV.")
                return False, None
            st.success(f"Loaded {len(df)} rows (header). Possibly advanced columns too.")

        num_kw= len(df)
        show_csv_cost_estimate(num_kw, gpt_model, num_clusters)

        # Preprocessing
        st.subheader("Keyword Preprocessing")
        advanced= spacy_available or textblob_available
        cleaned= preprocess_keywords(df["keyword"].tolist(), use_advanced= advanced)
        df["keyword_processed"]= cleaned
        st.success("Preprocessing done.")

        # Embeddings
        st.subheader("Generating Embeddings")
        emb= generate_embeddings(df, openai_available, openai_api_key)

        # PCA
        if emb.shape[1]> max_pca_components:
            st.subheader("Dimensionality Reduction (PCA)")
            try:
                pca= PCA()
                pca.fit(emb)
                cvar= np.cumsum(pca.explained_variance_ratio_)
                target_var= pca_variance/100.
                n_comp= np.argmax(cvar>= target_var)+1
                if n_comp==1 and len(cvar)>1:
                    n_comp= min(max_pca_components, len(cvar))
                st.info(f"Components for {pca_variance}% variance: {n_comp}")
                limit= min(n_comp, max_pca_components)
                pca2= PCA(n_components= limit)
                emb_reduced= pca2.fit_transform(emb)
                st.success(f"PCA applied: {limit} dims (~{pca_variance}% variance).")
            except Exception as e:
                st.error(f"PCA error: {str(e)}")
                emb_reduced= emb
        else:
            emb_reduced= emb
        
        # Clustering
        st.subheader("Advanced Semantic Clustering")
        labs= improved_clustering(emb_reduced, num_clusters=num_clusters)
        df["cluster_id"]= labs
        st.success(f"{len(df['cluster_id'].unique())} clusters found.")

        # Refinement
        st.subheader("Refinement")
        df= refine_clusters(df, emb_reduced)

        # Representative keywords
        st.subheader("Representative Keywords")
        cluster_reps={}
        uniq= df['cluster_id'].unique()
        pb= st.progress(0)
        for i, cid in enumerate(uniq):
            idxs= df[df['cluster_id']==cid].index.tolist()
            csize= len(idxs)
            n_rep= min(20, csize)
            c_embs= np.array([emb_reduced[x] for x in idxs])
            cent= np.mean(c_embs, axis=0)
            dist= [np.linalg.norm(emb_reduced[x]- cent) for x in idxs]
            sorted_idx= np.argsort(dist)[:n_rep]
            chosen= [idxs[x] for x in sorted_idx]
            rep_kws= df.loc[chosen,'keyword'].tolist()
            cluster_reps[cid]= rep_kws
            pb.progress(float(i+1)/ len(uniq))
        pb.progress(1.0)
        st.success("Representative keywords assigned.")

        # GPT naming + search intent
        if client:
            st.subheader("Generating Cluster Names & Intent (GPT)")
            try:
                results= generate_cluster_names(
                    clusters_with_reps= cluster_reps,
                    client= client,
                    model= gpt_model,
                    custom_prompt= user_prompt,
                    selected_language= st.session_state.get("language_choice","English")
                )
            except Exception as e:
                st.error(f"GPT naming error: {str(e)}")
                results= {c:(f"Cluster {c}",f"Desc {c}","informational") for c in uniq}
        else:
            results= {c:(f"Cluster {c}",f"Desc {c}","informational") for c in uniq}
        
        df['cluster_name']=''
        df['cluster_description']=''
        df['search_intent']=''
        df['representative']= False
        for cid,(cname,cdesc,cintent) in results.items():
            df.loc[df['cluster_id']== cid,'cluster_name']= cname
            df.loc[df['cluster_id']== cid,'cluster_description']= cdesc
            df.loc[df['cluster_id']== cid,'search_intent']= cintent
            for kw in cluster_reps[cid]:
                match_i= df[(df['cluster_id']== cid)&(df['keyword']== kw)].index
                if not match_i.empty:
                    df.loc[match_i,'representative']= True
        
        # Evaluate
        df= evaluate_cluster_quality(df, emb_reduced, cluster_col='cluster_id')

        # AI-based cluster evaluation
        if client:
            st.subheader("AI-based Cluster Splitting & Insights")
            try:
                splitted= evaluate_and_refine_clusters(df, client, model=gpt_model)
                st.session_state.cluster_evaluation= splitted
            except Exception as e:
                st.error(f"AI-based cluster evaluation error: {str(e)}")

        # Show total cost
        total_c= calculate_api_cost(num_kw, gpt_model, num_clusters)['total_cost']
        st.info(f"**Total cost** to process this CSV: ~${total_c:.4f}")

        return True, df
    except Exception as e:
        st.error(f"Error in pipeline: {str(e)}")
        return False, None


################################################################
#          STREAMLIT APP
################################################################

st.set_page_config(page_title="Advanced Semantic Keyword Clustering",
                   page_icon="🔍",
                   layout="wide")

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
.green-check {
    color: green;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Advanced Semantic Keyword Clustering</div>", unsafe_allow_html=True)

st.markdown("""
This tool takes a CSV of keywords (with or without header), 
performs advanced preprocessing, generates embeddings (OpenAI or fallback), 
clusters them with HDBSCAN/hierarchical/K-Means, 
assigns GPT-based names & search intent, and refines them. 
You can also propose an information architecture or content ideas if you have GPT.
""")

# We store pipeline results & states
if 'df_results' not in st.session_state:
    st.session_state.df_results= None
if 'process_complete' not in st.session_state:
    st.session_state.process_complete= False
if 'client' not in st.session_state:
    st.session_state.client= None

def check_openai_key_validity(key):
    if not key or key.strip()=="":
        return False
    if openai_available:
        try:
            os.environ["OPENAI_API_KEY"]= key
            test_cli= OpenAI()
            test_cli.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":"test"}],
                max_tokens=5
            )
            return True
        except:
            return False
    return False

st.sidebar.markdown("<div class='sub-header'>Configuration</div>", unsafe_allow_html=True)
csv_format= st.sidebar.selectbox("CSV Format", ["no_header","with_header"], index=0)
uploaded_file= st.sidebar.file_uploader("Upload your CSV", type=["csv"])

openai_api_key= st.sidebar.text_input("OpenAI API Key (optional)", type="password", help="We'll test if it's valid.")
valid_key= check_openai_key_validity(openai_api_key)
if valid_key:
    st.sidebar.markdown("<span class='green-check'>OpenAI Key is valid ✅</span>", unsafe_allow_html=True)
    st.session_state.client= OpenAI()
else:
    if openai_key_str := openai_api_key.strip():
        st.sidebar.warning("OpenAI key seems invalid or unreachable.")
    st.session_state.client= None

with st.sidebar.expander("Parameters Guide", expanded=True):
    st.markdown("""
**Number of clusters**  
- Higher => more & smaller clusters  
- Lower => fewer & bigger clusters

**PCA explained variance**  
- Higher => keep more variance => possibly better detail but slower  
- Lower => more dimension reduction => may lose nuance

**Max PCA components**  
- Hard limit on dimension after PCA

**Minimum / Maximum term frequency**  
- Affects which tokens are included in TF-IDF-based steps
- 1 to 10 => min_df, 50% to 100% => max_df
""")

num_clusters= st.sidebar.slider("Number of clusters",2,50,10)
pca_variance= st.sidebar.slider("PCA explained variance (%)",50,99,95)
max_pca_components= st.sidebar.slider("Max PCA components",10,300,100)
min_df= st.sidebar.slider("Minimum term frequency",1,10,1)
max_df= st.sidebar.slider("Maximum term frequency (%)",50,100,95)
gpt_model= st.sidebar.selectbox("GPT model", ["gpt-3.5-turbo","gpt-4"], index=0)

st.sidebar.markdown("### Custom Prompt for naming clusters")
default_prompt= (
    "You are an SEO expert. Name each cluster, give a short SEO description, "
    "and define search intent (informational/navigational/transactional/commercial). Return JSON only."
)
user_prompt= st.sidebar.text_area("Cluster Naming Prompt", value= default_prompt, height=120)

add_cost_calculator()

colS1, colS2= st.columns(2)
with colS1:
    if st.sidebar.button("Download Sample CSV Template"):
        sample_csv= generate_sample_csv()
        st.sidebar.download_button("Get CSV Header", data= sample_csv,
                                   file_name="sample_keyword_planner_template.csv",
                                   mime="text/csv", use_container_width=True)

start_btn= st.button("Start Clustering Pipeline", type="primary", use_container_width=True)
if start_btn and uploaded_file and not st.session_state.process_complete:
    success, dataf= run_clustering(
        uploaded_file= uploaded_file,
        openai_api_key= openai_api_key if valid_key else None,
        num_clusters= num_clusters,
        pca_variance= pca_variance,
        max_pca_components= max_pca_components,
        min_df= min_df,
        max_df= max_df,
        gpt_model= gpt_model,
        user_prompt= user_prompt,
        csv_format= csv_format
    )
    if success and dataf is not None:
        st.session_state.df_results= dataf
        st.session_state.process_complete= True
        st.success("✅ Semantic clustering completed!")
        st.experimental_rerun()

if st.session_state.process_complete and st.session_state.df_results is not None:
    df= st.session_state.df_results
    st.markdown("## Clustering Results")
    with st.expander("Visual Overview",expanded=True):
        st.subheader("Cluster Size Distribution")
        cluster_sizes= df.groupby(['cluster_id','cluster_name']).size().reset_index(name='count')
        cluster_sizes['label']= cluster_sizes.apply(lambda x: f"{x['cluster_name']} (ID:{x['cluster_id']})", axis=1)
        fig= px.bar(cluster_sizes, x='label', y='count', color='count',
                    labels={'count':'Number','label':'Cluster'},
                    title='Cluster Size')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Semantic Coherence per Cluster")
        coh_df= df.groupby(['cluster_id','cluster_name'])['cluster_coherence'].mean().reset_index()
        coh_df['label']= coh_df.apply(lambda x: f"{x['cluster_name']} (ID:{x['cluster_id']})", axis=1)
        fig2= px.bar(coh_df, x='label', y='cluster_coherence', color='cluster_coherence',
                     labels={'cluster_coherence':'Coherence','label':'Cluster'},
                     title='Coherence by Cluster')
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Explore Clusters",expanded=True):
        combos= [
            f"{row['cluster_name']} (ID:{row['cluster_id']})"
            for _, row in df.drop_duplicates(['cluster_id','cluster_name'])[['cluster_id','cluster_name']].iterrows()
        ]
        selected= st.selectbox("Select a cluster to inspect:", combos)
        if selected:
            cid= int(selected.split("ID:")[1].split(")")[0])
            cdf= df[df['cluster_id']==cid].copy()
            st.markdown(f"### {cdf['cluster_name'].iloc[0]}")
            st.markdown(f"**Description**: {cdf['cluster_description'].iloc[0]}")
            st.markdown(f"**Search Intent**: {cdf['search_intent'].iloc[0]}")
            st.markdown(f"**Coherence**: {cdf['cluster_coherence'].iloc[0]:.3f}")
            st.markdown(f"**Total keywords**: {len(cdf)}")
            reps= cdf[cdf['representative']==True]['keyword'].tolist()
            if reps:
                st.markdown("**Representative keywords**:")
                st.write(", ".join(reps[:15]))
            if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
                ev_dict= st.session_state.cluster_evaluation
                if cid in ev_dict:
                    st.subheader("AI Splitting / Additional Insights")
                    st.json(ev_dict[cid]['evaluation'])
            st.markdown("### All keywords")
            st.dataframe(cdf[['keyword']], use_container_width=True)

    with st.expander("Download Results",expanded=True):
        csv_full= df.to_csv(index=False)
        st.download_button("Download Full CSV", data= csv_full,
                           file_name="semantic_clustered_keywords.csv",
                           mime="text/csv", use_container_width=True)
        
        st.subheader("Clusters Summary")
        summary= df.groupby(['cluster_id','cluster_name','cluster_description','search_intent'])['keyword'].count().reset_index()
        summary.columns= ['ID','Name','Description','Search Intent','Number of Keywords']
        c_coh= df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary= summary.merge(c_coh, left_on='ID', right_on='cluster_id')
        summary.drop('cluster_id', axis=1, inplace=True)
        summary.rename(columns={'cluster_coherence':'Coherence'}, inplace=True)
        def representative_kws(cid):
            r= df[(df['cluster_id']==cid)&(df['representative']== True)]['keyword'].tolist()
            return ', '.join(r[:5])
        summary['Representative Keywords']= summary['ID'].apply(representative_kws)

        if 'cluster_evaluation' in st.session_state and st.session_state.cluster_evaluation:
            eval_ids= st.session_state.cluster_evaluation.keys()
            summary['AI Evaluation?']= summary['ID'].apply(lambda x:"Yes" if x in eval_ids else "No")
        else:
            summary['AI Evaluation?']= "No"
        
        st.dataframe(summary, use_container_width=True)
        csv_sum= summary.to_csv(index=False)
        st.download_button("Download Clusters Summary", data= csv_sum,
                           file_name="semantic_clusters_summary.csv", mime="text/csv",
                           use_container_width=True)

    st.markdown("---")
    st.subheader("Additional Actions")
    st.markdown("""
1. **Propose Information Architecture**: Creates site/page hierarchy from clusters.  
2. **Propose Content by Intent**: Creates content ideas per cluster, matching search intent.
""")
    colA,colB= st.columns(2)
    with colA:
        if st.button("Propose Information Architecture"):
            propose_information_architecture(df, selected_model=gpt_model)
    with colB:
        if st.button("Propose Content by Intent"):
            propose_contents_by_intent(df, selected_model=gpt_model)

    if st.button("Reset Pipeline"):
        st.session_state.df_results= None
        st.session_state.process_complete= False
        st.experimental_rerun()

with st.expander("How The Tool Works", expanded=True):
    st.markdown("""
1. **Upload CSV**: Select "no_header" (keywords only) or "with_header" (Keyword, search_volume, cpc, etc.).
2. **Configure**: Number of clusters, PCA variance, etc. Provide an optional OpenAI API Key for GPT features.
3. **Preprocessing**: spaCy/TextBlob/NLTK-based cleaning, lemmatization, bigrams, etc.
4. **Embeddings**: 
   - OpenAI if key is valid (up to 5k direct), or 
   - SentenceTransformers, or 
   - TF-IDF fallback.
5. **Clustering**: HDBSCAN → hierarchical → K-Means fallback.
6. **Refinement**: Outlier detection & merges for highly similar clusters.
7. **Evaluation**: Coherence, density, separation. Low coherence triggers warnings.
8. **GPT**: 
   - Cluster naming & search intent. 
   - (Optional) AI-based splitting suggestions.
9. **Architecture** & **Content** proposals: If GPT is available, we call it to generate structured outputs.
""")

st.markdown("""
---
<div style="text-align:center; color:#888;">
Advanced Semantic Keyword Clustering – with validated OpenAI key, complete param guide, working architecture/content proposals, and full pipeline.
</div>
""", unsafe_allow_html=True)
