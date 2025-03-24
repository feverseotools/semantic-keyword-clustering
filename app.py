import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Configuración de la página
st.set_page_config(
    page_title="Clustering Semántico de Keywords",
    page_icon="🔍",
    layout="wide"
)

# Estilos CSS para mejorar la apariencia
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
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.markdown("<div class='main-header'>Clustering Semántico de Keywords</div>", unsafe_allow_html=True)
st.markdown("""
Esta aplicación te permite agrupar keywords semánticamente similares utilizando embeddings y técnicas de clustering avanzadas.
Sube tu archivo CSV con las keywords y configura los parámetros para obtener clusters significativos.
""")

# Inicialización de sesión
if 'process_complete' not in st.session_state:
    st.session_state.process_complete = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = None
if 'nlp_loaded' not in st.session_state:
    st.session_state.nlp_loaded = False
if 'nltk_downloaded' not in st.session_state:
    st.session_state.nltk_downloaded = False

# Sidebar para la configuración
st.sidebar.markdown("<div class='sub-header'>Configuración</div>", unsafe_allow_html=True)

# 1. Subir CSV
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de keywords", type=['csv'])

# 2. API Key de OpenAI
openai_api_key = st.sidebar.text_input("API Key de OpenAI", type="password", help="Necesaria para generar embeddings y nombres de clusters")

# 3. Parámetros de clustering
st.sidebar.markdown("<div class='sub-header'>Parámetros</div>", unsafe_allow_html=True)
num_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=50, value=10, help="Número de grupos en los que dividir las keywords")
pca_variance = st.sidebar.slider("Varianza explicada PCA (%)", min_value=50, max_value=99, value=90, help="Porcentaje de varianza a mantener en la reducción de dimensionalidad")
max_pca_components = st.sidebar.slider("Máximo de componentes PCA", min_value=10, max_value=200, value=75, help="Número máximo de componentes PCA a utilizar")

# 4. Opciones avanzadas
st.sidebar.markdown("<div class='sub-header'>Opciones avanzadas</div>", unsafe_allow_html=True)
use_openai = st.sidebar.checkbox("Usar OpenAI para embeddings", value=False, help="Si está desactivado, se usará Sentence-BERT local o TF-IDF")
openai_model = st.sidebar.selectbox("Modelo de embeddings OpenAI", ["text-embedding-ada-002", "text-embedding-3-small"], index=0, disabled=not use_openai)
gpt_model = st.sidebar.selectbox("Modelo para nombrar clusters", ["gpt-3.5-turbo", "gpt-4"], index=0)
sample_size = st.sidebar.slider("Tamaño de muestra para embeddings", min_value=100, max_value=2000, value=1000, help="Número de keywords a utilizar para embeddings (ahorra costes)")

# Función para cargar NLTK y SpaCy
@st.cache_resource
def load_nlp_resources():
    with st.spinner("Cargando recursos de NLP..."):
        # Download NLTK resources if not already downloaded
        if not st.session_state.nltk_downloaded:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            st.session_state.nltk_downloaded = True
        
        # Load SpaCy model if not already loaded
        if not st.session_state.nlp_loaded:
            try:
                nlp = spacy.load("en_core_web_sm")
                st.session_state.nlp_loaded = True
            except OSError:
                st.warning("Descargando modelo de SpaCy (esto puede tardar un momento)...")
                os.system("python -m spacy download en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                st.session_state.nlp_loaded = True
            return nlp
    return None

# Función para preprocesar keywords
def preprocess_keywords(keywords, nlp):
    stop_words = set(stopwords.words('french')).union(stopwords.words('english'))
    processed_keywords = []
    
    progress_bar = st.progress(0)
    total = len(keywords)
    
    for i, keyword in enumerate(keywords):
        if not isinstance(keyword, str):
            processed_keywords.append("")
            continue

        doc = nlp(keyword.lower())  # Process keyword with SpaCy
        tokens = [token.text for token in doc if token.is_alpha and token.text.lower() not in stop_words]

        # Preserve named entities
        entities = [ent.text for ent in doc.ents]

        processed_text = ' '.join(tokens) if tokens else keyword.lower()
        processed_keywords.append(processed_text)
        
        # Update progress bar every 100 items
        if i % 100 == 0:
            progress_bar.progress(min(i / total, 1.0))
    
    progress_bar.progress(1.0)
    return processed_keywords

# Función para generar embeddings con Sentence-BERT
def generate_sbert_embeddings(texts):
    try:
        from sentence_transformers import SentenceTransformer
        st.info("Usando Sentence-BERT para embeddings locales...")

        # Crear barra de progreso
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Load a lightweight model that's good for keyword-style short text
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Generate embeddings locally
        progress_text.text("Generando embeddings con Sentence-BERT...")
        embeddings = model.encode(texts, show_progress_bar=False)
        
        progress_bar.progress(1.0)
        progress_text.text(f"✅ Generados {len(embeddings)} embeddings usando Sentence-BERT")
        
        return embeddings, True
    except ImportError:
        st.warning("Sentence-BERT no disponible, usando alternativa...")
        return None, False

# Función para generar embeddings con OpenAI
def sample_and_embed_keywords(keywords, client, sample_size=1000, model="text-embedding-ada-002"):
    """Intelligently sample keywords and generate embeddings"""
    N = len(keywords)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("Preparando muestreo de keywords...")
    
    if N <= sample_size:
        # For small datasets, embed everything
        sample_indices = list(range(N))
        full_sample = keywords
    else:
        # For large datasets, use intelligent sampling:
        # 1. Split keywords into chunks and take representatives from each
        chunk_size = N // (sample_size // 2)
        sample_indices = []

        # Get keywords from each chunk
        for i in range(0, N, chunk_size):
            chunk = keywords[i:i+chunk_size]
            # Take ~2 keywords from each chunk
            sample_count = max(1, min(2, len(chunk) // 5))
            chunk_indices = list(range(i, min(i+chunk_size, N)))
            if chunk_indices:
                selected = np.random.choice(chunk_indices, sample_count, replace=False)
                sample_indices.extend(selected)

        # 2. Add some completely random keywords to ensure diversity
        remaining = sample_size - len(sample_indices)
        if remaining > 0:
            remaining_indices = list(set(range(N)) - set(sample_indices))
            if remaining_indices:
                random_indices = np.random.choice(remaining_indices,
                                                min(remaining, len(remaining_indices)),
                                                replace=False)
                sample_indices.extend(random_indices)

        full_sample = [keywords[i] for i in sample_indices]
    
    progress_text.text(f"Generando embeddings para {len(full_sample)} keywords con OpenAI...")
    progress_bar.progress(0.2)
    
    # Generate embeddings for the sample
    try:
        # Filter out empty strings
        valid_samples = [(i, k) for i, k in enumerate(full_sample) if k]
        valid_indices = [i for i, _ in valid_samples]
        valid_keywords = [k for _, k in valid_samples]
        
        # Only make API call if we have valid keywords
        if valid_keywords:
            response = client.embeddings.create(
                model=model,
                input=valid_keywords
            )
            
            # Reindex the returned embeddings
            sample_embeddings_dict = {}
            for i, emb_data in enumerate(response.data):
                original_idx = valid_indices[i]
                sample_embeddings_dict[original_idx] = np.array(emb_data.embedding)
            
            # Convert to array format expected by rest of code
            sample_embeddings = np.zeros((len(full_sample), len(next(iter(sample_embeddings_dict.values())))))
            for idx, emb in sample_embeddings_dict.items():
                sample_embeddings[idx] = emb
        else:
            st.error("No valid keywords for embedding")
            return None, sample_indices
        
        progress_bar.progress(1.0)
        progress_text.text(f"✅ Generados embeddings para {len(sample_embeddings)} keywords de muestra")

        return sample_embeddings, sample_indices
    except Exception as e:
        st.error(f"Error generando embeddings: {str(e)}")
        return None, sample_indices

# Función para propagar embeddings a keywords similares
def propagate_embeddings(df, sample_embeddings, sample_indices, keyword_column='keyword_processed'):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Propagando embeddings a todas las keywords...")
    
    # Create TF-IDF matrix for all keywords
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[keyword_column].fillna(''))
    progress_bar.progress(0.3)
    
    # For each non-sampled keyword, find the most similar sampled keyword and use its embedding
    keyword_embeddings = np.zeros((len(df), sample_embeddings.shape[1]))
    for i in sample_indices:
        keyword_embeddings[i] = sample_embeddings[sample_indices.index(i)]
    
    progress_bar.progress(0.5)
    
    # Group non-sampled keywords into batches for faster processing
    non_sampled = [i for i in range(len(df)) if i not in sample_indices]
    batch_size = max(100, len(non_sampled) // 20)  # Split into ~20 batches
    total_batches = len(non_sampled) // batch_size + (1 if len(non_sampled) % batch_size != 0 else 0)
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(non_sampled))
        batch = non_sampled[start_idx:end_idx]
        
        for i in batch:
            similarities = cosine_similarity(
                tfidf_matrix[i:i+1],
                tfidf_matrix[sample_indices]
            )[0]
            most_similar_idx = sample_indices[np.argmax(similarities)]
            keyword_embeddings[i] = keyword_embeddings[most_similar_idx]
        
        # Update progress
        progress_bar.progress(0.5 + 0.5 * (batch_idx + 1) / total_batches)
    
    progress_bar.progress(1.0)
    progress_text.text(f"✅ Propagados embeddings a todas las {len(keyword_embeddings)} keywords")
    
    return keyword_embeddings

# Función para generar nombres de clusters
def generate_improved_cluster_names(clusters_with_representatives, client, model="gpt-3.5-turbo"):
    if not clusters_with_representatives:
        return {}

    results = {}
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text("Generando nombres y descripciones para los clusters...")

    # Process clusters in smaller batches to manage costs
    batch_size = 5
    total_batches = (len(clusters_with_representatives) + batch_size - 1) // batch_size
    
    for batch_idx, batch_start in enumerate(range(0, len(clusters_with_representatives), batch_size)):
        batch_end = min(batch_start + batch_size, len(clusters_with_representatives))
        batch_clusters = list(clusters_with_representatives.items())[batch_start:batch_end]

        try:
            # Step 1: Analyze clusters
            analysis_prompt = "I'll provide representative keywords for several clusters. For each cluster, analyze the keywords to identify common themes, topics, or categories.\n\n"

            # Track the cluster order to match the response
            cluster_order = []
            for cluster_id, keywords in batch_clusters:
                cluster_order.append(cluster_id)
                analysis_prompt += f"Cluster {cluster_id} representative keywords: {', '.join(keywords[:15])}\n\n"

            analysis_response = client.chat.completions.create(
                model=model,  # Use cheaper model for analysis
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=300
            )

            analysis_text = analysis_response.choices[0].message.content.strip()

            # Step 2: Generate names and descriptions based on analysis
            naming_prompt = f"""Based on the following analysis of keyword clusters, provide a specific name and description for each cluster.

Analysis:
{analysis_text}

For each cluster, provide:
1. A short, specific cluster name (3-5 words)
2. A one-sentence description that accurately represents the theme

Format your response as a JSON array, with each element containing cluster_id, cluster_name, and description for clusters {', '.join(map(str, cluster_order))}.
"""

            naming_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": naming_prompt}],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"}
            )

            naming_text = naming_response.choices[0].message.content.strip()

            try:
                data = json.loads(naming_text)

                # Handle different JSON structures that might be returned
                clusters_data = None
                if "clusters" in data:
                    clusters_data = data["clusters"]
                elif isinstance(data, list):
                    clusters_data = data
                else:
                    # Try to find any list in the response
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            clusters_data = value
                            break

                if not clusters_data:
                    raise ValueError("Could not find clusters data in JSON response")

                # Match clusters by position if IDs don't match
                if len(clusters_data) == len(cluster_order):
                    for i, cluster_info in enumerate(clusters_data):
                        actual_id = cluster_order[i]
                        results[actual_id] = (
                            cluster_info.get("cluster_name", f"Cluster {actual_id}"),
                            cluster_info.get("description", "No description provided.")
                        )
                else:
                    # Try matching by ID if available
                    for cluster_info in clusters_data:
                        if "cluster_id" in cluster_info:
                            # Try to match cluster IDs flexibly
                            cluster_id_str = str(cluster_info["cluster_id"]).strip()
                            for actual_id in cluster_order:
                                if str(actual_id) == cluster_id_str:
                                    results[actual_id] = (
                                        cluster_info.get("cluster_name", f"Cluster {actual_id}"),
                                        cluster_info.get("description", "No description provided.")
                                    )

            except json.JSONDecodeError as e:
                st.warning(f"Error analizando respuesta JSON para batch {batch_idx+1}/{total_batches}")
                # Handle the error gracefully - assign default names
                for i, cluster_id in enumerate(cluster_order):
                    results[cluster_id] = (f"Cluster {cluster_id}", "Error parsing JSON response")

            # Fill in any missing clusters from this batch
            for cluster_id, _ in batch_clusters:
                if cluster_id not in results:
                    results[cluster_id] = (f"Cluster {cluster_id}", "No description generated")

            # Update progress
            progress_bar.progress((batch_idx + 1) / total_batches)
            time.sleep(1)

        except Exception as e:
            st.warning(f"Error generando nombres para batch {batch_idx+1}/{total_batches}: {str(e)}")
            # Provide default names for all clusters in this batch
            for cluster_id, _ in batch_clusters:
                results[cluster_id] = (f"Cluster {cluster_id}", "Error generating description")
    
    progress_bar.progress(1.0)
    progress_text.text(f"✅ Nombres y descripciones generados para {len(results)} clusters")
    
    return results

# Función para calcular coherencia de clusters
def calculate_cluster_coherence(cluster_embeddings):
    """Calculate semantic coherence of a cluster based on embedding similarity"""
    if len(cluster_embeddings) <= 1:
        return 1.0  # Perfect coherence for single element

    # Calculate centroid
    centroid = np.mean(cluster_embeddings, axis=0)

    # Calculate average cosine similarity to centroid
    similarities = []
    for emb in cluster_embeddings:
        similarity = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
        similarities.append(similarity)

    return np.mean(similarities)

# Función principal para ejecutar el clustering
def run_clustering():
    if uploaded_file is None:
        st.warning("Por favor, sube un archivo CSV con keywords.")
        return
    
    if use_openai and not openai_api_key:
        st.warning("Se requiere una API Key de OpenAI para usar embeddings de OpenAI.")
        return
    
    st.info("Iniciando proceso de clustering semántico...")
    
    # Configurar cliente OpenAI si se proporciona la clave API
    client = None
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    
    # Cargar recursos NLP
    nlp = load_nlp_resources()
    if nlp is None:
        st.error("Error cargando recursos de NLP. Por favor, inténtalo de nuevo.")
        return
    
    # Cargar y procesar el CSV
    try:
        # Leer CSV
        df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
        st.success(f"✅ Cargadas {len(df)} keywords del archivo CSV")
        
        # Preprocesar keywords
        st.subheader("Preprocesamiento de Keywords")
        keywords_processed = preprocess_keywords(df["keyword"].tolist(), nlp)
        df['keyword_processed'] = keywords_processed
        st.success("✅ Keywords preprocesadas correctamente")
        
        # Generar embeddings
        st.subheader("Generación de Embeddings")
        
        if use_openai and client:
            # Usar OpenAI para embeddings
            sample_embeddings, sample_indices = sample_and_embed_keywords(
                df['keyword_processed'].tolist(),
                client,
                sample_size=min(sample_size, len(df)),
                model=openai_model
            )
            
            if sample_embeddings is None:
                st.warning("Usando TF-IDF como alternativa")
                vectorizer = TfidfVectorizer(max_features=300)
                tfidf_matrix = vectorizer.fit_transform(df['keyword_processed'].fillna(''))
                keyword_embeddings = tfidf_matrix.toarray()
            else:
                # Propagar embeddings a keywords similares
                keyword_embeddings = propagate_embeddings(df, sample_embeddings, sample_indices)
        else:
            # Usar Sentence-BERT local para embeddings
            embeddings, sbert_success = generate_sbert_embeddings(df['keyword_processed'].fillna(''))
            
            if sbert_success:
                keyword_embeddings = embeddings
            else:
                # Fallback a TF-IDF
                st.warning("Usando TF-IDF como alternativa para embeddings")
                vectorizer = TfidfVectorizer(max_features=300)
                tfidf_matrix = vectorizer.fit_transform(df['keyword_processed'].fillna(''))
                keyword_embeddings = tfidf_matrix.toarray()
        
        # Aplicar PCA
        st.subheader("Reducción de Dimensionalidad (PCA)")
        
        pca_progress = st.progress(0)
        pca_text = st.empty()
        pca_text.text("Analizando varianza explicada...")
        
        # Determine optimal number of components experimentally
        pca = PCA()
        pca.fit(keyword_embeddings)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        pca_progress.progress(0.3)
        
        # Find the number of components that explain the desired variance
        target_variance = pca_variance / 100.0
        n_components = np.argmax(cumulative_variance >= target_variance) + 1
        pca_text.text(f"Componentes para {pca_variance}% de varianza: {n_components}")
        pca_progress.progress(0.6)
        
        # Use that number (with a reasonable cap)
        max_components = min(n_components, max_pca_components)
        pca = PCA(n_components=max_components)
        keyword_embeddings = pca.fit_transform(keyword_embeddings)
        
        pca_progress.progress(1.0)
        pca_text.text(f"✅ PCA aplicado: {max_components} dimensiones ({pca_variance}% de varianza explicada)")
        
        # Aplicar clustering jerárquico
        st.subheader("Clustering Jerárquico")
        
        cluster_progress = st.progress(0)
        cluster_text = st.empty()
        cluster_text.text("Aplicando clustering jerárquico...")
        
        Z = linkage(keyword_embeddings, method="ward")
        cluster_progress.progress(0.5)
        
        df["cluster_id"] = fcluster(Z, t=num_clusters, criterion="maxclust")
        
        cluster_progress.progress(1.0)
        cluster_text.text(f"✅ Keywords agrupadas en {num_clusters} clusters")
        
        # Identificar keywords representativas para cada cluster
        st.subheader("Análisis de Clusters")
        
        rep_progress = st.progress(0)
        rep_text = st.empty()
        rep_text.text("Identificando keywords representativas...")
        
        clusters_with_representatives = {}
        for i, cluster_num in enumerate(df['cluster_id'].unique()):
            cluster_size = len(df[df['cluster_id'] == cluster_num])
            n_representatives = min(15, cluster_size)
            
            # Get indices of keywords in this cluster
            indices = df[df['cluster_id'] == cluster_num].index.tolist()
            
            # Calculate centroid of the cluster
            cluster_embeddings = np.array([keyword_embeddings[i] for i in indices])
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate distance to centroid for each keyword
            distances = [np.linalg.norm(keyword_embeddings[i] - centroid) for i in indices]
            
            # Get indices of keywords closest to centroid
            sorted_indices = np.argsort(distances)[:n_representatives]
            representative_indices = [indices[i] for i in sorted_indices]
            representative_keywords = df.iloc[representative_indices]['keyword'].tolist()
            
            clusters_with_representatives[cluster_num] = representative_keywords
            
            # Update progress
            rep_progress.progress((i+1) / len(df['cluster_id'].unique()))
        
        rep_progress.progress(1.0)
        rep_text.text(f"✅ Identificadas keywords representativas para {len(clusters_with_representatives)} clusters")
        
        # Generar nombres para los clusters si está disponible OpenAI
        if client:
            st.subheader("Generación de Nombres para Clusters")
            cluster_names = generate_improved_cluster_names(
                clusters_with_representatives, 
                client,
                model=gpt_model
            )
        else:
            st.warning("No se pueden generar nombres de clusters sin API Key de OpenAI")
            cluster_names = {k: (f"Cluster {k}", f"Grupo de keywords {k}") for k in df['cluster_id'].unique()}
        
        # Aplicar resultados al DataFrame
        df['cluster_name'] = ''
        df['cluster_description'] = ''
        df['representative'] = False
        
        for cluster_num, (name, description) in cluster_names.items():
            df.loc[df['cluster_id'] == cluster_num, 'cluster_name'] = name
            df.loc[df['cluster_id'] == cluster_num, 'cluster_description'] = description
            
            # Marcar keywords representativas
            for keyword in clusters_with_representatives.get(cluster_num, []):
                matching_indices = df[(df['cluster_id'] == cluster_num) & (df['keyword'] == keyword)].index
                if not matching_indices.empty:
                    df.loc[matching_indices, 'representative'] = True
        
        # Calcular coherencia para cada cluster
        st.subheader("Evaluación de Coherencia Semántica")
        
        coh_progress = st.progress(0)
        coh_text = st.empty()
        coh_text.text("Calculando coherencia semántica...")
        
        df['cluster_coherence'] = 0.0
        
        for i, cluster_num in enumerate(df['cluster_id'].unique()):
            cluster_indices = df[df['cluster_id'] == cluster_num].index.tolist()
            cluster_embeddings = np.array([keyword_embeddings[i] for i in cluster_indices])
            coherence = calculate_cluster_coherence(cluster_embeddings)
            df.loc[df['cluster_id'] == cluster_num, 'cluster_coherence'] = coherence
            
            # Update progress
            coh_progress.progress((i+1) / len(df['cluster_id'].unique()))
        
        coh_progress.progress(1.0)
        coh_text.text("✅ Coherencia semántica calculada para todos los clusters")
        
        # Guardar los resultados en la sesión
        st.session_state.df_results = df
        st.session_state.process_complete = True
        
        # Resumen final
        st.markdown("<div class='success-box'>✅ Clustering semántico completado con éxito!</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error durante el proceso: {str(e)}")
        return None

# Botón para ejecutar el clustering
if uploaded_file is not None and not st.session_state.process_complete:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Iniciar Clustering Semántico", type="primary", use_container_width=True):
            run_clustering()

# Mostrar resultados si el proceso está completo
if st.session_state.process_complete and st.session_state.df_results is not None:
    st.markdown("<div class='main-header'>Resultados del Clustering</div>", unsafe_allow_html=True)
    
    df = st.session_state.df_results
    
    # Pestaña para mostrar visualizaciones
    with st.expander("Visualizaciones"):
        # Gráfico de barras con tamaño de clusters
        st.subheader("Distribución de Clusters")
        
        cluster_sizes = df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
        cluster_sizes['label'] = cluster_sizes.apply(lambda x: f"{x['cluster_name']} (ID: {x['cluster_id']})", axis=1)
        
        fig = px.bar(
            cluster_sizes, 
            x='label', 
            y='count',
            color='count',
            labels={'count': 'Número de Keywords', 'label': 'Cluster'},
            title='Tamaño de cada Cluster',
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de coherencia de clusters
        st.subheader("Coherencia Semántica de Clusters")
        
        coherence_data = df.groupby(['cluster_id', 'cluster_name'])['cluster_coherence'].mean().reset_index()
        coherence_data['label'] = coherence_data.apply(lambda x: f"{x['cluster_name']} (ID: {x['cluster_id']})", axis=1)
        
        fig2 = px.bar(
            coherence_data,
            x='label',
            y='cluster_coherence',
            color='cluster_coherence',
            labels={'cluster_coherence': 'Coherencia', 'label': 'Cluster'},
            title='Coherencia Semántica por Cluster',
            color_continuous_scale=px.colors.sequential.Greens
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    # Pestaña para explorar clusters
    with st.expander("Explorar Clusters", expanded=True):
        # Selector de cluster
        cluster_options = [f"{row['cluster_name']} (ID: {row['cluster_id']})" for _, row in 
                          df.drop_duplicates(['cluster_id', 'cluster_name'])[['cluster_id', 'cluster_name']].iterrows()]
        selected_cluster = st.selectbox("Selecciona un cluster para explorar:", cluster_options)
        
        if selected_cluster:
            # Obtener ID del cluster seleccionado
            cluster_id = int(selected_cluster.split("ID: ")[1].split(")")[0])
            
            # Filtrar datos del cluster seleccionado
            cluster_df = df[df['cluster_id'] == cluster_id].copy()
            
            # Mostrar información del cluster
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {cluster_df['cluster_name'].iloc[0]}")
                st.markdown(f"**Descripción:** {cluster_df['cluster_description'].iloc[0]}")
                st.markdown(f"**Total keywords:** {len(cluster_df)}")
            
            with col2:
                st.markdown(f"**Coherencia semántica:** {cluster_df['cluster_coherence'].iloc[0]:.3f}")
                st.markdown("**Keywords representativas:**")
                rep_keywords = cluster_df[cluster_df['representative'] == True]['keyword'].tolist()
                if rep_keywords:
                    st.markdown("<ul>" + "".join([f"<li>{kw}</li>" for kw in rep_keywords[:10]]) + "</ul>", unsafe_allow_html=True)
            
            # Mostrar todas las keywords del cluster
            st.markdown("### Todas las keywords")
            st.dataframe(cluster_df[['keyword']], use_container_width=True)

    # Pestaña para descargar resultados
    with st.expander("Descargar Resultados"):
        # Opción de descarga de CSV completo
        csv = df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV con todos los resultados",
            data=csv,
            file_name="semantic_clustered_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Opción de descarga de resumen
        st.subheader("Resumen de Clusters")
        summary_df = df.groupby(['cluster_id', 'cluster_name', 'cluster_description'])['keyword'].count().reset_index()
        summary_df.columns = ['ID', 'Nombre', 'Descripción', 'Número de Keywords']
        
        # Añadir coherencia
        coherence_df = df.groupby('cluster_id')['cluster_coherence'].mean().reset_index()
        summary_df = summary_df.merge(coherence_df, left_on='ID', right_on='cluster_id')
        summary_df.drop('cluster_id', axis=1, inplace=True)
        summary_df.rename(columns={'cluster_coherence': 'Coherencia'}, inplace=True)
        
        # Añadir keywords representativas
        def get_rep_keywords(cluster_id):
            reps = df[(df['cluster_id'] == cluster_id) & (df['representative'] == True)]['keyword'].tolist()
            return ', '.join(reps[:5])
        
        summary_df['Keywords Representativas'] = summary_df['ID'].apply(get_rep_keywords)
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Descargar resumen
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Descargar resumen de clusters",
            data=csv_summary,
            file_name="semantic_clusters_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

# Botón para reiniciar
if st.session_state.process_complete:
    if st.button("Reiniciar", type="secondary", use_container_width=True):
        st.session_state.process_complete = False
        st.session_state.df_results = None
        st.experimental_rerun()

# Información adicional
with st.expander("Información sobre el Clustering Semántico"):
    st.markdown("""
    ### ¿Cómo funciona el clustering semántico?
    
    1. **Preprocesamiento**: Las keywords se limpian y normalizan eliminando stopwords y preservando entidades.
    
    2. **Embeddings**: Se generan vectores semánticos que capturan el significado de cada keyword usando:
       - **Sentence-BERT** (local, sin costo)
       - **OpenAI Embeddings** (requiere API key)
       - **TF-IDF** (fallback si las anteriores no están disponibles)
    
    3. **Reducción de dimensionalidad**: Se aplica PCA para reducir la dimensionalidad manteniendo la varianza deseada.
    
    4. **Clustering jerárquico**: Se agrupan las keywords en clusters basados en similitud semántica.
    
    5. **Análisis de clusters**: Se identifican keywords representativas y se evalúa la coherencia semántica.
    
    6. **Generación de nombres**: Opcionalmente, se usan modelos GPT para generar nombres y descripciones significativas.
    
    ### Consejos para mejores resultados
    
    - **Calidad de datos**: Asegúrate de que tus keywords estén limpias y sean relevantes.
    - **Número de clusters**: Experimenta con diferentes números para encontrar el óptimo.
    - **PCA**: Ajusta la varianza explicada según la complejidad de tus datos.
    - **Batch processing**: Para datasets grandes, considera procesarlos en lotes.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Desarrollado para clustering semántico de keywords | v1.0
    </div>
    """, 
    unsafe_allow_html=True
)
