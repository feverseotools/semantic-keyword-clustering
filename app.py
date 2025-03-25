import streamlit as st

# Instalar la versión correcta de OpenAI al inicio
st.write("Instalando dependencias...")
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "openai>=1.0.0"])
st.write("Dependencias instaladas. Reiniciando aplicación...")
st.experimental_rerun()

import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from io import StringIO

# Para OpenAI, importamos con manejo de errores
try:
    # Asegurarse de importar el cliente correcto
    from openai import OpenAI
    # Verificar la versión
    import openai
    st.sidebar.info(f"Versión de OpenAI: {openai.__version__}")
    openai_available = True
except ImportError:
    openai_available = False

# Descargar recursos de NLTK al inicio para evitar problemas posteriores
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    pass  # Continuar incluso si la descarga falla

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
Esta aplicación te permite agrupar keywords semánticamente similares utilizando técnicas de clustering avanzadas.
Sube tu archivo CSV con las keywords y configura los parámetros para obtener clusters significativos.
""")

# Inicialización de sesión
if 'process_complete' not in st.session_state:
    st.session_state.process_complete = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

# Sidebar para la configuración
st.sidebar.markdown("<div class='sub-header'>Configuración</div>", unsafe_allow_html=True)

# 1. Subir CSV
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de keywords", type=['csv'])

# 2. API Key de OpenAI (opcional)
openai_api_key = st.sidebar.text_input("API Key de OpenAI (opcional)", type="password", help="Necesaria solo para generar nombres de clusters")

# Mostrar estado de OpenAI
if openai_available:
    if openai_api_key:
        st.sidebar.success("✅ API Key proporcionada")
    else:
        st.sidebar.info("ℹ️ Sin API Key (los clusters tendrán nombres genéricos)")
else:
    st.sidebar.error("❌ Biblioteca OpenAI no disponible")

# 3. Parámetros de clustering
st.sidebar.markdown("<div class='sub-header'>Parámetros</div>", unsafe_allow_html=True)
num_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=50, value=10, help="Número de grupos en los que dividir las keywords")
pca_variance = st.sidebar.slider("Varianza explicada PCA (%)", min_value=50, max_value=99, value=90, help="Porcentaje de varianza a mantener en la reducción de dimensionalidad")
max_pca_components = st.sidebar.slider("Máximo de componentes PCA", min_value=10, max_value=200, value=75, help="Número máximo de componentes PCA a utilizar")

# 4. Opciones avanzadas
st.sidebar.markdown("<div class='sub-header'>Opciones avanzadas</div>", unsafe_allow_html=True)
min_df = st.sidebar.slider("Frecuencia mínima de términos", min_value=1, max_value=10, value=1, help="Ignora términos que aparecen en menos documentos que este")
max_df = st.sidebar.slider("Frecuencia máxima de términos (%)", min_value=50, max_value=100, value=95, help="Ignora términos que aparecen en más del N% de documentos")
gpt_model = st.sidebar.selectbox("Modelo para nombrar clusters", ["gpt-3.5-turbo", "gpt-4"], index=0)
# Función simplificada para preprocesar texto
def preprocess_text(text, use_lemmatization=True):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Convertir a minúsculas
        text = text.lower()
        
        # Tokenizar
        tokens = word_tokenize(text)
        
        # Cargar stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback básico si no se pueden cargar stopwords
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'in', 'on', 'to', 'for'}
        
        # Filtrar stopwords
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        # Lematización (opcional)
        if use_lemmatization:
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            except Exception as e:
                pass  # Continuar sin lematización si falla
        
        return " ".join(tokens)
    except Exception as e:
        # Manejo de errores para garantizar que siempre se devuelva algo
        return text.lower() if isinstance(text, str) else ""

# Función para preprocesar keywords
def preprocess_keywords(keywords, use_lemmatization=True):
    processed_keywords = []
    
    progress_bar = st.progress(0)
    total = len(keywords)
    
    for i, keyword in enumerate(keywords):
        processed_keywords.append(preprocess_text(keyword, use_lemmatization))
        
        # Update progress bar every 100 items
        if i % 100 == 0:
            progress_bar.progress(min(i / total, 1.0))
    
    progress_bar.progress(1.0)
    return processed_keywords

# Función para generar embeddings con TF-IDF
def generate_tfidf_embeddings(texts, min_df=1, max_df=0.95):
    st.info("Generando vectores TF-IDF para las keywords...")
    progress_bar = st.progress(0)
    
    try:
        # Crear un vectorizador con parámetros configurables
        vectorizer = TfidfVectorizer(
            max_features=300,  # Limitar características para prevenir problemas de memoria
            min_df=min_df,     # Ignorar términos que aparecen en menos de N documentos
            max_df=max_df,     # Ignorar términos que aparecen en más del N% de los documentos
            stop_words='english'
        )
        
        # Asegurar que no hay valores nulos
        clean_texts = [t if isinstance(t, str) and t else " " for t in texts]
        
        # Generar matriz TF-IDF
        progress_bar.progress(0.3)
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
        progress_bar.progress(0.8)
        
        # Convertir a array denso
        embeddings = tfidf_matrix.toarray()
        progress_bar.progress(1.0)
        
        st.success(f"✅ Generados {embeddings.shape[1]} vectores TF-IDF")
        return embeddings
    except Exception as e:
        st.error(f"Error generando embeddings TF-IDF: {str(e)}")
        # Último recurso: generar vectores aleatorios
        st.warning("Generando vectores aleatorios como último recurso")
        random_embeddings = np.random.rand(len(texts), 100)
        return random_embeddings

# Función para generar nombres de clusters con OpenAI
def generate_cluster_names(clusters_with_representatives, client, model="gpt-3.5-turbo"):
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
            # Prompt para analizar clusters
            analysis_prompt = "I'll provide representative keywords for several clusters. For each cluster, analyze the keywords to identify common themes, topics, or categories.\n\n"

            # Track the cluster order to match the response
            cluster_order = []
            for cluster_id, keywords in batch_clusters:
                cluster_order.append(cluster_id)
                analysis_prompt += f"Cluster {cluster_id} representative keywords: {', '.join(keywords[:15])}\n\n"

            analysis_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=300
            )

            analysis_text = analysis_response.choices[0].message.content.strip()

            # Prompt para generar nombres
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

            # Procesar la respuesta JSON
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

                if clusters_data:
                    # Match clusters by position if IDs don't match
                    if len(clusters_data) == len(cluster_order):
                        for i, cluster_info in enumerate(clusters_data):
                            actual_id = cluster_order[i]
                            results[actual_id] = (
                                cluster_info.get("cluster_name", f"Cluster {actual_id}"),
                                cluster_info.get("description", "No description provided.")
                            )
            except Exception as e:
                st.warning(f"Error analizando respuesta JSON para batch {batch_idx+1}/{total_batches}: {str(e)}")
                # Fallback para este batch
                for i, cluster_id in enumerate(cluster_order):
                    results[cluster_id] = (f"Cluster {cluster_id}", "Error parsing JSON response")

            # Progress update
            progress_bar.progress((batch_idx + 1) / total_batches)
            time.sleep(1)  # Evitar límites de rate

        except Exception as e:
            st.warning(f"Error generando nombres para batch {batch_idx+1}/{total_batches}: {str(e)}")
            # Proporcionar nombres predeterminados
            for cluster_id, _ in batch_clusters:
                results[cluster_id] = (f"Cluster {cluster_id}", "Error generating description")
    
    # Asegurar que todos los clusters tienen nombre
    for cluster_id in clusters_with_representatives.keys():
        if cluster_id not in results:
            results[cluster_id] = (f"Cluster {cluster_id}", f"Grupo de keywords {cluster_id}")
    
    progress_bar.progress(1.0)
    progress_text.text(f"✅ Nombres y descripciones generados para {len(results)} clusters")
    
    return results
    # Función para calcular coherencia de clusters
def calculate_cluster_coherence(cluster_embeddings):
    """Calculate semantic coherence of a cluster based on embedding similarity"""
    if len(cluster_embeddings) <= 1:
        return 1.0  # Perfect coherence for single element

    try:
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Calculate average cosine similarity to centroid
        similarities = []
        for emb in cluster_embeddings:
            # Evitar divisiones por cero
            norm_emb = np.linalg.norm(emb)
            norm_centroid = np.linalg.norm(centroid)
            if norm_emb > 0 and norm_centroid > 0:
                similarity = np.dot(emb, centroid) / (norm_emb * norm_centroid)
                similarities.append(similarity)
            else:
                similarities.append(0.0)

        return np.mean(similarities) if similarities else 0.0
    except Exception as e:
        st.warning(f"Error calculando coherencia: {str(e)}")
        return 0.5  # Valor predeterminado en caso de error

# Función principal para ejecutar el clustering
def run_clustering():
    if uploaded_file is None:
        st.warning("Por favor, sube un archivo CSV con keywords.")
        return
    
    st.info("Iniciando proceso de clustering semántico...")
    
    # Configurar cliente OpenAI si se proporciona la clave API
client = None
if openai_api_key and openai_available:
    try:
        # Verificamos que la API key no esté vacía
        if openai_api_key.strip() == "":
            st.info("No se ha proporcionado una API Key de OpenAI válida. Los clusters tendrán nombres genéricos.")
        else:
            # Establecemos la API key como variable de entorno
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
            # Creamos el cliente sin parámetros adicionales
            client = OpenAI()
            
            # Verificamos la conexión con una solicitud simple
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5
                )
                st.success("✅ Conexión con OpenAI establecida correctamente")
            except Exception as e:
                st.error(f"Error al verificar la conexión con OpenAI: {str(e)}")
                st.error("Posible causa: API Key inválida o problemas de conexión")
                client = None
    except Exception as e:
        st.error(f"Error configurando cliente OpenAI: {str(e)}")
        st.info("Continuando sin funcionalidades de OpenAI")
        client = None
elif not openai_available:
    st.warning("Biblioteca OpenAI no está disponible. Continuando sin funcionalidades de OpenAI.")
elif not openai_api_key or openai_api_key.strip() == "":
    st.info("No se ha proporcionado API Key de OpenAI. Los nombres de clusters serán genéricos.")
    
    # Cargar y procesar el CSV
    try:
        # Leer CSV
        try:
            df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
            st.success(f"✅ Cargadas {len(df)} keywords del archivo CSV")
        except Exception as e:
            st.error(f"Error leyendo CSV: {str(e)}")
            st.info("Intentando formato alternativo...")
            # Intentar otros formatos/separadores
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                df = pd.read_csv(StringIO(content), sep=None, engine='python', header=None)
                df.columns = ["keyword"]
                st.success(f"✅ Cargadas {len(df)} keywords del archivo CSV (formato alternativo)")
            except Exception as e2:
                st.error(f"No se pudo leer el archivo CSV: {str(e2)}")
                return
        
        # Preprocesar keywords
        st.subheader("Preprocesamiento de Keywords")
        st.info("Preprocesando keywords...")
        keywords_processed = preprocess_keywords(df["keyword"].tolist())
        df['keyword_processed'] = keywords_processed
        st.success("✅ Keywords preprocesadas correctamente")
        
        # Generar embeddings
        st.subheader("Generación de Vectores")
        keyword_embeddings = generate_tfidf_embeddings(
            df['keyword_processed'].fillna(''), 
            min_df=min_df, 
            max_df=max_df/100.0
        )
        
        # Aplicar PCA
        st.subheader("Reducción de Dimensionalidad (PCA)")
        
        try:
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
            # Si no hay suficientes componentes para la varianza deseada, usar el máximo
            if n_components == 1 and len(cumulative_variance) > 1:
                n_components = min(max_pca_components, len(cumulative_variance))
                
            pca_text.text(f"Componentes para {pca_variance}% de varianza: {n_components}")
            pca_progress.progress(0.6)
            
            # Use that number (with a reasonable cap)
            max_components = min(n_components, max_pca_components)
            pca = PCA(n_components=max_components)
            keyword_embeddings_reduced = pca.fit_transform(keyword_embeddings)
            
            pca_progress.progress(1.0)
            pca_text.text(f"✅ PCA aplicado: {max_components} dimensiones ({pca_variance}% de varianza explicada)")
        except Exception as e:
            st.error(f"Error aplicando PCA: {str(e)}")
            st.info("Continuando sin reducción de dimensionalidad")
            # En caso de error, mantener los embeddings originales
            keyword_embeddings_reduced = keyword_embeddings
        
        # Aplicar clustering jerárquico
        st.subheader("Clustering Jerárquico")
        
        try:
            cluster_progress = st.progress(0)
            cluster_text = st.empty()
            cluster_text.text("Aplicando clustering jerárquico...")
            
            Z = linkage(keyword_embeddings_reduced, method="ward")
            cluster_progress.progress(0.5)
            
            df["cluster_id"] = fcluster(Z, t=num_clusters, criterion="maxclust")
            
            cluster_progress.progress(1.0)
            cluster_text.text(f"✅ Keywords agrupadas en {num_clusters} clusters")
        except Exception as e:
            st.error(f"Error en clustering jerárquico: {str(e)}")
            st.info("Intentando clustering alternativo...")
            
            # Fallback: Asignar clusters de manera más básica
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                df["cluster_id"] = kmeans.fit_predict(keyword_embeddings_reduced) + 1  # +1 para empezar desde 1
                st.success("✅ Clustering completado usando K-Means como alternativa")
            except Exception as e2:
                st.error(f"Error en clustering alternativo: {str(e2)}")
                # Último recurso: asignar clusters aleatorios
                df["cluster_id"] = np.random.randint(1, num_clusters + 1, size=len(df))
                st.warning("⚠️ Se han asignado clusters aleatorios como último recurso")
        
        # Identificar keywords representativas para cada cluster
        st.subheader("Análisis de Clusters")
        
        rep_progress = st.progress(0)
        rep_text = st.empty()
        rep_text.text("Identificando keywords representativas...")
        
        clusters_with_representatives = {}
        try:
            for i, cluster_num in enumerate(df['cluster_id'].unique()):
                cluster_size = len(df[df['cluster_id'] == cluster_num])
                n_representatives = min(15, cluster_size)
                
                # Get indices of keywords in this cluster
                indices = df[df['cluster_id'] == cluster_num].index.tolist()
                
                # Calculate centroid of the cluster
                cluster_embeddings = np.array([keyword_embeddings_reduced[i] for i in indices])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distance to centroid for each keyword
                distances = [np.linalg.norm(keyword_embeddings_reduced[i] - centroid) for i in indices]
                
                # Get indices of keywords closest to centroid
                sorted_indices = np.argsort(distances)[:n_representatives]
                representative_indices = [indices[i] for i in sorted_indices]
                representative_keywords = df.iloc[representative_indices]['keyword'].tolist()
                
                clusters_with_representatives[cluster_num] = representative_keywords
                
                # Update progress
                rep_progress.progress((i+1) / len(df['cluster_id'].unique()))
            
            rep_progress.progress(1.0)
            rep_text.text(f"✅ Identificadas keywords representativas para {len(clusters_with_representatives)} clusters")
        except Exception as e:
            st.error(f"Error identificando keywords representativas: {str(e)}")
            # Fallback: tomar las primeras N keywords de cada cluster
            for cluster_num in df['cluster_id'].unique():
                cluster_keywords = df[df['cluster_id'] == cluster_num]['keyword'].tolist()
                clusters_with_representatives[cluster_num] = cluster_keywords[:min(15, len(cluster_keywords))]
            st.warning("Se han seleccionado keywords representativas básicas como alternativa")
        
        # Generar nombres para los clusters si está disponible OpenAI
        if client:
            st.subheader("Generación de Nombres para Clusters")
            try:
                cluster_names = generate_cluster_names(
                    clusters_with_representatives, 
                    client,
                    model=gpt_model
                )
            except Exception as e:
                st.error(f"Error generando nombres de clusters: {str(e)}")
                cluster_names = {k: (f"Cluster {k}", f"Grupo de keywords {k}") for k in df['cluster_id'].unique()}
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
        
        try:
            for i, cluster_num in enumerate(df['cluster_id'].unique()):
                cluster_indices = df[df['cluster_id'] == cluster_num].index.tolist()
                cluster_embeddings = np.array([keyword_embeddings_reduced[i] for i in cluster_indices])
                coherence = calculate_cluster_coherence(cluster_embeddings)
                df.loc[df['cluster_id'] == cluster_num, 'cluster_coherence'] = coherence
                
                # Update progress
                coh_progress.progress((i+1) / len(df['cluster_id'].unique()))
            
            coh_progress.progress(1.0)
            coh_text.text("✅ Coherencia semántica calculada para todos los clusters")
        except Exception as e:
            st.error(f"Error calculando coherencia semántica: {str(e)}")
            # Asignar un valor predeterminado de coherencia
            for cluster_num in df['cluster_id'].unique():
                df.loc[df['cluster_id'] == cluster_num, 'cluster_coherence'] = 0.5
            st.warning("Se ha asignado un valor de coherencia predeterminado")
        
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
    
    1. **Preprocesamiento**: Las keywords se limpian y normalizan eliminando stopwords y aplicando lematización.
    
    2. **Vectorización TF-IDF**: Convierte las keywords en vectores numéricos basados en la frecuencia de las palabras.
    
    3. **Reducción de dimensionalidad**: Se aplica PCA para reducir la dimensionalidad manteniendo la varianza deseada.
    
    4. **Clustering jerárquico**: Se agrupan las keywords en clusters basados en similitud semántica.
    
    5. **Análisis de clusters**: Se identifican keywords representativas y se evalúa la coherencia semántica.
    
    6. **Generación de nombres**: Opcionalmente, se usan modelos GPT para generar nombres y descripciones significativas.
    
    ### Consejos para mejores resultados
    
    - **Calidad de datos**: Asegúrate de que tus keywords estén limpias y sean relevantes.
    - **Número de clusters**: Experimenta con diferentes números para encontrar el óptimo.
    - **PCA**: Ajusta la varianza explicada según la complejidad de tus datos.
    - **Frecuencia de términos**: Ajusta los parámetros min_df y max_df para mejorar la relevancia.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Desarrollado para clustering semántico de keywords | Versión optimizada para Streamlit Free
    </div>
    """, 
    unsafe_allow_html=True
)
