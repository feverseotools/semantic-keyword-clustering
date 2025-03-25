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
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

# Intentar importar bibliotecas avanzadas con manejo de errores
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
        # Si el modelo no está descargado
        spacy_available = False
except ImportError:
    spacy_available = False

try:
    import hdbscan
    hdbscan_available = True
except ImportError:
    hdbscan_available = False

# Descargar recursos de NLTK al inicio para evitar problemas posteriores
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    pass  # Continuar incluso si la descarga falla

#############################
# FUNCIONES AUXILIARES MEJORADAS
#############################

# MEJORA 4: Preprocesamiento Semántico Mejorado
def enhanced_preprocessing(text, use_lemmatization=True):
    """Preprocesamiento mejorado con tratamiento de entidades y n-gramas"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Usar spaCy para un análisis lingüístico más avanzado
        if spacy_available:
            doc = nlp(text.lower())
            
            # Conservar entidades nombradas completas
            entities = [ent.text for ent in doc.ents]
            
            # Extraer tokens relevantes (no stopwords)
            tokens = []
            for token in doc:
                if not token.is_stop and token.is_alpha and len(token.text) > 1:
                    tokens.append(token.lemma_)
            
            # Extraer bigramas relevantes
            bigrams = []
            for i in range(len(doc) - 1):
                if (not doc[i].is_stop and not doc[i+1].is_stop and 
                    doc[i].is_alpha and doc[i+1].is_alpha):
                    bigrams.append(f"{doc[i].lemma_}_{doc[i+1].lemma_}")
            
            # Combinar todo preservando las entidades
            processed_parts = tokens + bigrams + entities
            return " ".join(processed_parts)
        else:
            # Fallback al método original si spaCy no está disponible
            return preprocess_text(text, use_lemmatization)
    except Exception as e:
        return text.lower() if isinstance(text, str) else ""

# Función original de preprocesamiento como fallback
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
def preprocess_keywords(keywords, use_advanced=True):
    processed_keywords = []
    
    progress_bar = st.progress(0)
    total = len(keywords)
    
    for i, keyword in enumerate(keywords):
        if use_advanced and spacy_available:
            processed_keywords.append(enhanced_preprocessing(keyword))
        else:
            processed_keywords.append(preprocess_text(keyword))
        
        # Update progress bar every 100 items
        if i % 100 == 0:
            progress_bar.progress(min(i / total, 1.0))
    
    progress_bar.progress(1.0)
    return processed_keywords

# MEJORA 1: Embeddings mejorados con prioridad a OpenAI y límite de 5000 keywords
def generate_embeddings(df, openai_available, openai_api_key=None):
    st.info("Generando embeddings para las keywords...")
    
    # Opción 1: Usar OpenAI si está disponible y se proporciona API key
    if openai_available and openai_api_key:
        try:
            st.info("Usando embeddings de OpenAI (alta precisión semántica)")
            # Configurar OpenAI
            os.environ["OPENAI_API_KEY"] = openai_api_key
            client = OpenAI()
            
            # Procesar en batches para minimizar costos
            keywords = df['keyword_processed'].fillna('').tolist()
            all_embeddings = []
            
            # Aumentado a 5000 keywords (en lugar de 1000)
            if len(keywords) > 5000:
                st.warning(f"Limitando a 5000 keywords representativas de las {len(keywords)} totales")
                # Seleccionar keywords estratégicamente (no solo las primeras)
                step = max(1, len(keywords) // 5000)
                sample_indices = list(range(0, len(keywords), step))[:5000]
                sample_keywords = [keywords[i] for i in sample_indices]
                
                progress_bar = st.progress(0)
                st.info("Procesando embeddings con OpenAI (esto puede tomar unos minutos)...")
                
                # Crear embeddings para muestra
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=sample_keywords
                )
                progress_bar.progress(0.5)
                
                # Extraer embeddings
                sample_embeddings = np.array([item.embedding for item in response.data])
                
                # Propagar embeddings al resto por similitud TF-IDF
                st.info("Propagando embeddings al resto de keywords...")
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(keywords)
                
                all_embeddings = np.zeros((len(keywords), len(sample_embeddings[0])))
                # Asignar embeddings a la muestra
                for i, idx in enumerate(sample_indices):
                    all_embeddings[idx] = sample_embeddings[i]
                
                # Para el resto, encontrar el más similar en TF-IDF
                remaining_indices = [i for i in range(len(keywords)) if i not in sample_indices]
                for i, idx in enumerate(remaining_indices):
                    similarities = cosine_similarity(
                        tfidf_matrix[idx:idx+1],
                        tfidf_matrix[sample_indices]
                    )[0]
                    most_similar_idx = sample_indices[np.argmax(similarities)]
                    all_embeddings[idx] = all_embeddings[most_similar_idx]
                    
                    # Actualizar progreso para la segunda mitad
                    if i % 100 == 0:
                        progress_bar.progress(0.5 + min(0.5, (i / len(remaining_indices) * 0.5)))
                
                progress_bar.progress(1.0)
            else:
                # Si son menos de 5000, procesar todas
                progress_bar = st.progress(0)
                st.info(f"Procesando embeddings para todas las {len(keywords)} keywords con OpenAI...")
                
                # Procesar en lotes de 1000 para evitar límites de API
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
            st.success(f"✅ Generados embeddings de {embeddings.shape[1]} dimensiones usando OpenAI")
            return embeddings
                
        except Exception as e:
            st.error(f"Error generando embeddings con OpenAI: {str(e)}")
            st.info("Intentando con Sentence Transformers como alternativa...")
    
    # Opción 2: Usar Sentence Transformers como fallback (sin costo)
    if sentence_transformers_available:
        try:
            st.success("Usando SentenceTransformer como fallback (sin costo)")
            
            # Usa un modelo multilingüe si tus keywords están en varios idiomas
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            progress_bar = st.progress(0)
            keywords = df['keyword_processed'].fillna('').tolist()
            
            # Procesar en batches para evitar problemas de memoria
            batch_size = 512
            all_embeddings = []
            
            for i in range(0, len(keywords), batch_size):
                batch = keywords[i:i+batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                progress_bar.progress(min(1.0, (i + batch_size) / len(keywords)))
                
            progress_bar.progress(1.0)
            embeddings = np.array(all_embeddings)
            st.success(f"✅ Generados embeddings de {embeddings.shape[1]} dimensiones usando SentenceTransformer")
            return embeddings
        except Exception as e:
            st.error(f"Error con SentenceTransformer: {str(e)}")
    
    # Opción 3: Fallback a TF-IDF (menos preciso) como último recurso
    st.warning("Usando TF-IDF como último recurso (menos preciso semánticamente)")
    return generate_tfidf_embeddings(df['keyword_processed'].fillna(''))

# Función original TF-IDF como fallback
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

# MEJORA 2: Algoritmo de clustering mejorado
def improved_clustering(embeddings, num_clusters=None, min_cluster_size=5):
    st.info("Aplicando algoritmos de clustering avanzados...")
    
    # Determinar automáticamente el número óptimo de clusters si no se especifica
    if num_clusters is None:
        try:
            from sklearn.metrics import silhouette_score
            
            st.info("Buscando número óptimo de clusters...")
            sil_scores = []
            max_clusters = min(30, len(embeddings) // 5)
            range_n_clusters = range(2, max(3, max_clusters))
            
            progress_bar = st.progress(0)
            
            # Calcular score de silueta para diferentes números de clusters
            for i, n_clusters in enumerate(range_n_clusters):
                # Usar K-Means para la prueba por ser más rápido
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=2)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calcular silueta (si hay suficientes muestras)
                if len(set(cluster_labels)) > 1:
                    try:
                        # Usar muestra para calcular silueta si hay muchos datos
                        if len(embeddings) > 5000:
                            sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
                            sample_score = silhouette_score(
                                embeddings[sample_indices], 
                                cluster_labels[sample_indices]
                            )
                        else:
                            sample_score = silhouette_score(embeddings, cluster_labels)
                        sil_scores.append(sample_score)
                    except:
                        sil_scores.append(0)
                else:
                    sil_scores.append(0)
                    
                progress_bar.progress((i + 1) / len(range_n_clusters))
                    
            # Seleccionar el número de clusters con mejor score
            if sil_scores:
                best_num_clusters = range_n_clusters[np.argmax(sil_scores)]
                st.success(f"Número óptimo de clusters determinado: {best_num_clusters}")
                num_clusters = best_num_clusters
            else:
                st.warning("No se pudo determinar el número óptimo de clusters. Usando valor predeterminado.")
        except Exception as e:
            st.error(f"Error determinando número óptimo de clusters: {str(e)}")
    
    # Probar HDBSCAN si está disponible (mejor para clusters de forma irregular)
    if hdbscan_available:
        try:
            st.info("Aplicando HDBSCAN para detección de clusters de forma natural...")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Verificar si HDBSCAN encontró una estructura razonable
            # Limitar el número máximo de clusters a un valor razonable
            unique_clusters = np.unique(cluster_labels)
            non_noise_clusters = [c for c in unique_clusters if c != -1]
            
            if len(non_noise_clusters) > 1 and len(non_noise_clusters) <= num_clusters * 2:
                st.success(f"HDBSCAN identificó {len(non_noise_clusters)} clusters naturales")
                
                # Reasignar cluster -1 (ruido) al cluster más cercano
                if -1 in unique_clusters:
                    noise_indices = np.where(cluster_labels == -1)[0]
                    for idx in noise_indices:
                        # Encontrar el centroide más cercano
                        min_dist = float('inf')
                        nearest_cluster = non_noise_clusters[0]
                        
                        for cluster in non_noise_clusters:
                            cluster_points = embeddings[cluster_labels == cluster]
                            if len(cluster_points) > 0:
                                centroid = np.mean(cluster_points, axis=0)
                                dist = np.linalg.norm(embeddings[idx] - centroid)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_cluster = cluster
                        
                        cluster_labels[idx] = nearest_cluster
                
                # Reasignar IDs para que empiecen desde 1
                old_to_new = {old_id: new_id + 1 for new_id, old_id in enumerate(np.unique(cluster_labels))}
                cluster_labels = np.array([old_to_new[label] for label in cluster_labels])
                
                return cluster_labels
        except Exception as e:
            st.warning(f"Error con HDBSCAN: {str(e)}. Usando clustering jerárquico.")
    
    # Fallback a clustering jerárquico
    try:
        st.info("Aplicando clustering jerárquico aglomerativo...")
        # Probar diferentes métodos de linkage para encontrar el mejor
        methods = ['ward', 'complete', 'average']
        best_method = 'ward'  # Valor predeterminado
        
        # Si el dataset no es demasiado grande, probar diferentes métodos
        if len(embeddings) < 5000:
            coherence_scores = []
            
            for method in methods:
                try:
                    Z = linkage(embeddings, method=method)
                    labels = fcluster(Z, t=num_clusters, criterion="maxclust")
                    
                    # Calcular coherencia promedio
                    coherence = 0
                    for cluster_id in np.unique(labels):
                        cluster_vectors = embeddings[labels == cluster_id]
                        if len(cluster_vectors) > 1:
                            centroid = np.mean(cluster_vectors, axis=0)
                            dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
                            coherence += np.mean(1 / (1 + dists))
                    
                    coherence_scores.append(coherence / len(np.unique(labels)))
                except:
                    coherence_scores.append(0)
            
            if coherence_scores:
                best_method = methods[np.argmax(coherence_scores)]
                st.success(f"Método de linkage óptimo: {best_method}")
        
        # Aplicar clustering con el mejor método
        Z = linkage(embeddings, method=best_method)
        labels = fcluster(Z, t=num_clusters, criterion="maxclust")
        
        return labels
        
    except Exception as e:
        st.error(f"Error en clustering jerárquico: {str(e)}")
        
        # Último recurso: K-Means
        st.warning("Usando K-Means como alternativa")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings) + 1  # +1 para empezar desde 1

# MEJORA 3: Refinamiento Post-Clustering
def refine_clusters(df, embeddings, original_cluster_column='cluster_id'):
    """Refina los clusters identificando y corrigiendo asignaciones pobres"""
    st.info("Refinando clusters para mejorar coherencia semántica...")
    
    # Guardar las asignaciones originales
    df['original_cluster'] = df[original_cluster_column]
    
    # 1. Identificar outliers semánticos en cada cluster
    outliers = []
    for cluster_id in df[original_cluster_column].unique():
        # Obtener índices de este cluster
        cluster_indices = df[df[original_cluster_column] == cluster_id].index.tolist()
        
        if len(cluster_indices) <= 3:  # Clusters muy pequeños, no refinar
            continue
            
        # Calcular centroide del cluster
        cluster_embeddings = np.array([embeddings[i] for i in cluster_indices])
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calcular distancias al centroide
        distances = [np.linalg.norm(embeddings[i] - centroid) for i in cluster_indices]
        
        # Normalizar distancias para este cluster
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist == 0:
            continue
            
        normalized_distances = [(d - mean_dist) / std_dist for d in distances]
        
        # Identificar outliers (keywords muy lejanas al centroide)
        for i, norm_dist in enumerate(normalized_distances):
            if norm_dist > 2.0:  # Más de 2 desviaciones estándar
                outliers.append((cluster_indices[i], cluster_id, norm_dist))
    
    # 2. Reasignar outliers a clusters más apropiados
    reassigned = 0
    for idx, original_cluster, _ in outliers:
        keyword_embedding = embeddings[idx]
        
        # Encontrar cluster más cercano (excluyendo el original)
        min_distance = float('inf')
        best_cluster = original_cluster
        
        for cluster_id in df[original_cluster_column].unique():
            if cluster_id == original_cluster:
                continue
                
            # Obtener índices de este cluster
            cluster_indices = df[df[original_cluster_column] == cluster_id].index.tolist()
            
            # Calcular centroide
            cluster_embeddings = np.array([embeddings[i] for i in cluster_indices])
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calcular distancia
            distance = np.linalg.norm(keyword_embedding - centroid)
            
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_id
        
        # Reasignar si encontramos un cluster mejor
        if best_cluster != original_cluster:
            df.loc[idx, original_cluster_column] = best_cluster
            reassigned += 1
    
    # 3. Combinar clusters demasiado similares
    similar_pairs = []
    clusters = df[original_cluster_column].unique()
    
    for i, cluster1 in enumerate(clusters):
        for cluster2 in clusters[i+1:]:
            # Calcular centroides
            indices1 = df[df[original_cluster_column] == cluster1].index.tolist()
            indices2 = df[df[original_cluster_column] == cluster2].index.tolist()
            
            if len(indices1) < 3 or len(indices2) < 3:
                continue  # Ignorar clusters muy pequeños
                
            centroid1 = np.mean(np.array([embeddings[i] for i in indices1]), axis=0)
            centroid2 = np.mean(np.array([embeddings[i] for i in indices2]), axis=0)
            
            # Calcular similitud coseno
            similarity = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
            
            if similarity > 0.8:  # Umbral alto para fusionar
                similar_pairs.append((cluster1, cluster2, similarity))
    
    # Ordenar por similitud para combinar primero los más similares
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Combinar clusters (manteniendo el ID más bajo)
    clusters_merged = 0
    processed_clusters = set()
    
    for cluster1, cluster2, _ in similar_pairs:
        if cluster1 in processed_clusters or cluster2 in processed_clusters:
            continue  # Evitar combinar clusters ya procesados
            
        # Elegir el ID más bajo para mantener
        keep_id = min(cluster1, cluster2)
        remove_id = max(cluster1, cluster2)
        
        # Reasignar keywords del cluster a eliminar
        df.loc[df[original_cluster_column] == remove_id, original_cluster_column] = keep_id
        
        processed_clusters.add(remove_id)
        clusters_merged += 1
        
        # Limitar número de fusiones
        if clusters_merged >= len(clusters) // 4:  # Máximo 25% de fusiones
            break
    
    st.success(f"Refinamiento completado: {reassigned} keywords reasignadas, {clusters_merged} clusters fusionados.")
    return df

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
            # Prompt mejorado para analizar clusters
            analysis_prompt = """I'll provide representative keywords for several clusters. For each cluster, analyze the keywords to identify:
1. Common themes, topics, or categories
2. User intent or purpose behind these keywords
3. Semantic relationships between words
4. Any distinctive patterns that make this cluster unique

Be thorough and insightful in your analysis.

"""

            # Track the cluster order to match the response
            cluster_order = []
            for cluster_id, keywords in batch_clusters:
                cluster_order.append(cluster_id)
                analysis_prompt += f"Cluster {cluster_id} representative keywords: {', '.join(keywords[:20])}\n\n"

            analysis_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500
            )

            analysis_text = analysis_response.choices[0].message.content.strip()

            # Prompt mejorado para generar nombres más precisos
            naming_prompt = f"""Based on the following analysis of keyword clusters, provide a specific name and description for each cluster.

Analysis:
{analysis_text}

For each cluster, provide:
1. A specific, descriptive cluster name (3-5 words) that clearly identifies the semantic theme
2. A concise description (1-2 sentences) that accurately represents the semantic relationship between the keywords

Your names should be concrete and specific, not generic. Focus on semantic meaning, not just superficial word patterns.

Format your response as a JSON array, with each element containing cluster_id, cluster_name, and description for clusters {', '.join(map(str, cluster_order))}.
"""

            naming_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": naming_prompt}],
                temperature=0.2,
                max_tokens=600,
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

# MEJORA 5: Evaluación avanzada de clusters
def evaluate_cluster_quality(df, embeddings, cluster_column='cluster_id'):
    """Evalúa la calidad de los clusters usando múltiples métricas"""
    st.subheader("Evaluación Avanzada de Calidad de Clusters")
    
    metrics = {
        'silhouette': [],
        'density': [],
        'separation': [],
        'coherence': []
    }
    
    # Calcular centroides de todos los clusters
    centroids = {}
    for cluster_id in df[cluster_column].unique():
        indices = df[df[cluster_column] == cluster_id].index.tolist()
        centroids[cluster_id] = np.mean(np.array([embeddings[i] for i in indices]), axis=0)
    
    # Evaluar cada cluster
    cluster_progress = st.progress(0)
    for i, cluster_id in enumerate(df[cluster_column].unique()):
        indices = df[df[cluster_column] == cluster_id].index.tolist()
        cluster_vectors = np.array([embeddings[i] for i in indices])
        centroid = centroids[cluster_id]
        
        # 1. Densidad (distancia promedio al centro)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        density = 1 / (1 + np.mean(distances)) if distances else 0
        metrics['density'].append((cluster_id, density))
        
        # 2. Coherencia (similitud coseno promedio entre vectores)
        coherence = calculate_cluster_coherence(cluster_vectors)
        metrics['coherence'].append((cluster_id, coherence))
        
        # 3. Separación (distancia mínima a otro centroide)
        min_separation = float('inf')
        for other_id, other_centroid in centroids.items():
            if other_id != cluster_id:
                separation = np.linalg.norm(centroid - other_centroid)
                min_separation = min(min_separation, separation)
        
        if min_separation != float('inf'):
            metrics['separation'].append((cluster_id, min_separation))
            
        cluster_progress.progress((i + 1) / len(df[cluster_column].unique()))
    
    # Visualizar métricas
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de coherencia vs tamaño
        coherence_data = pd.DataFrame(metrics['coherence'], columns=['cluster_id', 'score'])
        coherence_data = coherence_data.merge(
            df.groupby(cluster_column)['keyword'].count().reset_index(),
            left_on='cluster_id', right_on=cluster_column
        )
        coherence_data = coherence_data.merge(
            df.drop_duplicates(cluster_column)[['cluster_id', 'cluster_name']],
            on='cluster_id'
        )
        
        fig = px.scatter(
            coherence_data, 
            x='score', 
            y='keyword', 
            color='score',
            size='keyword',
            hover_data=['cluster_name'],
            labels={
                'score': 'Coherencia Semántica', 
                'keyword': 'Tamaño del Cluster'
            },
            title='Relación entre Coherencia y Tamaño',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de separación vs densidad
        if metrics['separation']:
            separation_data = pd.DataFrame(metrics['separation'], columns=['cluster_id', 'separation'])
            density_data = pd.DataFrame(metrics['density'], columns=['cluster_id', 'density'])
            
            combined_data = separation_data.merge(density_data, on='cluster_id')
            combined_data = combined_data.merge(
                df.drop_duplicates(cluster_column)[['cluster_id', 'cluster_name']],
                on='cluster_id'
            )
            
            fig2 = px.scatter(
                combined_data,
                x='separation',
                y='density',
                color='density',
                hover_data=['cluster_name'],
                labels={
                    'separation': 'Separación entre Clusters',
                    'density': 'Densidad del Cluster'
                },
                title='Separación vs Densidad',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Identificar clusters problemáticos
    st.subheader("Diagnóstico de Clusters")
    
    # Calcular umbrales
    coherence_threshold = np.percentile([x[1] for x in metrics['coherence']], 25)
    problematic = [x[0] for x in metrics['coherence'] if x[1] < coherence_threshold]
    
    # Añadir coherencia al dataframe original
    for cluster_id, coherence in metrics['coherence']:
        df.loc[df[cluster_column] == cluster_id, 'cluster_coherence'] = coherence
    
    if problematic:
        st.warning(f"Clusters con baja coherencia semántica: {problematic}")
        st.info("""
        Recomendaciones para mejorar:
        - Considera aumentar el número de clusters
        - Revisa las keywords en estos clusters específicos
        - Prueba usar embeddings de mayor calidad
        - Considera la posibilidad de dividir estos clusters manualmente
        """)
    else:
        st.success("Todos los clusters tienen buena coherencia semántica")
        
    return df

# Función básica para calcular coherencia
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

# Función principal para ejecutar el clustering mejorado
def run_clustering(uploaded_file, openai_api_key, num_clusters, pca_variance, max_pca_components, min_df, max_df, gpt_model):
    """Ejecuta el proceso completo de clustering y devuelve los resultados"""
    if uploaded_file is None:
        st.warning("Por favor, sube un archivo CSV con keywords.")
        return False, None
    
    st.info("Iniciando proceso de clustering semántico avanzado...")
    
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
        st.info("No se ha proporcionado API Key de OpenAI. Se usará SentenceTransformers como alternativa gratuita.")
    
    try:
        # Cargar y procesar el CSV
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
                return False, None
        
        # Preprocesar keywords
        st.subheader("Preprocesamiento de Keywords")
        st.info("Preprocesando keywords con análisis semántico mejorado...")
        
        # MEJORA 4: Usar el preprocesamiento semántico mejorado
        use_advanced = spacy_available
        if use_advanced:
            st.success("Usando preprocesamiento avanzado con análisis lingüístico")
        else:
            st.info("Usando preprocesamiento estándar (SpaCy no disponible)")
            
        keywords_processed = preprocess_keywords(df["keyword"].tolist(), use_advanced=use_advanced)
        df['keyword_processed'] = keywords_processed
        st.success("✅ Keywords preprocesadas correctamente")
        
        # Generar embeddings mejorados
        st.subheader("Generación de Vectores Semánticos")
        
        # MEJORA 1: Usar embeddings de alta calidad
        keyword_embeddings = generate_embeddings(df, openai_available, openai_api_key)
        
        # Aplicar PCA si los embeddings son de alta dimensionalidad
        if keyword_embeddings.shape[1] > max_pca_components:
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
        else:
            # No necesita PCA si la dimensionalidad ya es adecuada
            keyword_embeddings_reduced = keyword_embeddings
            st.info(f"Dimensionalidad de embeddings adecuada ({keyword_embeddings.shape[1]}). No se requiere PCA.")
        
        # Aplicar clustering mejorado
        st.subheader("Clustering Semántico Avanzado")
        
        # MEJORA 2: Usar algoritmo de clustering mejorado
        try:
            cluster_labels = improved_clustering(keyword_embeddings_reduced, num_clusters=num_clusters)
            df["cluster_id"] = cluster_labels
            st.success(f"✅ Keywords agrupadas en {len(df['cluster_id'].unique())} clusters semánticos")
        except Exception as e:
            st.error(f"Error en clustering avanzado: {str(e)}")
            st.info("Intentando clustering alternativo...")
            
            # Fallback: Asignar clusters de manera más básica
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                df["cluster_id"] = kmeans.fit_predict(keyword_embeddings_reduced) + 1
                st.success("✅ Clustering completado usando K-Means como alternativa")
            except Exception as e2:
                st.error(f"Error en clustering alternativo: {str(e2)}")
                # Último recurso: asignar clusters aleatorios
                df["cluster_id"] = np.random.randint(1, num_clusters + 1, size=len(df))
                st.warning("⚠️ Se han asignado clusters aleatorios como último recurso")
        
        # MEJORA 3: Refinar clusters
        st.subheader("Refinamiento de Clusters")
        df = refine_clusters(df, keyword_embeddings_reduced)
        num_clusters_after_refinement = len(df['cluster_id'].unique())
        st.success(f"✅ Refinamiento completado: {num_clusters_after_refinement} clusters finales")
        
        # Identificar keywords representativas para cada cluster
        st.subheader("Análisis de Clusters")
        
        rep_progress = st.progress(0)
        rep_text = st.empty()
        rep_text.text("Identificando keywords representativas...")
        
        clusters_with_representatives = {}
        try:
            for i, cluster_num in enumerate(df['cluster_id'].unique()):
                cluster_size = len(df[df['cluster_id'] == cluster_num])
                n_representatives = min(20, cluster_size)
                
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
                clusters_with_representatives[cluster_num] = cluster_keywords[:min(20, len(cluster_keywords))]
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
        
        # MEJORA 5: Evaluación avanzada de calidad de clusters
        df = evaluate_cluster_quality(df, keyword_embeddings_reduced)
        
        # Devolver los resultados
        return True, df
        
    except Exception as e:
        st.error(f"Error durante el proceso: {str(e)}")
        return False, None

#############################
# APLICACIÓN PRINCIPAL
#############################

# Configuración de la página
st.set_page_config(
    page_title="Clustering Semántico Avanzado de Keywords",
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
    .highlight {
        background-color: #fffbcc;
        padding: 0.2rem 0.5rem;
        border-radius: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.markdown("<div class='main-header'>Clustering Semántico Avanzado de Keywords</div>", unsafe_allow_html=True)
st.markdown("""
Esta aplicación te permite agrupar keywords semánticamente similares utilizando técnicas avanzadas de NLP y clustering.
Sube tu archivo CSV con las keywords y configura los parámetros para obtener clusters de alta correlación semántica.
""")

# Mostrar estado de librerías avanzadas
with st.expander("Estado de bibliotecas semánticas", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        if openai_available:
            st.success("✅ OpenAI disponible (con API Key)")
        else:
            st.warning("⚠️ OpenAI no disponible")
            
        if sentence_transformers_available:
            st.success("✅ SentenceTransformers disponible (sin costo)")
        else:
            st.warning("⚠️ SentenceTransformers no disponible")
            st.markdown("""
            Para instalar:
            ```
            pip install sentence-transformers
            ```
            """)
    
    with col2:
        if spacy_available:
            st.success("✅ SpaCy disponible")
        else:
            st.warning("⚠️ SpaCy no disponible")
            
        if hdbscan_available:
            st.success("✅ HDBSCAN disponible")
        else:
            st.warning("⚠️ HDBSCAN no disponible")
            
    with col3:
        # Información de instalación
        st.info("""
        Para más funcionalidades:
        ```
        pip install sentence-transformers spacy hdbscan
        python -m spacy download en_core_web_sm
        ```
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

# 2. API Key de OpenAI (ajustado el mensaje para indicar preferencia)
openai_api_key = st.sidebar.text_input(
    "API Key de OpenAI (recomendado)",
    type="password", 
    help="Proporciona tu API Key de OpenAI para embeddings de alta calidad (hasta 5000 keywords). Si no se proporciona, se usará SentenceTransformers como alternativa gratuita."
)

# Mostrar estado de procesamiento semántico
if openai_available:
    if openai_api_key:
        st.sidebar.success("✅ API Key proporcionada - Se usará OpenAI para embeddings de alta precisión")
    else:
        if sentence_transformers_available:
            st.sidebar.info("ℹ️ Sin API Key - Se usará SentenceTransformers como alternativa gratuita")
        else:
            st.sidebar.warning("⚠️ Sin API Key ni SentenceTransformers - Se usará TF-IDF (precisión reducida)")
else:
    if sentence_transformers_available:
        st.sidebar.info("ℹ️ OpenAI no disponible - Se usará SentenceTransformers como alternativa gratuita")
    else:
        st.sidebar.error("❌ Métodos avanzados no disponibles - Se usará TF-IDF (precisión reducida)")

# 3. Parámetros de clustering
st.sidebar.markdown("<div class='sub-header'>Parámetros</div>", unsafe_allow_html=True)

# Panel de explicación de parámetros - Colocado antes de los sliders
with st.sidebar.expander("ℹ️ Guía de Parámetros", expanded=False):
    st.markdown("""
    ### Guía de Parámetros de Clustering
    
    Aquí encontrarás explicaciones sobre cada parámetro y cómo ajustarlo para obtener mejores resultados:
    
    #### Número de clusters
    **¿Qué es?** El número de grupos en los que se dividirán tus keywords.
    
    **Cómo usarlo:** 
    - **↑ Aumentar** si necesitas una división más detallada y específica por temas.
    - **↓ Disminuir** si prefieres grupos más generales y amplios.
    
    **Resultado:**
    - **Valores altos** (15-30): Muchos grupos pequeños y muy específicos.
    - **Valores bajos** (5-10): Pocos grupos pero más amplios temáticamente.
    - **Ideal:** Generalmente entre 8-15 para 1000 keywords. Aumenta proporcionalmente con la cantidad de keywords.
    
    ---
    
    #### Varianza explicada PCA (%)
    **¿Qué es?** Determina cuánta información original se conserva al simplificar los datos. Piensa en esto como el "nivel de detalle" que se mantiene.
    
    **Cómo usarlo:**
    - **↑ Aumentar** para mayor precisión y preservar más matices semánticos.
    - **↓ Disminuir** para acelerar el procesamiento con conjuntos grandes.
    
    **Resultado:**
    - **Valores altos** (95-99%): Mayor precisión semántica pero más lento.
    - **Valores bajos** (80-90%): Procesamiento más rápido pero puede perderse algunos matices.
    - **Ideal:** 90-95% ofrece un buen equilibrio entre precisión y velocidad.
    
    ---
    
    #### Máximo de componentes PCA
    **¿Qué es?** Limita la complejidad máxima del modelo de análisis. Similar a establecer un límite para evitar un exceso de complejidad.
    
    **Cómo usarlo:**
    - **↑ Aumentar** para datasets grandes o con alta diversidad temática.
    - **↓ Disminuir** para datasets más pequeños o centrados en un solo tema.
    
    **Resultado:**
    - **Valores altos** (100-200): Captura más relaciones complejas entre palabras.
    - **Valores bajos** (30-75): Más eficiente pero puede simplificar demasiado.
    - **Ideal:** Entre 75-100 para la mayoría de casos.
    
    ---
    
    #### Frecuencia mínima de términos
    **¿Qué es?** Ignora palabras que aparecen en muy pocas keywords. Ayuda a filtrar palabras raras o errores tipográficos.
    
    **Cómo usarlo:**
    - **↑ Aumentar** para eliminar términos poco comunes y posible ruido.
    - **↓ Disminuir** para incluir términos poco frecuentes que podrían ser importantes.
    
    **Resultado:**
    - **Valores altos** (3-5): Elimina más términos raros, clustering más "limpio".
    - **Valores bajos** (1-2): Conserva términos poco comunes, puede mantener más ruido.
    - **Ideal:** 1-2 para datasets pequeños, 2-3 para datasets grandes (+5000 keywords).
    
    ---
    
    #### Frecuencia máxima de términos (%)
    **¿Qué es?** Ignora palabras que aparecen en un alto porcentaje de keywords. Similar a eliminar "palabras comodín" que están en todas partes.
    
    **Cómo usarlo:**
    - **↑ Aumentar** para incluir más términos comunes.
    - **↓ Disminuir** para filtrar palabras demasiado genéricas.
    
    **Resultado:**
    - **Valores altos** (90-100%): Incluye casi todos los términos, incluso los muy comunes.
    - **Valores bajos** (70-85%): Enfoque en palabras más distintivas, ignorando las genéricas.
    - **Ideal:** 85-95% funciona bien para la mayoría de datasets.
    """)
    
    st.info("""
    **Consejo:** Si no estás seguro, mantén los valores predeterminados. La aplicación está optimizada para funcionar bien con estos parámetros en la mayoría de los casos.
    
    Para datasets grandes (+5000 keywords), considera aumentar ligeramente el número de clusters y reducir la varianza explicada PCA para mantener tiempos de procesamiento razonables.
    """)

# Sliders para los parámetros con descripciones mejoradas
num_clusters = st.sidebar.slider(
    "Número de clusters", 
    min_value=2, 
    max_value=50, 
    value=10, 
    help="Número de grupos en los que se dividirán tus keywords. Más clusters = grupos más específicos."
)

pca_variance = st.sidebar.slider(
    "Varianza explicada PCA (%)", 
    min_value=50, 
    max_value=99, 
    value=95, 
    help="Porcentaje de información que se conserva. Mayor valor = mayor precisión pero más lento."
)

max_pca_components = st.sidebar.slider(
    "Máximo de componentes PCA", 
    min_value=10, 
    max_value=300, 
    value=100, 
    help="Límite de complejidad del modelo. Mayor valor = captura más relaciones complejas."
)

# 4. Opciones avanzadas
st.sidebar.markdown("<div class='sub-header'>Opciones avanzadas</div>", unsafe_allow_html=True)

min_df = st.sidebar.slider(
    "Frecuencia mínima de términos", 
    min_value=1, 
    max_value=10, 
    value=1, 
    help="Ignora términos poco frecuentes. Mayor valor = elimina más palabras raras."
)

max_df = st.sidebar.slider(
    "Frecuencia máxima de términos (%)", 
    min_value=50, 
    max_value=100, 
    value=95, 
    help="Ignora términos demasiado comunes. Menor valor = elimina más palabras genéricas."
)

gpt_model = st.sidebar.selectbox(
    "Modelo para nombrar clusters", 
    ["gpt-3.5-turbo", "gpt-4"], 
    index=0,
    help="GPT-4 proporciona nombres más precisos pero es más costoso y lento."
)

# Botón para ejecutar el clustering
if uploaded_file is not None and not st.session_state.process_complete:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Iniciar Clustering Semántico Avanzado", type="primary", use_container_width=True):
            success, results = run_clustering(
                uploaded_file, 
                openai_api_key, 
                num_clusters, 
                pca_variance, 
                max_pca_components, 
                min_df, 
                max_df,
                gpt_model
            )
            if success:
                st.session_state.df_results = results
                st.session_state.process_complete = True
                st.markdown("<div class='success-box'>✅ Clustering semántico completado con éxito!</div>", unsafe_allow_html=True)

# Mostrar resultados si el proceso está completo
if st.session_state.process_complete and st.session_state.df_results is not None:
    st.markdown("<div class='main-header'>Resultados del Clustering</div>", unsafe_allow_html=True)
    
    df = st.session_state.df_results
    
    # Pestaña para mostrar visualizaciones
    with st.expander("Visualizaciones", expanded=True):
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
with st.expander("Información sobre el Clustering Semántico Avanzado"):
    st.markdown("""
    ### ¿Cómo funciona este clustering semántico avanzado?
    
    1. **Preprocesamiento Lingüístico**: Las keywords se analizan usando NLP avanzado para extraer entidades nombradas, bigramas relevantes y tokens significativos.
    
    2. **Embeddings de Alta Calidad**: Se utilizan modelos de embeddings de última generación:
       - OpenAI Embeddings (hasta 5000 keywords) si se proporciona API key
       - Sentence Transformers (sin costo) como alternativa o fallback
       - TF-IDF como último recurso
    
    3. **Reducción Inteligente de Dimensionalidad**: PCA optimizado para preservar las relaciones semánticas más importantes.
    
    4. **Clustering Avanzado**: Algoritmos que descubren automáticamente la estructura óptima:
       - HDBSCAN para detectar clusters de forma natural
       - Clustering jerárquico aglomerativo optimizado
       - Determinación automática del número de clusters
    
    5. **Refinamiento Post-Clustering**: Identifica y corrige asignaciones problemáticas:
       - Detección de outliers semánticos
       - Fusión de clusters muy similares
       - Reasignación de keywords mal clasificadas
    
    6. **Evaluación Multi-Métrica**: Análisis riguroso de la calidad de los clusters:
       - Coherencia semántica interna
       - Densidad y compacidad
       - Separación entre clusters
       - Diagnóstico de clusters problemáticos
    
    ### Consejos para obtener mejores resultados
    
    - **Calidad de keywords**: El clustering funciona mejor cuando las keywords están relacionadas con un mismo dominio o industria.
    
    - **Preprocesamiento**: Asegúrate de que tus keywords no contengan errores ortográficos o caracteres extraños.
    
    - **API Key de OpenAI**: Proporciona una API Key para embeddings de mayor calidad, aunque SentenceTransformers ofrece buenos resultados sin costo.
    
    - **Número de clusters**: Considera usar la determinación automática del número óptimo de clusters.
    
    - **Evaluación iterativa**: Examina los clusters con baja coherencia y considera ajustar parámetros o dividirlos.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Desarrollado para clustering semántico avanzado de keywords | Versión 2.1 con OpenAI/SentenceTransformers híbrido
    </div>
    """, 
    unsafe_allow_html=True
)
