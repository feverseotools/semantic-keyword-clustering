# Add at the top of the file
"""
PDF Report Generator for Semantic Keyword Clustering.

This module handles PDF generation from cluster analysis data.
"""

__all__ = ['PDFReport', 'add_pdf_export_button', 'create_download_link', 'sanitize_filename']

# Add version info
__version__ = "0.1.0"

import os
import tempfile
import base64
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from io import BytesIO
from PIL import Image as PILImage
from collections import Counter
import time
import logging
import shutil
import concurrent.futures
import contextlib
from contextlib import ExitStack

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('export_pdf')

class PDFReport:
    """
    Class for generating PDF reports from keyword clustering results.
    
    Attributes:
        df (pandas.DataFrame): The clustered keywords dataframe
        cluster_evaluation (dict): Dictionary containing cluster evaluation data
        app_name (str): Name of the application to show in report
        language (str): Language code for report localization ('en', 'es')
        temp_dir (str): Path to temporary directory for image generation
    """
# Additional import at the top of the file
import contextlib
from contextlib import ExitStack


def __init__(self, df, cluster_evaluation=None, app_name="Advanced Semantic Keyword Clustering", language="en"):
    """
    Initialize the PDF report generator with improved resource management.
    
    Args:
        df (pandas.DataFrame): The clustered keywords dataframe
        cluster_evaluation (dict, optional): Dictionary containing cluster evaluation data
        app_name (str, optional): Name of the application
        language (str, optional): Language code ('en', 'es')
    """
    self.df = df.copy()  # Create a copy to avoid modifying the original
    self.cluster_evaluation = cluster_evaluation if cluster_evaluation else {}
    self.app_name = app_name
    self.language = language
    self.resource_stack = ExitStack()
    
    # Create temp directory with error handling
    try:
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
    except Exception as e:
        logger.error(f"Failed to create temporary directory: {str(e)}")
        self.temp_dir = None
        
    # Initialize styles
    self.styles = getSampleStyleSheet()
    self.custom_styles = {}
    self.setup_custom_styles()
    
    # Load translations
    self.translations = self._get_translations()
    
    # Validate inputs
    self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data to ensure required columns exist."""
        required_columns = ['cluster_id', 'keyword']
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns in dataframe: {missing_columns}")
            # Add dummy columns if needed to prevent errors
            for col in missing_columns:
                self.df[col] = "Unknown" if col == 'keyword' else 0
        
        # Ensure cluster_id is an integer
        try:
            self.df['cluster_id'] = self.df['cluster_id'].astype(int)
        except Exception as e:
            logger.warning(f"Failed to convert cluster_id to int: {str(e)}")
    
def __del__(self):
    """Cleanup resources when the object is destroyed."""
    self.cleanup()

def cleanup(self):
    """Explicitly clean up resources - can be called manually or by __del__."""
    if hasattr(self, 'resource_stack'):
        self.resource_stack.close()
    
    try:
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    def _get_translations(self):
        """Get translations dictionary based on language."""
        translations = {
            "en": {
                "report_title": "Results Report",
                "generated_on": "Generated on",
                "total_keywords": "Total Keywords",
                "number_of_clusters": "Number of Clusters",
                "total_search_volume": "Total Search Volume",
                "cluster_distribution": "Cluster Distribution",
                "cluster_size_title": "Size of Each Cluster",
                "cluster": "Cluster",
                "num_keywords": "Number of Keywords",
                "search_intent_analysis": "Search Intent Analysis",
                "intent_distribution": "Search Intent Distribution",
                "intent_by_cluster": "Search Intent by Cluster",
                "cluster_details": "Cluster Details",
                "total_keywords_in_cluster": "Total Keywords",
                "description": "Description",
                "representative_keywords": "Representative Keywords",
                "primary_search_intent": "Primary Search Intent",
                "customer_journey_phase": "Customer Journey Phase",
                "top_keywords_by_volume": "Top Keywords by Search Volume",
                "keyword": "Keyword",
                "search_volume": "Search Volume",
                "conclusions": "Conclusions and Recommendations",
                "prioritize_rec": "Prioritize clusters with high search volume and semantic coherence for content development.",
                "adapt_content_rec": "Adapt content type to the predominant search intent in each cluster:",
                "informational_rec": "• Informational: Explanatory articles, tutorials, comprehensive guides",
                "commercial_rec": "• Commercial: Comparisons, reviews, best product lists",
                "transactional_rec": "• Transactional: Product pages, categories, special offers",
                "navigational_rec": "• Navigational: Brand pages, contact, help",
                "journey_rec": "Consider the customer journey phase when developing your content strategy for each cluster.",
                "coherent_clusters": "Most Coherent Clusters (Recommended for Development)",
                "coherence": "Coherence",
                "split_clusters": "Clusters Recommended for Subdivision",
                "report_footer": "Report generated by Advanced Semantic Keyword Clustering Tool",
                "semantic_coherence": "Semantic Coherence of Clusters",
                "coherence_by_cluster": "Semantic Coherence by Cluster",
                "coherence_description": "Higher values indicate more closely related keywords within clusters",
                "customer_journey_analysis": "Customer Journey Analysis",
                "journey_phase_distribution": "Distribution of Clusters Across Customer Journey Phases",
                "search_intent_to_journey": "Flow from Search Intent to Customer Journey Phase",
                "intent_score_distribution": "Search Intent Score Distribution",
                "early_phase": "Early (Research Phase)",
                "middle_phase": "Middle (Consideration Phase)",
                "late_phase": "Late (Purchase Phase)"
            },
            "es": {
                "report_title": "Informe de Resultados",
                "generated_on": "Generado el",
                "total_keywords": "Total de Keywords",
                "number_of_clusters": "Número de Clusters",
                "total_search_volume": "Volumen de Búsqueda Total",
                "cluster_distribution": "Distribución de Clusters",
                "cluster_size_title": "Tamaño de Cada Cluster",
                "cluster": "Cluster",
                "num_keywords": "Número de Keywords",
                "search_intent_analysis": "Análisis de Intención de Búsqueda",
                "intent_distribution": "Distribución de Intención de Búsqueda",
                "intent_by_cluster": "Intención de Búsqueda por Cluster",
                "cluster_details": "Detalles de los Clusters",
                "total_keywords_in_cluster": "Total de Keywords",
                "description": "Descripción",
                "representative_keywords": "Keywords Representativas",
                "primary_search_intent": "Intención de Búsqueda Principal",
                "customer_journey_phase": "Fase del Customer Journey",
                "top_keywords_by_volume": "Top Keywords por Volumen de Búsqueda",
                "keyword": "Keyword",
                "search_volume": "Vol. Búsqueda",
                "conclusions": "Conclusiones y Recomendaciones",
                "prioritize_rec": "Priorice los clusters con mayor volumen de búsqueda y coherencia semántica para el desarrollo de contenido.",
                "adapt_content_rec": "Adapte el tipo de contenido a la intención de búsqueda predominante en cada cluster:",
                "informational_rec": "• Informacional: Artículos explicativos, tutoriales, guías completas",
                "commercial_rec": "• Comercial: Comparativas, reviews, listas de los mejores productos",
                "transactional_rec": "• Transaccional: Páginas de producto, categorías, ofertas especiales",
                "navigational_rec": "• Navegacional: Páginas de marca, contacto, ayuda",
                "journey_rec": "Considere la fase del customer journey al desarrollar su estrategia de contenido para cada cluster.",
                "coherent_clusters": "Clusters Más Coherentes (Recomendados para Desarrollo)",
                "coherence": "Coherencia",
                "split_clusters": "Clusters Recomendados para Subdivisión",
                "report_footer": "Informe generado por Advanced Semantic Keyword Clustering Tool",
                "semantic_coherence": "Coherencia Semántica de los Clusters",
                "coherence_by_cluster": "Coherencia Semántica por Cluster",
                "coherence_description": "Valores más altos indican keywords más relacionadas entre sí dentro de los clusters",
                "customer_journey_analysis": "Análisis del Customer Journey",
                "journey_phase_distribution": "Distribución de Clusters por Fases del Customer Journey",
                "search_intent_to_journey": "Flujo de Intención de Búsqueda a Fase del Customer Journey",
                "intent_score_distribution": "Distribución de Puntuación de Intención de Búsqueda",
                "early_phase": "Inicial (Fase de Investigación)",
                "middle_phase": "Media (Fase de Consideración)",
                "late_phase": "Final (Fase de Compra)"
            }
        }
        
        # Default to English if language not available
        return translations.get(self.language, translations["en"])
    
    def setup_custom_styles(self):
        """Create custom styles for PDF elements."""
        # Create custom styles
        self.custom_styles['Title'] = ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        self.custom_styles['Subtitle'] = ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkblue
        )
        
        self.custom_styles['Normal'] = ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8
        )
        
        self.custom_styles['SmallText'] = ParagraphStyle(
            name='CustomSmallText',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=6
        )
        
        self.custom_styles['ClusterName'] = ParagraphStyle(
            name='CustomClusterName',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
    
def plotly_to_image(self, fig, width=7.5*inch, height=4*inch, filename=None, max_retries=3, retry_delay=1.0):
    """
    Convert Plotly figure to ReportLab Image with enhanced retry logic.
    
    Args:
        fig: A plotly figure object
        width: Desired width in ReportLab units
        height: Desired height in ReportLab units
        filename: Optional filename for the temporary image
        max_retries: Number of retries on failure
        retry_delay: Delay between retries in seconds
            
    Returns:
        ReportLab Image object or None on failure
    """
    if not self.temp_dir or not os.path.exists(self.temp_dir):
        logger.error("Temporary directory not available")
        return None
    
    if filename is None:
        filename = f"temp_plot_{np.random.randint(10000)}.png"
    
    # Save the figure as a PNG file
    img_path = os.path.join(self.temp_dir, filename)
    
    # Add retry logic for more resilience
    for retry in range(max_retries):
        try:
            # Adjust figure layout for better PDF rendering
            fig.update_layout(
                margin=dict(l=50, r=50, t=70, b=150),
                font=dict(size=10)  # Smaller font for PDF
            )
            
            # Write with increased timeout for complex charts
            pio.write_image(fig, img_path, format='png', width=900, height=500, scale=2, engine='kaleido')
            
            # Check if the file was created successfully
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                # Create ReportLab Image
                img = Image(img_path, width=width, height=height)
                return img
            else:
                raise ValueError("Image file was not created properly")
        
        except Exception as e:
            logger.warning(f"Error on retry {retry+1}/{max_retries}: {str(e)}")
            if retry < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying, with increasing delay
                retry_delay *= 1.5  # Exponential backoff
            else:
                # On final failure, create a placeholder
                try:
                    # Create a simple placeholder image
                    placeholder = PILImage.new('RGB', (900, 500), color=(240, 240, 240))
                    # Add text to the placeholder
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(placeholder)
                    try:
                        # Try to load a font, fall back to default
                        font = ImageFont.truetype("Arial", 20)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    draw.text((450, 250), "Chart generation failed", 
                             fill=(0, 0, 0), font=font, anchor="mm")
                    placeholder.save(img_path)
                    return Image(img_path, width=width, height=height)
                except Exception as placeholder_error:
                    logger.error(f"Failed to create placeholder image: {str(placeholder_error)}")
                    return None
    
    def generate_summary_page(self, doc_elements):
        """
        Generate the summary page of the report.
        
        Args:
            doc_elements: List to which elements will be added
            
        Returns:
            Updated doc_elements list
        """
        # Title
        doc_elements.append(Paragraph(f"{self.app_name} - {self.translations['report_title']}", self.custom_styles['Title']))
        doc_elements.append(Spacer(1, 0.1*inch))
        
        # Date and timestamp
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        doc_elements.append(Paragraph(f"{self.translations['generated_on']}: {current_time}", self.custom_styles['Normal']))
        doc_elements.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        num_keywords = len(self.df)
        num_clusters = len(self.df['cluster_id'].unique())
        
        summary_text = [
            f"{self.translations['total_keywords']}: {num_keywords}",
            f"{self.translations['number_of_clusters']}: {num_clusters}"
        ]
        
        # Add search volume if available
        if 'search_volume' in self.df.columns:
            try:
                # Ensure search_volume is numeric
                self.df['search_volume'] = pd.to_numeric(self.df['search_volume'], errors='coerce')
                total_volume = self.df['search_volume'].sum()
                summary_text.append(f"{self.translations['total_search_volume']}: {int(total_volume):,}")
            except Exception as e:
                logger.warning(f"Error calculating total search volume: {str(e)}")
        
        for text in summary_text:
            doc_elements.append(Paragraph(text, self.custom_styles['Normal']))
        
        doc_elements.append(Spacer(1, 0.2*inch))
        return doc_elements

    def process_clusters_in_batches(self, batch_size=50):
    """
    Process clusters in batches to handle large datasets more efficiently.
    
    Args:
        batch_size: Number of clusters to process in each batch
        
    Returns:
        List of processed cluster IDs
    """
    unique_clusters = self.df['cluster_id'].unique()
    total_clusters = len(unique_clusters)
    processed_clusters = []
    
    logger.info(f"Processing {total_clusters} clusters in batches of {batch_size}")
    
    for i in range(0, total_clusters, batch_size):
        batch_end = min(i + batch_size, total_clusters)
        batch_clusters = unique_clusters[i:batch_end]
        
        # Process this batch
        logger.info(f"Processing batch of clusters {i+1} to {batch_end}")
        # Filter dataframe to only include these clusters
        batch_df = self.df[self.df['cluster_id'].isin(batch_clusters)].copy()
        
        # Process batch operations here
        # For example, calculate metrics, prepare visualizations, etc.
        
        processed_clusters.extend(batch_clusters)
    
    return processed_clusters
    
    def generate_cluster_distribution_chart(self, doc_elements):
        """
        Generate cluster distribution chart for the PDF.
        
        Args:
            doc_elements: List to which elements will be added
            
        Returns:
            Updated doc_elements list
        """
        doc_elements.append(Paragraph(self.translations["cluster_distribution"], self.custom_styles['Subtitle']))
        doc_elements.append(Spacer(1, 0.1*inch))
        
        try:
            # Create a Plotly figure
            cluster_sizes = self.df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
            
            # Limit to top 20 clusters for PDF if too many
            if len(cluster_sizes) > 20:
                cluster_sizes = cluster_sizes.sort_values('count', ascending=False).head(20)
                doc_elements.append(Paragraph(
                    "Showing top 20 clusters by size. See the application for the complete visualization.", 
                    self.custom_styles['SmallText']
                ))
            
            # Shorten labels for readability
            cluster_sizes['label'] = cluster_sizes.apply(
                lambda x: f"{x['cluster_name'][:25]}{'...' if len(x['cluster_name']) > 25 else ''} (ID: {x['cluster_id']})", 
                axis=1
            )
            
            fig = go.Figure(data=[
                go.Bar(
                    x=cluster_sizes['label'],
                    y=cluster_sizes['count'],
                    marker_color='royalblue'
                )
            ])
            
            fig.update_layout(
                title=self.translations["cluster_size_title"],
                xaxis_title=self.translations["cluster"],
                yaxis_title=self.translations["num_keywords"],
                margin=dict(l=50, r=50, t=70, b=200),  # Adjust margins for better label display
                xaxis_tickangle=-45,  # Rotate x-axis labels
                height=600  # Set figure height
            )
            
            # Convert the figure to an image and add to document
            img = self.plotly_to_image(fig, filename="cluster_distribution.png")
            if img:
                doc_elements.append(img)
            doc_elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            error_msg = f"Error generating cluster distribution chart: {str(e)}"
            logger.error(error_msg)
            doc_elements.append(Paragraph(error_msg, self.custom_styles['Normal']))
        
        # Add semantic coherence chart if available
        if 'cluster_coherence' in self.df.columns:
            try:
                doc_elements.append(Paragraph(self.translations["semantic_coherence"], self.custom_styles['Subtitle']))
                doc_elements.append(Spacer(1, 0.1*inch))
                
                coherence_data = self.df.groupby(['cluster_id', 'cluster_name'])['cluster_coherence'].mean().reset_index()
                
                # Limit to top 20 clusters for PDF if too many
                if len(coherence_data) > 20:
                    coherence_data = coherence_data.sort_values('cluster_coherence', ascending=False).head(20)
                    doc_elements.append(Paragraph(
                        "Showing top 20 clusters by coherence. See the application for the complete visualization.", 
                        self.custom_styles['SmallText']
                    ))
                
                coherence_data['label'] = coherence_data.apply(
                    lambda x: f"{x['cluster_name'][:25]}{'...' if len(x['cluster_name']) > 25 else ''} (ID: {x['cluster_id']})", 
                    axis=1
                )
                
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=coherence_data['label'],
                        y=coherence_data['cluster_coherence'],
                        marker=dict(
                            color=coherence_data['cluster_coherence'],
                            colorscale='Viridis'
                        )
                    )
                ])
                
                fig2.update_layout(
                    title=self.translations["coherence_by_cluster"],
                    xaxis_title=self.translations["cluster"],
                    yaxis_title=self.translations["coherence"],
                    margin=dict(l=50, r=50, t=70, b=200),
                    xaxis_tickangle=-45,
                    height=600
                )
                
                img2 = self.plotly_to_image(fig2, filename="coherence_distribution.png")
                if img2:
                    doc_elements.append(img2)
                doc_elements.append(Paragraph(self.translations["coherence_description"], self.custom_styles['Normal']))
                doc_elements.append(Spacer(1, 0.2*inch))
            except Exception as e:
                error_msg = f"Error generating coherence chart: {str(e)}"
                logger.error(error_msg)
                doc_elements.append(Paragraph(error_msg, self.custom_styles['Normal']))
        
        return doc_elements
    
    def generate_search_intent_charts(self, doc_elements):
        """
        Generate search intent charts if intent data is available.
        
        Args:
            doc_elements: List to which elements will be added
            
        Returns:
            Updated doc_elements list
        """
        if not self.cluster_evaluation:
            return doc_elements
        
        try:
            doc_elements.append(Paragraph(self.translations["search_intent_analysis"], self.custom_styles['Subtitle']))
            doc_elements.append(Spacer(1, 0.1*inch))
            
            # Collect intent data from all clusters
            intent_data = []
            for c_id, data in self.cluster_evaluation.items():
                if 'intent_classification' in data:
                    # Find cluster name safely
                    cluster_rows = self.df[self.df['cluster_id'] == c_id]
                    cluster_name = f"Cluster {c_id}"
                    if not cluster_rows.empty and 'cluster_name' in cluster_rows.columns:
                        cluster_name = cluster_rows['cluster_name'].iloc[0]
                    
                    # Get primary intent and count
                    primary_intent = data['intent_classification'].get('primary_intent', 'Unknown')
                    count = len(self.df[self.df['cluster_id'] == c_id])
                    
                    # Get scores if available
                    scores = data['intent_classification'].get('scores', {})
                    
                    intent_data.append({
                        'cluster_id': c_id,
                        'cluster_name': cluster_name,
                        'primary_intent': primary_intent,
                        'count': count,
                        'informational_score': scores.get('Informational', 0),
                        'navigational_score': scores.get('Navigational', 0),
                        'transactional_score': scores.get('Transactional', 0),
                        'commercial_score': scores.get('Commercial', 0)
                    })
            
            if intent_data:
                # Create intent distribution pie chart
                intent_counts = {}
                for item in intent_data:
                    intent_counts[item['primary_intent']] = intent_counts.get(item['primary_intent'], 0) + item['count']
                
                labels = list(intent_counts.keys())
                values = list(intent_counts.values())
                
                intent_colors = {
                    'Informational': 'rgb(33, 150, 243)',
                    'Navigational': 'rgb(76, 175, 80)',
                    'Transactional': 'rgb(255, 152, 0)',
                    'Commercial': 'rgb(156, 39, 176)',
                    'Mixed Intent': 'rgb(158, 158, 158)',
                    'Unknown': 'rgb(158, 158, 158)'
                }
                
                colors = [intent_colors.get(label, 'rgb(158, 158, 158)') for label in labels]
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels, 
                        values=values, 
                        marker_colors=colors,
                        textinfo='label+percent'
                    )
                ])
                
                fig.update_layout(
                    title=self.translations["intent_distribution"],
                    margin=dict(l=50, r=50, t=70, b=50),
                    height=500
                )
                
                # Convert the figure to an image and add to document
                img = self.plotly_to_image(fig, filename="intent_distribution.png")
                if img:
                    doc_elements.append(img)
                doc_elements.append(Spacer(1, 0.2*inch))
                
                # Create bar chart showing intent by cluster
                df_intent = pd.DataFrame(intent_data)
                
                # Limit to top 10 clusters for readability
                if len(df_intent) > 10:
                    df_intent = df_intent.sort_values('count', ascending=False).head(10)
                
                fig2 = go.Figure()
                
                for intent in intent_counts.keys():
                    df_filtered = df_intent[df_intent['primary_intent'] == intent]
                    if not df_filtered.empty:
                        fig2.add_trace(go.Bar(
                            x=df_filtered['cluster_name'],
                            y=df_filtered['count'],
                            name=intent,
                            marker_color=intent_colors.get(intent, 'rgb(158, 158, 158)')
                        ))
                
                fig2.update_layout(
                    title=self.translations["intent_by_cluster"],
                    xaxis_title=self.translations["cluster"],
                    yaxis_title=self.translations["num_keywords"],
                    barmode='stack',
                    margin=dict(l=50, r=50, t=70, b=150),
                    xaxis_tickangle=-45,
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Convert the figure to an image and add to document
                img2 = self.plotly_to_image(fig2, filename="intent_by_cluster.png")
                if img2:
                    doc_elements.append(img2)
                
                # Add intent score distribution chart
                doc_elements.append(PageBreak())
                doc_elements.append(Paragraph(self.translations["intent_score_distribution"], self.custom_styles['Subtitle']))
                
                # Create data for the heatmap
                top_clusters = df_intent.sort_values('count', ascending=False).head(8)  # Limit to top 8 for readability
                
                # Create a stacked bar chart for intent scores
                fig3 = go.Figure()
                cluster_names = top_clusters['cluster_name'].tolist()
                
                # Add each intent type as a separate bar
                fig3.add_trace(go.Bar(
                    x=cluster_names,
                    y=top_clusters['informational_score'],
                    name='Informational',
                    marker_color=intent_colors['Informational']
                ))
                
                fig3.add_trace(go.Bar(
                    x=cluster_names,
                    y=top_clusters['commercial_score'],
                    name='Commercial',
                    marker_color=intent_colors['Commercial']
                ))
                
                fig3.add_trace(go.Bar(
                    x=cluster_names,
                    y=top_clusters['transactional_score'],
                    name='Transactional',
                    marker_color=intent_colors['Transactional']
                ))
                
                fig3.add_trace(go.Bar(
                    x=cluster_names,
                    y=top_clusters['navigational_score'],
                    name='Navigational',
                    marker_color=intent_colors['Navigational']
                ))
                
                fig3.update_layout(
                    title=self.translations["intent_score_distribution"],
                    xaxis_title=self.translations["cluster"],
                    yaxis_title='Intent Score (%)',
                    barmode='group',
                    margin=dict(l=50, r=50, t=70, b=150),
                    xaxis_tickangle=-45,
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                img3 = self.plotly_to_image(fig3, filename="intent_scores.png")
                if img3:
                    doc_elements.append(img3)
                
                # Generate customer journey analysis if available
                journey_phases = []
                for c_id, data in self.cluster_evaluation.items():
                    if 'intent_flow' in data:
                        journey_phase = data['intent_flow'].get('journey_phase', 'Unknown')
                        # Find cluster name safely
                        cluster_rows = self.df[self.df['cluster_id'] == c_id]
                        cluster_name = f"Cluster {c_id}"
                        if not cluster_rows.empty and 'cluster_name' in cluster_rows.columns:
                            cluster_name = cluster_rows['cluster_name'].iloc[0]
                        
                        count = len(self.df[self.df['cluster_id'] == c_id])
                        
                        journey_phases.append({
                            'cluster_id': c_id,
                            'cluster_name': cluster_name,
                            'journey_phase': journey_phase,
                            'count': count
                        })
                
                if journey_phases:
                    try:
                        doc_elements.append(PageBreak())
                        doc_elements.append(Paragraph(self.translations["customer_journey_analysis"], self.custom_styles['Subtitle']))
                        doc_elements.append(Spacer(1, 0.1*inch))
                        
                        # Count clusters in each journey phase
                        phase_counts = Counter([item['journey_phase'] for item in journey_phases])
                        
                        # Create journey phase visualization
                        phase_order = [
                            self.translations["early_phase"], 
                            "Research-to-Consideration Transition",
                            self.translations["middle_phase"], 
                            "Consideration-to-Purchase Transition",
                            self.translations["late_phase"],
                            "Mixed Journey Stages",
                            "Unknown"
                        ]
                        
                        # Filter to only phases that exist in our data
                        phase_order = [phase for phase in phase_order if phase in phase_counts]
                        
                        phase_colors = {
                            self.translations["early_phase"]: "#43a047",
                            "Research-to-Consideration Transition": "#26a69a",
                            self.translations["middle_phase"]: "#1e88e5",
                            "Consideration-to-Purchase Transition": "#7b1fa2",
                            self.translations["late_phase"]: "#ff9800",
                            "Mixed Journey Stages": "#757575",
                            "Unknown": "#9e9e9e"
                        }
                        
                        phases = list(phase_counts.keys())
                        counts = list(phase_counts.values())
                        
                        fig4 = go.Figure(data=[
                            go.Bar(
                                x=phases,
                                y=counts,
                                marker=dict(
                                    color=[phase_colors.get(phase, "#9e9e9e") for phase in phases]
                                )
                            )
                        ])
                        
                        fig4.update_layout(
                            title=self.translations["journey_phase_distribution"],
                            xaxis_title="Journey Phase",
                            yaxis_title=self.translations["num_keywords"],
                            margin=dict(l=50, r=50, t=70, b=150),
                            height=600
                        )
                        
                        img4 = self.plotly_to_image(fig4, filename="journey_phases.png")
                        if img4:
                            doc_elements.append(img4)
                        doc_elements.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        error_msg = f"Error generating journey analysis: {str(e)}"
                        logger.error(error_msg)
                        doc_elements.append(Paragraph(error_msg, self.custom_styles['Normal']))
        except Exception as e:
            error_msg = f"Error generating search intent charts: {str(e)}"
            logger.error(error_msg)
            doc_elements.append(Paragraph(error_msg, self.custom_styles['Normal']))
        
        return doc_elements
                       
                       fig4.update_layout(
                           title=self.translations["journey_phase_distribution"],
                           xaxis_title="Journey Phase",
                           yaxis_title=self.translations["num_keywords"],
                           margin=dict(l=50, r=50, t=70, b=150),
                           height=600
                       )
                       
                       img4 = self.plotly_to_image(fig4, filename="journey_phases.png")
                       if img4:
                           doc_elements.append(img4)
                       doc_elements.append(Spacer(1, 0.2*inch))
                   except Exception as e:
                       error_msg = f"Error generating journey analysis: {str(e)}"
                       logger.error(error_msg)
                       doc_elements.append(Paragraph(error_msg, self.custom_styles['Normal']))
       except Exception as e:
           error_msg = f"Error generating search intent charts: {str(e)}"
           logger.error(error_msg)
           doc_elements.append(Paragraph(error_msg, self.custom_styles['Normal']))
       
       return doc_elements
   
   def generate_clusters_detail(self, doc_elements):
       """
       Generate detailed information for each cluster.
       
       Args:
           doc_elements: List to which elements will be added
           
       Returns:
           Updated doc_elements list
       """
       doc_elements.append(PageBreak())
       doc_elements.append(Paragraph(self.translations["cluster_details"], self.custom_styles['Subtitle']))
       doc_elements.append(Spacer(1, 0.1*inch))
       
       try:
           # Get top 10 clusters by keyword count for detailed view
           cluster_sizes = self.df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
           top_clusters = cluster_sizes.sort_values('count', ascending=False).head(10)
           
           for _, row in top_clusters.iterrows():
               c_id = row['cluster_id']
               c_name = row['cluster_name']
               c_count = row['count']
               
               doc_elements.append(Paragraph(f"{self.translations['cluster']}: {c_name} (ID: {c_id})", self.custom_styles['ClusterName']))
               doc_elements.append(Paragraph(f"{self.translations['total_keywords_in_cluster']}: {c_count}", self.custom_styles['Normal']))
               
               # Get cluster description if available
               cluster_desc_rows = self.df[(self.df['cluster_id'] == c_id) & ('cluster_description' in self.df.columns)]
               c_desc = cluster_desc_rows['cluster_description'].iloc[0] if not cluster_desc_rows.empty else ""
               if c_desc:
                   doc_elements.append(Paragraph(f"{self.translations['description']}: {c_desc}", self.custom_styles['Normal']))
               
               # Get representative keywords
               if 'representative' in self.df.columns:
                   rep_keywords = self.df[(self.df['cluster_id'] == c_id) & (self.df['representative'] == True)]['keyword'].tolist()
                   if rep_keywords:
                       doc_elements.append(Paragraph(f"{self.translations['representative_keywords']}: {', '.join(rep_keywords[:10])}", self.custom_styles['Normal']))
               
               # Search intent information if available
               if c_id in self.cluster_evaluation:
                   intent_data = self.cluster_evaluation[c_id].get('intent_classification', {})
                   primary_intent = intent_data.get('primary_intent', 'Unknown')
                   
                   doc_elements.append(Paragraph(f"{self.translations['primary_search_intent']}: {primary_intent}", self.custom_styles['Normal']))
                   
                   # Customer journey if available
                   if 'intent_flow' in self.cluster_evaluation[c_id]:
                       journey_phase = self.cluster_evaluation[c_id]['intent_flow'].get('journey_phase', 'Unknown')
                       doc_elements.append(Paragraph(f"{self.translations['customer_journey_phase']}: {journey_phase}", self.custom_styles['Normal']))
               
               # Top keywords by search volume if available
               if 'search_volume' in self.df.columns:
                   try:
                       # Make sure search_volume is numeric
                       cluster_df = self.df[self.df['cluster_id'] == c_id].copy()
                       cluster_df['search_volume'] = pd.to_numeric(cluster_df['search_volume'], errors='coerce')
                       top_kws = cluster_df.sort_values('search_volume', ascending=False).head(10)
                       
                       if not top_kws.empty:
                           doc_elements.append(Paragraph(f"{self.translations['top_keywords_by_volume']}:", self.custom_styles['Normal']))
                           
                           data = [[self.translations['keyword'], self.translations['search_volume']]]
                           for _, kw_row in top_kws.iterrows():
                               data.append([kw_row['keyword'], f"{int(kw_row['search_volume']):,}"])
                           
                           # Create a table for the top keywords
                           table = Table(data, colWidths=[4*inch, 1.5*inch])
                           table.setStyle(TableStyle([
                               ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                               ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                           ]))
                           
                           doc_elements.append(table)
                   except Exception as e:
                       error_msg = f"Error displaying top keywords: {str(e)}"
                       logger.warning(error_msg)
                       doc_elements.append(Paragraph(error_msg, self.custom_styles['SmallText']))
               
               doc_elements.append(Spacer(1, 0.3*inch))
       
       except Exception as e:
           error_msg = f"Error generating cluster details: {str(e)}"
           logger.error(error_msg)
           doc_elements.append(Paragraph(error_msg, self.custom_styles['Normal']))
       
       return doc_elements
   
   def generate_conclusion(self, doc_elements):
       """
       Generate conclusion section with recommendations.
       
       Args:
           doc_elements: List to which elements will be added
           
       Returns:
           Updated doc_elements list
       """
       doc_elements.append(PageBreak())
       doc_elements.append(Paragraph(self.translations["conclusions"], self.custom_styles['Subtitle']))
       doc_elements.append(Spacer(1, 0.1*inch))
       
       # General recommendations
       general_recommendations = [
           self.translations["prioritize_rec"],
           self.translations["adapt_content_rec"],
           self.translations["informational_rec"],
           self.translations["commercial_rec"],
           self.translations["transactional_rec"],
           self.translations["navigational_rec"],
           self.translations["journey_rec"]
       ]
       
       for rec in general_recommendations:
           doc_elements.append(Paragraph(rec, self.custom_styles['Normal']))
       
       doc_elements.append(Spacer(1, 0.2*inch))
       
       # Specific recommendations based on data
       if self.cluster_evaluation:
           try:
               # Find most coherent clusters
               coherent_clusters = []
               for c_id, data in self.cluster_evaluation.items():
                   coherence = data.get('coherence_score', 0)
                   
                   # Find cluster name safely
                   cluster_rows = self.df[self.df['cluster_id'] == c_id]
                   c_name = f"Cluster {c_id}"
                   if not cluster_rows.empty and 'cluster_name' in cluster_rows.columns:
                       c_name = cluster_rows['cluster_name'].iloc[0]
                   
                   if coherence >= 7:  # High coherence threshold
                       coherent_clusters.append((c_id, c_name, coherence))
               
               if coherent_clusters:
                   doc_elements.append(Paragraph(f"{self.translations['coherent_clusters']}:", self.custom_styles['Normal']))
                   
                   coherent_clusters.sort(key=lambda x: x[2], reverse=True)
                   for c_id, c_name, coherence in coherent_clusters[:5]:
                       doc_elements.append(Paragraph(f"• {c_name} (ID: {c_id}) - {self.translations['coherence']}: {coherence:.1f}/10", self.custom_styles['Normal']))
               
               # Find clusters that need splitting
               split_clusters = []
               for c_id, data in self.cluster_evaluation.items():
                   split_suggestion = data.get('split_suggestion', '')
                   
                   # Find cluster name safely
                   cluster_rows = self.df[self.df['cluster_id'] == c_id]
                   c_name = f"Cluster {c_id}"
                   if not cluster_rows.empty and 'cluster_name' in cluster_rows.columns:
                       c_name = cluster_rows['cluster_name'].iloc[0]
                   
                   if isinstance(split_suggestion, str) and 'yes' in split_suggestion.lower():
                       split_clusters.append((c_id, c_name))
               
               if split_clusters:
                   doc_elements.append(Spacer(1, 0.1*inch))
                   doc_elements.append(Paragraph(f"{self.translations['split_clusters']}:", self.custom_styles['Normal']))
                   
                   for c_id, c_name in split_clusters[:5]:
                       doc_elements.append(Paragraph(f"• {c_name} (ID: {c_id})", self.custom_styles['Normal']))
           except Exception as e:
               error_msg = f"Error generating recommendations: {str(e)}"
               logger.error(error_msg)
               doc_elements.append(Paragraph(error_msg, self.custom_styles['SmallText']))
       
       doc_elements.append(Spacer(1, 0.3*inch))
       
       # Footer with attribution
       doc_elements.append(Paragraph(self.translations['report_footer'], self.custom_styles['SmallText']))
       
       return doc_elements
   
def generate_pdf(self, output_file='clustering_report.pdf'):
    """
    Generate the complete PDF report with improved resource management.
    
    Args:
        output_file: Optional filename for the PDF
        
    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = BytesIO()
    
    # Create the PDF document
    try:
        # Process any large data in batches if needed
        if len(self.df) > 10000:  # Threshold for batch processing
            logger.info("Large dataset detected, processing in batches")
            self.process_clusters_in_batches()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Container for the elements to add to the PDF
        doc_elements = []
        
        # Generate each section
        doc_elements = self.generate_summary_page(doc_elements)
        doc_elements = self.generate_cluster_distribution_chart(doc_elements)
        doc_elements = self.generate_search_intent_charts(doc_elements)
        doc_elements = self.generate_clusters_detail(doc_elements)
        doc_elements = self.generate_conclusion(doc_elements)
        
        # Build the PDF
        doc.build(doc_elements)
        
        # Return the buffer
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        error_msg = f"Error generating PDF: {str(e)}"
        logger.error(error_msg)
        
        # If PDF generation fails completely, return a simple error PDF
        try:
            simple_doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch
            )
            
            simple_elements = []
            simple_elements.append(Paragraph("Error Generating PDF Report", self.styles['Title']))
            simple_elements.append(Spacer(1, 0.2*inch))
            simple_elements.append(Paragraph(f"An error occurred: {str(e)}", self.styles['Normal']))
            simple_elements.append(Spacer(1, 0.1*inch))
            simple_elements.append(Paragraph("Please try again with a smaller dataset or contact support.", self.styles['Normal']))
            
            simple_doc.build(simple_elements)
            
            buffer.seek(0)
            return buffer
        except Exception as fallback_error:
            logger.critical(f"Critical error creating fallback PDF: {str(fallback_error)}")
            # Last resort: return an empty buffer
            buffer.seek(0)
            return buffer
    finally:
        # Ensure we clean up resources
        self.cleanup()


def create_download_link(buffer, filename="report.pdf", text="Download PDF Report"):
   """
   Create a download link for Streamlit with improved error handling.
   
   Args:
       buffer: BytesIO buffer containing the PDF
       filename: Filename for the downloaded file
       text: Link text to display
       
   Returns:
       HTML string with download link
   """
   try:
       pdf_data = buffer.getvalue()
       b64_pdf = base64.b64encode(pdf_data).decode()
       href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">{text}</a>'
       return href
   except Exception as e:
       logger.error(f"Error creating download link: {str(e)}")
       st.error(f"Error creating download link: {str(e)}")
       return "Error generating download link. Please try again."


def sanitize_filename(filename):
    """
    Sanitize a filename to prevent path traversal or invalid characters with enhanced security.
    
    Args:
        filename: Input filename
        
    Returns:
        Sanitized filename
    """
    import re
    import os.path
    
    # Replace invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    filename = re.sub(invalid_chars, '_', filename)
    
    # Remove any leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure the filename doesn't escape its directory
    filename = os.path.basename(filename)
    
    # Limit length
    if len(filename) > 128:
        name, ext = os.path.splitext(filename)
        filename = name[:124] + ext
    
    # Ensure we have a valid filename
    if not filename or filename == '.':
        filename = "report.pdf"
    
    return filename


def add_pdf_export_button(df, cluster_evaluation=None, language="en"):
    """
    Add PDF export button to Streamlit app with enhanced error handling and resource management.
    
    Args:
        df: Dataframe containing clustered keywords
        cluster_evaluation: Dictionary with cluster evaluation data
        language: Language code ('en', 'es')
    """
    languages = {
        "en": "English",
        "es": "Spanish"
    }
    
    # Language selection
    selected_language = st.selectbox(
        "Select language for the report",
        options=list(languages.keys()),
        format_func=lambda x: languages.get(x, x),
        index=0  # Default to English
    )
    
    button_text = "🔍 Generate PDF Report" if selected_language == "en" else "🔍 Generar Informe PDF"
    
    # Add a download button as a placeholder in case the worker is busy
    if st.button(button_text, use_container_width=True):
        try:
            # Use a spinner to show progress
            with st.spinner("Generating PDF report..." if selected_language == "en" else "Generando el informe PDF..."):
                # Check dependencies before trying to generate
                missing_deps = []
                try:
                    import reportlab
                except ImportError:
                    missing_deps.append("reportlab")
                
                try:
                    import kaleido
                except ImportError:
                    missing_deps.append("kaleido")
                
                try:
                    from PIL import Image as PILImage
                except ImportError:
                    missing_deps.append("pillow")
                
                # If missing dependencies, show a helpful message
                if missing_deps:
                    missing_str = ", ".join(missing_deps)
                    st.error(f"Missing required dependencies: {missing_str}")
                    st.info(f"Please install them with: pip install {missing_str}")
                    return
                
                # Data validation
                if df is None or len(df) == 0:
                    st.error("No data available to generate report")
                    return
                
                if 'cluster_id' not in df.columns:
                    st.error("Required column 'cluster_id' not found in data")
                    return
                
                # Create the report
                pdf_report = PDFReport(df, cluster_evaluation, language=selected_language)
                
                try:
                    pdf_buffer = pdf_report.generate_pdf()
                    
                    # Create download link
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    filename = sanitize_filename(f"clustering_report_{timestamp}.pdf")
                    
                    # Display success message and download link
                    success_message = "✅ PDF report generated successfully" if selected_language == "en" else "✅ Informe PDF generado correctamente"
                    st.success(success_message)
                    
                    download_text = "Download PDF Report" if selected_language == "en" else "Descargar Informe PDF"
                    st.markdown(create_download_link(pdf_buffer, filename, download_text), unsafe_allow_html=True)
                    
                    # Display preview if possible
                    preview_title = "### Report Preview" if selected_language == "en" else "### Vista previa del informe"
                    st.markdown(preview_title)
                    
                    warning_text = "The preview may not be available on all platforms. If you can't see it, download the PDF file directly." if selected_language == "en" else "La vista previa puede no estar disponible en todas las plataformas. Si no puedes verla, descarga el archivo PDF directamente."
                    st.warning(warning_text)
                    
                    try:
                        # Try to display the PDF inline
                        base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    except Exception as preview_error:
                        info_text = f"Preview not available: {str(preview_error)}. Please download the PDF to view it." if selected_language == "en" else f"Vista previa no disponible: {str(preview_error)}. Por favor, descarga el PDF para verlo."
                        st.info(info_text)
                finally:
                    # Explicitly clean up the report object's resources
                    if 'pdf_report' in locals():
                        pdf_report.cleanup()
                
        except Exception as e:
            error_text = f"Error generating PDF report: {str(e)}" if selected_language == "en" else f"Error al generar el informe PDF: {str(e)}"
            st.error(error_text)
            
            # Provide more details in case of common errors
            if "kaleido" in str(e).lower():
                st.info("It appears you're missing the kaleido library needed for chart export. Try installing it with: pip install kaleido")
            elif "reportlab" in str(e).lower():
                st.info("It appears you're missing the reportlab library needed for PDF generation. Try installing it with: pip install reportlab")
