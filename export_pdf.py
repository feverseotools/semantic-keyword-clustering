import os
import tempfile
import base64
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image as PILImage

class PDFReport:
    """
    Class for generating PDF reports from clustering results
    """
    def __init__(self, df, cluster_evaluation=None, app_name="Advanced Semantic Keyword Clustering"):
        self.df = df
        self.cluster_evaluation = cluster_evaluation if cluster_evaluation else {}
        self.app_name = app_name
        self.temp_dir = tempfile.mkdtemp()
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Create custom styles for PDF elements"""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='Normal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='SmallText',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='ClusterName',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))
    
    def plotly_to_image(self, fig, width=7.5*inch, height=4*inch, filename=None):
        """Convert Plotly figure to ReportLab Image"""
        if filename is None:
            filename = f"temp_plot_{np.random.randint(10000)}.png"
        
        # Save the figure as a PNG file
        img_path = os.path.join(self.temp_dir, filename)
        pio.write_image(fig, img_path, format='png', width=900, height=500, scale=2)
        
        # Create ReportLab Image
        img = Image(img_path, width=width, height=height)
        return img
    
    def generate_summary_page(self, doc_elements):
        """Generate the summary page"""
        # Title
        doc_elements.append(Paragraph(f"{self.app_name} - Informe de Resultados", self.styles['Title']))
        doc_elements.append(Spacer(1, 0.1*inch))
        
        # Date and timestamp
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        doc_elements.append(Paragraph(f"Generado el: {current_time}", self.styles['Normal']))
        doc_elements.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        num_keywords = len(self.df)
        num_clusters = len(self.df['cluster_id'].unique())
        
        summary_text = [
            f"Total de Keywords: {num_keywords}",
            f"Número de Clusters: {num_clusters}"
        ]
        
        # Add search volume if available
        if 'search_volume' in self.df.columns:
            total_volume = self.df['search_volume'].sum()
            summary_text.append(f"Volumen de Búsqueda Total: {total_volume:,}")
        
        for text in summary_text:
            doc_elements.append(Paragraph(text, self.styles['Normal']))
        
        doc_elements.append(Spacer(1, 0.2*inch))
        return doc_elements
    
    def generate_cluster_distribution_chart(self, doc_elements):
        """Generate cluster distribution chart for the PDF"""
        doc_elements.append(Paragraph("Distribución de Clusters", self.styles['Subtitle']))
        doc_elements.append(Spacer(1, 0.1*inch))
        
        # Create a Plotly figure
        cluster_sizes = self.df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
        cluster_sizes['label'] = cluster_sizes.apply(lambda x: f"{x['cluster_name']} (ID: {x['cluster_id']})", axis=1)
        
        fig = go.Figure(data=[
            go.Bar(
                x=cluster_sizes['label'],
                y=cluster_sizes['count'],
                marker_color='royalblue'
            )
        ])
        
        fig.update_layout(
            title="Tamaño de Cada Cluster",
            xaxis_title="Cluster",
            yaxis_title="Número de Keywords",
            margin=dict(l=50, r=50, t=70, b=200),  # Adjust margins for better label display
            xaxis_tickangle=-45,  # Rotate x-axis labels
            height=600  # Set figure height
        )
        
        # Convert the figure to an image and add to document
        img = self.plotly_to_image(fig, filename="cluster_distribution.png")
        doc_elements.append(img)
        doc_elements.append(Spacer(1, 0.2*inch))
        
        return doc_elements
    
    def generate_search_intent_charts(self, doc_elements):
        """Generate search intent charts if intent data is available"""
        if not self.cluster_evaluation:
            return doc_elements
        
        doc_elements.append(Paragraph("Análisis de Intención de Búsqueda", self.styles['Subtitle']))
        doc_elements.append(Spacer(1, 0.1*inch))
        
        # Collect intent data from all clusters
        intent_data = []
        for c_id, data in self.cluster_evaluation.items():
            if 'intent_classification' in data:
                cluster_name = self.df[self.df['cluster_id'] == c_id]['cluster_name'].iloc[0] if not self.df[self.df['cluster_id'] == c_id].empty else f"Cluster {c_id}"
                primary_intent = data['intent_classification'].get('primary_intent', 'Unknown')
                count = len(self.df[self.df['cluster_id'] == c_id])
                
                intent_data.append({
                    'cluster_id': c_id,
                    'cluster_name': cluster_name,
                    'primary_intent': primary_intent,
                    'count': count
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
                title="Distribución de Intención de Búsqueda",
                margin=dict(l=50, r=50, t=70, b=50),
                height=500
            )
            
            # Convert the figure to an image and add to document
            img = self.plotly_to_image(fig, filename="intent_distribution.png")
            doc_elements.append(img)
            doc_elements.append(Spacer(1, 0.2*inch))
            
            # Create bar chart showing intent by cluster
            df_intent = pd.DataFrame(intent_data)
            
            # Limit to top 10 clusters for readability if needed
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
                title="Intención de Búsqueda por Cluster",
                xaxis_title="Cluster",
                yaxis_title="Número de Keywords",
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
            doc_elements.append(img2)
        
        return doc_elements
    
    def generate_clusters_detail(self, doc_elements):
        """Generate detailed information for each cluster"""
        doc_elements.append(PageBreak())
        doc_elements.append(Paragraph("Detalles de los Clusters", self.styles['Subtitle']))
        doc_elements.append(Spacer(1, 0.1*inch))
        
        # Get top 10 clusters by keyword count for detailed view
        cluster_sizes = self.df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
        top_clusters = cluster_sizes.sort_values('count', ascending=False).head(10)
        
        for _, row in top_clusters.iterrows():
            c_id = row['cluster_id']
            c_name = row['cluster_name']
            c_count = row['count']
            
            doc_elements.append(Paragraph(f"Cluster: {c_name} (ID: {c_id})", self.styles['ClusterName']))
            doc_elements.append(Paragraph(f"Total de Keywords: {c_count}", self.styles['Normal']))
            
            # Get cluster description if available
            c_desc = self.df[self.df['cluster_id'] == c_id]['cluster_description'].iloc[0] if not self.df[self.df['cluster_id'] == c_id].empty else ""
            if c_desc:
                doc_elements.append(Paragraph(f"Descripción: {c_desc}", self.styles['Normal']))
            
            # Get representative keywords
            rep_keywords = self.df[(self.df['cluster_id'] == c_id) & (self.df['representative'] == True)]['keyword'].tolist()
            if rep_keywords:
                doc_elements.append(Paragraph(f"Keywords Representativas: {', '.join(rep_keywords[:10])}", self.styles['Normal']))
            
            # Search intent information if available
            if c_id in self.cluster_evaluation:
                intent_data = self.cluster_evaluation[c_id].get('intent_classification', {})
                primary_intent = intent_data.get('primary_intent', 'Unknown')
                
                doc_elements.append(Paragraph(f"Intención de Búsqueda Principal: {primary_intent}", self.styles['Normal']))
                
                # Customer journey if available
                if 'intent_flow' in self.cluster_evaluation[c_id]:
                    journey_phase = self.cluster_evaluation[c_id]['intent_flow'].get('journey_phase', 'Unknown')
                    doc_elements.append(Paragraph(f"Fase del Customer Journey: {journey_phase}", self.styles['Normal']))
            
            # Top keywords by search volume if available
            if 'search_volume' in self.df.columns:
                top_kws = self.df[self.df['cluster_id'] == c_id].sort_values('search_volume', ascending=False).head(10)
                
                if not top_kws.empty:
                    doc_elements.append(Paragraph("Top Keywords por Volumen de Búsqueda:", self.styles['Normal']))
                    
                    data = [['Keyword', 'Vol. Búsqueda']]
                    for _, kw_row in top_kws.iterrows():
                        data.append([kw_row['keyword'], f"{kw_row['search_volume']:,}"])
                    
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
            
            doc_elements.append(Spacer(1, 0.3*inch))
        
        return doc_elements
    
    def generate_conclusion(self, doc_elements):
        """Generate conclusion section with recommendations"""
        doc_elements.append(PageBreak())
        doc_elements.append(Paragraph("Conclusiones y Recomendaciones", self.styles['Subtitle']))
        doc_elements.append(Spacer(1, 0.1*inch))
        
        # General recommendations
        general_recommendations = [
            "Priorice los clusters con mayor volumen de búsqueda y coherencia semántica para el desarrollo de contenido.",
            "Adapte el tipo de contenido a la intención de búsqueda predominante en cada cluster:",
            "• Informacional: Artículos explicativos, tutoriales, guías completas",
            "• Comercial: Comparativas, reviews, listas de los mejores productos",
            "• Transaccional: Páginas de producto, categorías, ofertas especiales",
            "• Navegacional: Páginas de marca, contacto, ayuda",
            "Considere la fase del customer journey al desarrollar su estrategia de contenido para cada cluster."
        ]
        
        for rec in general_recommendations:
            doc_elements.append(Paragraph(rec, self.styles['Normal']))
        
        doc_elements.append(Spacer(1, 0.2*inch))
        
        # Specific recommendations based on data
        if self.cluster_evaluation:
            # Find most coherent clusters
            coherent_clusters = []
            for c_id, data in self.cluster_evaluation.items():
                coherence = data.get('coherence_score', 0)
                c_name = self.df[self.df['cluster_id'] == c_id]['cluster_name'].iloc[0] if not self.df[self.df['cluster_id'] == c_id].empty else f"Cluster {c_id}"
                
                if coherence >= 7:  # High coherence threshold
                    coherent_clusters.append((c_id, c_name, coherence))
            
            if coherent_clusters:
                doc_elements.append(Paragraph("Clusters Más Coherentes (Recomendados para Desarrollo):", self.styles['Normal']))
                
                coherent_clusters.sort(key=lambda x: x[2], reverse=True)
                for c_id, c_name, coherence in coherent_clusters[:5]:
                    doc_elements.append(Paragraph(f"• {c_name} (ID: {c_id}) - Coherencia: {coherence:.1f}/10", self.styles['Normal']))
            
            # Find clusters that need splitting
            split_clusters = []
            for c_id, data in self.cluster_evaluation.items():
                split_suggestion = data.get('split_suggestion', '')
                c_name = self.df[self.df['cluster_id'] == c_id]['cluster_name'].iloc[0] if not self.df[self.df['cluster_id'] == c_id].empty else f"Cluster {c_id}"
                
                if 'yes' in split_suggestion.lower():
                    split_clusters.append((c_id, c_name))
            
            if split_clusters:
                doc_elements.append(Spacer(1, 0.1*inch))
                doc_elements.append(Paragraph("Clusters Recomendados para Subdivisión:", self.styles['Normal']))
                
                for c_id, c_name in split_clusters[:5]:
                    doc_elements.append(Paragraph(f"• {c_name} (ID: {c_id})", self.styles['Normal']))
        
        doc_elements.append(Spacer(1, 0.3*inch))
        
        # Footer with attribution
        doc_elements.append(Paragraph("Informe generado por Advanced Semantic Keyword Clustering Tool", self.styles['SmallText']))
        
        return doc_elements
    
    def generate_pdf(self, output_file='clustering_report.pdf'):
        """Generate the complete PDF report"""
        buffer = BytesIO()
        
        # Create the PDF document
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

def create_download_link(buffer, filename="report.pdf", text="Descargar Informe en PDF"):
    """Create a download link for Streamlit"""
    pdf_data = buffer.getvalue()
    b64_pdf = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">{text}</a>'
    return href

def add_pdf_export_button(df, cluster_evaluation=None):
    """Add PDF export button to Streamlit app"""
    if st.button("📊 Generar Informe PDF", use_container_width=True):
        with st.spinner("Generando el informe PDF..."):
            try:
                # Create the report
                pdf_report = PDFReport(df, cluster_evaluation)
                pdf_buffer = pdf_report.generate_pdf()
                
                # Create download link
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"clustering_report_{timestamp}.pdf"
                
                # Display success message and download link
                st.success("✅ Informe PDF generado correctamente")
                st.markdown(create_download_link(pdf_buffer, filename), unsafe_allow_html=True)
                
                # Display preview if possible
                st.markdown("### Vista previa del informe")
                st.warning("La vista previa puede no estar disponible en todas las plataformas. Si no puedes verla, descarga el archivo PDF directamente.")
                
                try:
                    # Try to display the PDF
                    base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except:
                    st.info("Vista previa no disponible. Por favor, descarga el PDF para verlo.")
                
            except Exception as e:
                st.error(f"Error al generar el informe PDF: {str(e)}")
