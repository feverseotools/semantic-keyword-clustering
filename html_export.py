"""
HTML Report Generator for Semantic Keyword Clustering.

This module handles HTML visualization from cluster analysis data.
"""

__all__ = ['HTMLReport', 'add_html_export_button', 'generate_html_report']

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import StringIO
import tempfile
import os
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('html_export')

class HTMLReport:
    """
    Class for generating interactive HTML reports from keyword clustering results.
    
    Attributes:
        df (pandas.DataFrame): The clustered keywords dataframe
        cluster_evaluation (dict): Dictionary containing cluster evaluation data
        app_name (str): Name of the application to show in report
        language (str): Language code for report localization ('en', 'es')
    """
    
    def __init__(self, df, cluster_evaluation=None, app_name="Advanced Semantic Keyword Clustering", language="en"):
        """
        Initialize the HTML report generator.
        
        Args:
            df (pandas.DataFrame): The clustered keywords dataframe
            cluster_evaluation (dict, optional): Dictionary containing cluster evaluation data
            app_name (str, optional): Name of the application
            language (str, optional): Language code ('en', 'es')
        """
        self.df = df.copy()
        self.cluster_evaluation = cluster_evaluation if cluster_evaluation else {}
        self.app_name = app_name
        self.language = language
        
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
    
    def _get_translations(self):
        """Get translations dictionary based on language."""
        translations = {
            "en": {
                "report_title": "Semantic Keyword Clustering Results",
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
                "semantic_coherence": "Semantic Coherence of Clusters",
                "customer_journey_analysis": "Customer Journey Analysis"
            },
            "es": {
                "report_title": "Resultados de Clustering Semántico de Keywords",
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
                "semantic_coherence": "Coherencia Semántica de los Clusters",
                "customer_journey_analysis": "Análisis del Customer Journey"
            }
        }
        
        # Default to English if language not available
        return translations.get(self.language, translations["en"])
    
    def generate_html(self):
        """
        Generate the HTML report.
        
        Returns:
            str: HTML content as a string
        """
        # Start building HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="{self.language}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.app_name} - {self.translations['report_title']}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
                .chart-container {{ margin-bottom: 30px; }}
                .cluster-details {{ margin-top: 20px; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
                .intent-info {{ background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px; margin-bottom: 10px; }}
                .intent-nav {{ background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 10px; margin-bottom: 10px; }}
                .intent-trans {{ background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; margin-bottom: 10px; }}
                .intent-comm {{ background-color: #f3e5f5; border-left: 5px solid #9c27b0; padding: 10px; margin-bottom: 10px; }}
                .intent-mixed {{ background-color: #f5f5f5; border-left: 5px solid #9e9e9e; padding: 10px; margin-bottom: 10px; }}
                .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 0.8em; color: #6c757d; }}
                .table-container {{ overflow-x: auto; }}
                .tab-content {{ padding: 20px; border: 1px solid #dee2e6; border-top: 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{self.app_name}</h1>
                    <h3>{self.translations['report_title']}</h3>
                    <p>{self.translations['generated_on']}: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
                    
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">{self.translations['total_keywords']}</h5>
                                    <p class="card-text display-4">{len(self.df):,}</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">{self.translations['number_of_clusters']}</h5>
                                    <p class="card-text display-4">{len(self.df['cluster_id'].unique()):,}</p>
                                </div>
                            </div>
                        </div>
        """
        
        # Add search volume if available
        if 'search_volume' in self.df.columns:
            try:
                self.df['search_volume'] = pd.to_numeric(self.df['search_volume'], errors='coerce')
                total_volume = self.df['search_volume'].sum()
                html += f"""
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">{self.translations['total_search_volume']}</h5>
                                    <p class="card-text display-4">{int(total_volume):,}</p>
                                </div>
                            </div>
                        </div>
                """
            except Exception as e:
                logger.warning(f"Error calculating total search volume: {str(e)}")
        
        html += """
                    </div>
                </div>
                
                <!-- Navigation Tabs -->
                <ul class="nav nav-tabs" id="reportTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab" aria-controls="overview" aria-selected="true">Overview</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="intent-tab" data-bs-toggle="tab" data-bs-target="#intent" type="button" role="tab" aria-controls="intent" aria-selected="false">Search Intent</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="journey-tab" data-bs-toggle="tab" data-bs-target="#journey" type="button" role="tab" aria-controls="journey" aria-selected="false">Customer Journey</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="clusters-tab" data-bs-toggle="tab" data-bs-target="#clusters" type="button" role="tab" aria-controls="clusters" aria-selected="false">Cluster Details</button>
                    </li>
                </ul>
                
                <div class="tab-content" id="reportTabsContent">
        """
        
        # Overview Tab Content
        html += self._generate_overview_tab()
        
        # Search Intent Tab Content
        html += self._generate_intent_tab()
        
        # Customer Journey Tab Content
        html += self._generate_journey_tab()
        
        # Cluster Details Tab Content
        html += self._generate_clusters_tab()
        
        # Close tab content and add footer
        html += """
                </div>
                
                <div class="footer text-center">
                    <p>Generated by Advanced Semantic Keyword Clustering Tool</p>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // Load all Plotly graphs when the DOM is ready
                document.addEventListener('DOMContentLoaded', function() {
                    // Activate the first tab by default
                    var firstTabEl = document.querySelector('#reportTabs button[data-bs-toggle="tab"]')
                    new bootstrap.Tab(firstTabEl).show()
                    
                    // Resize Plotly graphs when a tab is shown (to prevent render issues)
                    var tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]')
                    tabEls.forEach(function(tabEl) {
                        tabEl.addEventListener('shown.bs.tab', function (event) {
                            window.dispatchEvent(new Event('resize'));
                        })
                    })
                });
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _generate_overview_tab(self):
        """Generate the Overview tab content with cluster distribution charts."""
        overview_html = """
            <div class="tab-pane fade show active" id="overview" role="tabpanel" aria-labelledby="overview-tab">
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Cluster Distribution</h3>
                        <div class="chart-container" id="cluster-distribution-chart"></div>
                    </div>
                </div>
        """
        
        # Add code to generate the cluster distribution chart
        cluster_sizes = self.df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
        
        # Limit to top 20 clusters if there are too many
        if len(cluster_sizes) > 20:
            cluster_sizes = cluster_sizes.sort_values('count', ascending=False).head(20)
            overview_html += """
                <div class="alert alert-info">
                    Showing top 20 clusters by size. The dataset contains more clusters.
                </div>
            """
        
        # Shorten cluster names for display
        cluster_sizes['label'] = cluster_sizes.apply(
            lambda x: f"{x['cluster_name'][:25]}{'...' if len(x['cluster_name']) > 25 else ''} (ID: {x['cluster_id']})", 
            axis=1
        )
        
        # Create Plotly figure JSON
        fig = px.bar(
            cluster_sizes,
            x='label',
            y='count',
            color='count',
            labels={'count': 'Number of Keywords', 'label': 'Cluster'},
            title='Size of Each Cluster',
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        # Add plotly chart to the HTML
        overview_html += f"""
            <script>
                var clusterDistData = {fig.to_json()};
                Plotly.newPlot('cluster-distribution-chart', clusterDistData.data, clusterDistData.layout);
            </script>
        """
        
        # Add semantic coherence chart if available
        if 'cluster_coherence' in self.df.columns:
            overview_html += """
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Semantic Coherence</h3>
                        <div class="chart-container" id="coherence-chart"></div>
                    </div>
                </div>
            """
            
            coherence_data = self.df.groupby(['cluster_id', 'cluster_name'])['cluster_coherence'].mean().reset_index()
            
            # Limit to top 20 if needed
            if len(coherence_data) > 20:
                coherence_data = coherence_data.sort_values('cluster_coherence', ascending=False).head(20)
            
            coherence_data['label'] = coherence_data.apply(
                lambda x: f"{x['cluster_name'][:25]}{'...' if len(x['cluster_name']) > 25 else ''} (ID: {x['cluster_id']})", 
                axis=1
            )
            
            fig2 = px.bar(
                coherence_data,
                x='label',
                y='cluster_coherence',
                color='cluster_coherence',
                labels={'cluster_coherence': 'Coherence', 'label': 'Cluster'},
                title='Semantic Coherence by Cluster',
                color_continuous_scale=px.colors.sequential.Greens
            )
            fig2.update_layout(xaxis_tickangle=-45)
            
            overview_html += f"""
                <script>
                    var coherenceData = {fig2.to_json()};
                    Plotly.newPlot('coherence-chart', coherenceData.data, coherenceData.layout);
                </script>
            """
        
        # Close the tab content
        overview_html += """
            </div>
        """
        
        return overview_html
    
    def _generate_intent_tab(self):
        """Generate the Search Intent tab content with intent analysis."""
        if not self.cluster_evaluation:
            return """
                <div class="tab-pane fade" id="intent" role="tabpanel" aria-labelledby="intent-tab">
                    <div class="alert alert-warning mt-4">
                        No search intent data available. Run the clustering with an OpenAI API key to enable intent analysis.
                    </div>
                </div>
            """
        
        intent_html = """
            <div class="tab-pane fade" id="intent" role="tabpanel" aria-labelledby="intent-tab">
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Search Intent Distribution</h3>
                        <div class="chart-container" id="intent-distribution-chart"></div>
                    </div>
                </div>
        """
        
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
                title='Search Intent Distribution',
                margin=dict(l=50, r=50, t=70, b=50),
                height=500
            )
            
            intent_html += f"""
                <script>
                    var intentDistData = {fig.to_json()};
                    Plotly.newPlot('intent-distribution-chart', intentDistData.data, intentDistData.layout);
                </script>
            """
            
            # Add intent by cluster bar chart
            intent_html += """
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Intent by Cluster</h3>
                        <div class="chart-container" id="intent-by-cluster-chart"></div>
                    </div>
                </div>
            """
            
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
                title='Search Intent by Cluster',
                xaxis_title='Cluster',
                yaxis_title='Number of Keywords',
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
            
            intent_html += f"""
                <script>
                    var intentByClusterData = {fig2.to_json()};
                    Plotly.newPlot('intent-by-cluster-chart', intentByClusterData.data, intentByClusterData.layout);
                </script>
            """
            
            # Add intent score distribution chart
            intent_html += """
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Intent Score Distribution</h3>
                        <div class="chart-container" id="intent-score-chart"></div>
                    </div>
                </div>
            """
            
            # Create data for the chart
            top_clusters = df_intent.sort_values('count', ascending=False).head(8)
            
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
                title='Intent Score Distribution',
                xaxis_title='Cluster',
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
            
            intent_html += f"""
                <script>
                    var intentScoreData = {fig3.to_json()};
                    Plotly.newPlot('intent-score-chart', intentScoreData.data, intentScoreData.layout);
                </script>
            """
        
        # Close the tab content
        intent_html += """
            </div>
        """
        
        return intent_html
    
    def _generate_journey_tab(self):
        """Generate the Customer Journey tab content."""
        if not self.cluster_evaluation:
            return """
                <div class="tab-pane fade" id="journey" role="tabpanel" aria-labelledby="journey-tab">
                    <div class="alert alert-warning mt-4">
                        No customer journey data available. Run the clustering with an OpenAI API key to enable journey analysis.
                    </div>
                </div>
            """
        
        # Check if we have journey data
        has_journey_data = False
        for c_id, data in self.cluster_evaluation.items():
            if 'intent_flow' in data:
                has_journey_data = True
                break
        
        if not has_journey_data:
            return """
                <div class="tab-pane fade" id="journey" role="tabpanel" aria-labelledby="journey-tab">
                    <div class="alert alert-warning mt-4">
                        No customer journey data available. Run the clustering with an OpenAI API key to enable journey analysis.
                    </div>
                </div>
            """
        
        journey_html = """
            <div class="tab-pane fade" id="journey" role="tabpanel" aria-labelledby="journey-tab">
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Customer Journey Phase Distribution</h3>
                        <div class="chart-container" id="journey-phase-chart"></div>
                    </div>
                </div>
        """
        
        # Generate journey phases analysis
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
            # Count clusters in each journey phase
            from collections import Counter
            phase_counts = Counter([item['journey_phase'] for item in journey_phases])
            
            phase_colors = {
                "Early (Research Phase)": "#43a047",
                "Research-to-Consideration Transition": "#26a69a",
                "Middle (Consideration Phase)": "#1e88e5",
                "Consideration-to-Purchase Transition": "#7b1fa2",
                "Late (Purchase Phase)": "#ff9800",
                "Mixed Journey Stages": "#757575",
                "Unknown": "#9e9e9e"
            }
            
            phases = list(phase_counts.keys())
            counts = list(phase_counts.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=phases,
                    y=counts,
                    marker=dict(
                        color=[phase_colors.get(phase, "#9e9e9e") for phase in phases]
                    )
                )
            ])
            
            fig.update_layout(
                title='Journey Phase Distribution',
                xaxis_title='Journey Phase',
                yaxis_title='Number of Clusters',
                margin=dict(l=50, r=50, t=70, b=100),
                height=500
            )
            
            journey_html += f"""
                <script>
                    var journeyPhaseData = {fig.to_json()};
                    Plotly.newPlot('journey-phase-chart', journeyPhaseData.data, journeyPhaseData.layout);
                </script>
            """
            
            # Add table with clusters by journey phase
            journey_html += """
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Clusters by Journey Phase</h3>
                        <div class="table-container">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Journey Phase</th>
                                        <th>Cluster Name</th>
                                        <th>Primary Intent</th>
                                        <th>Keyword Count</th>
                                    </tr>
                                </thead>
                                <tbody>
            """
            
            # Sort phases in typical journey order
            phase_order = [
                "Early (Research Phase)", 
                "Research-to-Consideration Transition",
                "Middle (Consideration Phase)", 
                "Consideration-to-Purchase Transition",
                "Late (Purchase Phase)",
                "Mixed Journey Stages",
                "Unknown"
            ]
            
            journey_df = pd.DataFrame(journey_phases)
            
            # Add primary intent
            for i, row in journey_df.iterrows():
                c_id = row['cluster_id']
                if c_id in self.cluster_evaluation and 'intent_classification' in self.cluster_evaluation[c_id]:
                    journey_df.at[i, 'primary_intent'] = self.cluster_evaluation[c_id]['intent_classification'].get('primary_intent', 'Unknown')
                else:
                    journey_df.at[i, 'primary_intent'] = 'Unknown'
            
            # Order by journey phase
            journey_df['phase_order'] = journey_df['journey_phase'].apply(
                lambda x: phase_order.index(x) if x in phase_order else len(phase_order)
            )
            journey_df = journey_df.sort_values(['phase_order', 'count'], ascending=[True, False])
            
            # Generate table rows
            for _, row in journey_df.iterrows():
                phase_class = ""
                if "Early" in row['journey_phase']:
                    phase_class = "table-success"
                elif "Middle" in row['journey_phase']:
                    phase_class = "table-primary"
                elif "Late" in row['journey_phase']:
                    phase_class = "table-warning"
                elif "Transition" in row['journey_phase']:
                    phase_class = "table-info"
                
                journey_html += f"""
                    <tr class="{phase_class}">
                        <td>{row['journey_phase']}</td>
                        <td>{row['cluster_name']}</td>
                        <td>{row['primary_intent']}</td>
                        <td>{row['count']:,}</td>
                    </tr>
                """
            
            journey_html += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            """
            
            # Add journey overview and explanations
            journey_html += """
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h4>Understanding the Customer Journey</h4>
                            </div>
                            <div class="card-body">
                                <p>The customer journey typically flows through these stages:</p>
                                <div class="card mb-3 border-success">
                                    <div class="card-body">
                                        <h5 class="card-title text-success">Research Phase (Informational)</h5>
                                        <p class="card-text">Users are learning about solutions to their problems. Content should focus on education, explanations, and comprehensive guides.</p>
                                    </div>
                                </div>
                                <div class="card mb-3 border-primary">
                                    <div class="card-body">
                                        <h5 class="card-title text-primary">Consideration Phase (Commercial)</h5>
                                        <p class="card-text">Users are comparing options and evaluating alternatives. Content should focus on comparisons, reviews, and feature breakdowns.</p>
                                    </div>
                                </div>
                                <div class="card mb-3 border-warning">
                                    <div class="card-body">
                                        <h5 class="card-title text-warning">Purchase Phase (Transactional)</h5>
                                        <p class="card-text">Users are ready to make a purchase. Content should focus on product details, special offers, and clear calls to action.</p>
                                    </div>
                                </div>
                            </div>
                            <div class="card-footer">
                                <p class="mb-0">Create content that matches the journey phase of your target audience to maximize engagement and conversions.</p>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Close the tab content
        journey_html += """
            </div>
        """
        
        return journey_html
    
    def _generate_clusters_tab(self):
        """Generate the Cluster Details tab content."""
        clusters_html = """
            <div class="tab-pane fade" id="clusters" role="tabpanel" aria-labelledby="clusters-tab">
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Cluster Details</h3>
                        <div class="alert alert-info">
                            This tab shows details for the top 10 clusters by size. The information includes representative keywords, search intent, and journey phase if available.
                        </div>
                        
                        <div class="input-group mb-3">
                            <input type="text" class="form-control" placeholder="Search clusters..." id="clusterSearch">
                            <button class="btn btn-outline-secondary" type="button" onclick="filterClusters()">Search</button>
                        </div>
                    </div>
                </div>
                
                <div class="accordion" id="clusterAccordion">
        """
        
        # Get top clusters by keyword count
        cluster_sizes = self.df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
        top_clusters = cluster_sizes.sort_values('count', ascending=False).head(10)
        
        for idx, row in enumerate(top_clusters.itertuples()):
            c_id = row.cluster_id
            c_name = row.cluster_name
            c_count = row.count
            
            # Get cluster description if available
            cluster_desc_rows = self.df[(self.df['cluster_id'] == c_id) & ('cluster_description' in self.df.columns)]
            c_desc = cluster_desc_rows['cluster_description'].iloc[0] if not cluster_desc_rows.empty else ""
            
            # Get representative keywords
            rep_keywords = []
            if 'representative' in self.df.columns:
                rep_keywords = self.df[(self.df['cluster_id'] == c_id) & (self.df['representative'] == True)]['keyword'].tolist()
            
            # If no marked representatives, just get top 10 keywords
            if not rep_keywords:
                rep_keywords = self.df[self.df['cluster_id'] == c_id]['keyword'].tolist()[:10]
            else:
                rep_keywords = rep_keywords[:10]  # Limit to 10
            
            # Search intent and journey phase
            intent_info = ""
            journey_info = ""
            if c_id in self.cluster_evaluation:
                if 'intent_classification' in self.cluster_evaluation[c_id]:
                    intent_data = self.cluster_evaluation[c_id]['intent_classification']
                    primary_intent = intent_data.get('primary_intent', 'Unknown')
                    
                    # Choose CSS class based on intent
                    intent_class = "intent-mixed"
                    if primary_intent == "Informational":
                        intent_class = "intent-info"
                    elif primary_intent == "Navigational":
                        intent_class = "intent-nav"
                    elif primary_intent == "Transactional":
                        intent_class = "intent-trans"
                    elif primary_intent == "Commercial":
                        intent_class = "intent-comm"
                    
                    # Build intent info HTML
                    intent_info = f"""
                        <div class="{intent_class}">
                            <h5>Primary Search Intent: {primary_intent}</h5>
                            <p>{self.cluster_evaluation[c_id].get('search_intent', '')}</p>
                        </div>
                    """
                
                if 'intent_flow' in self.cluster_evaluation[c_id]:
                    journey_phase = self.cluster_evaluation[c_id]['intent_flow'].get('journey_phase', 'Unknown')
                    journey_info = f"""
                        <div class="mb-3">
                            <h5>Customer Journey Phase:</h5>
                            <p><span class="badge bg-primary">{journey_phase}</span></p>
                        </div>
                    """
            
            # Top keywords by search volume
            keywords_table = ""
            if 'search_volume' in self.df.columns:
                try:
                    cluster_df = self.df[self.df['cluster_id'] == c_id].copy()
                    cluster_df['search_volume'] = pd.to_numeric(cluster_df['search_volume'], errors='coerce')
                    top_kws = cluster_df.sort_values('search_volume', ascending=False).head(10)
                    
                    if not top_kws.empty:
                        keywords_table = """
                            <div class="mt-3">
                                <h5>Top Keywords by Search Volume:</h5>
                                <div class="table-container">
                                    <table class="table table-sm">
                                        <thead>
                                            <tr>
                                                <th>Keyword</th>
                                                <th>Search Volume</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                        """
                        
                        for _, kw_row in top_kws.iterrows():
                            keywords_table += f"""
                                <tr>
                                    <td>{kw_row['keyword']}</td>
                                    <td>{int(kw_row['search_volume']):,}</td>
                                </tr>
                            """
                        
                        keywords_table += """
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        """
                except Exception as e:
                    logger.warning(f"Error displaying top keywords: {str(e)}")
            
            # Add accordion item for this cluster
            clusters_html += f"""
                <div class="accordion-item" data-cluster-name="{c_name.lower()}" data-cluster-id="{c_id}">
                    <h2 class="accordion-header" id="heading{idx}">
                        <button class="accordion-button {'collapsed' if idx > 0 else ''}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{idx}" aria-expanded="{'true' if idx == 0 else 'false'}" aria-controls="collapse{idx}">
                            <strong>{c_name}</strong> <span class="badge bg-secondary ms-2">ID: {c_id}</span> <span class="badge bg-info ms-2">{c_count:,} keywords</span>
                        </button>
                    </h2>
                    <div id="collapse{idx}" class="accordion-collapse collapse {'show' if idx == 0 else ''}" aria-labelledby="heading{idx}" data-bs-parent="#clusterAccordion">
                        <div class="accordion-body">
                            <p><strong>Description:</strong> {c_desc}</p>
                            
                            <div class="mb-3">
                                <h5>Representative Keywords:</h5>
                                <p>{', '.join(rep_keywords)}</p>
                            </div>
                            
                            {intent_info}
                            {journey_info}
                            {keywords_table}
                        </div>
                    </div>
                </div>
            """
        
        # Add JavaScript for cluster search
        clusters_html += """
                </div>
                
                <script>
                    function filterClusters() {
                        const searchTerm = document.getElementById('clusterSearch').value.toLowerCase();
                        const clusterItems = document.querySelectorAll('.accordion-item');
                        
                        clusterItems.forEach(item => {
                            const clusterName = item.getAttribute('data-cluster-name');
                            const clusterId = item.getAttribute('data-cluster-id');
                            
                            if (clusterName.includes(searchTerm) || clusterId.includes(searchTerm)) {
                                item.style.display = '';
                            } else {
                                item.style.display = 'none';
                            }
                        });
                    }
                </script>
            </div>
        """
        
        return clusters_html


def generate_html_report(df, cluster_evaluation=None, language="en"):
    """
    Generate an HTML report from clustering results.
    
    Args:
        df: Dataframe containing clustered keywords
        cluster_evaluation: Dictionary with cluster evaluation data
        language: Language code ('en', 'es')
        
    Returns:
        str: HTML report as a string
    """
    try:
        html_report = HTMLReport(df, cluster_evaluation, language=language)
        html_content = html_report.generate_html()
        return html_content
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        # Return a simple error HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error Generating Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .error {{ color: red; background-color: #ffebee; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Error Generating HTML Report</h1>
            <div class="error">
                <p>An error occurred: {str(e)}</p>
                <p>Please try again with a smaller dataset or contact support.</p>
            </div>
        </body>
        </html>
        """


def add_html_export_button(df, cluster_evaluation=None, language="en"):
    """
    Add HTML export button to Streamlit app.
    
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
        "Select language for the HTML report",
        options=list(languages.keys()),
        format_func=lambda x: languages.get(x, x),
        index=0  # Default to English
    )
    
    button_text = "📊 Generate Interactive HTML Report" if selected_language == "en" else "📊 Generar Informe HTML Interactivo"
    
    if st.button(button_text, use_container_width=True):
        try:
            # Use a spinner to show progress
            with st.spinner("Generating HTML report..." if selected_language == "en" else "Generando el informe HTML..."):
                # Data validation
                if df is None or len(df) == 0:
                    st.error("No data available to generate report")
                    return
                
                # Generate HTML report
                html_content = generate_html_report(df, cluster_evaluation, language=selected_language)
                
                # Create a download link
                b64_html = base64.b64encode(html_content.encode()).decode()
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"keyword_clusters_report_{timestamp}.html"
                
                # Display success message and download link
                success_message = "✅ HTML report generated successfully" if selected_language == "en" else "✅ Informe HTML generado correctamente"
                st.success(success_message)
                
                download_text = "Download HTML Report" if selected_language == "en" else "Descargar Informe HTML"
                href = f'<a href="data:text/html;base64,{b64_html}" download="{filename}">{download_text}</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Show a preview option
                preview_title = "### HTML Report Preview" if selected_language == "en" else "### Vista previa del informe HTML"
                st.markdown(preview_title)
                
                warning_text = "The preview below is limited. For the full interactive experience, download the HTML file and open it in your browser." if selected_language == "en" else "La vista previa es limitada. Para la experiencia interactiva completa, descarga el archivo HTML y ábrelo en tu navegador."
                st.warning(warning_text)
                
                # Show a preview in an iframe (limited functionality in Streamlit)
                st.components.v1.html(html_content, height=600, scrolling=True)
                
        except Exception as e:
            error_text = f"Error generating HTML report: {str(e)}" if selected_language == "en" else f"Error al generar el informe HTML: {str(e)}"
            st.error(error_text)
