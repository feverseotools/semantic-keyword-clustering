"""
Excel Export Module for Semantic Keyword Clustering.

This module handles Excel export functionality for the clustering results.
"""

__all__ = ['ExcelReport', 'add_excel_export_button', 'generate_excel_report']

import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import logging
from datetime import datetime
from collections import Counter
import tempfile
import os
import re

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False
    logger.warning("xlsxwriter not available - Excel export will be limited")

# Update the generate_excel method in ExcelReport class
def generate_excel(self):
    """Generate an Excel report with multiple worksheets."""
    if not XLSXWRITER_AVAILABLE:
        raise ImportError("xlsxwriter is required for Excel export. Please install it with: pip install xlsxwriter")
    
    # Rest of the method remains the same...

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('excel_export')

class ExcelReport:
    """
    Class for generating Excel reports from keyword clustering results.
    
    Attributes:
        df (pandas.DataFrame): The clustered keywords dataframe
        cluster_evaluation (dict): Dictionary containing cluster evaluation data
        language (str): Language code for report localization ('en', 'es')
    """
    
    def __init__(self, df, cluster_evaluation=None, language="en"):
        """
        Initialize the Excel report generator.
        
        Args:
            df (pandas.DataFrame): The clustered keywords dataframe
            cluster_evaluation (dict, optional): Dictionary containing cluster evaluation data
            language (str, optional): Language code ('en', 'es')
        """
        self.df = df.copy()
        self.cluster_evaluation = cluster_evaluation if cluster_evaluation else {}
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
                "summary": "Summary",
                "clusters": "Clusters",
                "keywords": "Keywords",
                "search_intent": "Search Intent",
                "journey_analysis": "Journey Analysis",
                "recommendations": "Recommendations",
                "total_keywords": "Total Keywords",
                "number_of_clusters": "Number of Clusters",
                "total_search_volume": "Total Search Volume",
                "cluster_id": "Cluster ID",
                "cluster_name": "Cluster Name",
                "cluster_description": "Cluster Description",
                "keyword_count": "Keyword Count",
                "search_vol": "Search Volume",
                "coherence": "Coherence",
                "representative_keywords": "Representative Keywords",
                "primary_intent": "Primary Intent",
                "intent_informational": "Informational Score",
                "intent_navigational": "Navigational Score",
                "intent_transactional": "Transactional Score",
                "intent_commercial": "Commercial Score",
                "journey_phase": "Journey Phase",
                "keyword": "Keyword",
                "search_volume": "Search Volume",
                "split_suggestion": "Split Suggestion",
                "additional_info": "Additional Info"
            },
            "es": {
                "summary": "Resumen",
                "clusters": "Clusters",
                "keywords": "Keywords",
                "search_intent": "Intención de Búsqueda",
                "journey_analysis": "Análisis de Journey",
                "recommendations": "Recomendaciones",
                "total_keywords": "Total de Keywords",
                "number_of_clusters": "Número de Clusters",
                "total_search_volume": "Volumen de Búsqueda Total",
                "cluster_id": "ID de Cluster",
                "cluster_name": "Nombre de Cluster",
                "cluster_description": "Descripción de Cluster",
                "keyword_count": "Cantidad de Keywords",
                "search_vol": "Vol. de Búsqueda",
                "coherence": "Coherencia",
                "representative_keywords": "Keywords Representativas",
                "primary_intent": "Intención Principal",
                "intent_informational": "Puntuación Informacional",
                "intent_navigational": "Puntuación Navegacional",
                "intent_transactional": "Puntuación Transaccional",
                "intent_commercial": "Puntuación Comercial",
                "journey_phase": "Fase del Journey",
                "keyword": "Keyword",
                "search_volume": "Volumen de Búsqueda",
                "split_suggestion": "Sugerencia de División",
                "additional_info": "Info Adicional"
            }
        }
        
        # Default to English if language not available
        return translations.get(self.language, translations["en"])
    
    def generate_excel(self):
        """
        Generate an Excel report with multiple worksheets.
        
        Returns:
            BytesIO: Buffer containing the Excel file
        """
        # Create a BytesIO object to store the Excel file
        output = io.BytesIO()
        
        try:
            # Create a pandas Excel writer using XlsxWriter as the engine
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Create formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D9E1F2',
                    'border': 1
                })
                
                # Add the Summary worksheet
                self._add_summary_worksheet(writer, header_format)
                
                # Add the Clusters worksheet
                self._add_clusters_worksheet(writer, header_format)
                
                # Add the Keywords worksheet
                self._add_keywords_worksheet(writer, header_format)
                
                # Add the Search Intent worksheet if available
                if self.cluster_evaluation:
                    self._add_search_intent_worksheet(writer, header_format)
                    
                    # Add the Journey Analysis worksheet
                    self._add_journey_analysis_worksheet(writer, header_format)
                
                # Add the Recommendations worksheet
                self._add_recommendations_worksheet(writer, header_format)
            
            # Reset the buffer position to the beginning
            output.seek(0)
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating Excel file: {str(e)}")
            raise
    
    def _add_summary_worksheet(self, writer, header_format):
        """Add the Summary worksheet to the Excel file."""
        # Create a summary dataframe with basic info
        summary_data = {
            'Metric': [
                self.translations['total_keywords'],
                self.translations['number_of_clusters']
            ],
            'Value': [
                len(self.df),
                len(self.df['cluster_id'].unique())
            ]
        }
        
        # Add search volume if available
        if 'search_volume' in self.df.columns:
            try:
                self.df['search_volume'] = pd.to_numeric(self.df['search_volume'], errors='coerce')
                total_volume = self.df['search_volume'].sum()
                summary_data['Metric'].append(self.translations['total_search_volume'])
                summary_data['Value'].append(int(total_volume))
            except Exception as e:
                logger.warning(f"Error calculating total search volume: {str(e)}")
        
        # Create a dataframe and write to Excel
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name=self.translations['summary'], index=False)
        
        # Get the xlsxwriter worksheet object
        worksheet = writer.sheets[self.translations['summary']]
        
        # Apply formatting
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 15)
        
        # Add chart of cluster sizes
        try:
            cluster_sizes = self.df.groupby('cluster_id').size().reset_index(name='count')
            top_clusters = cluster_sizes.sort_values('count', ascending=False).head(10)
            
            # Create a chart
            chart = writer.book.add_chart({'type': 'column'})
            
            # Add series to the chart
            cluster_sizes_df = pd.DataFrame({
                'Cluster ID': top_clusters['cluster_id'],
                'Count': top_clusters['count']
            })
            cluster_sizes_df.to_excel(writer, sheet_name=self.translations['summary'], startrow=len(summary_df) + 2, index=False)
            
            # Write a title
            worksheet.write(len(summary_df) + 1, 0, 'Top 10 Clusters by Size', header_format)
            
            # Add a chart
            chart.add_series({
                'name': 'Keyword Count',
                'categories': f"='{self.translations['summary']}'!$A${len(summary_df) + 4}:$A${len(summary_df) + 13}",
                'values': f"='{self.translations['summary']}'!$B${len(summary_df) + 4}:$B${len(summary_df) + 13}",
                'fill': {'color': '#4472C4'}
            })
            
            chart.set_title({'name': 'Top 10 Clusters by Size'})
            chart.set_x_axis({'name': 'Cluster ID'})
            chart.set_y_axis({'name': 'Number of Keywords'})
            
            worksheet.insert_chart(len(summary_df) + 15, 0, chart, {'x_scale': 1.5, 'y_scale': 1.5})
            
        except Exception as e:
            logger.warning(f"Error adding cluster size chart: {str(e)}")
    
    def _add_clusters_worksheet(self, writer, header_format):
        """Add the Clusters worksheet to the Excel file."""
        # Create a clusters dataframe with relevant information
        cluster_data = []
        
        for c_id in self.df['cluster_id'].unique():
            cluster_rows = self.df[self.df['cluster_id'] == c_id]
            c_count = len(cluster_rows)
            
            # Get the cluster name and description
            c_name = ""
            c_desc = ""
            if 'cluster_name' in self.df.columns and not cluster_rows.empty:
                c_name = cluster_rows['cluster_name'].iloc[0]
            
            if 'cluster_description' in self.df.columns and not cluster_rows.empty:
                c_desc = cluster_rows['cluster_description'].iloc[0]
            
            # Get the search volume if available
            search_volume = 0
            if 'search_volume' in self.df.columns:
                try:
                    cluster_rows['search_volume'] = pd.to_numeric(cluster_rows['search_volume'], errors='coerce')
                    search_volume = cluster_rows['search_volume'].sum()
                except Exception as e:
                    logger.warning(f"Error calculating search volume for cluster {c_id}: {str(e)}")
            
            # Get coherence if available
            coherence = 0
            if 'cluster_coherence' in self.df.columns and not cluster_rows.empty:
                coherence = cluster_rows['cluster_coherence'].mean()
            
            # Get representative keywords
            rep_keywords = []
            if 'representative' in self.df.columns:
                rep_keywords = self.df[(self.df['cluster_id'] == c_id) & (self.df['representative'] == True)]['keyword'].tolist()
                
            # If no marked representatives, just get top 5 keywords
            if not rep_keywords:
                rep_keywords = cluster_rows['keyword'].tolist()[:5]
            else:
                rep_keywords = rep_keywords[:5]  # Limit to 5
            
            # Get search intent if available
            primary_intent = "Unknown"
            info_score = 0
            nav_score = 0
            trans_score = 0
            comm_score = 0
            journey_phase = "Unknown"
            split_suggestion = ""
            additional_info = ""
            
            if self.cluster_evaluation and c_id in self.cluster_evaluation:
                if 'intent_classification' in self.cluster_evaluation[c_id]:
                    intent_data = self.cluster_evaluation[c_id]['intent_classification']
                    primary_intent = intent_data.get('primary_intent', 'Unknown')
                    
                    scores = intent_data.get('scores', {})
                    info_score = scores.get('Informational', 0)
                    nav_score = scores.get('Navigational', 0)
                    trans_score = scores.get('Transactional', 0)
                    comm_score = scores.get('Commercial', 0)
                
                if 'intent_flow' in self.cluster_evaluation[c_id]:
                    journey_phase = self.cluster_evaluation[c_id]['intent_flow'].get('journey_phase', 'Unknown')
                
                split_suggestion = self.cluster_evaluation[c_id].get('split_suggestion', '')
                additional_info = self.cluster_evaluation[c_id].get('additional_info', '')
            
            # Add to cluster data
            cluster_data.append({
                self.translations['cluster_id']: c_id,
                self.translations['cluster_name']: c_name,
                self.translations['cluster_description']: c_desc,
                self.translations['keyword_count']: c_count,
                self.translations['search_vol']: int(search_volume),
                self.translations['coherence']: coherence,
                self.translations['representative_keywords']: ', '.join(rep_keywords),
                self.translations['primary_intent']: primary_intent,
                self.translations['intent_informational']: info_score,
                self.translations['intent_navigational']: nav_score,
                self.translations['intent_transactional']: trans_score,
                self.translations['intent_commercial']: comm_score,
                self.translations['journey_phase']: journey_phase,
                self.translations['split_suggestion']: split_suggestion,
                self.translations['additional_info']: additional_info
            })
        
        # Create dataframe and write to Excel
        clusters_df = pd.DataFrame(cluster_data)
        clusters_df.to_excel(writer, sheet_name=self.translations['clusters'], index=False)
        
        # Get the xlsxwriter worksheet object
        worksheet = writer.sheets[self.translations['clusters']]
        
        # Apply formatting
        for col_num, value in enumerate(clusters_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths based on content
        for i, col in enumerate(clusters_df.columns):
            column_width = max(len(str(val)) for val in clusters_df[col])
            column_width = max(column_width, len(col)) + 2  # Add a little extra space
            
            # Cap the width at 50 characters
            column_width = min(column_width, 50)
            
            worksheet.set_column(i, i, column_width)
    
    def _add_keywords_worksheet(self, writer, header_format):
        """Add the Keywords worksheet to the Excel file."""
        # Create a copy of the dataframe with selected columns
        columns_to_include = ['cluster_id', 'keyword']
        
        # Add optional columns if they exist
        if 'cluster_name' in self.df.columns:
            columns_to_include.append('cluster_name')
        
        if 'search_volume' in self.df.columns:
            columns_to_include.append('search_volume')
            
        if 'cluster_coherence' in self.df.columns:
            columns_to_include.append('cluster_coherence')
            
        if 'representative' in self.df.columns:
            columns_to_include.append('representative')
        
        # Build DataFrame for export
        keywords_df = self.df[columns_to_include].copy()
        
        # Rename columns to translated versions
        column_translations = {
            'cluster_id': self.translations['cluster_id'],
            'keyword': self.translations['keyword'],
            'cluster_name': self.translations['cluster_name'],
            'search_volume': self.translations['search_volume'],
            'cluster_coherence': self.translations['coherence'],
            'representative': self.translations['representative_keywords']
        }
        
        keywords_df.rename(columns={col: column_translations.get(col, col) for col in keywords_df.columns}, inplace=True)
        
        # Write to Excel
        keywords_df.to_excel(writer, sheet_name=self.translations['keywords'], index=False)
        
        # Get the xlsxwriter worksheet object
        worksheet = writer.sheets[self.translations['keywords']]
        
        # Apply formatting
        for col_num, value in enumerate(keywords_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Set up filters
        worksheet.autofilter(0, 0, len(keywords_df), len(keywords_df.columns) - 1)
        
        # Adjust column widths based on content
        for i, col in enumerate(keywords_df.columns):
            column_width = max(min(max(len(str(val)) for val in keywords_df[col]), 40), len(col)) + 2
            worksheet.set_column(i, i, column_width)
    
    def _add_search_intent_worksheet(self, writer, header_format):
        """Add the Search Intent worksheet to the Excel file."""
        if not self.cluster_evaluation:
            return
        
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
                
                # Get evidence for primary intent
                evidence = data['intent_classification'].get('evidence', {})
                evidence_for_primary = ', '.join(evidence.get(primary_intent, [])[:3])  # Top 3 pieces of evidence
                
                # Get search intent description
                search_intent = data.get('search_intent', '')
                
                intent_data.append({
                    self.translations['cluster_id']: c_id,
                    self.translations['cluster_name']: cluster_name,
                    self.translations['primary_intent']: primary_intent,
                    self.translations['keyword_count']: count,
                    self.translations['intent_informational']: scores.get('Informational', 0),
                    self.translations['intent_navigational']: scores.get('Navigational', 0),
                    self.translations['intent_transactional']: scores.get('Transactional', 0),
                    self.translations['intent_commercial']: scores.get('Commercial', 0),
                    'Evidence': evidence_for_primary,
                    'Search Intent Description': search_intent
                })
        
        if intent_data:
            # Create dataframe and write to Excel
            intent_df = pd.DataFrame(intent_data)
            intent_df.to_excel(writer, sheet_name=self.translations['search_intent'], index=False)
            
            # Get the xlsxwriter worksheet object
            worksheet = writer.sheets[self.translations['search_intent']]
            
            # Apply formatting
            for col_num, value in enumerate(intent_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set up conditional formatting for intent scores
            score_columns = [
                self.translations['intent_informational'],
                self.translations['intent_navigational'],
                self.translations['intent_transactional'],
                self.translations['intent_commercial']
            ]
            
            for i, col in enumerate(intent_df.columns):
                if col in score_columns:
                    col_idx = intent_df.columns.get_loc(col)
                    worksheet.conditional_format(1, col_idx, len(intent_df), col_idx, {
                        'type': '3_color_scale',
                        'min_color': '#FFFFFF',
                        'mid_color': '#FFEB84',
                        'max_color': '#63BE7B',
                        'min_type': 'num',
                        'min_value': 0,
                        'mid_type': 'num',
                        'mid_value': 50,
                        'max_type': 'num',
                        'max_value': 100
                    })
            
            # Adjust column widths
            for i, col in enumerate(intent_df.columns):
                column_width = max(min(max(len(str(val)) for val in intent_df[col]), 40), len(col)) + 2
                worksheet.set_column(i, i, column_width)
            
            # Add intent distribution chart
            try:
                intent_counts = Counter([item[self.translations['primary_intent']] for item in intent_data])
                
                # Create a chart
                chart = writer.book.add_chart({'type': 'pie'})
                
                # Add series to the chart
                intent_dist_df = pd.DataFrame({
                    'Intent': list(intent_counts.keys()),
                    'Count': list(intent_counts.values())
                })
                intent_dist_df.to_excel(writer, sheet_name=self.translations['search_intent'], 
                                        startrow=len(intent_df) + 2, index=False)
                
                # Write a title
                worksheet.write(len(intent_df) + 1, 0, 'Intent Distribution', header_format)
                
                # Add a chart
                chart.add_series({
                    'name': 'Intent Distribution',
                    'categories': f"='{self.translations['search_intent']}'!$A${len(intent_df) + 4}:$A${len(intent_df) + 3 + len(intent_counts)}",
                    'values': f"='{self.translations['search_intent']}'!$B${len(intent_df) + 4}:$B${len(intent_df) + 3 + len(intent_counts)}"
                })
                
                chart.set_title({'name': 'Distribution of Search Intent Across Clusters'})
                chart.set_style(10)  # Colorful style
                
                worksheet.insert_chart(len(intent_df) + 2, 3, chart, {'x_scale': 1.5, 'y_scale': 1.5})
                
            except Exception as e:
                logger.warning(f"Error adding intent distribution chart: {str(e)}")
    
    def _add_journey_analysis_worksheet(self, writer, header_format):
        """Add the Journey Analysis worksheet to the Excel file."""
        if not self.cluster_evaluation:
            return
        
        # Check if we have journey data
        has_journey_data = False
        for c_id, data in self.cluster_evaluation.items():
            if 'intent_flow' in data:
                has_journey_data = True
                break
        
        if not has_journey_data:
            return
        
        # Collect journey data from all clusters
        journey_data = []
        for c_id, data in self.cluster_evaluation.items():
            if 'intent_flow' in data:
                # Find cluster name safely
                cluster_rows = self.df[self.df['cluster_id'] == c_id]
                cluster_name = f"Cluster {c_id}"
                if not cluster_rows.empty and 'cluster_name' in cluster_rows.columns:
                    cluster_name = cluster_rows['cluster_name'].iloc[0]
                
                # Get journey phase and count
                journey_phase = data['intent_flow'].get('journey_phase', 'Unknown')
                count = len(self.df[self.df['cluster_id'] == c_id])
                
                # Get intent distribution
                intent_dist = data['intent_flow'].get('intent_distribution', {})
                
                # Get top keywords with their intents
                keyword_sample = data['intent_flow'].get('keyword_sample', [])
                keyword_examples = ', '.join([f"{item.get('keyword')} ({item.get('intent')})" for item in keyword_sample[:3]])
                
                # Get primary intent
                primary_intent = 'Unknown'
                if 'intent_classification' in data:
                    primary_intent = data['intent_classification'].get('primary_intent', 'Unknown')
                
                journey_data.append({
                    self.translations['cluster_id']: c_id,
                    self.translations['cluster_name']: cluster_name,
                    self.translations['journey_phase']: journey_phase,
                    self.translations['primary_intent']: primary_intent,
                    self.translations['keyword_count']: count,
                    'Informational %': intent_dist.get('Informational', 0),
                    'Commercial %': intent_dist.get('Commercial', 0),
                    'Transactional %': intent_dist.get('Transactional', 0),
                    'Navigational %': intent_dist.get('Navigational', 0),
                    'Keyword Examples': keyword_examples
                })
        
        if journey_data:
            # Create dataframe and write to Excel
            journey_df = pd.DataFrame(journey_data)
            journey_df.to_excel(writer, sheet_name=self.translations['journey_analysis'], index=False)
            
            # Get the xlsxwriter worksheet object
            worksheet = writer.sheets[self.translations['journey_analysis']]
            
            # Apply formatting
            for col_num, value in enumerate(journey_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Adjust column widths
            for i, col in enumerate(journey_df.columns):
                column_width = max(min(max(len(str(val)) for val in journey_df[col]), 40), len(col)) + 2
                worksheet.set_column(i, i, column_width)
            
            # Add journey phase distribution chart
            try:
                from collections import Counter
                phase_counts = Counter([item[self.translations['journey_phase']] for item in journey_data])
                
                # Create a chart
                chart = writer.book.add_chart({'type': 'column'})
                
                # Add series to the chart
                phase_dist_df = pd.DataFrame({
                    'Phase': list(phase_counts.keys()),
                    'Count': list(phase_counts.values())
                })
                phase_dist_df.to_excel(writer, sheet_name=self.translations['journey_analysis'], 
                                        startrow=len(journey_df) + 2, index=False)
                
                # Write a title
                worksheet.write(len(journey_df) + 1, 0, 'Journey Phase Distribution', header_format)
                
                # Add a chart
                chart.add_series({
                    'name': 'Journey Phases',
                    'categories': f"='{self.translations['journey_analysis']}'!$A${len(journey_df) + 4}:$A${len(journey_df) + 3 + len(phase_counts)}",
                    'values': f"='{self.translations['journey_analysis']}'!$B${len(journey_df) + 4}:$B${len(journey_df) + 3 + len(phase_counts)}",
                    'fill': {'color': '#4472C4'}
                })
                
                chart.set_title({'name': 'Distribution of Clusters Across Customer Journey Phases'})
                chart.set_x_axis({'name': 'Journey Phase'})
                chart.set_y_axis({'name': 'Number of Clusters'})
                
                worksheet.insert_chart(len(journey_df) + 2, 3, chart, {'x_scale': 1.5, 'y_scale': 1.5})
                
            except Exception as e:
                logger.warning(f"Error adding journey phase chart: {str(e)}")
    
    def _add_recommendations_worksheet(self, writer, header_format):
        """Add the Recommendations worksheet to the Excel file."""
        # Create recommendations based on clustering results
        recommendations = []
        
        # General recommendations
        recommendations.append({
            'Category': 'General',
            'Recommendation': 'Prioritize clusters with high search volume and semantic coherence for content development.',
            'Details': 'Focus on clusters with high search volume and good coherence scores (> 0.7) for the best ROI.'
        })
        
        recommendations.append({
            'Category': 'Content Strategy',
            'Recommendation': 'Adapt content type to the predominant search intent in each cluster.',
            'Details': ('• Informational: Explanatory articles, tutorials, comprehensive guides\n'
                       '• Commercial: Comparisons, reviews, best product lists\n'
                       '• Transactional: Product pages, categories, special offers\n'
                       '• Navigational: Brand pages, contact, help')
        })
        
        recommendations.append({
            'Category': 'Customer Journey',
            'Recommendation': 'Create content that matches the customer journey phase of each cluster.',
            'Details': ('• Early (Research): Educational content, answers to basic questions\n'
                       '• Middle (Consideration): Comparison content, reviews, feature breakdowns\n'
                       '• Late (Purchase): Product details, pricing, special offers')
        })
        
        # Specific recommendations based on data
        if self.cluster_evaluation:
            # Find most coherent clusters
            coherent_clusters = []
            for c_id, data in self.cluster_evaluation.items():
                coherence = data.get('coherence_score', 0)
                
                if coherence >= 7:  # High coherence threshold
                    # Find cluster name safely
                    cluster_rows = self.df[self.df['cluster_id'] == c_id]
                    c_name = f"Cluster {c_id}"
                    if not cluster_rows.empty and 'cluster_name' in cluster_rows.columns:
                        c_name = cluster_rows['cluster_name'].iloc[0]
                    
                    coherent_clusters.append((c_id, c_name, coherence))
            
            if coherent_clusters:
                coherent_clusters.sort(key=lambda x: x[2], reverse=True)
                top_coherent = coherent_clusters[:5]
                
                coherent_details = '\n'.join([f"• {name} (ID: {cid}) - Coherence: {coh:.1f}/10" for cid, name, coh in top_coherent])
                
                recommendations.append({
                    'Category': 'Prioritization',
                    'Recommendation': 'Focus on these highly coherent clusters first',
                    'Details': coherent_details
                })
            
            # Find clusters that need splitting
            split_clusters = []
            for c_id, data in self.cluster_evaluation.items():
                split_suggestion = data.get('split_suggestion', '')
                
                if isinstance(split_suggestion, str) and 'yes' in split_suggestion.lower():
                    # Find cluster name safely
                    cluster_rows = self.df[self.df['cluster_id'] == c_id]
                    c_name = f"Cluster {c_id}"
                    if not cluster_rows.empty and 'cluster_name' in cluster_rows.columns:
                        c_name = cluster_rows['cluster_name'].iloc[0]
                    
                    split_clusters.append((c_id, c_name))
            
            if split_clusters:
                split_details = '\n'.join([f"• {name} (ID: {cid})" for cid, name in split_clusters[:5]])
                
                recommendations.append({
                    'Category': 'Cluster Refinement',
                    'Recommendation': 'Consider splitting these clusters for more targeted content',
                    'Details': split_details
                })
        
        # Create dataframe and write to Excel
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df.to_excel(writer, sheet_name=self.translations['recommendations'], index=False)
        
        # Get the xlsxwriter worksheet object
        worksheet = writer.sheets[self.translations['recommendations']]
        
        # Apply formatting
        for col_num, value in enumerate(recommendations_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust row heights for details column
        for row_num in range(1, len(recommendations_df) + 1):
            details = recommendations_df.iloc[row_num - 1]['Details']
            if details:
                # Count newlines to determine row height
                newlines = details.count('\n') + 1
                row_height = 15 * max(1, newlines)  # 15 points per line
                worksheet.set_row(row_num, row_height)
        
        # Adjust column widths
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:B', 40)
        worksheet.set_column('C:C', 60)
        
        # Add text wrap format for details
        wrap_format = writer.book.add_format({'text_wrap': True, 'valign': 'top'})
        for row_num in range(1, len(recommendations_df) + 1):
            worksheet.write(row_num, 2, recommendations_df.iloc[row_num - 1]['Details'], wrap_format)


def generate_excel_report(df, cluster_evaluation=None, language="en"):
    """
    Generate an Excel report from clustering results.
    
    Args:
        df: Dataframe containing clustered keywords
        cluster_evaluation: Dictionary with cluster evaluation data
        language: Language code ('en', 'es')
        
    Returns:
        BytesIO: Buffer containing the Excel file
    """
    try:
        excel_report = ExcelReport(df, cluster_evaluation, language=language)
        excel_buffer = excel_report.generate_excel()
        return excel_buffer
    except Exception as e:
        logger.error(f"Error generating Excel report: {str(e)}")
        # Return a simple error Excel file
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create a simple error worksheet
            pd.DataFrame([
                {'Error': f"An error occurred: {str(e)}"},
                {'Error': "Please try again with a smaller dataset or contact support."}
            ]).to_excel(writer, sheet_name='Error', index=False)
        
        output.seek(0)
        return output


def add_excel_export_button(df, cluster_evaluation=None, language="en"):
    """
    Add Excel export button to Streamlit app.
    
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
        "Select language for the Excel report",
        options=list(languages.keys()),
        format_func=lambda x: languages.get(x, x),
        index=0  # Default to English
    )
    
    button_text = "📊 Generate Excel Report" if selected_language == "en" else "📊 Generar Informe Excel"
    
    if st.button(button_text, use_container_width=True):
        try:
            # Use a spinner to show progress
            with st.spinner("Generating Excel report..." if selected_language == "en" else "Generando el informe Excel..."):
                # Data validation
                if df is None or len(df) == 0:
                    st.error("No data available to generate report")
                    return
                
                # Generate Excel report
                excel_buffer = generate_excel_report(df, cluster_evaluation, language=selected_language)
                
                # Create a download link
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"keyword_clusters_{timestamp}.xlsx"
                
                # Display success message and download link
                success_message = "✅ Excel report generated successfully" if selected_language == "en" else "✅ Informe Excel generado correctamente"
                st.success(success_message)
                
                # Convert to b64 for download
                b64 = base64.b64encode(excel_buffer.getvalue()).decode()
                download_text = "Download Excel Report" if selected_language == "en" else "Descargar Informe Excel"
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{download_text}</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            error_text = f"Error generating Excel report: {str(e)}" if selected_language == "en" else f"Error al generar el informe Excel: {str(e)}"
            st.error(error_text)
