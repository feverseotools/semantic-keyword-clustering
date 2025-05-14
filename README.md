Advanced Semantic Keyword Clustering
A powerful tool for clustering keywords based on semantic similarity, search intent analysis, and customer journey mapping.

Semantic Clustering: Group keywords by meaning, not just lexical similarity
Search Intent Analysis: Classify keywords into informational, navigational, transactional, and commercial intents
Customer Journey Mapping: Understand which stage of the customer journey your clusters represent
Multi-language Support: Process keywords in English, Spanish, French, German, Polish and more
Interactive Visualizations: Explore your keyword clusters with intuitive charts and graphs
Export Options: Download results as CSV, Excel, JSON, or comprehensive PDF reports
AI-Powered Analysis: Get intelligent recommendations for content planning (OpenAI API optional)

Installation
Option 1: Quick Install (Basic Features)
bash# Clone the repository
git clone https://github.com/yourusername/semantic-keyword-clustering.git
cd semantic-keyword-clustering

# Install with basic features
pip install .

# Download required NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Create models directory for caching
mkdir -p ./models
Option 2: Advanced Install with Custom Features
bash# Install with selected extra features
pip install ".[openai,export]"  # For OpenAI integration and PDF/Excel export

# For all features (including development tools)
pip install ".[full]"

# Optional: Install spaCy language models
python -m spacy download en_core_web_sm  # English
python -m spacy download es_core_news_sm  # Spanish
# Add more languages as needed
Option 3: Traditional requirements.txt Install
bashpip install -r requirements.txt

# Then follow the post-installation instructions in the file
For Conda Users
bashconda env create -f environment.yml
conda activate semantic-clustering
Quick Start
bash# Run the Streamlit application
streamlit run semantic_clustering/app.py
Then open your browser at http://localhost:8501
Usage Guide

Upload Your Keywords:

Upload a CSV file with keywords
Choose either "No Header" (one keyword per line) or "With Header" (Keyword Planner format)


Configure Clustering:

Set the number of clusters
Adjust PCA variance for dimensionality reduction
Specify your preferred language


Optional: Add OpenAI API Key:

For enhanced embeddings and cluster naming
Estimate costs with the built-in calculator


Launch the Clustering Process:

Click the "Start Advanced Semantic Clustering" button
Watch as your keywords are processed and analyzed


Explore Results:

Visualize cluster sizes and coherence
Analyze search intent distribution
Map keywords to customer journey phases
View detailed cluster contents


Export Your Data:

Download as CSV, Excel, JSON, or comprehensive PDF report
Share insights with your team



CSV Format Options
The tool accepts two CSV formats:
1. No Header Format (Simple)
running shoes
nike shoes
adidas sneakers
2. With Header Format (Keyword Planner style)
Keyword,search_volume,competition,cpc
running shoes,5400,0.75,1.25
nike shoes,8900,0.82,1.78
adidas sneakers,3200,0.65,1.12
Search Intent Categories
Keywords are classified into four intent categories:

Informational: Users seeking information or answers (e.g., "how to", "what is")
Navigational: Users looking for a specific website or page (e.g., brand names, login pages)
Transactional: Users ready to make a purchase (e.g., "buy", "discount", "shop")
Commercial: Users researching before purchasing (e.g., "best", "reviews", "vs")

Customer Journey Analysis
The tool maps keyword clusters to stages in the customer journey:

Early Phase (Research): Dominated by informational queries
Middle Phase (Consideration): Primarily commercial comparison queries
Late Phase (Purchase): Mostly transactional buying queries

Advanced Configuration
For more control, edit these parameters:
python# In semantic_clustering/app.py
NUM_CLUSTERS = 10  # Default number of clusters
PCA_VARIANCE = 95  # Default PCA explained variance percentage
MAX_PCA_COMPONENTS = 100  # Maximum PCA components
Extending the Tool
Adding New Languages

Add your language to SPACY_LANGUAGE_MODELS in app.py
Install the corresponding spaCy model
Select your language in the UI

Custom Export Formats
Implement new export functions in semantic_clustering/export_pdf.py
Requirements

Python 3.8+
4GB RAM minimum (8GB recommended for larger datasets)
OpenAI API key (optional)

License
MIT License - See LICENSE file for details.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Troubleshooting
Common Issues

OpenAI API errors: Verify your API key and check your usage limits
Memory errors: Try reducing your dataset size or increasing available RAM
PDF generation fails: Install required dependencies (pip install ".[export]")

Getting Help

Open an issue on GitHub
Check the FAQ section in the documentation


Made with ❤️ for SEO and content professionals
