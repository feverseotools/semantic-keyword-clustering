#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from setuptools import setup, find_packages

def post_install():
    """Run post-installation tasks."""
    print("\n" + "="*80)
    print("Running post-installation tasks for Semantic Keyword Clustering")
    print("="*80)

    # Download NLTK data
    try:
        import nltk
        print("Downloading NLTK resources...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK resources downloaded successfully")
    except Exception as e:
        print(f"⚠️ Error downloading NLTK resources: {e}")
        print("   You may need to download them manually using:")
        print("   python -c \"import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')\"")

    # Create models directory
    try:
        models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        print(f"✅ Created models directory at: {models_dir}")
    except Exception as e:
        print(f"⚠️ Error creating models directory: {e}")

    # Check if spaCy is installed
    try:
        import spacy
        spacy_installed = True
        print("✅ spaCy is installed")
    except ImportError:
        spacy_installed = False
        print("ℹ️ spaCy is not installed. You can install it with: pip install spacy")

    # If spaCy is installed, ask about downloading language models
    if spacy_installed:
        languages = {
            'en': 'en_core_web_sm',   # English
            'es': 'es_core_news_sm',  # Spanish
            'fr': 'fr_core_news_sm',  # French
            'de': 'de_core_news_sm',  # German
            'pl': 'pl_core_news_sm',  # Polish
        }
        
        print("\nAvailable spaCy language models:")
        for lang_code, model in languages.items():
            print(f"  {lang_code}: {model}")
        
        if "install" in sys.argv and "--no-prompt" not in sys.argv:
            install_models = input("\nDo you want to install any spaCy language models? (y/n): ").strip().lower()
            if install_models == 'y':
                selected_langs = input("Enter language codes separated by spaces (e.g., 'en es'): ").strip().split()
                
                for lang in selected_langs:
                    if lang in languages:
                        model = languages[lang]
                        print(f"Installing {model}...")
                        try:
                            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
                            print(f"✅ Installed {model}")
                        except Exception as e:
                            print(f"⚠️ Error installing {model}: {e}")
                    else:
                        print(f"⚠️ Unknown language code: {lang}")
        else:
            print("\nTo install spaCy language models, use: python -m spacy download <model_name>")
            print("For example: python -m spacy download en_core_web_sm")

    print("\n" + "="*80)
    print("Semantic Keyword Clustering installation complete!")
    print("="*80)
    print("To run the application: streamlit run semantic_clustering/app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Setup is primarily configured in pyproject.toml,
    # but we use setup.py for the post-install hook
    setup(
        name="semantic-keyword-clustering",
        packages=find_packages(),
        # Include package data
        include_package_data=True,
        # All other configurations are in pyproject.toml
    )
    
    # Run post-install tasks if this is an install command
    if "install" in sys.argv:
        post_install()
