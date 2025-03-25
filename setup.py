import spacy
import os

# Descargar el modelo de SpaCy si no existe
if not spacy.util.is_package("en_core_web_sm"):
    os.system("python -m spacy download en_core_web_sm")
