import sys
import os

# Ajoute dynamiquement la racine du projet au PYTHONPATH 
# pour que les tests puissent importer le dossier 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
