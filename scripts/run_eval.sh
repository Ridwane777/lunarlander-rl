#!/bin/bash

# LunarLander RLScript d'évaluation
# Ce script prépare l'environnement et lance l'évaluation déterministe

# Se placer à la racine du projet
cd "$(dirname "$0")/.." || exit 1

echo "============================================================"
echo "🚀 Démarrage du Benchmark d'Évaluation (LunarLander-v3)"
echo "============================================================"

# 1. Vérification de Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur : python3 n'est pas installé sur ce système."
    exit 1
fi

# 2. Gestion de l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel (venv)..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Échec de la création de l'environnement virtuel. Le paquet python3-venv est-il installé ?"
        exit 1
    fi
fi

echo "🔄 Activation de l'environnement virtuel..."
    source venv/bin/activate

# 3. Installation des dépendances
if [ -f "requirements.txt" ]; then
    echo "📥 Vérification et installation des dépendances (patientez un instant)..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
else
    echo "⚠️  Attention : Fichier requirements.txt introuvable !"
fi

# 4. Vérification de la présence des modèles
if ! ls models/*.pth 1> /dev/null 2>&1; then
    echo ""
    echo "❌ Erreur : Aucun modèle (.pth) n'a été trouvé dans le dossier models/."
    echo "   Veuillez entraîner vos agents d'abord en lançant : python train.py"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "🎯 Évaluation des agents pré-entraînés sur 100 épisodes."
echo "   Cette étape va prendre quelques minutes..."
echo "============================================================"
echo ""

# 5. Lancer l'évaluation
python3 eval.py

echo ""
echo "============================================================"
echo "✅ Évaluation terminée avec succès !"
echo "📊 Graphiques comparatifs : plots/benchmark/"
echo "🎬 Vidéos des atterrissages : videos/evals/"
echo "============================================================"
