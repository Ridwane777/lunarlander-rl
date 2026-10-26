#!/bin/bash

# Evaluation script
# Sets up the environment and runs the deterministic evaluation

# Move to the project root
cd "$(dirname "$0")/.." || exit 1

echo "============================================================"
echo "🚀 Starting evaluation benchmark (LunarLander-v3)"
echo "============================================================"

# 1. Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed on this system."
    exit 1
fi

# 2. Gestion de l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment (venv)..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create the virtual environment. Is python3-venv installed?"
        exit 1
    fi
fi

echo "🔄 Activation de l'environnement virtuel..."
    source venv/bin/activate

# 3. Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📥 Checking and installing dependencies (please wait)..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
else
    echo "⚠️  Attention : Fichier requirements.txt introuvable !"
fi

# 4. Check that models exist
if ! ls models/*.pth 1> /dev/null 2>&1; then
    echo ""
    echo "❌ Error: no model (.pth) found in models/."
    echo "   Train your agents first with: python train.py"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "🎯 Evaluating pre-trained agents over 100 episodes."
echo "   This will take a few minutes..."
echo "============================================================"
echo ""

# 5. Run the evaluation
python3 eval.py

echo ""
echo "============================================================"
echo "✅ Evaluation completed successfully!"
echo "📊 Graphiques comparatifs : plots/benchmark/"
echo "🎬 Landing videos: videos/evals/"
echo "============================================================"
