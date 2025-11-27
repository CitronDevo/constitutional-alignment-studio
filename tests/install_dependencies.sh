#!/bin/bash
# Installation script for test suite dependencies

echo "======================================================================"
echo "Installing Test Suite Dependencies"
echo "======================================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Install core dependencies
echo "Installing core dependencies..."
echo "----------------------------------------------------------------------"
pip install streamlit>=1.28.0 openai>=1.0.0 ollama>=0.6.1 google-generativeai>=0.8.5 pydantic>=2.0.0 nltk>=3.8.0

if [ $? -eq 0 ]; then
    echo "✓ Core dependencies installed successfully"
else
    echo "✗ Failed to install core dependencies"
    exit 1
fi

echo ""

# Install dev dependencies
echo "Installing development dependencies..."
echo "----------------------------------------------------------------------"
pip install pytest>=7.4.0 pytest-cov>=4.1.0

if [ $? -eq 0 ]; then
    echo "✓ Dev dependencies installed successfully"
else
    echo "✗ Failed to install dev dependencies"
    exit 1
fi

echo ""

# Download NLTK data
echo "Downloading NLTK data..."
echo "----------------------------------------------------------------------"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

if [ $? -eq 0 ]; then
    echo "✓ NLTK data downloaded successfully"
else
    echo "⚠ NLTK data download had issues (may not be critical)"
fi

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Verifying installation..."
python tests/verify_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ ALL SET! You can now run tests:"
    echo "  pytest tests/test_app.py -v"
    echo "======================================================================"
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "⚠ Some dependencies may still be missing. Please check the output above."
    echo "======================================================================"
    exit 1
fi
