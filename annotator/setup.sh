#!/bin/bash
# Setup script for YOLO Manuscript Annotator
# Works on Mac/Linux. For Windows, run the commands manually in PowerShell.

echo "=================================="
echo "YOLO Manuscript Annotator Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.8 or later from https://www.python.org/"
    exit 1
fi

echo "Found Python: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Mac/Linux
    source venv/bin/activate
fi

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To start annotating:"
echo "  1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     venv\\Scripts\\activate"
else
    echo "     source venv/bin/activate"
fi
echo ""
echo "  2. Run the annotator:"
echo "     python annotate_yolo.py --image your_image.png"
echo ""
echo "See README.md for full instructions."