#!/bin/bash
# Force Python 3.10 environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies with version constraints
pip install pysqlite3-binary==0.5.2
pip install chromadb==0.4.24 --no-deps
pip install -r requirements.txt
