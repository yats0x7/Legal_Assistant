import sys
import pandas as pd
import xgboost as xgb
import sklearn
import chromadb
import langchain
import torch

def check_setup():
    print("--- ⚖️ Legal AI Assistant: System Check ---")
    
    # Check Python Version
    print(f"✅ Python Version: {sys.version.split()[0]}")
    
    # Check ML Libraries
    print(f"✅ Pandas Version: {pd.__version__}")
    print(f"✅ Scikit-Learn Version: {sklearn.__version__}")
    print(f"✅ XGBoost Version: {xgb.__version__}")
    
    # Check Deep Learning (for Step 5 BERT)
    print(f"✅ PyTorch (AI Engine): {'Detected' if torch.backends.mps.is_available() else 'CPU Only'}")
    # Note: On Mac M1/M2/M3, 'mps' is the secret sauce that makes AI fast!
    
    # Check VectorDB (for Step 3 RAG)
    print(f"✅ ChromaDB: Ready")
    
    # A tiny "Hello World" ML Test
    # We are asking XGBoost to learn that if X=1, Y=1.
    X = [[1], [2], [3]]
    y = [0, 1, 1]
    model = xgb.XGBClassifier()
    model.fit(X, y)
    print("✅ XGBoost Test Run: Success!")
    
    print("\n🚀 Status: You are ready to build Step 1!")

if __name__ == "__main__":
    check_setup()