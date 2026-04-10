# Indian Legal Assistant

I built this project to act as a local, AI-powered legal assistant tailored for Indian law. I noticed a lot of existing LLMs still hallucinate about the old IPC and CrPC, so I designed this system to strictly follow the new post-July 2024 criminal laws (BNS, BNSS, BSA).

It's essentially a RAG pipeline hooked up to Ollama running locally. Before it even generates an answer, I added some machine learning classifiers on top (using XGBoost) to detect intent and sentiment. This helps the assistant know if you're just saying "hi", asking a civil law question, or if it's an actual emergency that needs immediate boilerplate safety steps.

**Disclaimer**: I am not a lawyer, and this is just an AI tool meant for educational purposes. Do not use this as actual legal counsel.

## How it works

1. **Intent & Sentiment Detection**: An XGBoost model categorizes your prompt to figure out the legal domain and checks if you're panicking/in an emergency.
2. **RAG (Knowledge Base)**: It runs a similarity search against a local ChromaDB filled with the latest statutes.
3. **LLM Engine**: It feeds the matched context into `dolphin-llama3:8b` via Ollama. I've strictly prompted it to act as a senior advocate that refuses to hallucinate and only cites modern 2024 laws.
4. **Safety Filter**: Finally, `toxic-bert` evaluates the generated text to block anything toxic or unsafe before showing it to you.

## Setup 

You'll need Python 3.9+ and Ollama installed on your machine.

1. Start by pulling the LLM: 
   ```bash
   ollama run dolphin-llama3:8b
   ```
2. Install the python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the local models (you only need to do this once):
   ```bash
   python train_intent.py
   # you can also run the sentiment training script here if needed
   ```
4. Build the database: You'll need to run the `ingest_*.py` scripts to embed your legal PDFs and JSONs into ChromaDB if it isn't already setup in `chroma_db/`.

## Running the app

Once everything is set up, just run the main script:
```bash
python main_assistant.py
```
*(There's also a `run_assistant.sh` script if you prefer using that).*

