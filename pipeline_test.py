import joblib
import os

def load_models():
    # Load Sentiment (Step 1)
    sent_model = joblib.load('models/sentiment_xgb.pkl')
    sent_vec = joblib.load('models/sentiment_vectorizer.pkl')
    
    # Load Intent (Step 2)
    intent_model = joblib.load('models/intent_model.pkl')
    intent_vec = joblib.load('models/intent_vectorizer.pkl')
    
    return sent_model, sent_vec, intent_model, intent_vec

def run_pipeline(text, models):
    sent_m, sent_v, intent_m, intent_v = models
    
    # Step 1: Sentiment
    s_pred = sent_m.predict(sent_v.transform([text]))[0]
    s_label = {0: "Neutral 😐", 1: "URGENT 🚨", 2: "Positive 🟢"}[s_pred]
    
    # Step 2: Intent
    i_pred = intent_m.predict(intent_v.transform([text]))[0]
    i_label = {0: "Criminal Law 👮", 1: "Family/Civil 🏠", 2: "Corporate 💼"}[i_pred]
    
    return s_label, i_label

if __name__ == "__main__":
    print("--- ⚖️ Legal AI Pipeline (Step 1 + 2) ---")
    
    try:
        models = load_models()
    except FileNotFoundError:
        print("Error: Models not found. Run train_sentiment.py and train_intent.py first.")
        exit()

    while True:
        text = input("\nEnter query (or 'q'): ")
        if text == 'q': break
        
        sentiment, intent = run_pipeline(text, models)
        
        print(f"   Analysis: [{sentiment}] -> Routing to: [{intent}]")