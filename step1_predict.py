import joblib
import os

def predict_sentiment(text):
    # Check if model exists
    if not os.path.exists('models/sentiment_xgb.pkl'):
        return "Error: Model not found. Run train_sentiment.py first."

    # Load the "Brain"
    model = joblib.load('models/sentiment_xgb.pkl')
    vectorizer = joblib.load('models/sentiment_vectorizer.pkl')

    # Transform input
    input_vec = vectorizer.transform([text])

    # Predict
    prediction = model.predict(input_vec)[0]
    
    # Map back to human words
    labels = {0: "Neutral 😐", 1: "URGENT/CRISIS 🚨", 2: "Positive 🟢"}
    return labels[prediction]

if __name__ == "__main__":
    print("--- ⚖️ Legal Sentiment Classifier ---")
    while True:
        user_input = input("\nEnter a legal query (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        result = predict_sentiment(user_input)
        print(f"Sentiment: {result}")