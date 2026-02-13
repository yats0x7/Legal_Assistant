from transformers import pipeline

# 1. Load the Safety Officer (BERT)
# We use 'unitary/toxic-bert' which is the gold standard for catching hate speech/toxicity.
# It will download about 200MB the first time.
print("⏳ Loading Safety Filter (BERT)...")
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

def check_safety(text):
    """
    Returns True if safe, False if toxic.
    """
    print("   🛡️ Scanning response for toxicity...")
    results = toxicity_classifier(text)
    
    # The model returns scores for: toxic, severe_toxic, obscene, threat, insult, identity_hate
    # We want to know if ANY of these scores are high (above 0.5)
    
    scores = results[0] # Get the list of scores
    is_safe = True
    
    for score in scores:
        if score['score'] > 0.5: # 50% confidence threshold
            print(f"      ⚠️ Flagged as {score['label']} ({score['score']:.2f})")
            is_safe = False
    
    if is_safe:
        print("      ✅ Content is Safe.")
    
    return is_safe

# --- Test Loop ---
if __name__ == "__main__":
    print("--- 🛡️ Legal Safety Filter Ready ---")
    
    # Test 1: Safe Legal Text
    test_safe = "Section 302 of the IPC prescribes punishment for murder."
    print(f"\nTesting: '{test_safe}'")
    check_safety(test_safe)
    
    # Test 2: Toxic Text (Simulated bad output)
    test_toxic = "You are an idiot and you deserve to go to jail."
    print(f"\nTesting: '{test_toxic}'")
    check_safety(test_toxic)