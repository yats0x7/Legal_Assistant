import pandas as pd
import os
import random

if not os.path.exists('data'):
    os.makedirs('data')

# 0 = Criminal Law (BNS, Police, Arrest)
# 1 = Family & Civil Law (Divorce, Property, Rent)
# 2 = Corporate & Tax Law (GST, Companies, Contracts)

# --- Templates ---
criminal_keywords = ["arrest", "bail", "police", "FIR", "jail", "murder", "theft", "assault", "warrant", "custody", "crime"]
family_keywords = ["divorce", "alimony", "child custody", "property", "rent", "landlord", "tenant", "will", "ancestral", "marriage"]
corporate_keywords = ["GST", "tax", "company", "startup", "incorporation", "cheque bounce", "contract", "agreement", "salary", "employment"]

templates = [
    "How to file a case for {keyword}?",
    "What is the punishment for {keyword} in India?",
    "My neighbor is involved in {keyword}.",
    "Is {keyword} legal in Delhi?",
    "Procedure to apply for {keyword}.",
    "Can I get a lawyer for {keyword}?",
    "Latest supreme court judgment on {keyword}.",
    "Rules regarding {keyword} 2024.",
    "I am facing an issue with {keyword}.",
    "Help me with {keyword} laws."
]

data = []

# Generate 100 Criminal Examples
for _ in range(100):
    text = random.choice(templates).format(keyword=random.choice(criminal_keywords))
    data.append([text, 0])

# Generate 100 Family/Civil Examples
for _ in range(100):
    text = random.choice(templates).format(keyword=random.choice(family_keywords))
    data.append([text, 1])

# Generate 100 Corporate Examples
for _ in range(100):
    text = random.choice(templates).format(keyword=random.choice(corporate_keywords))
    data.append([text, 2])

# Save
df = pd.DataFrame(data, columns=['text', 'intent'])
# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

csv_path = 'data/intent_data.csv'
df.to_csv(csv_path, index=False)
print(f"✅ Generated {len(df)} intent examples at: {csv_path}")