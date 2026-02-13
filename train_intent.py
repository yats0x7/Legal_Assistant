import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Data
print("⏳ Loading Intent Data...")
df = pd.read_csv('data/intent_data.csv')
X = df['text']
y = df['intent']

# 2. Vectorization
# ngram_range=(1,2) captures "cheque bounce" or "child custody" as single concepts
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 4. Train (Logistic Regression)
print("⏳ Training Intent Model...")

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Training Complete! Accuracy: {acc*100:.2f}%")

# 6. Save
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/intent_model.pkl')
joblib.dump(vectorizer, 'models/intent_vectorizer.pkl')
print("💾 Intent Model saved to 'models/' folder.")