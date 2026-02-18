import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
print("⏳ Loading Intent Data...")
df = pd.read_csv('data/intent_data.csv')
X = df['text']
y = df['intent']

label_names = {0: "Criminal Law", 1: "Family/Civil", 2: "Corporate", 3: "Greeting/Casual"}
print(f"   Classes: {dict(y.value_counts().sort_index())}")

# 2. Vectorization
# ngram_range=(1,2) captures "cheque bounce" or "child custody" as single concepts
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train (Logistic Regression with balanced class weights)
print("⏳ Training Intent Model (4 classes)...")

model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Training Complete! Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=[label_names[i] for i in sorted(label_names)]))

# 6. Save
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/intent_model.pkl')
joblib.dump(vectorizer, 'models/intent_vectorizer.pkl')
print("💾 Intent Model saved to 'models/' folder.")