import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1. Load Data
print("⏳ Loading data...")
df = pd.read_csv('data/sentiment_data.csv')
X = df['text']
y = df['label']

# 2. Vectorization (Convert text to numbers)
# We use TF-IDF. 'ngram_range=(1,2)' helps capture phrases like "police beating".
print("⏳ Vectorizing text...")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 4. Train XGBoost
print("⏳ Training XGBoost model...")
model = XGBClassifier(
    n_estimators=50,      # Number of "trees" in the forest
    max_depth=5,          # How deep each tree grows
    learning_rate=0.1,    # How fast it learns
    objective='multi:softmax', # For multi-class classification
    num_class=3           # 3 Classes: Neutral, Urgent, Positive
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Training Complete! Accuracy: {acc*100:.2f}%")

# 6. Save Model (Sustainability Step)
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/sentiment_xgb.pkl')
joblib.dump(vectorizer, 'models/sentiment_vectorizer.pkl')
print("💾 Model saved to 'models/' folder.")