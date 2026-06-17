import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("🏎️ Loading Predictive F1 dataset...")
df = pd.read_csv('data/f1_predictive_data.csv')

# Drop target label and database keys
X = df.drop(columns=['raceId', 'driverId', 'predict_pit_next_lap'])
y = df['predict_pit_next_lap']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🏎️ Training Anticipatory XGBoost Model...")
model = XGBClassifier(
    scale_pos_weight=7.0,
    n_estimators=180,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

print("\n=== ANTICIPATORY MODEL PERFORMANCE ===")
y_pred = model.predict(X_test)
print(f"Overall Predictive Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

with open("f1_pit_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("🎉 Model updated successfully!")
