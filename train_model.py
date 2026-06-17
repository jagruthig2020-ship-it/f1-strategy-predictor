import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("🏎️ Phase 1: Loading enriched F1 dataset...")
df = pd.read_csv('data/f1_processed_data.csv')

print("🏎️ Phase 2: Splitting features and target labels...")
# Drop target and database primary keys that confuse XGBoost
X = df.drop(columns=['raceId', 'driverId', 'pitted'])
y = df['pitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🏎️ Phase 3: Handling class imbalance via conservative weighting...")
# Instead of multiplying by 51 (which causes false alarms), we balance it carefully
scale_weight_value = 8.0 

print("🏎️ Phase 4: Training XGBoost Classifier with telemetry data...")
model = XGBClassifier(
    scale_pos_weight=scale_weight_value,
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

print("🏎️ Phase 5: Evaluating model performance...")
y_pred = model.predict(X_test)

print("\n=== MODEL PERFORMANCE REPORT ===")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Metrics:")
print(classification_report(y_test, y_pred))

print("\n🏎️ Phase 6: Exporting trained model file...")
with open("f1_pit_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("🎉 Success! Trained model saved as 'f1_pit_model.pkl'")