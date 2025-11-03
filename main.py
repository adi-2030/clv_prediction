import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Step 1: CSV file read karo
data = pd.read_csv("data/churn.csv")

# Step 2: Pehli 5 rows print karo
print("Sample Data:")
print(data.head())

# Step 3: Dataset ka size check karo
print("Dataset shape:", data.shape)

# Step 4: Columns dekhne ke liye
print("Columns:", data.columns)

# 'TotalCharges' ko numeric me convert karo, jo convert na ho sake wo NaN ban jayega
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Sirf 'TotalCharges' me jo missing ho un rows ko remove karo
data = data.dropna(subset=['TotalCharges'])

# Check karo
print("Missing values removed in 'TotalCharges':", data['TotalCharges'].isnull().sum())

data.reset_index(drop=True, inplace=True)

print("Updated dataset shape:", data.shape)
print(data.head())

# Categorical columns check karo
categorical_cols = data.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

data['Churn'] = data['Churn'].replace({'Yes':1, 'No':0})

# Churn ke alawa baaki string columns ko one-hot encode karo
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Churn'])

print("Dataset after encoding:", data.shape)

X = data.drop('Churn', axis=1)  # Features
y = data['Churn']               # Target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# End-to-End Telecom Churn Prediction
# -------------------------------

# Step 0: Libraries import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: CSV file read
data = pd.read_csv("data/churn.csv")
print("Sample Data:")
print(data.head())
print("Dataset shape:", data.shape)

# Step 2: TotalCharges numeric conversion
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Step 3: Missing values remove
data = data.dropna(subset=['TotalCharges'])
data.reset_index(drop=True, inplace=True)
print("Missing values removed. Updated shape:", data.shape)

# Step 4: Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

# Step 5: Encode target variable 'Churn'
data['Churn'] = data['Churn'].replace({'Yes':1, 'No':0})

# Step 6: One-hot encode remaining categorical columns
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Churn'])
print("Dataset after encoding:", data.shape)

# Step 7: Features and target split
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Step 9: Logistic Regression model train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 10: Prediction & Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# Optional: Feature importance or next steps
# -------------------------------
# - RandomForest / XGBoost model try kar sakte ho
# - Hyperparameter tuning
# - Feature selection

# CSV load
data = pd.read_csv("data/churn.csv")

# TotalCharges numeric conversion & missing values remove
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])
data.reset_index(drop=True, inplace=True)

# Target encode
data['Churn'] = data['Churn'].replace({'Yes':1, 'No':0})

# One-hot encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Churn'])

# Features & target split
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Prediction aur evaluation
y_pred_rf = rf_model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# -------------------------------
# Optimized Telecom Churn Prediction
# -------------------------------

# Step 0: Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load CSV
data = pd.read_csv("data/churn.csv")
print("Original dataset shape:", data.shape)

# Step 2: TotalCharges numeric & missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])
data.reset_index(drop=True, inplace=True)
print("After removing missing TotalCharges:", data.shape)

# Step 3: Encode target
data['Churn'] = data['Churn'].replace({'Yes':1, 'No':0}).astype(int)

# Step 4: One-hot encode categorical columns (drop_first=True to reduce dimensions)
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Churn'], drop_first=True)
print("After encoding:", data.shape)

# Step 5: Features & target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: RandomForest with basic tuning
rf_model = RandomForestClassifier(
    n_estimators=300,        # zyada trees
    max_depth=10,            # limit depth to avoid overfitting
    min_samples_leaf=5,      # minimum samples per leaf
    random_state=42
)
rf_model.fit(X_train, y_train)

# Step 8: Predict & evaluate RandomForest
y_pred_rf = rf_model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("RandomForest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Step 9: Feature importance plot
importances = rf_model.feature_importances_
features = X.columns
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
plt.title("Top 10 Important Features (RandomForest)")
plt.show()

# Step 10: XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Step 11: Predict & evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Step 12 (Optional): Top features for XGBoost
xgb_importances = xgb_model.feature_importances_
feat_imp_xgb = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_importances})
feat_imp_xgb = feat_imp_xgb.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_xgb.head(10))
plt.title("Top 10 Important Features (XGBoost)")
plt.show()

import xgboost


from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# -------------------------------
# Optimized Telecom Churn Prediction (RandomForest + SMOTE)
# -------------------------------

# Step 0: Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load CSV
data = pd.read_csv("data/churn.csv")
print("Original dataset shape:", data.shape)

# Step 2: TotalCharges numeric & missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])
data.reset_index(drop=True, inplace=True)
print("After removing missing TotalCharges:", data.shape)

# Step 3: Encode target (warning fix with astype(int))
data['Churn'] = data['Churn'].replace({'Yes':1, 'No':0}).astype(int)

# Step 4: One-hot encode categorical columns (drop_first=True to reduce dimensions)
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Churn'], drop_first=True)
print("Dataset after encoding:", data.shape)

# Step 5: Features & target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE, class distribution:\n", y_train_res.value_counts())

# Step 8: RandomForest with basic tuning
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
rf_model.fit(X_train_res, y_train_res)

# Step 9: Predict & evaluate
y_pred_rf = rf_model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("RandomForest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Step 10: Feature importance plot
importances = rf_model.feature_importances_
features = X.columns
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
plt.title("Top 10 Important Features (RandomForest + SMOTE)")
plt.show()

# -------------------------------
# Production-Ready Telecom Churn Prediction
# RandomForest + SMOTE + Top Feature Selection + Hyperparameter Tuning
# -------------------------------

# Step 0: Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load CSV
data = pd.read_csv("data/churn.csv")
print("Original dataset shape:", data.shape)

# Step 2: TotalCharges numeric & missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])
data.reset_index(drop=True, inplace=True)
print("After removing missing TotalCharges:", data.shape)

# Step 3: Encode target (fix warning)
data['Churn'] = data['Churn'].replace({'Yes':1, 'No':0}).astype(int)

# Step 4: One-hot encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Churn'], drop_first=True)
print("Dataset after encoding:", data.shape)

# Step 5: Features & target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE, class distribution:\n", y_train_res.value_counts())

# Step 8: Feature importance pre-selection
# Train basic RandomForest to get feature importances
rf_basic = RandomForestClassifier(n_estimators=200, random_state=42)
rf_basic.fit(X_train_res, y_train_res)

feat_imp = pd.DataFrame({'Feature': X_train_res.columns, 'Importance': rf_basic.feature_importances_})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# Select top 30 features
top_features = feat_imp['Feature'].head(30).tolist()
X_train_top = X_train_res[top_features]
X_test_top = X_test[top_features]
print("Selected top 30 features for model training.")

# Step 9: Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators':[200, 300, 400],
    'max_depth':[6, 8, 10],
    'min_samples_leaf':[3, 5, 7]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train_top, y_train_res)
print("Best parameters:", grid.best_params_)

# Step 10: Train final RandomForest with best params
rf_final = grid.best_estimator_
rf_final.fit(X_train_top, y_train_res)

# Step 11: Predict & evaluate
y_pred_final = rf_final.predict(X_test_top)
print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred_final))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))

# Step 12: Feature importance plot
importances = rf_final.feature_importances_
feat_imp_final = pd.DataFrame({'Feature': top_features, 'Importance': importances})
feat_imp_final = feat_imp_final.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_final.head(10))
plt.title("Top 10 Important Features (Final RandomForest)")
plt.show()

# -------------------------------
# Telecom Churn Prediction - Production Ready Pipeline
# -------------------------------

# Step 0: Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving/loading model

# Step 1: Load CSV
data = pd.read_csv("data/churn.csv")

# Step 2: TotalCharges numeric & missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])
data.reset_index(drop=True, inplace=True)

# Step 3: Encode target
data['Churn'] = data['Churn'].replace({'Yes':1, 'No':0}).astype(int)

# Step 4: One-hot encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Churn'], drop_first=True)

# Step 5: Features & target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: SMOTE for imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 8: Feature importance pre-selection
rf_basic = RandomForestClassifier(n_estimators=200, random_state=42)
rf_basic.fit(X_train_res, y_train_res)

feat_imp = pd.DataFrame({'Feature': X_train_res.columns, 'Importance': rf_basic.feature_importances_})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# Top 30 features
top_features = feat_imp['Feature'].head(30).tolist()
X_train_top = X_train_res[top_features]
X_test_top = X_test[top_features]

# Step 9: Hyperparameter tuning
param_grid = {
    'n_estimators':[200, 300, 400],
    'max_depth':[6, 8, 10],
    'min_samples_leaf':[3, 5, 7]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train_top, y_train_res)

# Step 10: Train final model
rf_final = grid.best_estimator_
rf_final.fit(X_train_top, y_train_res)

# Step 11: Evaluate final model
y_pred_final = rf_final.predict(X_test_top)
print("Final Model Accuracy:", accuracy_score(y_test, y_pred_final))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))

# Step 12: Feature importance plot
importances = rf_final.feature_importances_
feat_imp_final = pd.DataFrame({'Feature': top_features, 'Importance': importances})
feat_imp_final = feat_imp_final.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_final.head(10))
plt.title("Top 10 Important Features (Final RandomForest)")
plt.show()

# Step 13: Save model for future prediction
joblib.dump(rf_final, 'randomforest_churn_model.pkl')
joblib.dump(top_features, 'top_features.pkl')
print("Model and top features saved successfully!")

# -------------------------------
# Step 14: Predict new customers (example)
# -------------------------------
# Load model & top features
model = joblib.load('randomforest_churn_model.pkl')
top_features = joblib.load('top_features.pkl')

# Example new data (must match column names after one-hot encoding)
# new_data = pd.DataFrame([...])
# new_data_top = new_data[top_features]
# churn_pred = model.predict(new_data_top)
# print(churn_pred)

# import prob   <-- comment ya delete
