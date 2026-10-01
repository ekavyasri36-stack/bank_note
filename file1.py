# ============================================
# BANKNOTE AUTHENTICATION - SVM + SMOTE
# ============================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE


# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------


data = pd.read_csv("dataset/data_banknote_authentication.csv")
print("=" * 60)
print("BANKNOTE AUTHENTICATION SYSTEM")
print("=" * 60)

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())


# --------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------

X = data.drop("Class", axis=1)
y = data["Class"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nClass Distribution Before SMOTE:")
print(y.value_counts())


# --------------------------------------------
# 3. Split Dataset
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------
# 4. Apply SMOTE
# --------------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nClass Distribution After SMOTE:")
print(y_train_smote.value_counts())


# --------------------------------------------
# 5. Feature Scaling
# --------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------
# 6. Train SVM