

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1) Load
# -----------------------------
CSV_PATH = "podcast_dataset.csv"
TARGET = "Listening_Time_minutes"

data = pd.read_csv(CSV_PATH)
if TARGET not in data.columns:
    raise ValueError(f"'{TARGET}' column not found in {CSV_PATH}")

# Necessary columns we agreed on
CATEGORICAL_COLS = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]
NUMERIC_COLS = ["Episode_Length_minutes", "Host_Popularity_percentage", "Number_of_Ads"]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

df = data[FEATURES + [TARGET]].copy()
df = df.dropna(subset=[TARGET])

X = df[FEATURES]
y = df[TARGET].astype(float)

# -----------------------------
# 2) Preprocessing
# -----------------------------
# Numerics: impute + scale (MLPs need scaling)
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categoricals: impute + OneHot (dense is fine here; categories are small)
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ],
    remainder="drop"
)

# -----------------------------
# 3) Split
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# 4) Model (compact, early stopping)
# -----------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=1e-4,                 
    batch_size=256,           
    max_iter=200,              
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,   
    random_state=42
)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("mlp", mlp)
])

# -----------------------------
# 5) Train & Evaluate
# -----------------------------
model.fit(X_train, y_train)
preds = model.predict(X_valid)

mae = mean_absolute_error(y_valid, preds)
rmse = np.sqrt(mean_squared_error(y_valid, preds))
r2 = r2_score(y_valid, preds)

print("=== Neural Network (MLP) ===")
print(f"Valid MAE:  {mae:.4f}")
print(f"Valid RMSE: {rmse:.4f}")
print(f"Valid R^2:  {r2:.4f}")

# -----------------------------
# 6) Sanity checks
# -----------------------------
print("\nSanity checks:")
print(f"Target count: {y.notnull().sum()}  mean: {y.mean():.2f}  std: {y.std():.2f}  "
      f"min: {y.min():.2f}  max: {y.max():.2f}")
print(f"Train/Valid sizes: {len(y_train)} / {len(y_valid)}")

# -----------------------------
# 7) Save model
# -----------------------------
import joblib
joblib.dump(model, "podcast_model.pkl")


