import json, joblib, pandas as pd, shap, matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
MODEL_PATH = "api/model/model.pkl"
SCHEMA_PATH = "api/model/schema.json"
DATA_PATH = "data/raw/dataset.csv"   # sample rows for explainer context
TARGET = "{{TARGET}}"

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH) as f: schema = json.load(f)
features = schema["feature_names"]

df = pd.read_csv(DATA_PATH)

# Drop target if present
if TARGET in df.columns:
    df = df.drop(columns=[TARGET])

# Select only model features
X = df[features].head(500)

# Handle missing values
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

X.loc[:,numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
X.loc[:,cat_cols] = X[cat_cols].fillna("missing")

# Encode categorical columns
if cat_cols:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X.loc[:,cat_cols] = enc.fit_transform(X[cat_cols]) 

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure()
try:
    shap.summary_plot(shap_values, X, show=False)
except Exception:
    # for multi-output differences
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[-1], X, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("Saved shap_summary.png")
