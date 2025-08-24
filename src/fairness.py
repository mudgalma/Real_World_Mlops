import json, joblib, pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import f1_score, accuracy_score
import argparse
from sklearn.preprocessing import OrdinalEncoder
parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True)
args = parser.parse_args()

TARGET = args.target
MODEL_PATH = "api/model/model.pkl"
SCHEMA_PATH = "api/model/schema.json"
DATA_PATH = "data/raw/dataset.csv"
SENSITIVE_COL = "gender"  # if dataset lacks, we will synthesize

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH) as f: schema = json.load(f)
features = schema["feature_names"]

df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])

X = df[features]
y = df[TARGET]
# Preprocess (same as SHAP script)
if y.dtype == object or str(y.dtype).startswith("category"):
    if set(y.unique()) <= {"yes", "no"}:
        y = y.map({"no": 0, "yes": 1})
    else:
        y, _ = pd.factorize(y)
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

X.loc[:, numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
X.loc[:, cat_cols] = X[cat_cols].fillna("missing")

if cat_cols:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X.loc[:, cat_cols] = enc.fit_transform(X[cat_cols])
preds = model.predict(X)

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y, y_pred=preds, sensitive_features=df[SENSITIVE_COL]
)

print("By group metrics:")
print(mf.by_group)
print("\nOverall:")
print(mf.overall)
