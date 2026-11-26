"""Train and compare LogisticRegression vs RandomForest."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import NUMERIC_FEATURES

MIN_ROWS = 40          # minimum reviews needed to train
MIN_PER_CLASS = 5      # minimum critical and non-critical reviews


def check_data(df):
    """Return (ok, message) about whether the data is enough to train."""
    n = len(df)
    n_crit = int(df["is_critical"].sum())
    n_ok = n - n_crit
    if n < MIN_ROWS:
        return False, f"Only {n} reviews were collected. At least {MIN_ROWS} are needed to train the models."
    if n_crit < MIN_PER_CLASS or n_ok < MIN_PER_CLASS:
        return False, (f"Not enough examples of both classes (critical: {n_crit}, non-critical: {n_ok}). "
                       "Increase the number of reviews or add more apps.")
    return True, ""


def _preprocessor(n_rows):
    min_df = 2 if n_rows >= 100 else 1
    return ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=3000), "clean_text"),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])


def train_and_compare(df):
    """Train both models on the same split. Returns results dict."""
    X = df[["clean_text"] + NUMERIC_FEATURES]
    y = df["is_critical"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
    }
    rows, models, matrices = [], {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", _preprocessor(len(df))), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        })
        models[name] = pipe
        matrices[name] = pd.DataFrame(
            confusion_matrix(y_test, pred, labels=[0, 1]),
            index=["Actual: Normal", "Actual: Critical"],
            columns=["Predicted: Normal", "Predicted: Critical"])

    table = pd.DataFrame(rows).set_index("Model")
    # Winner = highest Recall (ties broken by F1, then Precision)
    best = table.sort_values(["Recall", "F1", "Precision"], ascending=False).index[0]
    return {"table": table, "models": models, "matrices": matrices, "best": best,
            "n_train": len(X_train), "n_test": len(X_test)}


def feature_importance(pipe, top=15):
    """Top features of a trained pipeline (RF importance or |LR coefficient|)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.replace("text__", "word: ").replace("num__", "") for n in names]
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    else:
        values = np.abs(clf.coef_[0])
    s = pd.Series(values, index=names).sort_values(ascending=False)
    return s.head(top)


def numeric_importance(pipe):
    """Importance of the engineered numeric features only."""
    s = feature_importance(pipe, top=100000)
    return s[s.index.isin(NUMERIC_FEATURES)].sort_values(ascending=False)
