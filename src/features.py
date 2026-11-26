"""Text cleaning and automatic feature engineering."""
import re

import numpy as np
import pandas as pd

# Critical problem keywords (English + Arabic), grouped by problem type
CRITICAL_KEYWORDS = {
    "Crash / freeze": ["crash", "crashes", "crashing", "freeze", "freezes", "frozen", "stuck", "force close",
                       "closes", "يعلق", "تعليق", "يقفل", "يطلع", "توقف", "يهنج", "معلق"],
    "Bug / not working": ["bug", "bugs", "error", "glitch", "broken", "not working", "doesnt work", "dont work",
                          "doesn't work", "خطا", "خلل", "عطل", "مشكله", "لا يعمل", "ما يشتغل", "مش شغال", "لا يشتغل"],
    "Login / account": ["login", "log in", "sign in", "otp", "verification", "verify", "account", "password",
                        "تسجيل", "الدخول", "حساب", "الحساب", "كلمه السر", "رمز التحقق", "الرمز"],
    "Payment / money": ["payment", "pay", "card", "charged", "refund", "money", "wallet", "transaction",
                        "دفع", "الدفع", "بطاقه", "فلوس", "استرجاع", "مبلغ", "خصم", "المحفظه"],
    "Speed / performance": ["slow", "lag", "laggy", "loading", "battery", "بطيء", "بطء", "تحميل", "ثقيل"],
    "Update": ["update", "updated", "new version", "تحديث", "التحديث"],
}

NUMERIC_FEATURES = ["text_length", "word_count", "exclamation_count", "thumbs_up",
                    "critical_keyword_count", "hour"]

_ARABIC_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْـ]")


def normalize_arabic(text):
    text = _ARABIC_DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    return text


def clean_text(text):
    """Lowercase, normalize Arabic, remove links, numbers and symbols."""
    text = str(text or "").lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = normalize_arabic(text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z؀-ۿ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_NORMALIZED_KEYWORDS = {cat: [clean_text(k) for k in words] for cat, words in CRITICAL_KEYWORDS.items()}


def _count_keywords(clean, words):
    padded = f" {clean} "
    return sum(padded.count(f" {w} ") for w in words if w)


def keyword_categories(clean):
    """Return the problem categories mentioned in a cleaned review."""
    return [cat for cat, words in _NORMALIZED_KEYWORDS.items() if _count_keywords(clean, words) > 0]


def build_features(df):
    """Clean texts, add numeric features and the target is_critical."""
    out = df.copy()
    out["content"] = out["content"].fillna("").astype(str)
    out = out[out["content"].str.strip() != ""]                 # drop empty reviews
    out = out.drop_duplicates(subset=["app_id", "review_id"])   # drop repeated reviews
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score"])
    out["thumbs_up"] = pd.to_numeric(out["thumbs_up"], errors="coerce").fillna(0).astype(int)
    out["at"] = pd.to_datetime(out["at"], errors="coerce")

    out["clean_text"] = out["content"].map(clean_text)
    out["text_length"] = out["content"].str.len()
    out["word_count"] = out["clean_text"].str.split().map(len)
    out["exclamation_count"] = out["content"].str.count("!")
    all_words = sum(_NORMALIZED_KEYWORDS.values(), [])
    out["critical_keyword_count"] = out["clean_text"].map(lambda t: _count_keywords(t, all_words))
    out["hour"] = out["at"].dt.hour.fillna(12).astype(int)
    out["problem_types"] = out["clean_text"].map(lambda t: ", ".join(keyword_categories(t)))
    out["is_critical"] = (out["score"] <= 2).astype(int)
    return out.reset_index(drop=True)


def single_review_features(text, thumbs_up=0, hour=12):
    """Build a one-row table for live prediction of a new review."""
    clean = clean_text(text)
    all_words = sum(_NORMALIZED_KEYWORDS.values(), [])
    return pd.DataFrame([{
        "clean_text": clean,
        "text_length": len(str(text)),
        "word_count": len(clean.split()),
        "exclamation_count": str(text).count("!"),
        "thumbs_up": int(thumbs_up),
        "critical_keyword_count": _count_keywords(clean, all_words),
        "hour": int(hour),
    }])


def problem_breakdown(df):
    """Share of critical reviews that mention each problem type, per app (%)."""
    crit = df[df["is_critical"] == 1]
    rows = {}
    for app, g in crit.groupby("app_name"):
        rows[app] = {cat: round(100 * g["problem_types"].str.contains(cat, regex=False).mean(), 1)
                     for cat in CRITICAL_KEYWORDS}
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def safe_float(x):
    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 0.0
