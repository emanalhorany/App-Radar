# 📡 App Radar — Early Warning & Competitor Intelligence for Mobile Apps

A dynamic cloud app that helps product managers **monitor user complaints** and **compare their app with competitors** on Google Play — live, for any app.

**Live app:** `https://YOUR-APP.streamlit.app`  ← replace with your link

## Business problem
App companies receive thousands of reviews. Critical complaints (crashes, payment or login failures) are easy to miss, and comparing with competitors by hand takes days. App Radar does it in minutes.

## How it works
1. **Live app search** — type any app name (e.g. Talabat, Careem, a bank). The app uses `google-play-scraper` `search()` to list matching apps. Choose **our app + 2–4 competitors** and the number of reviews per app.
2. **Automated scraping & feature engineering** — newest reviews of all apps are merged into one table with `app_name`. Texts are cleaned (Arabic normalisation included). Features: text length, word count, exclamation marks, likes, critical keyword count, posting hour. Target: `is_critical = 1` when rating ≤ 2★.
3. **Dynamic model benchmarking** — `LogisticRegression` vs `RandomForestClassifier` (`random_state=42`), using TF-IDF text features + scaled numeric features. Accuracy, Precision, Recall, F1 and confusion matrices are shown. The model with the **highest Recall** is selected automatically.
4. **Competitor dashboard & live triage** — `st.metric` cards, rating and critical-rate charts, problem-type breakdown, daily trend, and an `st.form` to test any new review with `predict_proba` and get a recommended action.
5. **AI executive report & export** — Gemini 2.5 Flash writes a comparative report (competitor weaknesses + our development priorities). A rule-based report is used if no key is set. Download the report (TXT) and cleaned data (CSV).
6. **Cloud deployment** — GitHub + Streamlit Community Cloud, with the API key stored in Secrets.

## Why Recall?
Missing a real critical complaint (false negative) means an unhappy user, a bad rating and possible churn. Checking a normal review by mistake (false positive) only costs a few seconds. So we prefer the model that **catches more critical reviews** (higher Recall), not the one with the highest overall Accuracy — especially because critical reviews are usually the minority class.

## Handling different data
- Apps with **no reviews** in the chosen language/country are skipped with a warning.
- Empty and duplicate reviews are removed.
- Training only starts if there are at least 40 reviews and 5 examples of each class; otherwise the user gets a clear message.
- `stratify` keeps the same critical/non-critical ratio in train and test; `class_weight="balanced"` handles class imbalance.

## Project structure
```
app-radar/
├── app.py                      # Streamlit interface
├── requirements.txt            # dependencies
├── README.md
├── PROPOSAL.md                 # freelance proposal (Upwork / Mostaql)
├── .gitignore
├── .streamlit/
│   ├── config.toml             # theme
│   └── secrets.toml.example    # how to add the key (no real key)
└── src/
    ├── scraper.py              # live search + review scraping
    ├── features.py             # cleaning + feature engineering
    ├── modeling.py             # training, comparison, feature importance
    └── report.py               # Gemini report, rule engine, triage actions
```

## Run locally
```bash
pip install -r requirements.txt
# optional: set your Gemini key
#   Windows CMD:  set GEMINI_API_KEY=your-key
#   Mac/Linux:    export GEMINI_API_KEY=your-key
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → choose the repo, branch `main`, file `app.py`.
3. **Advanced settings** → Python **3.12** → in **Secrets** add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Deploy.

## Security
No API key is written in the code. The key is read from Streamlit Secrets or an environment variable.

## Tech stack
Python · Streamlit · pandas · scikit-learn · google-play-scraper · Google Gemini 2.5 Flash
