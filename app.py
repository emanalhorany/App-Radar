"""App Radar: Early Warning & Competitor Intelligence for Mobile Apps.

Live Google Play search -> competitor reviews -> ML (Recall-based) -> Gemini executive report.
Run locally:  streamlit run app.py
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from src.features import build_features, keyword_categories, problem_breakdown, single_review_features, clean_text
from src.modeling import check_data, feature_importance, numeric_importance, train_and_compare
from src.report import get_api_key, gemini_report, summary_table, triage_action
from src.scraper import fetch_reviews, search_apps

st.set_page_config(page_title="App Radar", page_icon="📡", layout="wide")

# ---------------- Session state ----------------
for key, default in {"main_app": None, "competitors": {}, "main_results": [], "comp_results": [],
                     "data": None, "results": None, "report": None, "report_source": None}.items():
    st.session_state.setdefault(key, default)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query, lang, country):
    return search_apps(query, lang=lang, country=country, n_hits=10)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_reviews(app_id, app_name, count, lang, country):
    return fetch_reviews(app_id, app_name, count=count, lang=lang, country=country)


def app_label(a):
    stars = f" ★{a['score']:.1f}" if a.get("score") else ""
    return f"{a['title']} — {a['developer']}{stars}"


# ---------------- Header ----------------
st.title("📡 App Radar — Early Warning & Competitor Intelligence")
st.caption("Search any app on Google Play, compare it with competitors, detect critical complaints with "
           "machine learning, and get an AI executive report.")

# ---------------- Sidebar settings ----------------
st.sidebar.header("⚙️ Settings")
LANGS = {"English": "en", "Arabic": "ar"}
COUNTRIES = {"United States": "us", "Saudi Arabia": "sa", "UAE": "ae", "Egypt": "eg", "Jordan": "jo",
             "Kuwait": "kw", "Qatar": "qa", "United Kingdom": "gb"}
lang = LANGS[st.sidebar.selectbox("Review language", list(LANGS))]
country = COUNTRIES[st.sidebar.selectbox("Store country", list(COUNTRIES))]
n_reviews = st.sidebar.slider("Reviews per app", 100, 1000, 300, step=50)
st.sidebar.info("Tip: Arabic + Saudi Arabia works well for apps like Talabat, Careem or banks.")
if st.sidebar.button("🔄 Start a new analysis"):
    for k in ["main_app", "data", "results", "report", "report_source"]:
        st.session_state[k] = None
    st.session_state["competitors"] = {}
    st.rerun()

# ---------------- Step 1: search & select ----------------
st.header("1️⃣ Search & select apps")
col_main, col_comp = st.columns(2)

with col_main:
    st.subheader("🏠 Our app")
    q_main = st.text_input("App name", placeholder="e.g. Talabat", key="q_main")
    if st.button("Search", key="btn_main"):
        st.session_state["main_results"] = cached_search(q_main, lang, country)
        if not st.session_state["main_results"]:
            st.warning("No apps found. Try another name or change the country.")
    if st.session_state["main_results"]:
        options = st.session_state["main_results"]
        idx = st.selectbox("Choose our app", range(len(options)), format_func=lambda i: app_label(options[i]))
        if st.button("✅ Set as our app"):
            st.session_state["main_app"] = options[idx]
    if st.session_state["main_app"]:
        st.success(f"Our app: **{st.session_state['main_app']['title']}**")

with col_comp:
    st.subheader("🥊 Competitors (2 to 4)")
    q_comp = st.text_input("Competitor name", placeholder="e.g. Careem, HungerStation", key="q_comp")
    if st.button("Search", key="btn_comp"):
        st.session_state["comp_results"] = cached_search(q_comp, lang, country)
        if not st.session_state["comp_results"]:
            st.warning("No apps found. Try another name or change the country.")
    if st.session_state["comp_results"]:
        options = st.session_state["comp_results"]
        picks = st.multiselect("Choose competitors", range(len(options)), format_func=lambda i: app_label(options[i]))
        if st.button("➕ Add to competitors"):
            for i in picks:
                a = options[i]
                main_id = st.session_state["main_app"]["appId"] if st.session_state["main_app"] else None
                if a["appId"] != main_id:
                    st.session_state["competitors"][a["appId"]] = a
            if len(st.session_state["competitors"]) > 4:
                st.session_state["competitors"] = dict(list(st.session_state["competitors"].items())[:4])
                st.warning("Maximum 4 competitors. Only the first 4 were kept.")
    for app_id, a in list(st.session_state["competitors"].items()):
        c1, c2 = st.columns([5, 1])
        c1.write(f"• {a['title']}")
        if c2.button("✖", key=f"rm_{app_id}"):
            st.session_state["competitors"].pop(app_id)
            st.rerun()

n_comp = len(st.session_state["competitors"])
ready = st.session_state["main_app"] is not None and 2 <= n_comp <= 4
if not ready:
    st.info("Select our app and 2–4 competitors, then run the analysis.")

# ---------------- Step 2 & 3: scrape, features, train ----------------
if st.button("🚀 Run analysis", type="primary", disabled=not ready):
    apps = [st.session_state["main_app"]] + list(st.session_state["competitors"].values())
    frames, skipped = [], []
    progress = st.progress(0.0, text="Collecting reviews...")
    for i, a in enumerate(apps, 1):
        progress.progress(i / len(apps), text=f"Collecting reviews: {a['title']}")
        df_app = cached_reviews(a["appId"], a["title"], n_reviews, lang, country)
        if df_app.empty:
            skipped.append(a["title"])
        else:
            frames.append(df_app)
    progress.empty()
    if skipped:
        st.warning("No reviews found (in this language/country) for: " + ", ".join(skipped))
    if not frames:
        st.error("No reviews were collected. Try another language or country.")
        st.stop()

    data = build_features(pd.concat(frames, ignore_index=True))
    ok, msg = check_data(data)
    st.session_state["data"] = data
    st.session_state["report"] = None
    if ok:
        with st.spinner("Training and comparing models..."):
            st.session_state["results"] = train_and_compare(data)
    else:
        st.session_state["results"] = None
        st.error(msg)

data = st.session_state["data"]
results = st.session_state["results"]
if data is None:
    st.stop()

main_name = st.session_state["main_app"]["title"]
summary = summary_table(data)
breakdown = problem_breakdown(data)

tab_data, tab_models, tab_dash, tab_triage, tab_report = st.tabs(
    ["📄 Data", "🧠 Model benchmark", "📊 Competitor dashboard", "🚨 Live triage", "📝 AI report & export"])

# ---------------- Tab: data ----------------
with tab_data:
    st.subheader("Cleaned review data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", len(data))
    c2.metric("Apps", data["app_name"].nunique())
    c3.metric("Critical reviews (≤ 2★)", f"{100 * data['is_critical'].mean():.1f}%")
    st.dataframe(data[["app_name", "content", "score", "thumbs_up", "text_length", "exclamation_count",
                       "critical_keyword_count", "hour", "problem_types", "is_critical"]], height=400)

# ---------------- Tab: models ----------------
with tab_models:
    if results is None:
        st.warning("Models were not trained because the data is not enough. See the message above.")
    else:
        best = results["best"]
        st.subheader("Logistic Regression vs Random Forest")
        st.caption(f"Train: {results['n_train']} reviews | Test: {results['n_test']} reviews | random_state=42")
        st.dataframe(results["table"].style.format("{:.3f}").highlight_max(axis=0, color="#c8e6c9"))
        st.success(f"🏆 Selected model: **{best}** (highest Recall = "
                   f"{results['table'].loc[best, 'Recall']:.3f})")
        st.info("Why Recall? Missing a critical complaint (a false negative) is more costly than checking "
                "a normal review by mistake. Recall measures how many of the real critical reviews we catch.")
        m1, m2 = st.columns(2)
        for col, name in zip([m1, m2], results["models"]):
            with col:
                st.markdown(f"**Confusion matrix — {name}**")
                st.dataframe(results["matrices"][name])
        st.subheader("Most important features (selected model)")
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**Top 15 features**")
            st.bar_chart(feature_importance(results["models"][best]))
        with f2:
            st.markdown("**Engineered numeric features**")
            st.bar_chart(numeric_importance(results["models"][best]))

# ---------------- Tab: dashboard ----------------
with tab_dash:
    st.subheader(f"Our app vs competitors")
    ours = summary.loc[main_name] if main_name in summary.index else None
    comps = summary.drop(index=main_name, errors="ignore")
    if ours is not None:
        k1, k2, k3, k4 = st.columns(4)
        comp_rating = comps["Avg rating"].mean() if len(comps) else 0
        comp_crit = comps["Critical %"].mean() if len(comps) else 0
        k1.metric("Our avg rating", f"{ours['Avg rating']:.2f}", f"{ours['Avg rating'] - comp_rating:+.2f} vs competitors")
        k2.metric("Our critical reviews", f"{ours['Critical %']:.1f}%",
                  f"{ours['Critical %'] - comp_crit:+.1f} pts vs competitors", delta_color="inverse")
        k3.metric("Our reviews analysed", int(ours["Reviews"]))
        rank = summary["Critical %"].rank(method="min").loc[main_name]
        k4.metric("Our rank (fewest complaints)", f"{int(rank)} of {len(summary)}")
        if ours["Critical %"] > comp_crit + 5:
            st.error("⚠️ Early warning: our share of critical complaints is clearly higher than competitors.")
        elif ours["Critical %"] < comp_crit - 5:
            st.success("✅ We have fewer critical complaints than the competitors' average.")
        else:
            st.info("ℹ️ Our critical complaint level is close to the competitors' average.")
    st.dataframe(summary)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Average rating**")
        st.bar_chart(summary["Avg rating"])
    with d2:
        st.markdown("**Critical reviews (%)**")
        st.bar_chart(summary["Critical %"])
    if not breakdown.empty:
        st.markdown("**Problem types in critical reviews (% of each app's critical reviews)**")
        st.dataframe(breakdown.style.format("{:.1f}").background_gradient(cmap="Reds", axis=None))
    trend = data.dropna(subset=["at"]).copy()
    if not trend.empty:
        trend["date"] = trend["at"].dt.date
        daily = trend.pivot_table(index="date", columns="app_name", values="is_critical", aggfunc="mean") * 100
        st.markdown("**Daily critical review rate (%)**")
        st.line_chart(daily)

# ---------------- Tab: live triage ----------------
with tab_triage:
    st.subheader("Test a new review")
    if results is None:
        st.warning("Train the models first (need more data).")
    else:
        with st.form("triage_form"):
            text = st.text_area("Review text", placeholder="e.g. The app crashes every time I try to pay!!")
            t1, t2 = st.columns(2)
            likes = t1.number_input("Likes on the review", min_value=0, value=0)
            hour = t2.slider("Hour posted", 0, 23, datetime.now().hour)
            submitted = st.form_submit_button("Predict risk")
        if submitted:
            if not text.strip():
                st.warning("Please write a review first.")
            else:
                model = results["models"][results["best"]]
                row = single_review_features(text, likes, hour)
                prob = float(model.predict_proba(row)[0][1])
                cats = keyword_categories(clean_text(text))
                level, action, tips = triage_action(prob, cats)
                r1, r2 = st.columns(2)
                with r1:
                    st.metric("Probability of a critical review", f"{prob:.1%}")
                    st.progress(min(prob, 1.0))
                    if level == "HIGH":
                        st.error(f"🔴 Risk level: {level}")
                    elif level == "MEDIUM":
                        st.warning(f"🟠 Risk level: {level}")
                    else:
                        st.success(f"🟢 Risk level: {level}")
                    st.caption(f"Model used: {results['best']}")
                with r2:
                    st.markdown("**Recommended action**")
                    st.write(action)
                    for tip in tips:
                        st.write("• " + tip)
                    if cats:
                        st.write("**Detected problem types:** " + ", ".join(cats))

# ---------------- Tab: report & export ----------------
with tab_report:
    st.subheader("AI executive report")
    api_key = get_api_key(st.secrets if hasattr(st, "secrets") else None)
    if api_key:
        st.caption("✅ Gemini API key found (stored safely in Secrets).")
    else:
        st.caption("ℹ️ No GEMINI_API_KEY found — a rule-based report will be used.")
    rep_lang = st.radio("Report language", ["English", "Arabic"], horizontal=True)
    if st.button("✨ Generate report"):
        with st.spinner("Writing the report..."):
            text, source = gemini_report(api_key, summary, breakdown, data, main_name, rep_lang)
        st.session_state["report"], st.session_state["report_source"] = text, source
    if st.session_state["report"]:
        st.caption(f"Source: {st.session_state['report_source']}")
        rep = st.session_state["report"]
        if st.session_state["report_source"] == "Rule engine":
            rep = rep.replace("\n", "  \n")  # keep line breaks in markdown
        st.markdown(rep)

    st.subheader("Downloads")
    e1, e2 = st.columns(2)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    export_cols = ["app_name", "content", "clean_text", "score", "thumbs_up", "at", "text_length", "word_count",
                   "exclamation_count", "critical_keyword_count", "hour", "problem_types", "is_critical"]
    e1.download_button("⬇️ Cleaned data (CSV)", data[export_cols].to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"app_radar_data_{stamp}.csv", mime="text/csv")
    report_txt = st.session_state["report"] or "Generate the report first."
    header = f"App Radar Executive Report\nOur app: {main_name}\nDate: {stamp}\n\n{summary.to_string()}\n\n"
    e2.download_button("⬇️ Executive report (TXT)", (header + report_txt).encode("utf-8"),
                       file_name=f"app_radar_report_{stamp}.txt", mime="text/plain",
                       disabled=st.session_state["report"] is None)
