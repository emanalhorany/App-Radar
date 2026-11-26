"""Executive report with Gemini 2.5 Flash, plus a rule-based fallback."""
import os

import pandas as pd


def get_api_key(st_secrets=None):
    """Read GEMINI_API_KEY from Streamlit secrets or environment. Never hard-code it."""
    try:
        if st_secrets is not None and "GEMINI_API_KEY" in st_secrets:
            return st_secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def _build_prompt(summary, breakdown, samples, main_app, language):
    lang_line = "Write the report in Arabic." if language == "Arabic" else "Write the report in simple English."
    return f"""You are a senior mobile product analyst.
Our app is: {main_app}. The other apps are competitors.

Performance summary (from live Google Play reviews):
{summary.to_string()}

Share (%) of critical reviews mentioning each problem type:
{breakdown.to_string() if not breakdown.empty else 'not available'}

Sample critical reviews per app:
{samples}

Write a short executive report with these sections:
1. Executive summary (3 sentences).
2. Our app vs competitors: where we are better and where we are worse (use the numbers).
3. Main technical weaknesses of each competitor.
4. Top 3 development priorities for our app, ordered by urgency, each with a concrete action and owner team.
5. One risk to watch next month.
Be specific and practical. Do not invent numbers. {lang_line}"""


def critical_samples(df, per_app=12, max_len=220):
    parts = []
    for app, g in df[df["is_critical"] == 1].groupby("app_name"):
        top = g.sort_values("thumbs_up", ascending=False).head(per_app)
        lines = [f"- ({int(r.score)}★, {int(r.thumbs_up)} likes) {str(r.content)[:max_len]}" for r in top.itertuples()]
        parts.append(f"[{app}]\n" + "\n".join(lines))
    return "\n\n".join(parts)


def gemini_report(api_key, summary, breakdown, df, main_app, language="English"):
    """Return (text, source). Falls back to the rule-based report on any error."""
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = _build_prompt(summary, breakdown, critical_samples(df), main_app, language)
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            if resp.text:
                return resp.text, "Gemini 2.5 Flash"
        except Exception as e:
            return rule_based_report(summary, breakdown, main_app) + f"\n\n(Gemini was not available: {type(e).__name__})", "Rule engine"
    return rule_based_report(summary, breakdown, main_app), "Rule engine"


def rule_based_report(summary, breakdown, main_app):
    """Simple report built from the numbers, used when there is no API key."""
    lines = ["EXECUTIVE REPORT (rule-based)", ""]
    ours = summary.loc[main_app] if main_app in summary.index else None
    comps = summary.drop(index=main_app, errors="ignore")
    if ours is not None and len(comps):
        lines.append(f"Our app ({main_app}): average rating {ours['Avg rating']:.2f}, "
                     f"critical reviews {ours['Critical %']:.1f}%.")
        lines.append(f"Competitors average: rating {comps['Avg rating'].mean():.2f}, "
                     f"critical reviews {comps['Critical %'].mean():.1f}%.")
        better = comps[comps["Critical %"] < ours["Critical %"]].index.tolist()
        if better:
            lines.append("Competitors with fewer critical complaints than us: " + ", ".join(better) + ".")
        else:
            lines.append("We have the lowest share of critical complaints in this group.")
    lines += ["", "Main problems per app:"]
    if not breakdown.empty:
        for app, row in breakdown.iterrows():
            top = row.sort_values(ascending=False).head(2)
            top = [f"{k} ({v:.0f}%)" for k, v in top.items() if v > 0]
            lines.append(f"- {app}: " + (", ".join(top) if top else "no clear technical pattern"))
    lines += ["", "Development priorities for our app:"]
    actions = {
        "Crash / freeze": "Fix stability first: review crash logs (Firebase Crashlytics) and release a hotfix.",
        "Bug / not working": "Run a bug triage meeting and fix the top reported errors in the next sprint.",
        "Login / account": "Check the login and OTP flow; add clear error messages and a fast support path.",
        "Payment / money": "Audit failed payments and refunds with the finance team within 48 hours.",
        "Speed / performance": "Profile loading times and reduce app size; test on low-end phones.",
        "Update": "Review the latest release; consider a rollback or staged rollout.",
    }
    if not breakdown.empty and main_app in breakdown.index:
        for i, (cat, v) in enumerate(breakdown.loc[main_app].sort_values(ascending=False).head(3).items(), 1):
            if v > 0:
                lines.append(f"{i}. {cat} ({v:.0f}% of our critical reviews): {actions[cat]}")
    else:
        lines.append("1. Read the most-liked critical reviews and group them by problem type.")
    return "\n".join(lines)


def triage_action(prob, categories):
    """Management action for one new review, based on its risk."""
    if prob >= 0.7:
        level = "HIGH"
        base = "Escalate to the engineering team today and reply to the user within 24 hours."
    elif prob >= 0.4:
        level = "MEDIUM"
        base = "Add to the support queue and reply within 48 hours; track if similar reviews increase."
    else:
        level = "LOW"
        base = "No urgent action. Thank the user and keep monitoring."
    extra = {
        "Crash / freeze": "Ask for phone model and app version, and check crash logs.",
        "Payment / money": "Forward to the payments team to check the transaction.",
        "Login / account": "Check the login/OTP service status.",
        "Bug / not working": "Try to reproduce the bug and open a ticket.",
        "Speed / performance": "Check server response time and app performance on this device type.",
        "Update": "Check if the problem started after the latest update.",
    }
    tips = [extra[c] for c in categories if c in extra]
    return level, base, tips


def summary_table(df):
    g = df.groupby("app_name")
    return pd.DataFrame({
        "Reviews": g.size(),
        "Avg rating": g["score"].mean().round(2),
        "Critical %": (100 * g["is_critical"].mean()).round(1),
        "Avg likes": g["thumbs_up"].mean().round(1),
    }).sort_values("Critical %")
