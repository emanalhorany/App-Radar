"""Live search and review scraping from Google Play."""
import pandas as pd
from google_play_scraper import Sort, reviews, search


def search_apps(query, lang="en", country="us", n_hits=10):
    """Return a list of matching apps: [{appId, title, developer, score, icon}]."""
    if not query or not query.strip():
        return []
    try:
        results = search(query.strip(), lang=lang, country=country, n_hits=n_hits)
    except Exception:
        return []
    apps = []
    for r in results:
        if r.get("appId"):  # some results come back without an id, skip them
            apps.append({
                "appId": r["appId"],
                "title": r.get("title") or r["appId"],
                "developer": r.get("developer") or "",
                "score": r.get("score"),
                "icon": r.get("icon"),
            })
    return apps


def fetch_reviews(app_id, app_name, count=300, lang="en", country="us"):
    """Fetch the newest reviews of one app. Returns an empty table if none."""
    cols = ["app_id", "app_name", "review_id", "content", "score", "thumbs_up", "at"]
    try:
        result, _ = reviews(app_id, lang=lang, country=country, sort=Sort.NEWEST, count=count)
    except Exception:
        return pd.DataFrame(columns=cols)
    if not result:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(result)
    df = df.rename(columns={"reviewId": "review_id", "thumbsUpCount": "thumbs_up"})
    df["app_id"] = app_id
    df["app_name"] = app_name
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]
