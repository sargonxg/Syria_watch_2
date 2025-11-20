# Syria Watch Pulse (v2)

An upgraded Streamlit-based dashboard for Syria-focused news monitoring. It:

- Ingests multiple Syria-related RSS feeds.
- Scrapes full article text where possible.
- Stores everything in a local SQLite DB (`syria_monitor.db`).
- Tags items by topic and actor using a Syria-specific ontology.
- Runs two heuristic "agents":
  - **Source intelligence** (alignment / type / notes).
  - **Red-flag detector** (propaganda tone, vague sourcing, etc.).
- Assigns a **relevance score** (1–5) and a human-readable label (e.g. 🔥 Critical).
- Provides:
  - An **Analyst Dashboard** with filters, metrics, and CSV export of the filtered view.
  - A **Backend/Admin** tab with raw DB view and full-DB CSV export.
  - A **Source Intelligence** tab summarizing each outlet.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will create `syria_monitor.db` in the same folder.  
Use the sidebar to:

- Set the **lookback window** (24–168 hours).
- Trigger a refresh (ingest + classification).

## Best-practice notes

- All tagging, relevance, and red-flag logic is **heuristic**. Treat it as an aid to human judgment, not a substitute.
- For high-stakes use (policy, SitReps, talking points), always cross-check key stories across **diverse** outlets.
- Be mindful of each source's political alignment and institutional incentives when interpreting framing.
- If you run this regularly on a server, consider adding scheduling (cron / CI) and log monitoring.
