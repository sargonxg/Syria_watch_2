import streamlit as st
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dateutil import parser
import sqlite3
import requests

# ===============================
# CONFIGURATION & ONTOLOGY
# ===============================

# Expanded Source List (Local, Gov, Opposition, Kurdish, International)
NEWS_SOURCES = [
    # Independent / Opposition
    {'id': 'enab', 'name': 'Enab Baladi', 'url': 'https://english.enabbaladi.net/feed/'},
    {'id': 'zaman', 'name': 'Zaman Al Wasl', 'url': 'https://en.zamanalwsl.net/rss.php'},
    {'id': 'direct', 'name': 'Syria Direct', 'url': 'https://syriadirect.org/feed/'},
    {'id': 'halab', 'name': 'Halab Today', 'url': 'https://halabtodaytv.net/feed'},
    # Kurdish / SDF
    {'id': 'npa', 'name': 'North Press', 'url': 'https://npasyria.com/en/feed/'},
    {'id': 'hawar', 'name': 'Hawar News', 'url': 'https://hawarnews.com/en/feed/'},
    {'id': 'rojava', 'name': 'Rojava Info', 'url': 'https://rojavainformationcenter.org/feed/'},
    # Government / State
    {'id': 'sana', 'name': 'SANA (Gov)', 'url': 'https://sana.sy/en/?feed=rss2'},
    # Hyper-Local
    {'id': 'suwayda', 'name': 'Suwayda 24', 'url': 'https://suwayda24.com/feed/'},
    {'id': 'deir', 'name': 'DeirEzzor 24', 'url': 'https://deirezzor24.net/en/feed/'},
    # Additional / International Syria-focused
    {'id': 'observer', 'name': 'The Syrian Observer', 'url': 'https://syrianobserver.com/feed'},
]

# Ontology for Tags (The "Type" of Event)
TOPIC_KEYWORDS = {
    'Humanitarian': ['aid', 'refugee', 'camp', 'food', 'water', 'cholera', 'earthquake', 'unrwa', 'displacement', 'shelter', 'poverty', 'starvation'],
    'Military/Ground': ['shelling', 'clash', 'airstrike', 'air strike', 'bombing', 'killed', 'injured', 'attack', 'isis', 'islamic state', 'ied', 'drone', 'assassination', 'frontline', 'front line'],
    'Political': ['meeting', 'decree', 'election', 'minister', 'normalization', 'astana', 'geneva', 'un sc', 'security council', 'diplomacy', 'statement', 'agreement', 'talks', 'negotiation'],
    'Human Rights': ['arrest', 'torture', 'detainee', 'prison', 'detention', 'kidnap', 'kidnapping', 'execution', 'violation', 'forced', 'activist', 'enforced disappearance'],
}

# Ontology for Actors (Who is involved)
ACTOR_KEYWORDS = {
    'Regime/SAA': ['assad', 'regime', 'saa', 'syrian army', 'government forces', 'damascus', '4th division'],
    'SDF/Kurdish': ['sdf', 'kurdish', 'ypg', 'asayish', 'aanes', 'mazloum'],
    'Turkey/SNA': ['turkey', 'turkish', 'sna', 'national army', 'ankara', 'mercenaries'],
    'HTS/Idlib': ['hts', 'hayat tahrir', 'jolani', 'salvation government', 'idlib'],
    'Russia': ['russia', 'russian', 'moscow', 'putin', 'khmeimim'],
    'Iran/Militias': ['iran', 'tehran', 'militia', 'irgc', 'hezbollah'],
    'USA/Coalition': ['usa', 'american', 'coalition', 'washington', 'base', 'pentagon'],
    'Israel': ['israel', 'idf', 'tel aviv', 'golani brigade'],
}

# Source profiles (for the "source intelligence" agent)
SOURCE_PROFILES = {
    'Enab Baladi': {
        'alignment': 'Independent / Opposition-leaning',
        'type': 'Local Syrian outlet',
        'note': 'Community-rooted media founded during the uprising, often critical of Damascus.'
    },
    'Zaman Al Wasl': {
        'alignment': 'Opposition',
        'type': 'Syrian online newspaper',
        'note': 'Carries opposition narratives and leaks, sometimes with strong political framing.'
    },
    'Syria Direct': {
        'alignment': 'Independent',
        'type': 'Training-focused media NGO',
        'note': 'Trains Syrian journalists; aims for explanatory reporting and local voices.'
    },
    'Halab Today': {
        'alignment': 'Opposition-leaning',
        'type': 'TV / online outlet',
        'note': 'Aleppo-origin station with strong focus on northern Syria.'
    },
    'North Press': {
        'alignment': 'AANES / SDF-leaning',
        'type': 'Regional agency',
        'note': 'Covers northeast Syria with a perspective close to local self-administration.'
    },
    'Hawar News': {
        'alignment': 'SDF-leaning',
        'type': 'Agency / movement outlet',
        'note': 'Often amplifies AANES/SDF narratives and official positions.'
    },
    'Rojava Info': {
        'alignment': 'Pro-Rojava',
        'type': 'Research / information center',
        'note': 'Long-form and investigative content on northeast Syria and Kurdish actors.'
    },
    'SANA (Gov)': {
        'alignment': 'Government / State',
        'type': 'Official state agency',
        'note': 'Formal voice of Damascus; strong official framing and propaganda risk.'
    },
    'Suwayda 24': {
        'alignment': 'Local / Community',
        'type': 'Local news page',
        'note': 'Hyper-local coverage of Suwayda governorate, protests, and security incidents.'
    },
    'DeirEzzor 24': {
        'alignment': 'Local / Opposition-leaning',
        'type': 'Local network',
        'note': 'Granular reporting on Deir Ezzor, often focused on SDF, ISIS cells, and tribal dynamics.'
    },
    'The Syrian Observer': {
        'alignment': 'Curated / Mixed',
        'type': 'English-language aggregator',
        'note': 'Curates translations from diverse Syrian press; mix of views but editorial choices matter.'
    },
}

# ===============================
# DB INITIALIZATION & UTILITIES
# ===============================

DB_PATH = "syria_monitor.db"


def init_db():
    """Initialize SQLite database and ensure required columns exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Base table definition (with new columns)
    c.execute(
        '''CREATE TABLE IF NOT EXISTS articles (
               link TEXT PRIMARY KEY,
               title TEXT,
               source TEXT,
               published_date TIMESTAMP,
               summary TEXT,
               full_text TEXT,
               tags TEXT,
               actors TEXT,
               relevance_score REAL,
               red_flags TEXT,
               fetched_at TIMESTAMP
           )'''
    )
    # Basic migration for older DBs: ensure extra columns exist
    c.execute("PRAGMA table_info(articles)")
    cols = [row[1] for row in c.fetchall()]
    needed = {
        "full_text": "TEXT",
        "relevance_score": "REAL",
        "red_flags": "TEXT",
    }
    for col, col_type in needed.items():
        if col not in cols:
            c.execute(f"ALTER TABLE articles ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


def clean_html(html_content: str) -> str:
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ").strip()


def fetch_full_article(url: str) -> str:
    """Attempt to download and extract the full article text."""
    try:
        headers = {
            "User-Agent": "SyriaWatchPulse/1.0 (+for analytical use)"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")

        # Prefer <article> tag if present
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        # Avoid returning extremely long blobs
        return text.strip()[:12000]
    except Exception:
        return ""


# ===============================
# ANALYTIC AGENTS
# ===============================

def analyze_text(text: str):
    """Keyword-based ontological classification for topics and actors."""
    text_lower = text.lower()

    # Determine Tags
    found_tags = []
    for category, keywords in TOPIC_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            found_tags.append(category)
    if not found_tags:
        found_tags.append("General")

    # Determine Actors
    found_actors = []
    for actor, keywords in ACTOR_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            found_actors.append(actor)

    return ", ".join(found_tags), ", ".join(found_actors)


def evaluate_source_profile(source_name: str):
    """Return structured metadata for a given source."""
    meta = SOURCE_PROFILES.get(source_name, None)
    if not meta:
        return {
            "alignment": "Unknown",
            "type": "Unknown",
            "note": "No profile available yet. Treat with standard source verification discipline.",
        }
    return meta


def evaluate_red_flags(text: str, source_name: str) -> str:
    """Heuristic red-flag detector for propaganda / weak sourcing."""
    text_lower = text.lower()
    flags = []

    propaganda_phrases = [
        "heroic resistance", "heroic people", "steadfast people", "steadfastness",
        "martyr", "martyrs", "glorious victory", "crushing blow",
        "zionist entity", "traitorous", "treacherous", "puppet regime",
        "takfiri", "crusader", "axis of resistance"
    ]

    if any(p in text_lower for p in propaganda_phrases):
        flags.append("Propaganda / highly loaded language")

    if "according to activists" in text_lower or "according to local sources" in text_lower or "sources said" in text_lower:
        flags.append("Vague / anonymous sourcing")

    if "unconfirmed" in text_lower or "could not be independently verified" in text_lower:
        flags.append("Explicitly unverified information")

    if "sana" in source_name.lower():
        flags.append("Official state outlet (high propaganda risk)")

    if not flags:
        return "None detected"
    return "; ".join(flags)


def evaluate_relevance(title: str, text: str, tags: str, actors: str) -> float:
    """Assign a relevance score 1–5 based on political weight and human impact."""
    text_all = (title + " " + (text or "")).lower()
    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    actors_list = [a.strip() for a in (actors or "").split(",") if a.strip()]

    score = 1.0

    # High-level diplomacy / heads of state
    if any(w in text_all for w in [
        "president", "head of state", "king ", "emir ", "prime minister",
        "secretary-general", "secretary general", "foreign minister",
        "summit", "high-level", "high level"
    ]):
        score += 2.0

    if "security council" in text_all or "un sc" in text_all:
        score += 1.5

    if "Political" in tags_list:
        score += 1.0

    # Ground developments with casualties
    if "Military/Ground" in tags_list:
        score += 0.8

    if any(w in text_all for w in ["dozens of", "scores of", "massacre", "killed", "deaths", "casualties"]):
        score += 0.7

    # Local / lower-level governance
    if any(w in text_all for w in ["mayor", "municipal", "local council", "village head"]):
        score -= 0.5

    if any(a in actors_list for a in [
        "Regime/SAA", "Russia", "Iran/Militias", "USA/Coalition",
        "Israel", "Turkey/SNA", "SDF/Kurdish", "HTS/Idlib"
    ]):
        score += 0.3

    # Clamp to [1,5]
    score = max(1.0, min(5.0, score))
    return score


# ===============================
# DATA PIPELINE
# ===============================

def fetch_and_process_feeds(lookback_hours: int = 72) -> int:
    """Fetch RSS, scrape full articles, classify, and store in DB."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    new_count = 0
    cutoff_date = datetime.now() - timedelta(hours=lookback_hours)

    for source in NEWS_SOURCES:
        try:
            feed = feedparser.parse(source['url'])
            for entry in feed.entries:
                # Parse Date
                try:
                    if hasattr(entry, 'published'):
                        pub_date = parser.parse(entry.published)
                    elif hasattr(entry, 'updated'):
                        pub_date = parser.parse(entry.updated)
                    else:
                        pub_date = datetime.now()
                except Exception:
                    pub_date = datetime.now()

                pub_date = pub_date.replace(tzinfo=None)

                # FILTER: timeframe window
                if pub_date < cutoff_date:
                    continue

                # Check duplicates
                link = getattr(entry, "link", None)
                if not link:
                    continue

                c.execute("SELECT link FROM articles WHERE link=?", (link,))
                if c.fetchone():
                    continue

                # Process summary
                summary_raw = entry.get('summary', entry.get('description', ''))
                summary = clean_html(summary_raw)

                # Full article scrape
                full_text = fetch_full_article(link)
                analysis_text = (entry.title or "") + " " + (full_text or summary)

                # Ontological tagging
                tags, actors = analyze_text(analysis_text)

                # Red flags & relevance
                red_flags = evaluate_red_flags(analysis_text, source['name'])
                relevance_score = evaluate_relevance(entry.title or "", analysis_text, tags, actors)

                # Insert into DB
                c.execute(
                    '''INSERT OR REPLACE INTO articles
                       (link, title, source, published_date, summary, full_text,
                        tags, actors, relevance_score, red_flags, fetched_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (
                        link,
                        entry.title,
                        source['name'],
                        pub_date,
                        summary,
                        full_text,
                        tags,
                        actors,
                        relevance_score,
                        red_flags,
                        datetime.now(),
                    ),
                )
                new_count += 1

        except Exception as e:
            print(f"Error parsing {source['name']}: {e}")

    conn.commit()
    conn.close()
    return new_count


def load_data(lookback_hours: int = 72) -> pd.DataFrame:
    """Load data from DB for a given lookback window."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    query = "SELECT * FROM articles WHERE published_date >= ? ORDER BY published_date DESC"
    df = pd.read_sql_query(query, conn, params=(cutoff,))
    conn.close()
    return df


def load_all_data() -> pd.DataFrame:
    """Load the full DB (backend view)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles ORDER BY published_date DESC", conn)
    conn.close()
    return df


def relevance_label(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 4.5:
        return "🔥 Critical"
    if score >= 3.5:
        return "High"
    if score >= 2.5:
        return "Medium"
    if score >= 1.5:
        return "Low"
    return "Very low"


# ===============================
# STREAMLIT APP
# ===============================

st.set_page_config(page_title="Syria Conflict News Monitor", layout="wide")

# Ensure DB exists
init_db()

st.title("🇸🇾 Syria Conflict News Monitor")
st.markdown("**Syria Watch Pulse — structured monitoring of multi-source Syria news.**")

# Sidebar controls
st.sidebar.header("Control Panel")
lookback_hours = st.sidebar.slider(
    "Lookback window (hours)", min_value=24, max_value=168, value=72, step=24
)

if st.sidebar.button("🔄 Refresh Data Now"):
    with st.spinner("Scraping sources, fetching full articles & running ontology/flag analysis..."):
        init_db()
        count = fetch_and_process_feeds(lookback_hours=lookback_hours)
    st.sidebar.success(f"Found {count} new articles in the last {lookback_hours}h.")

# Tabs for analyst vs backend views
tab1, tab2, tab3 = st.tabs(["📊 Analyst Dashboard", "🗄 Backend / Admin", "🛰 Source Intelligence"])

# -------------------------------
# TAB 1: ANALYST DASHBOARD
# -------------------------------
with tab1:
    try:
        df = load_data(lookback_hours=lookback_hours)
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        # Filters
        st.subheader("Filtered View")

        col_filters1, col_filters2 = st.columns(2)

        with col_filters1:
            all_sources = sorted(df['source'].dropna().unique().tolist())
            selected_sources = st.multiselect("Filter by source", all_sources, default=all_sources)

        with col_filters2:
            all_tags_set = set()
            for t in df['tags'].dropna().tolist():
                for part in str(t).split(","):
                    part = part.strip()
                    if part:
                        all_tags_set.add(part)
            all_tags = sorted(list(all_tags_set))
            selected_tags = st.multiselect("Filter by topic", all_tags)

        filtered_df = df.copy()
        if selected_sources:
            filtered_df = filtered_df[filtered_df['source'].isin(selected_sources)]
        if selected_tags:
            mask = filtered_df['tags'].fillna("").apply(
                lambda x: any(tag in x for tag in selected_tags)
            )
            filtered_df = filtered_df[mask]

        # Add human-readable relevance label
        filtered_df['relevance_label'] = filtered_df['relevance_score'].apply(relevance_label)

        # Stats row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Articles in window", len(filtered_df))
        c2.metric(
            "Military events",
            len(filtered_df[filtered_df['tags'].fillna("").str.contains("Military/Ground")]),
        )
        c3.metric(
            "Humanitarian",
            len(filtered_df[filtered_df['tags'].fillna("").str.contains("Humanitarian")]),
        )
        c4.metric(
            "Political",
            len(filtered_df[filtered_df['tags'].fillna("").str.contains("Political")]),
        )
        most_common_actor = "N/A"
        actors_series = filtered_df['actors'].replace("", pd.NA).dropna()
        if not actors_series.empty:
            actor_tokens = []
            for a in actors_series.tolist():
                actor_tokens.extend([x.strip() for x in str(a).split(",") if x.strip()])
            if actor_tokens:
                most_common_actor = pd.Series(actor_tokens).mode()[0]
        c5.metric("Most referenced actor", most_common_actor)

        st.markdown("---")

        # Display table
        display_df = filtered_df[[
            'published_date', 'source', 'tags', 'actors',
            'relevance_label', 'red_flags', 'title', 'link', 'summary'
        ]].copy()
        display_df.rename(
            columns={
                'published_date': 'Time',
                'source': 'Source',
                'tags': 'Type',
                'actors': 'Actors',
                'relevance_label': 'Relevance',
                'red_flags': 'Red flags',
                'title': 'Title',
                'link': 'Link',
                'summary': 'Summary',
            },
            inplace=True,
        )

        st.dataframe(
            display_df,
            column_config={
                "Link": st.column_config.LinkColumn("Link"),
                "Time": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm"),
                "Type": st.column_config.TextColumn("Type"),
                "Actors": st.column_config.TextColumn("Actors involved"),
                "Relevance": st.column_config.TextColumn("Relevance"),
                "Red flags": st.column_config.TextColumn("Analytic flags"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # CSV download of filtered view
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download filtered view as CSV",
            data=csv_data,
            file_name="syria_watch_pulse_filtered.csv",
            mime="text/csv",
        )

    else:
        st.info("Database is empty or no news in the selected window. Use the sidebar to refresh data.")


# -------------------------------
# TAB 2: BACKEND / ADMIN
# -------------------------------
with tab2:
    st.subheader("Database Overview")

    try:
        admin_df = load_all_data()
    except Exception:
        admin_df = pd.DataFrame()

    if not admin_df.empty:
        total_rows = len(admin_df)
        min_date = admin_df['published_date'].min()
        max_date = admin_df['published_date'].max()
        last_fetch = admin_df['fetched_at'].max()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total articles in DB", total_rows)
        c2.metric("Oldest article", str(min_date))
        c3.metric("Last fetch", str(last_fetch))

        st.markdown("### Recent 100 rows (raw)")
        st.dataframe(
            admin_df.head(100),
            hide_index=True,
            use_container_width=True,
        )

        # Full DB export
        csv_db = admin_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download full DB as CSV",
            data=csv_db,
            file_name="syria_watch_pulse_full_db.csv",
            mime="text/csv",
        )
    else:
        st.info("No data in the DB yet. Trigger a refresh from the sidebar.")

    st.markdown("---")
    st.markdown(
        """**Operational notes / best practices:**  
        - Treat red-flag and relevance scores as advisory, not definitive.  
        - Always cross-check critical items with multiple, distinct sources.  
        - Consider running the app on a schedule (e.g. cron/CI) to keep the DB fresh.  
        - Respect source terms of use and robots.txt if you extend scraping further."""
    )


# -------------------------------
# TAB 3: SOURCE INTELLIGENCE
# -------------------------------
with tab3:
    st.subheader("Source Profiles & Bias Hints")

    source_names = sorted([s['name'] for s in NEWS_SOURCES])
    selected_source_name = st.selectbox("Select a source", source_names)

    profile = evaluate_source_profile(selected_source_name)

    st.markdown(f"### {selected_source_name}")
    st.markdown(f"**Alignment:** {profile['alignment']}")
    st.markdown(f"**Type:** {profile['type']}")
    st.markdown(f"**Notes:** {profile['note']}")

    st.markdown("---")
    st.markdown(
        """This panel is intentionally conservative:  
        it encodes *expected* alignments and institutional roles, not truth/falsity.  
        Use it to structure your own media-triage, not to discard sources wholesale."""
    )
