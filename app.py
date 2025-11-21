import streamlit as st
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dateutil import parser
import sqlite3
import requests
from fpdf import FPDF
import re

# ===============================
# CONFIGURATION & ONTOLOGY
# ===============================

# Expanded Source List
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

# Ontology for Tags
TOPIC_KEYWORDS = {
    'Humanitarian': ['aid', 'refugee', 'camp', 'food', 'water', 'cholera', 'earthquake', 'unrwa', 'displacement', 'shelter', 'poverty', 'starvation'],
    'Military/Ground': ['shelling', 'clash', 'airstrike', 'air strike', 'bombing', 'killed', 'injured', 'attack', 'isis', 'islamic state', 'ied', 'drone', 'assassination', 'frontline', 'front line'],
    'Political': ['meeting', 'decree', 'election', 'minister', 'normalization', 'astana', 'geneva', 'un sc', 'security council', 'diplomacy', 'statement', 'agreement', 'talks', 'negotiation'],
    'Human Rights': ['arrest', 'torture', 'detainee', 'prison', 'detention', 'kidnap', 'kidnapping', 'execution', 'violation', 'forced', 'activist', 'enforced disappearance'],
}

# Ontology for Actors
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

# Source profiles
SOURCE_PROFILES = {
    'Enab Baladi': {'alignment': 'Independent / Opposition-leaning', 'type': 'Local Syrian outlet', 'note': 'Community-rooted media founded during the uprising, often critical of Damascus.'},
    'Zaman Al Wasl': {'alignment': 'Opposition', 'type': 'Syrian online newspaper', 'note': 'Carries opposition narratives and leaks, sometimes with strong political framing.'},
    'Syria Direct': {'alignment': 'Independent', 'type': 'Training-focused media NGO', 'note': 'Trains Syrian journalists; aims for explanatory reporting and local voices.'},
    'Halab Today': {'alignment': 'Opposition-leaning', 'type': 'TV / online outlet', 'note': 'Aleppo-origin station with strong focus on northern Syria.'},
    'North Press': {'alignment': 'AANES / SDF-leaning', 'type': 'Regional agency', 'note': 'Covers northeast Syria with a perspective close to local self-administration.'},
    'Hawar News': {'alignment': 'SDF-leaning', 'type': 'Agency / movement outlet', 'note': 'Often amplifies AANES/SDF narratives and official positions.'},
    'Rojava Info': {'alignment': 'Pro-Rojava', 'type': 'Research / information center', 'note': 'Long-form and investigative content on northeast Syria and Kurdish actors.'},
    'SANA (Gov)': {'alignment': 'Government / State', 'type': 'Official state agency', 'note': 'Formal voice of Damascus; strong official framing and propaganda risk.'},
    'Suwayda 24': {'alignment': 'Local / Community', 'type': 'Local news page', 'note': 'Hyper-local coverage of Suwayda governorate, protests, and security incidents.'},
    'DeirEzzor 24': {'alignment': 'Local / Opposition-leaning', 'type': 'Local network', 'note': 'Granular reporting on Deir Ezzor, often focused on SDF, ISIS cells, and tribal dynamics.'},
    'The Syrian Observer': {'alignment': 'Curated / Mixed', 'type': 'English-language aggregator', 'note': 'Curates translations from diverse Syrian press; mix of views but editorial choices matter.'},
}

DB_PATH = "syria_monitor.db"

# ===============================
# PDF GENERATION CLASS
# ===============================

class BriefingPDF(FPDF):
    def header(self):
        # Set margins explicitly to ensure space exists
        self.set_margins(15, 15, 15) 
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Syria Conflict Monitor | Political Affairs Briefing', ln=True, align='C')
        self.ln(2)
        # Draw line
        self.set_draw_color(0, 0, 0)
        self.line(15, 25, 195, 25)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Generated by Syria Watch Pulse', align='C')

    def chapter_title(self, label):
        self.ln(4)
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(235, 235, 235) # Light gray
        self.cell(0, 8, f"{label}", fill=True, ln=True)
        self.ln(2)

    def article_item(self, title, source, text):
        # Title line
        self.set_font('Helvetica', 'B', 10)
        self.multi_cell(0, 5, f"{title} [{source}]")
        
        # Body text
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5, text)
        
        # Subtle separator
        self.ln(2)
        self.set_draw_color(220, 220, 220)
        self.line(self.get_x(), self.get_y(), 195, self.get_y())
        self.ln(3)
        self.set_draw_color(0, 0, 0) # Reset to black

# ===============================
# PDF GENERATOR LOGIC (ANCHOR + FILL)
# ===============================

def sanitize_text_for_pdf(text):
    """Clean text to prevent FPDF crashes on special characters."""
    if not text: return ""
    # Map common smart quotes/dashes to ASCII
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...',
        '\u00A0': ' ', # Non-breaking space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Force Latin-1, replacing unknowns with '?' to guarantee PDF render
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_briefing(df: pd.DataFrame, max_items_per_section: int = 10) -> bytes:
    if df.empty:
        return None

    # 1. SORT & PREPARE
    # We sort by relevance to ensure the 'Top 1' is actually the most important
    df_sorted = df.sort_values(by='relevance_score', ascending=False).copy()
    
    # Budget for ~2 pages of dense text (Title + Body)
    # A safe limit for FPDF 2-pager is around 5000-5500 characters total text.
    CHAR_BUDGET = 5200 
    current_char_count = 0
    used_links = set()

    # Containers
    final_sections = {
        "Political Developments": [],
        "Situation on the Ground": [],
        "Humanitarian and Human Rights": []
    }

    # Helper for density
    def get_dense_summary(row):
        full = row['full_text']
        summary = row['summary']
        # Prefer full text, fallback to summary
        raw_text = full if (full and len(str(full)) > 150) else summary
        if not raw_text: return "No details available."
        
        # Clean newlines to save vertical space
        clean = raw_text.replace("\n", " ").replace("\r", "").strip()
        
        # Cap length to ensure we fit more news. 500 chars is a solid paragraph.
        limit = 550 
        if len(clean) > limit:
            return clean[:limit].rsplit(' ', 1)[0] + "..."
        return clean

    # ---------------------------------------------------------
    # STEP 1: THE ANCHORS (Guarantee 1 per section if available)
    # ---------------------------------------------------------
    
    # Create temporary filtered lists to find the leaders
    pol_candidates = df_sorted[df_sorted['tags'].astype(str).str.contains("Political")]
    gnd_candidates = df_sorted[df_sorted['tags'].astype(str).str.contains("Military/Ground")]
    hum_candidates = df_sorted[df_sorted['tags'].astype(str).str.contains("Humanitarian|Human Rights")]

    def add_item_to_section(row, section_key):
        nonlocal current_char_count
        link = row['link']
        if link in used_links: return
        
        summary = get_dense_summary(row)
        # Calculate cost
        cost = len(summary) + len(row['title']) + 50 # +50 for metadata/spacing
        
        final_sections[section_key].append((row, summary))
        used_links.add(link)
        current_char_count += cost

    # Add Top 1 Political
    if not pol_candidates.empty:
        add_item_to_section(pol_candidates.iloc[0], "Political Developments")
    
    # Add Top 1 Ground
    if not gnd_candidates.empty:
        add_item_to_section(gnd_candidates.iloc[0], "Situation on the Ground")
        
    # Add Top 1 Humanitarian
    if not hum_candidates.empty:
        add_item_to_section(hum_candidates.iloc[0], "Humanitarian and Human Rights")

    # ---------------------------------------------------------
    # STEP 2: THE FILL (Prioritize Political)
    # ---------------------------------------------------------
    
    # Iterate through the MAIN sorted list again to fill remaining budget
    for _, row in df_sorted.iterrows():
        if current_char_count >= CHAR_BUDGET:
            break
            
        link = row['link']
        if link in used_links: continue
        
        tags = str(row['tags'])
        
        # Logic: Determine Section
        target_section = None
        if "Political" in tags:
            target_section = "Political Developments"
        elif "Military/Ground" in tags:
            target_section = "Situation on the Ground"
        elif "Humanitarian" in tags or "Human Rights" in tags:
            target_section = "Humanitarian and Human Rights"
            
        if target_section:
            # LOGIC: BIAS TOWARDS POLITICAL
            is_political = (target_section == "Political Developments")
            
            # We always add Political if space exists.
            # We only add others if Political section is already "healthy" (>=2 items)
            # to satisfy the requirement "possibly more political".
            pol_count = len(final_sections["Political Developments"])
            
            should_add = False
            if is_political:
                should_add = True
            else:
                if pol_count >= 2:
                    should_add = True
            
            if should_add and len(final_sections[target_section]) < max_items_per_section:
                add_item_to_section(row, target_section)

    # ---------------------------------------------------------
    # STEP 3: RENDER PDF
    # ---------------------------------------------------------
    pdf = BriefingPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Header Info
    total_analyzed = len(df)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 5, f"Analysis Window: Past 72 Hours | Articles Processed: {total_analyzed}", ln=True)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", ln=True)

    sections_order = [
        "Political Developments",
        "Situation on the Ground",
        "Humanitarian and Human Rights"
    ]

    for section in sections_order:
        items = final_sections[section]
        pdf.chapter_title(section)
        
        if not items:
            pdf.set_font('Helvetica', 'I', 9)
            pdf.cell(0, 6, "No significant events detected in this category.", ln=True)
        else:
            for row, summary_text in items:
                # Sanitize before render
                safe_title = sanitize_text_for_pdf(row['title'])
                safe_source = sanitize_text_for_pdf(row['source'])
                safe_text = sanitize_text_for_pdf(summary_text)
                
                pdf.article_item(safe_title, safe_source, safe_text)
                
    return pdf.output(dest='S').encode('latin-1')


# ===============================
# DB INITIALIZATION & UTILITIES
# ===============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
    c.execute("PRAGMA table_info(articles)")
    cols = [row[1] for row in c.fetchall()]
    needed = {"full_text": "TEXT", "relevance_score": "REAL", "red_flags": "TEXT"}
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
    """Attempt to download and extract the full article text without limits."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return ""
        
        soup = BeautifulSoup(resp.text, "html.parser")
        article_tag = soup.find("article")
        
        if article_tag:
            paragraphs = article_tag.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        # Join with newlines for structure
        text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs)
        return text.strip()
    except Exception:
        return ""

# ===============================
# ANALYTIC AGENTS
# ===============================

def analyze_text(text: str):
    text_lower = text.lower()
    found_tags = []
    for category, keywords in TOPIC_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            found_tags.append(category)
    if not found_tags:
        found_tags.append("General")

    found_actors = []
    for actor, keywords in ACTOR_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            found_actors.append(actor)

    return ", ".join(found_tags), ", ".join(found_actors)

def evaluate_source_profile(source_name: str):
    meta = SOURCE_PROFILES.get(source_name, None)
    if not meta:
        return {"alignment": "Unknown", "type": "Unknown", "note": "No profile available."}
    return meta

def evaluate_red_flags(text: str, source_name: str) -> str:
    text_lower = text.lower()
    flags = []
    propaganda_phrases = ["heroic resistance", "martyr", "zionist entity", "terrorist gangs", "crusader"]
    
    if any(p in text_lower for p in propaganda_phrases):
        flags.append("Propaganda / loaded language")
    if "according to activists" in text_lower or "sources said" in text_lower:
        flags.append("Vague sourcing")
    if "sana" in source_name.lower():
        flags.append("Official state outlet (High propaganda risk)")

    if not flags:
        return "None detected"
    return "; ".join(flags)

def evaluate_relevance(title: str, text: str, tags: str, actors: str) -> float:
    text_all = (title + " " + (text or "")).lower()
    tags_list = str(tags).split(",")
    
    score = 1.0
    if any(w in text_all for w in ["president", "foreign minister", "summit", "un sc"]): score += 2.0
    if "Political" in tags_list: score += 1.0
    if "Military/Ground" in tags_list: score += 0.8
    if any(w in text_all for w in ["killed", "massacre", "casualties"]): score += 0.7
    
    return max(1.0, min(5.0, score))

# ===============================
# DATA PIPELINE
# ===============================

def fetch_and_process_feeds(lookback_hours: int = 72) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_count = 0
    cutoff_date = datetime.now() - timedelta(hours=lookback_hours)

    for source in NEWS_SOURCES:
        try:
            feed = feedparser.parse(source['url'])
            for entry in feed.entries:
                try:
                    if hasattr(entry, 'published'): pub_date = parser.parse(entry.published)
                    elif hasattr(entry, 'updated'): pub_date = parser.parse(entry.updated)
                    else: pub_date = datetime.now()
                except: pub_date = datetime.now()
                
                pub_date = pub_date.replace(tzinfo=None)
                if pub_date < cutoff_date: continue

                link = getattr(entry, "link", None)
                if not link: continue

                c.execute("SELECT link FROM articles WHERE link=?", (link,))
                if c.fetchone(): continue

                summary = clean_html(entry.get('summary', entry.get('description', '')))
                full_text = fetch_full_article(link)
                
                analysis_text = (entry.title or "") + " " + (full_text or summary)
                tags, actors = analyze_text(analysis_text)
                red_flags = evaluate_red_flags(analysis_text, source['name'])
                relevance_score = evaluate_relevance(entry.title or "", analysis_text, tags, actors)

                c.execute(
                    '''INSERT OR REPLACE INTO articles (link, title, source, published_date, summary, full_text, tags, actors, relevance_score, red_flags, fetched_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (link, entry.title, source['name'], pub_date, summary, full_text, tags, actors, relevance_score, red_flags, datetime.now())
                )
                new_count += 1
        except Exception as e:
            print(f"Error: {e}")

    conn.commit()
    conn.close()
    return new_count

def load_data(lookback_hours: int = 72) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    df = pd.read_sql_query("SELECT * FROM articles WHERE published_date >= ? ORDER BY published_date DESC", conn, params=(cutoff,))
    conn.close()
    return df

# ===============================
# STREAMLIT APP
# ===============================

st.set_page_config(page_title="Syria Conflict News Monitor", layout="wide")
init_db()

st.title("🇸🇾 Syria Conflict News Monitor")
st.markdown("**Syria Watch Pulse — structured monitoring of multi-source Syria news.**")

st.sidebar.header("Control Panel")
lookback_hours = st.sidebar.slider("Lookback window (hours)", 24, 168, 72, 24)

if st.sidebar.button("🔄 Refresh Data Now"):
    with st.spinner("Fetching full articles (unlimited) & analyzing..."):
        init_db()
        count = fetch_and_process_feeds(lookback_hours)
    st.sidebar.success(f"Found {count} new articles.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Analyst Dashboard", "🗄 Backend / Admin", "🛰 Source Intelligence", "📝 PDF Briefing"])

with tab1:
    df = load_data(lookback_hours)
    if not df.empty:
        st.subheader("Filtered View")
        all_sources = sorted(df['source'].dropna().unique().tolist())
        sel_sources = st.multiselect("Filter by source", all_sources, default=all_sources)
        
        if sel_sources: df = df[df['source'].isin(sel_sources)]
        
        df['relevance_label'] = df['relevance_score'].apply(lambda s: "🔥 Critical" if s>=4.5 else ("High" if s>=3.5 else "Medium"))
        
        display_df = df[['published_date','source','tags','relevance_label','title','link']].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data available.")

with tab2:
    st.subheader("Database Overview")
    conn = sqlite3.connect(DB_PATH)
    admin_df = pd.read_sql_query("SELECT * FROM articles ORDER BY published_date DESC LIMIT 100", conn)
    conn.close()
    st.dataframe(admin_df, use_container_width=True)

with tab3:
    st.subheader("Source Profiles")
    s_name = st.selectbox("Select Source", sorted([s['name'] for s in NEWS_SOURCES]))
    prof = evaluate_source_profile(s_name)
    st.json(prof)

with tab4:
    st.subheader("📝 Political Affairs Briefing Generator (PDF)")
    st.markdown("Generates a dense, 2-page style situational report categorized by sector.")
    
    if 'df' in locals() and not df.empty:
        c1, c2 = st.columns([3,1])
        with c2:
            max_items = st.number_input("Max items per section", 1, 8, 5)
            if st.button("📄 Generate PDF"):
                try:
                    pdf_bytes = generate_pdf_briefing(df, max_items)
                    st.session_state['pdf_data'] = pdf_bytes
                    st.success("PDF Generated!")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
        
        with c1:
            if 'pdf_data' in st.session_state:
                st.markdown("### Download Ready")
                st.download_button(
                    label="⬇️ Download Briefing PDF",
                    data=st.session_state['pdf_data'],
                    file_name=f"Syria_SitRep_{datetime.now().strftime('%Y-%m-%d')}.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("No data loaded.")
