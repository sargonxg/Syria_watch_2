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

NEWS_SOURCES = [
    {'id': 'enab', 'name': 'Enab Baladi', 'url': 'https://english.enabbaladi.net/feed/'},
    {'id': 'zaman', 'name': 'Zaman Al Wasl', 'url': 'https://en.zamanalwsl.net/rss.php'},
    {'id': 'direct', 'name': 'Syria Direct', 'url': 'https://syriadirect.org/feed/'},
    {'id': 'halab', 'name': 'Halab Today', 'url': 'https://halabtodaytv.net/feed'},
    {'id': 'npa', 'name': 'North Press', 'url': 'https://npasyria.com/en/feed/'},
    {'id': 'hawar', 'name': 'Hawar News', 'url': 'https://hawarnews.com/en/feed/'},
    {'id': 'rojava', 'name': 'Rojava Info', 'url': 'https://rojavainformationcenter.org/feed/'},
    {'id': 'sana', 'name': 'SANA (Gov)', 'url': 'https://sana.sy/en/?feed=rss2'},
    {'id': 'suwayda', 'name': 'Suwayda 24', 'url': 'https://suwayda24.com/feed/'},
    {'id': 'deir', 'name': 'DeirEzzor 24', 'url': 'https://deirezzor24.net/en/feed/'},
    {'id': 'observer', 'name': 'The Syrian Observer', 'url': 'https://syrianobserver.com/feed'},
]

TOPIC_KEYWORDS = {
    'Humanitarian': ['aid', 'refugee', 'camp', 'food', 'water', 'cholera', 'earthquake', 'unrwa', 'displacement', 'shelter', 'poverty', 'starvation'],
    'Military/Ground': ['shelling', 'clash', 'airstrike', 'air strike', 'bombing', 'killed', 'injured', 'attack', 'isis', 'islamic state', 'ied', 'drone', 'assassination', 'frontline', 'front line'],
    'Political': ['meeting', 'decree', 'election', 'minister', 'normalization', 'astana', 'geneva', 'un sc', 'security council', 'diplomacy', 'statement', 'agreement', 'talks', 'negotiation'],
    'Human Rights': ['arrest', 'torture', 'detainee', 'prison', 'detention', 'kidnap', 'kidnapping', 'execution', 'violation', 'forced', 'activist', 'enforced disappearance'],
}

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
        # NOTE: Do NOT set margins here. They are set in __init__.
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Syria Conflict Monitor | Political Affairs Briefing', ln=True, align='C')
        self.ln(2)
        self.set_draw_color(0, 0, 0)
        self.line(10, 25, 200, 25) # Hardcoded line coordinates to be safe
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Generated by Syria Watch Pulse', align='C')

    def chapter_title(self, label):
        self.ln(4)
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(235, 235, 235)
        # Use epw (effective page width) to ensure no overflow
        self.cell(w=0, h=8, txt=f"{label}", fill=True, ln=True, align='L')
        self.ln(2)

    def article_item(self, title, source, text):
        # Check for empty text to avoid weird FPDF behavior
        if not text: text = "No summary available."
        
        self.set_font('Helvetica', 'B', 10)
        # Multi_cell with w=0 uses full available width
        self.multi_cell(w=0, h=5, txt=f"{title} [{source}]", align='L')
        
        self.set_font('Helvetica', '', 9)
        self.multi_cell(w=0, h=5, txt=text, align='L')
        
        self.ln(2)
        self.set_draw_color(220, 220, 220)
        self.line(self.get_x(), self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_draw_color(0, 0, 0)

# ===============================
# PDF GENERATOR LOGIC
# ===============================

def sanitize_text_for_pdf(text):
    """Aggressively clean text to prevent layout crashes."""
    if not text: return ""
    
    # 1. Replace Non-Breaking Spaces (The #1 cause of 'not enough horizontal space')
    text = text.replace('\u00A0', ' ')
    
    # 2. Map common smart quotes/dashes to ASCII
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...',
        '\t': ' ' # Remove tabs
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # 3. Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Force Latin-1 compatible, replace errors with ?
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_briefing(df: pd.DataFrame, max_items_per_section: int = 10) -> bytes:
    if df.empty: return None

    df_sorted = df.sort_values(by='relevance_score', ascending=False).copy()
    
    # Budget: Approx 5200 characters for 2 dense pages
    CHAR_BUDGET = 5200 
    current_char_count = 0
    used_links = set()

    final_sections = {
        "Political Developments": [],
        "Situation on the Ground": [],
        "Humanitarian and Human Rights": []
    }

    def get_dense_summary(row):
        full = row['full_text']
        summary = row['summary']
        raw_text = full if (full and len(str(full)) > 150) else summary
        if not raw_text: return "No details available."
        
        # Clean up
        clean = raw_text.replace("\n", " ").replace("\r", "").strip()
        
        # Hard cap to prevent one article eating the whole page
        limit = 550 
        if len(clean) > limit:
            return clean[:limit].rsplit(' ', 1)[0] + "..."
        return clean

    # --- STEP 1: GUARANTEED ANCHORS (1 per section) ---
    pol_candidates = df_sorted[df_sorted['tags'].astype(str).str.contains("Political")]
    gnd_candidates = df_sorted[df_sorted['tags'].astype(str).str.contains("Military/Ground")]
    hum_candidates = df_sorted[df_sorted['tags'].astype(str).str.contains("Humanitarian|Human Rights")]

    def add_item(row, section_key):
        nonlocal current_char_count
        link = row['link']
        if link in used_links: return
        
        summary = get_dense_summary(row)
        cost = len(summary) + len(row['title']) + 50
        final_sections[section_key].append((row, summary))
        used_links.add(link)
        current_char_count += cost

    if not pol_candidates.empty: add_item(pol_candidates.iloc[0], "Political Developments")
    if not gnd_candidates.empty: add_item(gnd_candidates.iloc[0], "Situation on the Ground")
    if not hum_candidates.empty: add_item(hum_candidates.iloc[0], "Humanitarian and Human Rights")

    # --- STEP 2: WEIGHTED FILL (Prioritize Political) ---
    for _, row in df_sorted.iterrows():
        if current_char_count >= CHAR_BUDGET: break
        
        link = row['link']
        if link in used_links: continue
        
        tags = str(row['tags'])
        target = None
        if "Political" in tags: target = "Political Developments"
        elif "Military/Ground" in tags: target = "Situation on the Ground"
        elif "Humanitarian" in tags or "Human Rights" in tags: target = "Humanitarian and Human Rights"
            
        if target:
            # Policy: Always add Political. Add others only if Political section is healthy (>=2)
            should_add = False
            if target == "Political Developments": should_add = True
            elif len(final_sections["Political Developments"]) >= 2: should_add = True
            
            if should_add and len(final_sections[target]) < max_items_per_section:
                add_item(row, target)

    # --- STEP 3: RENDER ---
    # Initialize FPDF with explicit format to prevent layout errors
    pdf = BriefingPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=10, top=15, right=10) # Set margins ONCE here
    
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Meta Header
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 5, f"Window: 72h | Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", ln=True)

    sections_order = ["Political Developments", "Situation on the Ground", "Humanitarian and Human Rights"]

    for section in sections_order:
        items = final_sections[section]
        pdf.chapter_title(section)
        
        if not items:
            pdf.set_font('Helvetica', 'I', 9)
            pdf.cell(0, 6, "No significant events.", ln=True)
        else:
            for row, summary_text in items:
                s_title = sanitize_text_for_pdf(row['title'])
                s_source = sanitize_text_for_pdf(row['source'])
                s_text = sanitize_text_for_pdf(summary_text)
                pdf.article_item(s_title, s_source, s_text)
                
    return pdf.output(dest='S').encode('latin-1')

# ===============================
# DB INITIALIZATION & UTILITIES
# ===============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles (link TEXT PRIMARY KEY, title TEXT, source TEXT, published_date TIMESTAMP, summary TEXT, full_text TEXT, tags TEXT, actors TEXT, relevance_score REAL, red_flags TEXT, fetched_at TIMESTAMP)''')
    c.execute("PRAGMA table_info(articles)")
    cols = [row[1] for row in c.fetchall()]
    needed = {"full_text": "TEXT", "relevance_score": "REAL", "red_flags": "TEXT"}
    for col, col_type in needed.items():
        if col not in cols: c.execute(f"ALTER TABLE articles ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()

def clean_html(html_content: str) -> str:
    if not html_content: return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ").strip()

def fetch_full_article(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200: return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        article_tag = soup.find("article")
        paragraphs = article_tag.find_all("p") if article_tag else soup.find_all("p")
        return "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs).strip()
    except: return ""

# ===============================
# ANALYTICS
# ===============================

def analyze_text(text: str):
    text_lower = text.lower()
    tags = [cat for cat, kws in TOPIC_KEYWORDS.items() if any(k in text_lower for k in kws)]
    actors = [act for act, kws in ACTOR_KEYWORDS.items() if any(k in text_lower for k in kws)]
    if not tags: tags.append("General")
    return ", ".join(tags), ", ".join(actors)

def evaluate_red_flags(text: str, source_name: str) -> str:
    text_lower = text.lower()
    flags = []
    if any(p in text_lower for p in ["heroic resistance", "martyr", "zionist entity", "crusader"]):
        flags.append("Propaganda")
    if "sources said" in text_lower: flags.append("Vague sourcing")
    if "sana" in source_name.lower(): flags.append("State Media")
    return "; ".join(flags) if flags else "None"

def evaluate_relevance(title: str, text: str, tags: str, actors: str) -> float:
    text_all = (title + " " + (text or "")).lower()
    score = 1.0
    if any(w in text_all for w in ["president", "un sc", "geneva"]): score += 2.0
    if "Political" in tags: score += 1.0
    if "Military/Ground" in tags: score += 0.8
    if any(w in text_all for w in ["killed", "massacre"]): score += 0.7
    return max(1.0, min(5.0, score))

def evaluate_source_profile(source_name: str):
    return SOURCE_PROFILES.get(source_name, {"alignment": "Unknown", "type": "Unknown", "note": "No profile."})

# ===============================
# DATA PIPELINE
# ===============================

def fetch_and_process_feeds(lookback_hours: int = 72) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_count = 0
    cutoff = datetime.now() - timedelta(hours=lookback_hours)

    for source in NEWS_SOURCES:
        try:
            feed = feedparser.parse(source['url'])
            for entry in feed.entries:
                try: pub_date = parser.parse(entry.get('published', entry.get('updated', str(datetime.now())))).replace(tzinfo=None)
                except: pub_date = datetime.now()
                
                if pub_date < cutoff: continue
                link = getattr(entry, "link", None)
                if not link: continue
                
                c.execute("SELECT link FROM articles WHERE link=?", (link,))
                if c.fetchone(): continue
                
                summary = clean_html(entry.get('summary', entry.get('description', '')))
                full_text = fetch_full_article(link)
                
                analysis_text = (entry.title or "") + " " + (full_text or summary)
                tags, actors = analyze_text(analysis_text)
                red_flags = evaluate_red_flags(analysis_text, source['name'])
                score = evaluate_relevance(entry.title or "", analysis_text, tags, actors)

                c.execute("INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?,?,?)", 
                          (link, entry.title, source['name'], pub_date, summary, full_text, tags, actors, score, red_flags, datetime.now()))
                new_count += 1
        except: pass
    conn.commit()
    conn.close()
    return new_count

def load_data(lookback_hours: int = 72) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles WHERE published_date >= ? ORDER BY published_date DESC", conn, params=(datetime.now() - timedelta(hours=lookback_hours),))
    conn.close()
    return df

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="Syria Conflict News Monitor", layout="wide")
init_db()

st.title("🇸🇾 Syria Conflict News Monitor")
st.sidebar.header("Control Panel")
lookback = st.sidebar.slider("Window (Hours)", 24, 168, 72, 24)

if st.sidebar.button("🔄 Refresh Data"):
    with st.spinner("Ingesting & Analyzing..."):
        init_db()
        count = fetch_and_process_feeds(lookback)
    st.sidebar.success(f"Fetched {count} new articles.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🗄 Admin", "🛰 Sources", "📝 PDF Briefing"])

with tab1:
    df = load_data(lookback)
    if not df.empty:
        st.subheader("Filtered View")
        sources = st.multiselect("Filter Source", sorted(df['source'].unique()), default=sorted(df['source'].unique()))
        if sources: df = df[df['source'].isin(sources)]
        
        df['Label'] = df['relevance_score'].apply(lambda x: "🔥 Critical" if x>=4.5 else ("High" if x>=3.5 else "Medium"))
        st.dataframe(df[['published_date','source','tags','Label','title','link']], use_container_width=True, hide_index=True)
    else: st.info("No data.")

with tab2:
    conn = sqlite3.connect(DB_PATH)
    st.dataframe(pd.read_sql_query("SELECT * FROM articles ORDER BY published_date DESC LIMIT 50", conn), use_container_width=True)
    conn.close()

with tab3:
    s = st.selectbox("Source Profile", sorted([x['name'] for x in NEWS_SOURCES]))
    st.json(evaluate_source_profile(s))

with tab4:
    st.subheader("📝 Political Affairs Briefing (PDF)")
    if 'df' in locals() and not df.empty:
        c1, c2 = st.columns([3,1])
        with c2:
            if st.button("Generate PDF"):
                try:
                    pdf_data = generate_pdf_briefing(df)
                    st.session_state['pdf'] = pdf_data
                    st.success("Success!")
                except Exception as e: st.error(f"Error: {e}")
        with c1:
            if 'pdf' in st.session_state:
                st.download_button("⬇️ Download PDF", st.session_state['pdf'], f"Syria_SitRep_{datetime.now().date()}.pdf", "application/pdf")
    else: st.warning("Load data first.")
