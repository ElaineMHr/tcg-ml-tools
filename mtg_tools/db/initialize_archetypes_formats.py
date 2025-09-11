import sqlite3
import csv

DB_PATH = "mtgcore.db"
CSV_PATH = "Archetypes_Processed.csv"

FORMATS = [
    "Standard", "Alchemy", "Pioneer", "Explorer", "Modern", "Historic", "Legacy",
    "Brawl", "Vintage", "Timeless", "Commander", "Pauper", "Oathbreaker", "Penny"
]

# --- Connect ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# --- Create tables ---
cur.executescript("""

CREATE TABLE IF NOT EXISTS formats (
    format_name TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS deck_formats (
    deck_id TEXT,
    format_name TEXT
);

CREATE TABLE IF NOT EXISTS archetypes (
    name TEXT PRIMARY KEY,
    is_main BOOLEAN DEFAULT 0,
    is_special BOOLEAN DEFAULT 0,
    is_obsolete BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS deck_archetypes (
    deck_id TEXT,
    archetype_name TEXT,
    source TEXT CHECK(source IN ('raw', 'manual', 'model'))
);
""")

# --- Insert formats ---
cur.executemany("INSERT OR IGNORE INTO formats (format_name) VALUES (?)", [(f,) for f in FORMATS])

# --- Parse Archetypes CSV ---
archetype_flags = {}

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # All possible tag columns
        mapping_tags = [t.strip() for t in row['mapping_tag'].split(",") if t.strip()]
        manual_tags = [t.strip() for t in row['manual_tag'].split(",") if t.strip()]
        model_tags = [t.strip() for t in row['model_tag'].split(",") if t.strip()]
        main_tags = [t.strip() for t in row['main_tag'].split(",") if t.strip()]
        special_tags = [t.strip() for t in row['special_tag'].split(",") if t.strip()]

        for tag in mapping_tags + manual_tags + model_tags:
            archetype_flags.setdefault(tag, {'main': False, 'special': False})

        for tag in main_tags:
            archetype_flags.setdefault(tag, {'main': False, 'special': False})
            archetype_flags[tag]['main'] = True

        for tag in special_tags:
            archetype_flags.setdefault(tag, {'main': False, 'special': False})
            archetype_flags[tag]['special'] = True

# --- Insert archetypes ---
for tag, flags in archetype_flags.items():
    cur.execute("""
        INSERT OR IGNORE INTO archetypes (name, is_main, is_special)
        VALUES (?, ?, ?)
    """, (tag, int(flags['main']), int(flags['special'])))

conn.commit()
conn.close()
print("Formats and archetypes initialized with raw/manual/model/mapping/main/special tags.")
