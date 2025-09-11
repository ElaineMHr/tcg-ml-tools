import sqlite3
import json
import csv

DB_PATH = "mtgcore.db"
DECK_FILE = "scraped_deck_data.json"
ARCHETYPE_CSV = "Archetypes_Processed.csv"

def populate_deck_archetypes():
    raw_to_mapping = {}
    raw_to_manual = {}

    with open(ARCHETYPE_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["raw_archetype"].strip()
            mappings = [t.strip() for t in row["mapping_tag"].split(",") if t.strip()]
            manuals = [t.strip() for t in row["manual_tag"].split(",") if t.strip()]
            if raw:
                raw_to_mapping[raw.lower()] = mappings
                raw_to_manual[raw.lower()] = manuals

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT deck_id, source_url FROM decks")
    url_to_deck_id = dict(cur.fetchall())

    with open(DECK_FILE, "r", encoding="utf-8") as f:
        deck_data = json.load(f)

    cur.execute("DELETE FROM deck_archetypes")

    rows = []
    unmatched = []

    for url, deck in deck_data.items():
        deck_id = url_to_deck_id.get(url)
        if not deck_id:
            continue

        weak_tag = deck.get("weak_archetype", "").strip()
        if not weak_tag:
            continue

        key = weak_tag.lower()
        mapping_tags = raw_to_mapping.get(key)
        if not mapping_tags:
            unmatched.append((deck_id, weak_tag))
            continue

        for tag in mapping_tags:
            rows.append((deck_id, tag, "raw"))

        for tag in raw_to_manual.get(key, []):
            rows.append((deck_id, tag, "manual"))

    cur.executemany("""
        INSERT OR IGNORE INTO deck_archetypes (deck_id, archetype_name, source)
        VALUES (?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(rows)} archetype mappings.")
    print(f"⚠️ {len(unmatched)} unmatched archetypes logged.")

    with open("unmatched_archetypes.log", "w", encoding="utf-8") as f:
        for deck_id, tag in unmatched:
            f.write(f"{deck_id}\t{tag}\n")

if __name__ == "__main__":
    populate_deck_archetypes()
