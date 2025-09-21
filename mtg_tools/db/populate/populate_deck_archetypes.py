import sqlite3
import json

DB_PATH = "mtgcore.db"
DECK_FILE = "scraped_deck_data.json"
MAPPING_FILE = "archetype_mappings.json"

def populate_deck_archetypes():
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        archetype_map = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get deck_id by URL
    cur.execute("SELECT source_url, deck_id FROM decks")
    url_to_deck_id = dict(cur.fetchall())

    # Load scraped deck data
    with open(DECK_FILE, "r", encoding="utf-8") as f:
        deck_data = json.load(f)

    cur.execute("DELETE FROM deck_archetypes")

    rows = []
    unmatched = []

    for url, deck in deck_data.items():
        deck_id = url_to_deck_id.get(url)
        if not deck_id:
            continue

        raw = deck.get("weak_archetype", "")
        if isinstance(raw, list):
            weak_tag = " ".join(raw).strip()
        elif isinstance(raw, str):
            weak_tag = raw.strip()
        else:
            continue

        if not weak_tag:
            continue

        mapping = archetype_map.get(weak_tag)
        if not mapping:
            unmatched.append((deck_id, weak_tag))
            continue

        for tag in mapping.get("mapping_tag", []):
            rows.append((deck_id, tag, "raw"))

        for tag in mapping.get("manual_tag", []):
            rows.append((deck_id, tag, "manual"))

    cur.executemany("""
        INSERT OR IGNORE INTO deck_archetypes (deck_id, archetype_name, source)
        VALUES (?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()

    print(f"Inserted {len(rows)} archetype mappings.")
    print(f"Logged {len(unmatched)} unmatched archetypes.")
    with open("unmatched_archetypes.log", "w", encoding="utf-8") as f:
        for deck_id, tag in unmatched:
            f.write(f"{deck_id}\t{tag}\n")

if __name__ == "__main__":
    populate_deck_archetypes()
