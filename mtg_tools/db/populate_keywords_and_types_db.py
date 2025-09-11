import json
import sqlite3
from pathlib import Path

def populate_keywords_and_types(db_path="mtgcore.db", data_dir="./keywords and types"):
    data_path = Path(data_dir)

    with open(data_path / "keyword-abilities.json", "r", encoding="utf-8") as f:
        keyword_abilities = json.load(f)
    with open(data_path / "keyword-actions.json", "r", encoding="utf-8") as f:
        keyword_actions = json.load(f)
    with open(data_path / "ability-words.json", "r", encoding="utf-8") as f:
        ability_words = json.load(f)

    keywords = [(kw, "ability") for kw in keyword_abilities] + \
               [(kw, "action") for kw in keyword_actions] + \
               [(kw, "word") for kw in ability_words]

    type_categories = {
        "supertypes": "supertype",
        "card-types": "type",
        "artifact-types": "subtype",
        "battle-types": "subtype",
        "creature-types": "subtype",
        "enchantment-types": "subtype",
        "land-types": "subtype",
        "planeswalker-types": "subtype",
        "spell-types": "subtype"
    }

    types = []
    for filename, category in type_categories.items():
        path = data_path / f"{filename}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                values = json.load(f)
                for value in values:
                    parent = None
                    if category == "subtype":
                        if "creature" in filename:
                            parent = "Creature"
                        elif "artifact" in filename:
                            parent = "Artifact"
                        elif "enchantment" in filename:
                            parent = "Enchantment"
                        elif "land" in filename:
                            parent = "Land"
                        elif "planeswalker" in filename:
                            parent = "Planeswalker"
                        elif "spell" in filename:
                            parent = "Spell"
                        elif "battle" in filename:
                            parent = "Battle"
                    types.append((value, category, parent))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS keywords (
        name TEXT PRIMARY KEY,
        source TEXT
    );

    CREATE TABLE IF NOT EXISTS types (
        name TEXT PRIMARY KEY,
        category TEXT CHECK(category IN ('supertype', 'type', 'subtype')),
        parent_type TEXT
    );

    DELETE FROM keywords;
    DELETE FROM types;
    """)

    cur.executemany("INSERT OR IGNORE INTO keywords (name, source) VALUES (?, ?)", keywords)
    cur.executemany("INSERT OR IGNORE INTO types (name, category, parent_type) VALUES (?, ?, ?)", types)

    conn.commit()
    conn.close()
    print("Keywords and types populated successfully.")

if __name__ == "__main__":
    populate_keywords_and_types()
