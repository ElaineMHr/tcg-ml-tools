import sqlite3

def create_deck_stats_table(db_path="mtgcore.db"):
    schema = """
    CREATE TABLE IF NOT EXISTS deck_stats (
        deck_id TEXT PRIMARY KEY,
        avg_cmc REAL,
        cmc_distribution TEXT,        -- JSON: {"0": 1, "1": 5, ..., "7+": 2}
        color_identity TEXT,          -- JSON: {"W": 5, "U": 3, ...}
        color_tag TEXT,               -- "mono-red", "dimir", "naya", "five-color", "colorless"
        has_companion TEXT,           -- Name or NULL
        main_tribe TEXT,              -- e.g., "Elf"
        tribe_percent REAL,           -- % of creatures with that tribe
        dominant_type TEXT,           -- e.g., "Creature", "Enchantment"
        common_keywords TEXT,         -- JSON: {"Flying": 12, "Trample": 4}
        common_types TEXT,            -- JSON: {"Elf": 10, "Goblin": 3}
        common_phrases TEXT           -- JSON: {"draw_cards": 6, "create_tokens": 4}
    );
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(schema)
    conn.commit()
    conn.close()
    print("✅ deck_stats table created.")

if __name__ == "__main__":
    create_deck_stats_table()
