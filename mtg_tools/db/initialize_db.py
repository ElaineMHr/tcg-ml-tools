import sqlite3

def initialize_mtg_database(db_path="mtgcore.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
    -- CARDS
    CREATE TABLE IF NOT EXISTS cards (
        card_id TEXT PRIMARY KEY,
        oracle_id TEXT NOT NULL,
        name TEXT NOT NULL,
        layout TEXT,
        cmc REAL,
        color_identity TEXT,
        is_alchemy BOOLEAN DEFAULT FALSE,
        base_oracle_id TEXT
    );

    CREATE TABLE IF NOT EXISTS card_faces (
        face_id TEXT PRIMARY KEY,
        card_id TEXT NOT NULL,
        face_index INTEGER,
        name TEXT,
        mana_cost TEXT,
        type_line TEXT,
        oracle_text TEXT,
        power TEXT,
        toughness TEXT,
        loyalty TEXT,
        produced_mana TEXT,
        image_art_crop TEXT,
        image_large TEXT,
        artist TEXT
    );

    CREATE TABLE IF NOT EXISTS card_printings (
        printing_id TEXT PRIMARY KEY,
        card_id TEXT NOT NULL,
        set_code TEXT,
        collector_number TEXT,
        language TEXT DEFAULT 'en',
        rarity TEXT,
        frame TEXT,
        frame_effect TEXT,
        full_art BOOLEAN,
        promo BOOLEAN,
        border_color TEXT,
        watermark TEXT,
        released_at TEXT,
        artist TEXT
    );

    CREATE TABLE IF NOT EXISTS card_types (
        face_id TEXT NOT NULL,
        type_value TEXT NOT NULL,
        type_category TEXT CHECK(type_category IN ('supertype', 'type', 'subtype'))
    );

    CREATE TABLE IF NOT EXISTS keywords (
        name TEXT PRIMARY KEY,
        source TEXT
    );

    CREATE TABLE IF NOT EXISTS card_keywords (
        face_id TEXT NOT NULL,
        keyword TEXT NOT NULL,
        keyword_class TEXT
    );

    CREATE TABLE IF NOT EXISTS types (
        name TEXT PRIMARY KEY,
        category TEXT CHECK(category IN ('supertype', 'type', 'subtype')),
        parent_type TEXT
    );

    -- DECKS
    CREATE TABLE IF NOT EXISTS decks (
        deck_id TEXT PRIMARY KEY,
        deck_hash TEXT,
        source TEXT,
        source_url TEXT UNIQUE,
        main_color_identity TEXT,
        companion_id TEXT,
        deck_title TEXT
    );

    CREATE TABLE IF NOT EXISTS deck_cards_maindeck (
        deck_id TEXT NOT NULL,
        card_id TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY (deck_id, card_id)
    );

    CREATE TABLE IF NOT EXISTS deck_cards_sideboard (
        deck_id TEXT NOT NULL,
        card_id TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY (deck_id, card_id)
    );

    CREATE TABLE IF NOT EXISTS decks_deduplicated (
        deck_hash TEXT PRIMARY KEY,
        representative_deck_id TEXT
    );
    
    -- DECK STATS
    CREATE TABLE IF NOT EXISTS deck_stats (
        deck_id        TEXT PRIMARY KEY,
        avg_cmc        REAL NOT NULL,
        main_tribe     TEXT,
        dominant_type  TEXT,

        cmc_0          INTEGER DEFAULT 0,
        cmc_1          INTEGER DEFAULT 0,
        cmc_2          INTEGER DEFAULT 0,
        cmc_3          INTEGER DEFAULT 0,
        cmc_4          INTEGER DEFAULT 0,
        cmc_5          INTEGER DEFAULT 0,
        cmc_6          INTEGER DEFAULT 0,
        cmc_7_plus     INTEGER DEFAULT 0,

        color_W        INTEGER DEFAULT 0,
        color_U        INTEGER DEFAULT 0,
        color_B        INTEGER DEFAULT 0,
        color_R        INTEGER DEFAULT 0,
        color_G        INTEGER DEFAULT 0,
        color_C        INTEGER DEFAULT 0,

        features_version TEXT DEFAULT 'v1',
        generated_at     TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    -- ARCHETYPE AND FORMAT
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

    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_mtg_database()
    print("Database initialized successfully.")
