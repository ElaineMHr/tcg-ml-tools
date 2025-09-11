# Database Schema (SQLite)

This document summarizes the tables created and populated by scripts in `mtg_tools/db/`.

Sources examined:
- `mtg_tools/db/initialize_cards_decks.py`
- `mtg_tools/db/initialize_archetypes_formats.py`
- `mtg_tools/db/initialize_deck_stats.py`
- `mtg_tools/db/populate/populate_decks.py` (for usage expectations)
- `mtg_tools/db/populate_cards.py` and `mtg_tools/db/populate_cards_db.py`
- `mtg_tools/db/populate_keywords_and_types_db.py`
- `mtg_tools/db/populate deck_formats.py`

Notes:
- The schema does not declare foreign keys, but relationships are implied and noted below.
- Several columns store JSON-encoded data in `TEXT` fields (documented per table).
- There are two alternative definitions of `deck_stats` across scripts; both are documented.

## Cards and Types

Table: `cards` (from `initialize_cards_decks.py`)
- card_id: TEXT, PRIMARY KEY
- oracle_id: TEXT, NOT NULL
- name: TEXT, NOT NULL
- layout: TEXT
- cmc: REAL
- color_identity: TEXT  (comma-separated W,U,B,R,G)
- is_alchemy: BOOLEAN DEFAULT FALSE
- base_oracle_id: TEXT  (oracle id of base card for Alchemy variants)

Table: `card_faces`
- face_id: TEXT, PRIMARY KEY
- card_id: TEXT, NOT NULL  (→ cards.card_id)
- face_index: INTEGER
- name: TEXT
- mana_cost: TEXT
- type_line: TEXT
- oracle_text: TEXT
- power: TEXT
- toughness: TEXT
- loyalty: TEXT
- produced_mana: TEXT  (comma-separated mana symbols)
- image_art_crop: TEXT  (relative image path if present)
- image_large: TEXT     (relative image path if present)
- artist: TEXT

Table: `card_printings`
- printing_id: TEXT, PRIMARY KEY  (often same as card_id for canonical pick)
- card_id: TEXT, NOT NULL         (→ cards.card_id)
- set_code: TEXT
- collector_number: TEXT
- language: TEXT DEFAULT 'en'
- rarity: TEXT
- frame: TEXT
- frame_effect: TEXT              (comma-separated list)
- full_art: BOOLEAN
- promo: BOOLEAN
- border_color: TEXT
- watermark: TEXT
- released_at: TEXT               (YYYY-MM-DD)
- artist: TEXT

Table: `card_types`
- face_id: TEXT, NOT NULL         (→ card_faces.face_id)
- type_value: TEXT, NOT NULL
- type_category: TEXT CHECK in ('supertype','type','subtype')

Table: `keywords` (populated from JSON catalogs)
- name: TEXT, PRIMARY KEY
- source: TEXT                    (e.g., 'ability','action','word')

Table: `card_keywords`
- face_id: TEXT, NOT NULL         (→ card_faces.face_id)
- keyword: TEXT, NOT NULL         (→ keywords.name)
- keyword_class: TEXT             (optional classifier)

Table: `types` (populated from JSON catalogs)
- name: TEXT, PRIMARY KEY
- category: TEXT CHECK in ('supertype','type','subtype')
- parent_type: TEXT               (e.g., 'Creature' for creature subtypes)

## Decks and Cards-in-Decks

Table: `decks` (from `initialize_cards_decks.py`)
- deck_id: TEXT, PRIMARY KEY
- deck_hash: TEXT                 (content hash of main deck)
- source: TEXT                    (scraper/source identifier)
- source_url: TEXT, UNIQUE        (original URL)
- main_color_identity: TEXT
- companion_id: TEXT              (→ cards.card_id if resolved)
- deck_title: TEXT

Table: `deck_cards_maindeck`
- deck_id: TEXT, NOT NULL         (→ decks.deck_id)
- card_id: TEXT, NOT NULL         (→ cards.card_id)
- count: INTEGER, NOT NULL
- PRIMARY KEY (deck_id, card_id)

Table: `deck_cards_sideboard`
- deck_id: TEXT, NOT NULL         (→ decks.deck_id)
- card_id: TEXT, NOT NULL         (→ cards.card_id)
- count: INTEGER, NOT NULL
- PRIMARY KEY (deck_id, card_id)

Table: `decks_deduplicated`
- deck_hash: TEXT, PRIMARY KEY
- representative_deck_id: TEXT    (→ decks.deck_id)

## Formats and Archetypes

Table: `formats` (from `initialize_archetypes_formats.py`)
- format_name: TEXT, PRIMARY KEY  (e.g., 'Standard','Modern',...)

Table: `deck_formats`
- deck_id: TEXT                   (→ decks.deck_id)
- format_name: TEXT               (→ formats.format_name)

Table: `archetypes`
- name: TEXT, PRIMARY KEY
- is_main: BOOLEAN DEFAULT 0
- is_special: BOOLEAN DEFAULT 0
- is_obsolete: BOOLEAN DEFAULT 0

Table: `deck_archetypes`
- deck_id: TEXT                   (→ decks.deck_id)
- archetype_name: TEXT            (→ archetypes.name)
- source: TEXT CHECK in ('raw','manual','model')

## Deck Statistics (two variants in code)

Variant A: defined in `initialize_cards_decks.py`
- deck_id: TEXT, PRIMARY KEY      (→ decks.deck_id)
- avg_cmc: REAL
- drop_curve: TEXT                (JSON)
- type_counts: TEXT               (JSON)
- keyword_counts: TEXT            (JSON)
- dominant_types: TEXT            (JSON)
- dominant_keywords: TEXT         (JSON)
- color_counts: TEXT              (JSON)
- color_percentages: TEXT         (JSON)
- has_splash: BOOLEAN DEFAULT FALSE
- creature_count: INTEGER DEFAULT 0
- noncreature_count: INTEGER DEFAULT 0

Variant B: defined in `initialize_deck_stats.py`
- deck_id: TEXT, PRIMARY KEY      (→ decks.deck_id)
- avg_cmc: REAL
- cmc_distribution: TEXT          (JSON like {"0":1,...,"7+":2})
- color_identity: TEXT            (JSON map of WUBRG counts)
- color_tag: TEXT                 (e.g., 'mono-red','dimir','naya','five-color','colorless')
- has_companion: TEXT             (companion name or NULL)
- main_tribe: TEXT                (e.g., 'Elf')
- tribe_percent: REAL             (% of creatures in main tribe)
- dominant_type: TEXT             (e.g., 'Creature','Enchantment')
- common_keywords: TEXT           (JSON frequency map)
- common_types: TEXT              (JSON frequency map)
- common_phrases: TEXT            (JSON frequency map)

Implementation note: These scripts create the same table name `deck_stats` with different columns. Use one variant consistently in your pipeline to avoid mismatches.

## Populators and Expectations

Cards (`populate_cards.py` / `populate_cards_db.py`)
- Expect `keywords` and `types` to be pre-populated.
- Populate `cards`, `card_faces`, `card_printings`, `card_types`, and `card_keywords`.

Decks (`populate/populate_decks.py`)
- Reads `scraped_deck_data.json` and resolves names to `cards.card_id`.
- Inserts into `decks`, `deck_cards_maindeck`, `deck_cards_sideboard`.
- Computes and stores a stable `deck_hash`.

Archetypes (`populate_deck_archetypes.py`)
- Maps a deck’s weak archetype to multiple archetype tags into `deck_archetypes` with source labels.

Formats (`populate deck_formats.py`)
- Derives legal format memberships into `deck_formats` using card legality JSON stored in `cards` (per script’s expectation on a `legalities` field if present).

## Implied Relationships (no FK constraints declared)

- `card_faces.card_id` → `cards.card_id`
- `card_printings.card_id` → `cards.card_id`
- `card_types.face_id` → `card_faces.face_id`
- `card_keywords.face_id` → `card_faces.face_id`; `card_keywords.keyword` → `keywords.name`
- `deck_cards_*.(deck_id, card_id)` → (`decks.deck_id`, `cards.card_id`)
- `decks_deduplicated.representative_deck_id` → `decks.deck_id`
- `deck_formats.deck_id` → `decks.deck_id`; `deck_formats.format_name` → `formats.format_name`
- `deck_archetypes.deck_id` → `decks.deck_id`; `deck_archetypes.archetype_name` → `archetypes.name`

## Known Mismatches

- `populate/populate_decks.py` inserts into a `decks` column named `format`, which does not exist in the `initialize_cards_decks.py` schema. Either remove that column from the insert or add a `format` column to `decks` if you need it.
- Two conflicting `deck_stats` definitions exist; align your initialization and consumers on one.

