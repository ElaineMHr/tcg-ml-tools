import json
import sqlite3
import hashlib
import uuid
import difflib
import re
from collections import defaultdict, Counter
from mtg_tools.preprocess.deck_properties import who_is_companion


def generate_hash(decklist):
    normalized = sorted([f"{name.strip().lower()}:{count}" for name, count in decklist.items()])
    return hashlib.sha256(",".join(normalized).encode("utf-8")).hexdigest()


def insert_double_slash(name):
    # Function for the case, when a card cannot be found.
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' // ', name)


def normalize_slashes(name):
    # Function for the case, when a card cannot be found.
    return re.sub(r'\s+', ' ', name.replace('/', ' ')).strip()


def populate_decks(db_path="mtgcore.db", deck_file="scraped_deck_data.json"):
    with open(deck_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    DELETE FROM deck_cards_maindeck;
    DELETE FROM deck_cards_sideboard;
    DELETE FROM decks;
    """)
    print("Cleared previous deck data.")

    manual_name_map = {
        "rick, steadfast leader": "greymond, avacyn's stalwart",
        "________ goblin": "_____ goblin",
        "lucille": "gisa's favorite shovel",
        "glenn, the voice of calm": "gregor, shrewd magistrate"
    }

    # Fetch cards and faces
    cur.execute("""
    SELECT name, cards.card_id, released_at
    FROM cards
    JOIN card_printings ON cards.card_id = card_printings.card_id
    WHERE promo = 0 AND full_art = 0 AND language = 'en'
    """)
    card_map = defaultdict(list)
    for name, card_id, released_at in cur.fetchall():
        card_map[name.strip().lower()].append((card_id, released_at))

    cur.execute("SELECT name, card_id, face_index FROM card_faces")
    face_map = defaultdict(list)
    for name, card_id, face_index in cur.fetchall():
        face_map[name.strip().lower()].append((card_id, face_index))

    default_card = {}
    for name, entries in card_map.items():
        chosen_id = sorted(entries, key=lambda x: x[1] or "0000-00-00", reverse=True)[0][0]
        default_card[name] = chosen_id

    skipped = []
    face_mapped = []
    fuzzy_mapped = []
    remapped = []

    all_card_names = list(default_card.keys())
    all_face_names = list(face_map.keys())

    resolved_cache = {}

    def resolve_name(original, deck_id):
        if original in resolved_cache:
            return resolved_cache[original]

        attempts = [
            insert_double_slash(original),
            manual_name_map.get(original.strip().lower()),
            normalize_slashes(original)
        ]

        for attempt in attempts:
            if not attempt:
                continue
            norm = attempt.strip().lower()
            if norm in default_card:
                remapped.append((deck_id, original, attempt, default_card[norm]))
                resolved_cache[original] = default_card[norm]
                return default_card[norm]
            if norm in face_map:
                face_mapped.append((deck_id, original, face_map[norm][0][0], face_map[norm][0][1]))
                resolved_cache[original] = face_map[norm][0][0]
                return face_map[norm][0][0]

        norm = original.strip().lower()
        close = difflib.get_close_matches(norm, all_card_names, n=1, cutoff=0.9)
        if close:
            match = close[0]
            fuzzy_mapped.append((deck_id, original, match, default_card[match], "card", difflib.SequenceMatcher(None, norm, match).ratio()))
            resolved_cache[original] = default_card[match]
            return default_card[match]

        close = difflib.get_close_matches(norm, all_face_names, n=1, cutoff=0.9)
        if close:
            match = close[0]
            fuzzy_mapped.append((deck_id, original, match, face_map[match][0][0], "face", difflib.SequenceMatcher(None, norm, match).ratio()))
            resolved_cache[original] = face_map[match][0][0]
            return face_map[match][0][0]

        resolved_cache[original] = None
        return None

    for url, deck in data.items():
        if not deck or not deck.get("decklist"):
            continue

        maindeck = deck.get("decklist", [])
        sideboard = deck.get("sideboard", [])
        source = deck.get("source", "unknown")

        decklist = Counter()
        for card in maindeck:
            decklist[card["name"].strip()] += card["count"]

        side = Counter()
        for card in sideboard:
            side[card["name"].strip()] += card["count"]

        companion = who_is_companion(maindeck, sideboard)
        if companion:
            side.pop(companion, None)
            if companion not in decklist:
                decklist[companion] = 1

        deck_hash = generate_hash(decklist)
        deck_id = str(uuid.uuid4())

        mapped_main = []
        mapped_side = []
        failed = False

        for name, count in decklist.items():
            card_id = resolve_name(name, deck_id)
            if not card_id:
                skipped.append((url, name))
                failed = True
                break
            mapped_main.append((card_id, count))

        if failed:
            continue

        for name, count in side.items():
            card_id = resolve_name(name, deck_id)
            if not card_id:
                skipped.append((url, name))
                failed = True
                break
            mapped_side.append((card_id, count))

        if failed:
            continue

        cur.execute("""
        INSERT INTO decks (
            deck_id, deck_hash, source, source_url, 
            companion_id, main_color_identity
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            deck_id, deck_hash, source, url, companion, ""
        ))

        for cid, count in mapped_main:
            cur.execute("INSERT INTO deck_cards_maindeck (deck_id, card_id, count) VALUES (?, ?, ?)",
                        (deck_id, cid, count))

        for cid, count in mapped_side:
            cur.execute("INSERT INTO deck_cards_sideboard (deck_id, card_id, count) VALUES (?, ?, ?)",
                        (deck_id, cid, count))

    conn.commit()
    conn.close()

    with open("skipped_decks.log", "w", encoding="utf-8") as f:
        for url, name in skipped:
            f.write(f"{url}\t{name}\n")

    with open("face_name_mappings.log", "w", encoding="utf-8") as f:
        for deck_id, name, card_id, face_index in face_mapped:
            f.write(f"{deck_id}\t{name}\t{card_id}\tface={face_index}\n")

    with open("fuzzy_name_mappings.log", "w", encoding="utf-8") as f:
        for deck_id, name, match, card_id, source, ratio in fuzzy_mapped:
            f.write(f"{deck_id}\t{name}\t{match}\t{card_id}\t{source}\t{ratio:.3f}\n")

    with open("remapped_names.log", "w", encoding="utf-8") as f:
        for deck_id, original, remapped_name, matched_id in remapped:
            f.write(f"{deck_id}\t{original}\t{remapped_name}\t{matched_id}\n")

    print("Decks populated. Skipped, remapped, fuzzy, and face matches logged.")


if __name__ == "__main__":
    populate_decks()
