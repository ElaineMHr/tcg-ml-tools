import sqlite3
import ast
from collections import defaultdict, Counter

DB_PATH = "mtgcore.db"

def populate_deck_formats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get card legalities and restricted formats
    cur.execute("SELECT card_id, legalities FROM cards")
    card_legalities = {}
    card_restricted = defaultdict(set)

    for cid, raw in cur.fetchall():
        if not raw:
            continue
        try:
            data = ast.literal_eval(raw)
            legal = set()
            for fmt, status in data.items():
                if status == "legal":
                    legal.add(fmt)
                elif status == "restricted":
                    legal.add(fmt)
                    card_restricted[cid].add(fmt)
            card_legalities[cid] = legal
        except:
            continue

    # Load full decklist
    cur.execute("SELECT deck_id, card_id, count FROM deck_cards_maindeck")
    maindeck = defaultdict(list)
    for deck_id, cid, count in cur.fetchall():
        maindeck[deck_id].append((cid, count))

    cur.execute("SELECT deck_id, card_id, count FROM deck_cards_sideboard")
    sideboard = defaultdict(list)
    for deck_id, cid, count in cur.fetchall():
        sideboard[deck_id].append((cid, count))

    cur.execute("DELETE FROM deck_formats")

    rows = []
    skipped = []

    all_deck_ids = set(maindeck.keys()) | set(sideboard.keys())

    for deck_id in all_deck_ids:
        all_cards = maindeck[deck_id] + sideboard[deck_id]
        counter = Counter()
        legal_sets = []

        for cid, count in all_cards:
            counter[cid] += count
            legal = card_legalities.get(cid)
            if legal is None:
                break
            legal_sets.append(legal)
        else:
            if not legal_sets:
                continue
            formats = set.intersection(*legal_sets)
            for fmt in formats:
                legal = True
                for cid, c in counter.items():
                    if fmt in card_restricted[cid] and c > 1:
                        legal = False
                        break
                if legal:
                    rows.append((deck_id, fmt.title()))
            continue

        skipped.append(deck_id)

    cur.executemany("""
        INSERT OR IGNORE INTO deck_formats (deck_id, format_name)
        VALUES (?, ?)
    """, rows)

    conn.commit()
    conn.close()

    with open("unresolved_deck_formats.log", "w", encoding="utf-8") as f:
        for deck_id in skipped:
            f.write(f"{deck_id}\n")

    print(f"✅ Inserted {len(rows)} format entries.")
    print(f"⚠️ Skipped {len(skipped)} decks due to unresolved card legalities.")

if __name__ == "__main__":
    populate_deck_formats()
