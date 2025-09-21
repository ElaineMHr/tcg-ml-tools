import os
import re
import sqlite3
from pathlib import Path
from collections import Counter
from contextlib import closing

# --- Path to DB (fixed relative) ---
DB_PATH = "mtgcore.db"

# --- Config ---
CREATURE_THRESHOLD = 0.50         # ≥ 50% of creature cards (see notes) → main_tribe
CMC_BUCKETS = [0,1,2,3,4,5,6]     # 7+ collapses into cmc_7_plus
SUPER_TYPES = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Land"]
CANDIDATE_TYPES = ["Creature","Instant","Sorcery","Artifact","Enchantment","Planeswalker"]  # dominance ignores Land
COLOR_KEYS  = ["W","U","B","R","G","C"]  # include colorless

# --- Helpers ---
def bucket_cmc(cmc):
    try:
        v = int(round(float(cmc)))
    except Exception:
        v = 0
    return "cmc_7_plus" if v >= 7 else f"cmc_{v}"

def parse_color_identity(ci_text):
    """cards.color_identity is comma-separated: 'U,R' or 'G' or ''."""
    if not ci_text:
        return []
    parts = [p.strip() for p in str(ci_text).split(",")]
    return [p for p in parts if p in COLOR_KEYS]

def parse_type_line(type_line):
    """
    Returns (supertypes_present, creature_subtypes_list).
    e.g., 'Legendary Creature — Elf Druid'
    """
    if not type_line:
        return set(), []
    parts = re.split(r"\s+—\s+|\s+-\s+", type_line)  # em dash or hyphen
    left = parts[0]
    right = parts[1] if len(parts) > 1 else ""
    supers = {tok for tok in left.split() if tok in SUPER_TYPES}
    subs = [tok for tok in right.split() if tok] if "Creature" in supers else []
    return supers, subs

def iter_deck_ids(conn: sqlite3.Connection):
    # Use maindeck presence to determine which decks to compute
    for (deck_id,) in conn.execute("SELECT DISTINCT deck_id FROM deck_cards_maindeck;"):
        yield deck_id

def fetch_rows_for_deck(conn: sqlite3.Connection, deck_id: str):
    """
    Returns rows of (count, cmc, type_line, color_identity) for a deck's maindeck.
    Uses primary face (face_index=0) for type_line when available.
    """
    sql = """
    SELECT m.count,
           c.cmc,
           f.type_line,
           c.color_identity
    FROM deck_cards_maindeck m
    JOIN cards c ON c.card_id = m.card_id
    LEFT JOIN card_faces f ON f.card_id = c.card_id AND (f.face_index = 0 OR f.face_index IS NULL)
    WHERE m.deck_id = ?;
    """
    return conn.execute(sql, (deck_id,)).fetchall()

def compute_stats(rows):
    """
    rows: list of tuples (count, cmc, type_line, color_identity)
    Returns dict of column -> value.
    """
    cmc_counts   = Counter({f"cmc_{b}": 0 for b in CMC_BUCKETS}); cmc_counts["cmc_7_plus"] = 0
    color_counts = Counter({k: 0 for k in COLOR_KEYS})
    super_counts = Counter({k: 0 for k in SUPER_TYPES})
    tribe_counts = Counter()

    nonland_cards_total = 0
    cmc_weighted_sum = 0.0

    for cnt, cmc, type_line, ci in rows:
        cnt = int(cnt or 0)
        supers, subs = parse_type_line(type_line)

        # avg cmc excludes lands
        if "Land" not in supers:
            nonland_cards_total += cnt
            try:
                cmc_weighted_sum += float(cmc) * cnt
            except Exception:
                pass

        # cmc buckets (includes lands; change here if you want to exclude them)
        cmc_counts[bucket_cmc(cmc)] += cnt

        # dominant type tally (count all supertypes; we'll pick from non-land later)
        for st in supers:
            super_counts[st] += cnt

        # tribes (only for creature cards)
        if "Creature" in supers:
            for sub in subs:
                tribe_counts[sub] += cnt

        # color identity counts
        for color in parse_color_identity(ci):
            color_counts[color] += cnt

    avg_cmc = (cmc_weighted_sum / nonland_cards_total) if nonland_cards_total > 0 else 0.0

    # Dominant type among non-land supertypes only
    if any(super_counts[t] for t in CANDIDATE_TYPES):
        dominant_type = max(CANDIDATE_TYPES, key=lambda t: super_counts[t])
    else:
        dominant_type = None

    # Main tribe (>= 50% of counted creature subtype mentions)
    main_tribe = None
    if any(tribe_counts.values()):
        top_sub, top_cnt = max(tribe_counts.items(), key=lambda kv: kv[1])
        total_creatures_mentions = sum(tribe_counts.values())
        if total_creatures_mentions > 0 and (top_cnt / total_creatures_mentions) >= CREATURE_THRESHOLD:
            main_tribe = top_sub

    return {
        "avg_cmc": round(avg_cmc, 4),
        "main_tribe": main_tribe,
        "dominant_type": dominant_type,
        "cmc_0": cmc_counts["cmc_0"], "cmc_1": cmc_counts["cmc_1"], "cmc_2": cmc_counts["cmc_2"],
        "cmc_3": cmc_counts["cmc_3"], "cmc_4": cmc_counts["cmc_4"], "cmc_5": cmc_counts["cmc_5"],
        "cmc_6": cmc_counts["cmc_6"], "cmc_7_plus": cmc_counts["cmc_7_plus"],
        "color_W": color_counts["W"], "color_U": color_counts["U"], "color_B": color_counts["B"],
        "color_R": color_counts["R"], "color_G": color_counts["G"], "color_C": color_counts["C"],
    }

def upsert(conn: sqlite3.Connection, deck_id: str, s: dict):
    conn.execute(
        """
        INSERT INTO deck_stats (
          deck_id, avg_cmc, main_tribe, dominant_type,
          cmc_0, cmc_1, cmc_2, cmc_3, cmc_4, cmc_5, cmc_6, cmc_7_plus,
          color_W, color_U, color_B, color_R, color_G, color_C,
          features_version, generated_at
        ) VALUES (?,?,?,?, ?,?,?,?,?, ?,?,?,
                  ?,?,?,?, ?,?, 'v1', CURRENT_TIMESTAMP)
        ON CONFLICT(deck_id) DO UPDATE SET
          avg_cmc=excluded.avg_cmc,
          main_tribe=excluded.main_tribe,
          dominant_type=excluded.dominant_type,
          cmc_0=excluded.cmc_0, cmc_1=excluded.cmc_1, cmc_2=excluded.cmc_2, cmc_3=excluded.cmc_3,
          cmc_4=excluded.cmc_4, cmc_5=excluded.cmc_5, cmc_6=excluded.cmc_6, cmc_7_plus=excluded.cmc_7_plus,
          color_W=excluded.color_W, color_U=excluded.color_U, color_B=excluded.color_B,
          color_R=excluded.color_R, color_G=excluded.color_G, color_C=excluded.color_C,
          features_version='v1',
          generated_at=CURRENT_TIMESTAMP;
        """,
        (
            deck_id,
            s["avg_cmc"], s["main_tribe"], s["dominant_type"],
            s["cmc_0"], s["cmc_1"], s["cmc_2"], s["cmc_3"],
            s["cmc_4"], s["cmc_5"], s["cmc_6"], s["cmc_7_plus"],
            s["color_W"], s["color_U"], s["color_B"],
            s["color_R"], s["color_G"], s["color_C"],
        )
    )

def main():
    
    with closing(sqlite3.connect((DB_PATH))) as conn:
        
        # Gather deck ids
        deck_ids = list(iter_deck_ids(conn))
        print(f"Found {len(deck_ids)} decks with maindeck rows.")

        # Populate in one transaction (faster)
        processed = 0
        with conn:
            for deck_id in deck_ids:
                rows = fetch_rows_for_deck(conn, deck_id)
                stats = compute_stats(rows)
                upsert(conn, deck_id, stats)
                processed += 1
                if processed % 500 == 0:
                    print(f"- processed {processed} decks...")

        n = conn.execute("SELECT COUNT(*) FROM deck_stats;").fetchone()[0]
        print(f"deck_stats populated: {n} rows")

        # Tiny preview
        try:
            preview = conn.execute("""
                SELECT deck_id, avg_cmc, dominant_type, main_tribe
                FROM deck_stats
                ORDER BY avg_cmc DESC
                LIMIT 5;
            """).fetchall()
            if preview:
                print("Top 5 avg_cmc preview:")
                for row in preview:
                    print("  ", row)
        except Exception:
            pass

if __name__ == "__main__":
    main()
