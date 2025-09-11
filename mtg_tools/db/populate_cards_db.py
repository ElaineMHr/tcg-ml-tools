import json
import sqlite3
from pathlib import Path

def parse_type_line(type_line, supertypes, types_dict):
    if "—" in type_line:
        left, right = map(str.strip, type_line.split("—", 1))
        subtypes = right.split()
    else:
        left = type_line.strip()
        subtypes = []
    parts = left.split()
    sups = [t for t in parts if t in supertypes]
    typs = [t for t in parts if t in types_dict]
    return sups, typs, subtypes

def populate_cards(db_path, cards_json_path, image_dir):
    with open(cards_json_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM keywords")
    keyword_set = {row[0] for row in cur.fetchall()}

    cur.execute("SELECT name, category FROM types")
    type_map = {row[0]: row[1] for row in cur.fetchall()}
    supertypes = {name for name, cat in type_map.items() if cat == "supertype"}
    types_dict = {name for name, cat in type_map.items() if cat == "type"}
    subtypes_set = {name for name, cat in type_map.items() if cat == "subtype"}

    missing_keywords = set()
    missing_types = set()
    skipped_cards = []

    for card in cards:
        card_id = card["id"]
        name = card["name"]
        layout = card.get("layout", "")
        cmc = card.get("cmc", 0.0)
        color_identity = ",".join(card.get("color_identity", []))
        is_alchemy = name.lower().startswith("a-")

        # Resolve oracle_id safely
        oracle_id = card.get("oracle_id")
        if not oracle_id:
            faces = card.get("card_faces", [])
            face_ids = {face.get("oracle_id") for face in faces if face.get("oracle_id")}
            if len(face_ids) == 1:
                oracle_id = face_ids.pop()
            else:
                skipped_cards.append((card_id, name))
                continue

        base_oracle_id = oracle_id if is_alchemy else None

        cur.execute("""
        INSERT OR IGNORE INTO cards (card_id, oracle_id, name, layout, cmc, color_identity, is_alchemy, base_oracle_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (card_id, oracle_id, name, layout, cmc, color_identity, is_alchemy, base_oracle_id))

        faces = card.get("card_faces", [card])
        for face_index, face in enumerate(faces):
            face_id = f"{card_id}_{face_index}"
            mana_cost = face.get("mana_cost")
            type_line = face.get("type_line")
            oracle_text = face.get("oracle_text")
            power = face.get("power")
            toughness = face.get("toughness")
            loyalty = face.get("loyalty")
            produced_mana = ",".join(face.get("produced_mana", []))
            artist = face.get("artist", card.get("artist"))
            art_crop = f"{face_id}.jpg" if Path(f"{image_dir}/{face_id}.jpg").exists() else None
            image_large = f"{face_id}.png" if Path(f"{image_dir}/{face_id}.png").exists() else None

            cur.execute("""
            INSERT OR IGNORE INTO card_faces (
                face_id, card_id, face_index, name, mana_cost, type_line, oracle_text,
                power, toughness, loyalty, produced_mana, image_art_crop, image_large, artist
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                face_id, card_id, face_index, face.get("name"), mana_cost, type_line, oracle_text,
                power, toughness, loyalty, produced_mana, art_crop, image_large, artist
            ))

            sups, typs, subs = parse_type_line(type_line or "", supertypes, types_dict)
            for t in sups:
                cur.execute("INSERT INTO card_types (face_id, type_value, type_category) VALUES (?, ?, ?)", (face_id, t, "supertype"))
            for t in typs:
                cur.execute("INSERT INTO card_types (face_id, type_value, type_category) VALUES (?, ?, ?)", (face_id, t, "type"))
            for t in subs:
                if t in subtypes_set:
                    cur.execute("INSERT INTO card_types (face_id, type_value, type_category) VALUES (?, ?, ?)", (face_id, t, "subtype"))
                else:
                    missing_types.add(t)

            for kw in face.get("keywords", []):
                if kw in keyword_set:
                    cur.execute("INSERT INTO card_keywords (face_id, keyword) VALUES (?, ?)", (face_id, kw))
                else:
                    missing_keywords.add(kw)

        cur.execute("""
        INSERT OR IGNORE INTO card_printings (
            printing_id, card_id, set_code, collector_number, language, rarity, frame,
            frame_effect, full_art, promo, border_color, watermark, released_at, artist
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_id, card_id, card.get("set"), card.get("collector_number"), card.get("lang", "en"),
            card.get("rarity"), card.get("frame"), ",".join(card.get("frame_effects", [])),
            card.get("full_art", False), card.get("promo", False), card.get("border_color"),
            card.get("watermark"), card.get("released_at"), card.get("artist")
        ))

    conn.commit()
    conn.close()

    with open("missing_keywords.log", "w", encoding="utf-8") as f:
        for kw in sorted(missing_keywords):
            f.write(kw + "\n")
    with open("missing_types.log", "w", encoding="utf-8") as f:
        for t in sorted(missing_types):
            f.write(t + "\n")
    with open("skipped_cards.log", "w", encoding="utf-8") as f:
        for cid, name in skipped_cards:
            f.write(f"{cid}: {name}\n")

    print("Cards populated. Missing values logged.")

if __name__ == "__main__":
    populate_cards("mtgcore.db", "./card-database/default-cards-20250413090942.json", ".")
