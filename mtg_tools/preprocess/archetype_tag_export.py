import json

def tag_printer_unique():
    scraped_file = "scraped_deck_data.json"

    with open(scraped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_archetype_phrases = set()

    for url, deck in data.items():
        if deck is None:
            continue  # Skip if entry is null

        archetypes = deck.get("weak_archetype", [])

        # Handle string vs list inconsistencies
        if isinstance(archetypes, str):
            archetypes = [archetypes]

        # Join weak_archetype list into one phrase per deck
        if archetypes:
            joined = " ".join(archetypes).strip()
            if joined:
                unique_archetype_phrases.add(joined)

    with open("archetype_phrases_all.json", "w", encoding="utf-8") as f:
        json.dump(sorted(unique_archetype_phrases), f, indent=2)

def tag_printer_unique_with_counter():
    import json
    from collections import Counter
    scraped_file = "scraped_deck_data.json"

    with open(scraped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    archetype_counter = Counter()

    for url, deck in data.items():
        if deck is None:
            continue  # Skip if entry is null

        archetypes = deck.get("weak_archetype", [])

        # Handle string vs list inconsistencies
        if isinstance(archetypes, str):
            archetypes = [archetypes]

        # Join weak_archetype list into one phrase per deck
        if archetypes:
            joined = " ".join(archetypes).strip()
            if joined:
                archetype_counter[joined] += 1

    # Output as JSON: list of {archetype: ..., count: ...}
    output = [{"archetype": k, "count": v} for k, v in archetype_counter.most_common()]

    with open("archetype_phrase_counts.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Extracted {len(archetype_counter)} unique archetype phrases.")

def tag_printer():
    scraped_file = "scraped_deck_data.json"

    with open(scraped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_archetype_phrases = []

    for url, deck in data.items():
        if deck is None:
            continue  # Skip if entry is null

        archetypes = deck.get("weak_archetype", [])

        # Handle string vs list inconsistencies
        if isinstance(archetypes, str):
            archetypes = [archetypes]

        # Join weak_archetype list into one phrase per deck
        if archetypes:
            joined = " ".join(archetypes).strip()
            if joined:
                unique_archetype_phrases.append(joined)

    return unique_archetype_phrases

def tag_printer_archetypes():
    scraped_file = "scraped_deck_data.json"

    with open(scraped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    archetype_phrases = []

    for url, deck in data.items():
        if deck is None:
            continue  # Skip if entry is null

        archetypes = deck.get("weak_archetype", [])

        # Handle string vs list inconsistencies
        if isinstance(archetypes, str):
            archetype_phrases.append(archetypes)

    return archetype_phrases

if __name__ == "__main__":
    tag_printer_unique_with_counter()
