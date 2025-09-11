import json

# === SETTINGS ===
input_file = "scraped_deck_data_archetypes.json"
output_file = "scraped_deck_data_no_commander_phrase.json"
empty_decklist_file = "decks_with_empty_decklist.json"

# === FORMAT RANKING ===
format_ranking = {
    "Standard": 0,
    "Pioneer": 1,
    "Modern": 2,
    "Legacy": 3,
    "Vintage": 4
}

# === COMMANDER FORMATS TO SKIP and LIMITED ===
commander_formats = {
    "Brawl", "Canadian Highlander", "Commander", "Commander Pauper", "Duel", "Duel Commander",
    "Gladiator", "Highlander", "Historicbrawl", "MTGO Commander", "Oathbreaker",
    "Paupercommander", "Predh", "Standardbrawl", "cEDH", "Limited"
}

# === HELPERS ===
def is_commander_format(format_list):
    return any(fmt in commander_formats for fmt in format_list)

def reduce_format_tags(tags):
    ranked = [t for t in tags if t in format_ranking]
    unranked = [t for t in tags if t not in format_ranking]
    if len(ranked) > 1:
        most_restrictive = min(ranked, key=lambda x: format_ranking[x])
        return [most_restrictive] + unranked
    return tags  # return as-is if only 1 or none are ranked

# === MAIN PROCESSING ===
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = {}
empty_decklist_urls = []
count = 0

for url, deck in raw_data.items():
    count += 1
    print("Loading Deck",count)
    if not isinstance(deck, dict):
        continue

    decklist = deck.get("decklist", [])
    sideboard = deck.get("sideboard", [])

    # Record decks with empty decklists
    if not decklist:
        empty_decklist_urls.append(url)
        continue

    format_tags = deck.get("format", [])

    # Convert to list if it's a string
    if isinstance(format_tags, str):
        format_tags = [format_tags]

    # Skip if commander-style
    if is_commander_format(format_tags):
        continue

    # Reduce format tags if needed
    reduced_tags = reduce_format_tags(format_tags)
    deck["format"] = reduced_tags

    # Normalize weak_archetype field
    archetypes = deck.get("weak_archetype", [])
    if isinstance(archetypes, str):
        archetypes = [archetypes]

    joined_archetype = " ".join(archetypes).strip()
    if joined_archetype:
        deck["weak_archetype"] = joined_archetype

    cleaned_data[url] = deck

# === SAVE RESULTS ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)

with open(empty_decklist_file, "w", encoding="utf-8") as f:
    json.dump(empty_decklist_urls, f, indent=2)

print(f"✅ Saved {len(cleaned_data)} cleaned decks to '{output_file}'")
print(f"✅ Found {len(empty_decklist_urls)} decks with empty decklists — saved to '{empty_decklist_file}'")
