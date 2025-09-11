"""
deck_properties.py

This module provides boolean checks for common deck properties,
such as whether a deck is singleton, has a companion, contains only lands,
or contains a commander.

Each function returns True/False based on the input deck structure.

This module also provides an identification of the commander for decks that contain a commander,
which returns a string or None based on the deck,
and a total amount of cards in a deck, excluding the sideboard.
"""

def is_singleton(decklist):
    """
    Returns True if all cards have a count of 1 (singleton format), excluding basic lands.
    """
    basic_lands = {
        "Plains", "Island", "Swamp", "Mountain", "Forest",
        "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
        "Snow-Covered Mountain", "Snow-Covered Forest",
        "Wastes"
    }
    for card in decklist:
        if card["count"] > 1 and card["name"] not in basic_lands:
            return False
    return True

def is_companion(decklist, sideboard):
    """
    Returns True if the deck has a known companion in the main deck or sideboard.
    """
    companions = {
        "Gyruda, Doom of Depths",
        "Jegantha, the Wellspring",
        "Kaheera, the Orphanguard",
        "Keruga, the Macrosage",
        "Lurrus of the Dream-Den",
        "Lutri, the Spellchaser",
        "Obosh, the Preypiercer",
        "Umori, the Collector",
        "Yorion, Sky Nomad",
        "Zirda, the Dawnwaker"
    }
    first_main = decklist[0]["name"] if decklist else None
    first_side = sideboard[0]["name"] if sideboard else None

    return first_main in companions or first_side in companions

def who_is_companion(decklist, sideboard):
    """
    Returns the companion of a deck (string) or None.
    """
    companions = {
        "Gyruda, Doom of Depths",
        "Jegantha, the Wellspring",
        "Kaheera, the Orphanguard",
        "Keruga, the Macrosage",
        "Lurrus of the Dream-Den",
        "Lutri, the Spellchaser",
        "Obosh, the Preypiercer",
        "Umori, the Collector",
        "Yorion, Sky Nomad",
        "Zirda, the Dawnwaker"
    }
    first_main = decklist[0]["name"] if decklist else None
    first_side = sideboard[0]["name"] if sideboard else None

    matches = [
        card for card in companions
        if (first_main and first_main.lower() in card.lower())
           or (first_side and first_side.lower() in card.lower())
    ]

    for card in matches:
        if card:
            return card
    return None

def total_card_count(decklist):
    """
    Returns the total amount of cards in a decklist.
    """
    sum = 0
    for card in decklist:
        sum += card["count"]
    return sum

def is_commander(decklist, sideboard, card_database):
    """
    Returns True if the deck has a known commander in the main deck or sideboard.
    """
    first_main = decklist[0]["name"] if decklist else None
    first_side = sideboard[0]["name"] if sideboard else None

    matches = [
        card for card in card_database
        if (first_main and first_main.lower() in card.get("name", "").lower())
           or (first_side and first_side.lower() in card.get("name", "").lower())
    ]

    for card in matches:
        if 'leadershipSkills' in card:
            return True
    return False

def who_is_commander(decklist, sideboard, card_database):
    """
    Returns the commander of a deck (string) or None.
    """
    first_main = decklist[0]["name"] if decklist else None
    first_side = sideboard[0]["name"] if sideboard else None

    matches = [
        card for card in card_database
        if (first_main and first_main.lower() in card.get("name", "").lower())
           or (first_side and first_side.lower() in card.get("name", "").lower())
    ]

    for card in matches:
        if 'leadershipSkills' in card:
            return card['name']
    return None

def is_only_lands(decklist):
    """
    Returns True if the decklist contains only basic lands.
    """
    basic_lands = {
        "Plains", "Island", "Swamp", "Mountain", "Forest",
        "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
        "Snow-Covered Mountain", "Snow-Covered Forest",
        "Wastes"
    }
    for card in decklist:
        if card["name"] not in basic_lands:
            return False
    return True

# import json
#
# with open("scraped_deck_data_no_commander_phrase.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# with open("./card-database/AllPrintings.json", "r", encoding="utf-8") as f:
#     all_data = json.load(f)
#
# card_database = []
# for set_data in all_data["data"].values():
#     card_database.extend(set_data["cards"])
#
# counter_singleton = 0
# counter_companion = 0
# counter_singleton_commander = 0
# null_counter = 0
# for _, deck in data.items():
#     decklist = deck.get("decklist",[])
#     sideboard = deck.get("sideboard",[])
#     singleton = is_singleton(decklist)
#     companion = is_companion(decklist,sideboard)
#     only_lands = is_only_lands(decklist)
#     if singleton:
#         counter_singleton += 1
#         commander = who_is_commander(decklist, sideboard, card_database)
#         if commander and not companion and not only_lands:
#             counter_singleton_commander += 1
#             print(commander)
#             print(total_card_count(decklist))
#             print(deck.get("url",[]))
#
#     if companion and singleton:
#         print("Companion:")
#         print(total_card_count(decklist))
#         print(deck.get("url",[]))
#         counter_companion += 1
#     if not decklist:
#         null_counter += 1
#     if companion:
#         print(who_is_companion(decklist,sideboard))
#
# print("Singleton:", counter_singleton)
# print("Singleton & Commander:", counter_singleton_commander)
# print("Companion:", counter_companion)
# print("Null:", null_counter)
