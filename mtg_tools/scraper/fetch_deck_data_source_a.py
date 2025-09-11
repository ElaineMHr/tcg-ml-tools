import requests
import re
from bs4 import BeautifulSoup

def fetch_deck_data_source_a(deck_url):
    """Fetch details of an individual deck using requests and BeautifulSoup."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    response = requests.get(deck_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch deck: {deck_url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract metadata (deck name, format, event, date, player)
    deck_name_tag = soup.select("div.event_title")[1]
    format_tag = soup.select_one("div.meta_arch")
    archetype_tag = soup.select_one("a[href^='archetype']")
    player_tag = soup.select_one("a.player_big")
    event_tag = soup.select("div.event_title")[0]

    deck_name = deck_name_tag.text.strip() if deck_name_tag else "Unknown Deck"
    deck_format = format_tag.text.strip() if format_tag else "Unknown Format"
    event_name = event_tag.text.strip() if event_tag else "Unknown Event"
    player_name = player_tag.text.strip() if player_tag else "Unknown Player"
    archetype_name = archetype_tag.text.strip() if archetype_tag else "Unknown Archetype"
    archetype_name = archetype_name.replace(" decks","").split(" ")

    types_tag = soup.select("div.O14")
    card_count = 0
    cards = soup.select("div.deck_line.hover_tr")
    start_index = 0
    decklist = []
    sideboard = []
    is_commander = False

    for row in types_tag:
        text = row.text.strip().split(" ", 1)
        if "SIDEBOARD" in text[0]:
            break
        try:
            card_count += int(text[0])
        except ValueError:
            is_commander = True
            continue

    placement = (match.group(1) if (match := re.match(r"#(\d+(?:-\d+)?)", deck_name)) else "Unknown Placement")

    # Remove the # and number
    deck_name = re.sub(r"^#\d+(?:-\d+)?\s*", "", deck_name)

    # Remove the player name and " - " before it
    deck_name = re.sub(rf"\s*-\s*{re.escape(player_name)}$", "", deck_name)

    # Extract main deck and sideboard

    if is_commander:
        start_index = 1
        count, card_name = cards[0].text.strip().split(" ", 1)
        card_data = {"name": card_name, "count": int(count)}
        sideboard.append(card_data)

    for row in cards[start_index:]:
        count, card_name = row.text.strip().split(" ", 1)
        card_data = {"name": card_name, "count": int(count)}
        if card_count > 0:
            decklist.append(card_data)
            card_count -= int(count)
        else:
            sideboard.append(card_data)

    deck_data = {
        "name": deck_name,
        "format": [deck_format],
        "weak_archetype": archetype_name,
        "decklist": decklist,
        "sideboard": sideboard,
        "source": "Source A",
        "url": deck_url,
        "event": event_name,
        "placement": placement
    }

    return deck_data
