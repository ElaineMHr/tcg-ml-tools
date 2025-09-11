import requests
from bs4 import BeautifulSoup


def fetch_deck_data_source_b(deck_url):
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
    deck_name_tag = soup.select_one("h1.deck-name")
    format_tag = soup.select("div.legal-format")
    deck_format = []
    for row in format_tag:
        deck_format.append(row.text.strip()) if row.text else "Unknown Format"

    deck_name = deck_name_tag.text.strip() if deck_name_tag else "Unknown Deck"

    types_tag = soup.select("h2.section")
    card_count = 0
    for row in types_tag:
        text = row.text.strip().split(" ", 1)
        if "Sideboard" in text[0]:
            break
        card_count += int(text[1])

    # Extract main deck and sideboard
    decklist = []
    sideboard = []

    for row in soup.select("a.card"):
        count, card_name = row.text.strip().split(" ", 1)
        try:
            count = int(count)
        except ValueError:
            continue
        card_data = {"name": card_name.strip(), "count": count}
        if card_count > 0:
            decklist.append(card_data)
            card_count -= int(count)
        else:
            sideboard.append(card_data)

    deck_data = {
        "name": deck_name,
        "format": deck_format,
        "weak_archetype": "Unknown Archetype",
        "decklist": decklist,
        "sideboard": sideboard,
        "source": "source b",
        "url": deck_url,
        "event": "Unknown Event",
        "placement": "Unknown Placement"
    }

    return deck_data
