import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

source_b_url = os.environ["SOURCE_B_URL"]

# Initialize Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")  # Headless mode (remove if you want to see the browser)
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


# Function to extract deck data with retries and stale element handling
def extract_deck_data(max_attempts=10):
    """Extracts deck data from the current page, retrying stale elements with increasing wait times."""

    attempt = 0
    deck_data = []

    while attempt < max_attempts:
        try:
            # Wait for elements to be present
            WebDriverWait(driver, 360).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "td.deck-name"))
            )

            # Re-find all `td.deck-name` elements each attempt
            td_elements = driver.find_elements(By.CSS_SELECTOR, "td.deck-name")

            for row in td_elements:
                try:
                    deck_link_element = row.find_element(By.TAG_NAME, "a")
                    deck_link = deck_link_element.get_attribute('href') if deck_link_element else "No Link"

                    deck_data.append({
                        "site": "source b",
                        "link": deck_link,
                        "archetype": "Unknown Archetype",
                        "format": "Unknown Format",
                        "status": "not processed"
                    })

                except StaleElementReferenceException:
                    print(f"Stale element found in row. Attempt {attempt + 1}/{max_attempts}. Restarting extraction...")
                    attempt += 1
                    time.sleep(2 * attempt)  # Increase wait time between attempts
                    break  # Restart extraction from scratch

            # If we successfully extract data without a stale element issue, return it
            return deck_data

        except StaleElementReferenceException:
            print(f"StaleElementReferenceException at main level. Retrying...")
            attempt += 1
            time.sleep(2 * attempt)  # Increase wait time with each retry

    # If we reach the max attempts, extract what we can while skipping any stale elements
    print("Max retries reached. Skipping stale elements and proceeding with available data.")
    deck_data = []

    td_elements = driver.find_elements(By.CSS_SELECTOR, "td.deck-name")

    for row in td_elements:
        try:
            deck_link_element = row.find_element(By.TAG_NAME, "a")
            deck_link = deck_link_element.get_attribute('href') if deck_link_element else "No Link"

            deck_data.append({
                "site": "source b",
                "link": deck_link,
                "archetype": "Unknown Archetype",
                "format": "Unknown Format",
                "status": "not processed"
            })

        except StaleElementReferenceException:
            print(f"Skipping single stale element.")

    return deck_data


# Function to scrape all decks from all pages
def scrape_source_b_decks(url):
    """Scrapes all decks from Source B, iterating through all pages."""

    print(f"Scraping: {url}")
    driver.get(url)

    # Wait for the first page to load
    WebDriverWait(driver, 360).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "td.deck-name"))
    )

    page_counter = 1

    while True:
        print(f"Scraping Page: {page_counter}")

        # Extract fresh data for the current page, handling stale elements correctly
        all_deck_data.extend(extract_deck_data())

        try:
            # Store current page source to track changes
            old_page_source = driver.page_source

            # Find "Next" button
            next_button = WebDriverWait(driver, 360).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.next.paginate"))
            )

            # Click "Next"
            driver.execute_script("arguments[0].click();", next_button)

            # Wait until the new page loads (detects change in content)
            WebDriverWait(driver, 360).until(
                lambda d: d.page_source != old_page_source
            )

            # Small wait to ensure elements are fully loaded
            time.sleep(1.5)
            page_counter += 1

        except TimeoutException:
            print("No more pages found. Stopping.")
            break  # Stop when no more pages exist


# Example: List of URLs
url_list = [
    source_b_url
]

# Initialize all_deck_data
all_deck_data = []

# Loop through the URLs
for url in url_list:
    scrape_source_b_decks(url)
    print("Source B: Data saved to all_deck_data")

# Write scraped data to JSON
with open("source_b_deck_links.json", "w", encoding="utf-8") as f:
    json.dump(all_deck_data, f, indent=4, ensure_ascii=False)

print("Data saved to source_b_deck_links.json")

# Close Selenium after all URLs are processed
driver.quit()
