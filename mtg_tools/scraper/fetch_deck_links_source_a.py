import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import NoAlertPresentException, TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

source_a_url = os.environ["SOURCE_A_URL"]

# Initialize Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")  # Headless mode
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to close popups or confirmation alerts
def close_alert():
    """Attempts to close any confirmation alert or cookie consent popup."""
    try:
        # Handle JavaScript alert popups
        WebDriverWait(driver, 2).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        print("JavaScript alert detected. Closing it...")
        alert.dismiss()
        time.sleep(1)

    except (NoAlertPresentException, TimeoutException):
        pass  # No JavaScript alert found, continue normally

    try:
        # Handle Cookie Consent Banner
        cookie_button_xpath = "//*[@id='sd-cmp']/div[2]/div[1]/div/div[1]/div/div/div[2]/div/button[2]/span"
        cookie_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, cookie_button_xpath))
        )
        print("Cookie consent detected. Accepting it...")
        cookie_button.click()
        time.sleep(2)  # Allow page update after clicking

    except (TimeoutException, NoSuchElementException):
        print("No cookie popup found. Continuing.")

# Function to extract deck data
def extract_deck_data():
    """Extracts deck data from the current page."""
    deck_data = []

    try:
        # Ensure elements are present before scraping
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "td.S12 a"))
        )
    except TimeoutException:
        print("No deck data found on the page.")
        return deck_data  # Return empty list if no data is found

    soup = BeautifulSoup(driver.page_source, "html.parser")
    elements = soup.select("td.S12 a")
    for row in elements:
        deck_link = source_a_url + row["href"]

        deck_data.append({
            "site": "source a",
            "link": deck_link,
            "archetype": "Unknown Archetype",
            "format": "Unknown Format",
            "status": "not processed"
        })

    return deck_data

# Function to scrape all decks from all pages with retry mechanism
def scrape_source_a_decks(url):
    """Scrapes all decks from a single Source A URL, iterating through all pages."""
    global all_deck_data  # Use global variable to store deck data

    print(f"Source A: Scraping: {url}")
    driver.get(url)
    time.sleep(1.5)  # Let page load

    # Close any confirmation alerts if present
    close_alert()

    page_counter = 1
    old_page_source = None  # Store old page to detect changes

    try:
        while True:
            print(f"Scraping Page: {page_counter}")

            # Extract data and save it
            all_deck_data.extend(extract_deck_data())

            # Store current page source before clicking "Next"
            new_page_source = driver.page_source

            # If the page source hasn't changed, we assume we're stuck and break
            if new_page_source == old_page_source:
                print("Page did not change after clicking 'Next'. Stopping to prevent an infinite loop.")
                break

            old_page_source = new_page_source  # Update for next iteration

            # Retry mechanism for "Next" button
            next_button_found = False
            attempts = 0
            max_attempts = 5

            while attempts < max_attempts:
                try:
                    # Find "Next" button
                    next_buttons = driver.find_elements(By.CLASS_NAME, "Nav_norm")
                    for btn in next_buttons:
                        if btn.text.strip().lower() == "next":
                            # Scroll slightly before clicking
                            button_y = btn.location['y']
                            driver.execute_script(f"window.scrollTo(0, {button_y - 200});")

                            # Click "Next"
                            btn.click()

                            # Wait until page actually changes
                            WebDriverWait(driver, 60).until(lambda d: d.page_source != old_page_source)

                            time.sleep(1)  # Let page stabilize
                            next_button_found = True
                            break  # Exit inner loop

                    if next_button_found:
                        break  # Exit retry loop

                except Exception:
                    attempts += 1
                    wait_time = 2 * attempts
                    print(f"'Next' button not found. Retrying... ({attempts}/{max_attempts})")
                    time.sleep(wait_time)

            if not next_button_found:
                print("Source A: No more pages found after retries. Stopping.")
                break  # Stop when no more pages exist

            page_counter += 1

    except KeyboardInterrupt:
        print("Script interrupted manually. Saving progress before exiting.")
        save_progress()
        print("Progress saved. Exiting.")
        exit(0)

def save_progress():
    """Saves scraped data to JSON before exiting."""
    with open("source_a_deck_links.json", "w", encoding="utf-8") as f:
        json.dump(all_deck_data, f, indent=4, ensure_ascii=False)
    print("Data saved to source_a_deck_links.json")

# Example: List of URLs
url_list = [
    source_a_url + "/search"
]

# Initialize all_deck_data
all_deck_data = []

# Loop through the URLs
for url in url_list:
    scrape_source_a_decks(url)
    print("Source A: Data saved to all_deck_data")

# Save final results after scraping
save_progress()

# Close Selenium after all URLs are processed
driver.quit()
