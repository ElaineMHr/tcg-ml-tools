from mtg_tools.scraper.fetch_deck_data_source_a import fetch_deck_data_source_a as fetch_source_a
from mtg_tools.scraper.fetch_deck_data_source_b import fetch_deck_data_source_b as fetch_source_b

import json
import threading
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from itertools import cycle
from math import floor

fetchers = {
    "source_a": fetch_source_a,
    "source_b": fetch_source_b
}

file_map = {
    "source_a": "source_a_deck_links.json",
    "source_b": "source_b_deck_links.json"
}

result_file = "scraped_deck_data.json"
save_every = 25000
pattern_slots = 20
max_per_source = 10

lock = threading.Lock()
terminate_event = threading.Event()
deck_lists = {}
deck_files = {}
fetch_queue = []
progress_counter = 0
last_save_checkpoint = 0
initial_processed = 0
total_to_process = 0
start_time = time.time()


def build_interleaving_pattern():
    total = {k: len([d for d in deck_lists[k] if d.get("status") != "processed"]) for k in deck_lists}
    total_urls = sum(total.values())
    if total_urls == 0:
        return []
    weights = {k: min((v / total_urls) * pattern_slots, max_per_source) for k, v in total.items()}
    rounded = {k: floor(w) for k, w in weights.items()}
    leftover = pattern_slots - sum(rounded.values())

    remainders = sorted(((k, weights[k] - rounded[k]) for k in weights if rounded[k] < max_per_source), key=lambda x: -x[1])
    rem_iter = cycle([k for k, _ in remainders])
    added = 0
    while added < leftover:
        k = next(rem_iter)
        if rounded[k] < max_per_source:
            rounded[k] += 1
            added += 1
    return [(k, rounded[k]) for k in rounded if rounded[k] > 0]


def load_deck_files():
    global initial_processed, total_to_process
    for site, file in file_map.items():
        with open(file, "r", encoding="utf-8") as f:
            decks = json.load(f)
            deck_lists[site] = decks
            deck_files[site] = file
    initial_processed = sum(1 for decks in deck_lists.values() for d in decks if d.get("status") == "processed")
    total_to_process = sum(len(decks) for decks in deck_lists.values())


def load_results():
    global results
    if Path(result_file).exists():
        with open(result_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                results = json.loads(content)
                print(f"Loaded {len(results)} previously scraped results.")
            else:
                results = {}
                print("Results file was empty. Starting fresh.")
    else:
        results = {}
        print("No previous results file found. Starting fresh.")


def resync_missing_results():
    global initial_processed
    print("Checking for sync issues between deck status and saved results...")
    missing = []
    fixed = 0

    for site, decks in deck_lists.items():
        for deck in decks:
            url = deck["link"]
            if deck.get("status") == "processed" and url not in results:
                # Marked as processed, but result missing → requeue
                missing.append((site, deck))
            elif deck.get("status") != "processed" and url in results:
                # Result exists but not marked → fix status
                deck["status"] = "processed"
                fixed += 1

    if fixed:
        print(f"Updated status to 'processed' for {fixed} decks already in results.")

    if missing:
        print(f"Found {len(missing)} decks marked as processed but not saved. Adding back to queue.")
        fetch_queue.extend(missing)
        initial_processed -= len(missing)
    else:
        print("All processed decks have corresponding results.")

def refill_queue(pattern):
    grouped = {site: [deck for deck in deck_lists[site] if deck.get("status") != "processed"] for site in deck_lists}
    temp_queue = []
    while any(grouped.values()):
        for site, count in pattern:
            for _ in range(count):
                if grouped[site]:
                    temp_queue.append((site, grouped[site].pop(0)))
    return temp_queue


def save_all():
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for site, file in deck_files.items():
        with open(file, "w", encoding="utf-8") as f:
            json.dump(deck_lists[site], f, indent=2)


def signal_handler(sig, frame):
    print("\n[!] Keyboard interrupt received. Saving progress...")
    terminate_event.set()
    save_all()
    print("Progress saved. Exiting.")
    exit(0)


def format_eta(seconds_per_deck, remaining):
    remaining_seconds = seconds_per_deck * remaining
    minutes = int(remaining_seconds // 60)
    seconds = int(remaining_seconds % 60)
    return f"ETA: {minutes}m {seconds}s"


def worker():
    global progress_counter, last_save_checkpoint
    count = 0
    while not terminate_event.is_set():
        with lock:
            if not fetch_queue:
                return
            site, deck = fetch_queue.pop(0)
            if deck.get("status") == "processed":
                continue
            url = deck["link"]

        try:
            data = fetchers[site](url)
            with lock:
                results[url] = data
                deck["status"] = "processed"
                progress_counter += 1
                current_total = initial_processed + progress_counter

                elapsed = time.time() - start_time
                decks_done = max(progress_counter, 1)
                seconds_per_deck = elapsed / decks_done
                remaining = total_to_process - current_total
                eta = format_eta(seconds_per_deck, remaining)

                print(f"[+] [{current_total}/{total_to_process}] Decks processed | {eta}")

                if current_total - last_save_checkpoint >= save_every:
                    print("Auto-saving progress...")
                    save_all()
                    last_save_checkpoint = current_total

                if count % 10000 == 0 and count > 0:
                    print(f"[~] Thread processed {count}. Refreshing pattern...")
                    pattern = build_interleaving_pattern()
                    fetch_queue.extend(refill_queue(pattern))

                count += 1
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")


def run_workers():
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            for _ in range(max_workers):
                executor.submit(worker)
            while not terminate_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[!] Manual interrupt. Stopping workers...")
            terminate_event.set()
            executor.shutdown(wait=False)
            save_all()
            print("Progress saved. Exiting.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    load_deck_files()
    load_results()
    resync_missing_results()

    # Recalculate after any updates to status flags
    initial_processed = sum(
        1 for decks in deck_lists.values()
        for d in decks if d.get("status") == "processed"
    )
    print(f"Already processed: {initial_processed} / {total_to_process}")

    pattern = build_interleaving_pattern()
    fetch_queue.extend(refill_queue(pattern))

    run_workers()

    save_all()
    print(f"All scraping complete. Progress saved.")
