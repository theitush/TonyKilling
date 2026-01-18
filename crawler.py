import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

def get_all_minutes():
    """
    Scrape ratetony.com/minutes and extract all minute data using Selenium
    Supports resume functionality - will skip already scraped entries
    """
    url = "https://www.ratetony.com/minutes"
    output_file = 'data/ratetony_scraped_data.json'

    print(f"[DEBUG] Starting scraper for {url}")

    # Load existing data if available
    existing_data = []
    scraped_names = set()

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            scraped_names = {entry['name'] + '|' + entry['episode'] for entry in existing_data}
            print(f"[DEBUG] Loaded {len(existing_data)} existing entries from {output_file}")
    except FileNotFoundError:
        print(f"[DEBUG] No existing data found, starting fresh")
    except Exception as e:
        print(f"[WARN] Could not load existing data: {e}")

    # Setup Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    chrome_options.add_argument('--window-size=1920,1080')

    driver = None
    try:
        print("[DEBUG] Initializing Selenium WebDriver...")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        # Wait for minute cards to load
        print("[DEBUG] Waiting for page to load...")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "minute-card"))
        )

        # Give extra time for all cards to render
        time.sleep(2)

        # Find all minute cards
        minute_cards = driver.find_elements(By.CLASS_NAME, "minute-card")
        total = len(minute_cards)
        print(f"[DEBUG] Found {total} minute cards!")

        minutes_data = existing_data
        new_count = 0
        skipped_count = 0

        for i, card in enumerate(minute_cards):
            try:
                # Extract basic info from the card
                name = card.find_element(By.CLASS_NAME, "owner-name").text
                episode = card.find_element(By.CLASS_NAME, "episode-title").text

                # Check if already scraped
                unique_key = name + '|' + episode
                if unique_key in scraped_names:
                    skipped_count += 1
                    if (i + 1) % 50 == 0:
                        print(f"[DEBUG] Progress: {i + 1}/{total} ({new_count} new, {skipped_count} skipped)")
                    continue

                score = card.find_element(By.CLASS_NAME, "score-float-value").text

                # Get YouTube link
                youtube_link = "N/A"
                try:
                    play_btn = card.find_element(By.CLASS_NAME, "icon-btn-watch")
                    youtube_link = play_btn.get_attribute("href")
                except:
                    pass

                # Click the transcript button to open modal
                print(f"[DEBUG] Processing NEW card {i + 1}/{total}: {name}")
                try:
                    transcript_btn = card.find_element(By.CLASS_NAME, "icon-btn-transcript")
                    driver.execute_script("arguments[0].click();", transcript_btn)

                    # Wait for modal to open and get full transcript
                    time.sleep(1.5)  # Wait for modal animation

                    # Try to find the full transcript in the modal
                    try:
                        # First try to find the specific transcript content element
                        # Look for common transcript container classes
                        transcript_elem = None
                        selectors = [
                            ".transcript-full",
                            ".transcript-content",
                            "[role='dialog'] .transcript",
                            "[role='dialog'] p",
                            "[role='dialog'] div[class*='transcript']"
                        ]

                        for selector in selectors:
                            try:
                                transcript_elem = driver.find_element(By.CSS_SELECTOR, selector)
                                if transcript_elem and len(transcript_elem.text.strip()) > 100:
                                    transcript = transcript_elem.text.strip()
                                    break
                            except:
                                continue

                        # If no specific element found, get modal text and clean it
                        if not transcript_elem:
                            modal = driver.find_element(By.CSS_SELECTOR, "[role='dialog']")
                            modal_text = modal.text

                            # Remove common header/footer elements
                            lines = modal_text.split('\n')
                            # Skip first few lines (TRANSCRIPT, name, MIN #, TIMECODE, KT: OUTT)
                            # and last few lines (Open on YouTube, Close, See Episode)
                            cleaned_lines = []
                            skip_keywords = ['TRANSCRIPT', 'MIN #', 'TIMECODE', 'KT:', 'Open on YouTube', 'Close', 'See Episode', 'See Profile']

                            for line in lines:
                                line_stripped = line.strip()
                                # Skip if line matches skip keywords or is the name/episode we already have
                                if any(keyword in line_stripped for keyword in skip_keywords):
                                    continue
                                if line_stripped == name or line_stripped == episode:
                                    continue
                                if line_stripped:  # Keep non-empty lines
                                    cleaned_lines.append(line_stripped)

                            transcript = '\n'.join(cleaned_lines).strip()
                    except:
                        # Fallback to preview if modal not found
                        transcript = card.find_element(By.CLASS_NAME, "transcript-preview").text

                    # Close the modal (press ESC or click close button)
                    driver.find_element(By.TAG_NAME, "body").send_keys('\ue00c')  # ESC key
                    time.sleep(0.5)

                except Exception as e:
                    print(f"[WARN] Could not get full transcript for {name}: {e}")
                    # Fallback to preview
                    try:
                        transcript = card.find_element(By.CLASS_NAME, "transcript-preview").text
                    except:
                        transcript = "N/A"

                # Create new entry
                new_entry = {
                    'name': name,
                    'episode': episode,
                    'youtube_link': youtube_link,
                    'score': score,
                    'transcript': transcript
                }

                # Append to data and save immediately
                minutes_data.append(new_entry)
                scraped_names.add(unique_key)
                new_count += 1

                # Write to disk immediately
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minutes_data, f, indent=2, ensure_ascii=False)

                print(f"[DEBUG] ✓ Saved! ({new_count} new entries total)")

            except Exception as e:
                print(f"[ERROR] Failed to parse card {i}: {e}")
                continue

        print(f"\n✓ Scraping complete!")
        print(f"  Total entries: {len(minutes_data)}")
        print(f"  New entries: {new_count}")
        print(f"  Skipped (already scraped): {skipped_count}")
        print(f"  Data saved to: {output_file}")

        return minutes_data

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Always close the browser
        if driver:
            print("[DEBUG] Closing browser...")
            driver.quit()

if __name__ == "__main__":
    get_all_minutes()
