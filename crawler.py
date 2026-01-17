from bs4 import BeautifulSoup
import requests
import json

def get_all_minutes():
    """
    Scrape ratetony.com/minutes and extract all minute data
    """
    url = "https://www.ratetony.com/minutes"
    print(f"[DEBUG] Starting scraper for {url}")
    
    try:
        print("[DEBUG] Fetching page with requests...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"[DEBUG] Page retrieved ({len(response.text)} bytes)")
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for article elements with class "minute-card"
        print("[DEBUG] Searching for minute cards...")
        minute_cards = soup.find_all('article', class_='minute-card')
        total = len(minute_cards)
        print(f"[DEBUG] Found {total} minute cards!")
        
        # Extract data from each card
        minutes_data = []
        
        for i, card in enumerate(minute_cards):
            try:
                # Name
                name_elem = card.find('h3', class_='owner-name')
                name = name_elem.get_text(strip=True) if name_elem else "N/A"
                
                # Episode info
                episode_elem = card.find('p', class_='episode-title')
                episode = episode_elem.get_text(strip=True) if episode_elem else "N/A"
                
                # YouTube link
                youtube_link = "N/A"
                play_btn = card.find('a', class_='icon-btn-watch')
                if play_btn and play_btn.get('href'):
                    youtube_link = play_btn.get('href')
                
                # Score
                score_elem = card.find('span', class_='score-float-value')
                score = score_elem.get_text(strip=True) if score_elem else "N/A"
                
                # Transcript
                transcript_elem = card.find('div', class_='transcript-preview')
                transcript = transcript_elem.get_text(strip=True) if transcript_elem else "N/A"
                
                minutes_data.append({
                    'name': name,
                    'episode': episode,
                    'youtube_link': youtube_link,
                    'score': score,
                    'transcript': transcript
                })
                
                if (i + 1) % 100 == 0:
                    print(f"[DEBUG] Processed {i + 1}/{total} cards...")
                    
            except Exception as e:
                print(f"[ERROR] Failed to parse card {i}: {e}")
                continue
        
        print(f"\n✓ Successfully scraped {len(minutes_data)} minutes!")
        
        # Save to JSON
        output_file = 'ratetony_scraped_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(minutes_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Data saved to {output_file}")
        
        # Display first 3 entries as preview
        print("\n=== First 3 entries ===")
        for i, minute in enumerate(minutes_data[:3]):
            print(f"\n[{i+1}] {minute['name']}")
            print(f"    Episode: {minute['episode']}")
            print(f"    Score: {minute['score']}")
            print(f"    YouTube: {minute['youtube_link']}")
            print(f"    Transcript: {minute['transcript'][:100]}...")
        
        return minutes_data
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    get_all_minutes()
