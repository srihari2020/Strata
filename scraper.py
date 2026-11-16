import os
import requests
import time

os.makedirs("data", exist_ok=True)

def scrape_wikipedia_archaeology():
    print("Initiating global crawler...")
    search_url = "https://en.wikipedia.org/w/api.php"
    
    # Wikipedia requires a descriptive User-Agent
    headers = {
        "User-Agent": "StrataArchaeologyBot/1.0 (https://github.com/yourusername/strata; your@email.com)"
    }
    
    target_sites = [
        "Göbekli_Tepe", 
        "Machu_Picchu", 
        "Mohenjo-daro", 
        "Pompeii", 
        "Great_Pyramid_of_Giza",
        "Stonehenge",
        "Petra"
    ]
    
    for site in target_sites:
        print(f"Scraping data for: {site.replace('_', ' ')}...")
        params = {
            "action": "query",
            "format": "json",
            "titles": site,
            "prop": "extracts",
            "explaintext": True
        }
        
        try:
            response = requests.get(search_url, params=params, headers=headers)
            
            # Check if the request was successful before parsing JSON
            if response.status_code == 200:
                data = response.json()
                pages = data['query']['pages']
                for page_id, page_info in pages.items():
                    if 'extract' in page_info:
                        content = page_info['extract']
                        
                        filename = f"data/{site}.txt"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Successfully saved {len(content.split())} words to {filename}")
            else:
                print(f"Failed with status code: {response.status_code}")
                
            time.sleep(1) # Be polite to the API
            
        except Exception as e:
            print(f"Failed to scrape {site}: {e}")

if __name__ == "__main__":
    scrape_wikipedia_archaeology()
    print("\nScraping complete. Ready for ingestion.")
