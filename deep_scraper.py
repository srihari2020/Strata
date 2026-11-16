import os
import requests
import time

os.makedirs("data", exist_ok=True)

def scrape_deep_web_academic():
    print("Initiating deep academic crawler (Crossref API)...")
    
    # Politeness header required by academic APIs
    headers = {
        "User-Agent": "StrataArchaeologyBot/1.0 (mailto:strata@local.dev)"
    }
    
    # Target peer-reviewed archaeological excavations with available abstracts
    url = "https://api.crossref.org/works"
    params = {
        "query": "archaeology excavation site findings",
        "select": "title,abstract",
        "rows": 50,  # Number of academic papers to rip
        "filter": "has-abstract:true"
    }
    
    print("Querying global academic databases...")
    try:
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            
            extracted_text = ""
            for item in items:
                title = item.get("title", [""])[0]
                abstract = item.get("abstract", "")
                
                if title and abstract:
                    # Clean up academic XML tags common in these databases
                    abstract = abstract.replace('<jats:p>', '').replace('</jats:p>', '').replace('<jats:title>', '')
                    extracted_text += f"Title: {title}\nExcavation Abstract: {abstract}\n\n"
                    
            if extracted_text:
                filename = "data/deep_academic_journals.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                print(f"Successfully ripped {len(items)} peer-reviewed papers into {filename}")
        else:
            print(f"API blocked request with status: {response.status_code}")
            
    except Exception as e:
        print(f"Crawler encountered an error: {e}")

if __name__ == "__main__":
    scrape_deep_web_academic()
