import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import os

# --- 1. DEFINITIONS ---

high_qids = ["Q937", "Q935", "Q7186", "Q1035", "Q307", "Q9036", "Q7251", "Q7259", "Q529", "Q9106", "Q40904", "Q7085", "Q9021", "Q9130", "Q39246", "Q17714", "Q9095", "Q8750", "Q8963", "Q619"]
medium_qids = ["Q47480", "Q65989", "Q8753", "Q58978", "Q56189", "Q9123", "Q132537", "Q48983", "Q7474", "Q9047", "Q17455", "Q39607", "Q43393", "Q41257", "Q84296", "Q153243", "Q41688", "Q37463", "Q177681", "Q949"]
low_qids = ["Q163415", "Q182031", "Q186497", "Q184571", "Q80905", "Q178344", "Q172840", "Q173028", "Q183655", "Q130113", "Q44286", "Q133267", "Q155790", "Q57100", "Q76797", "Q83552", "Q1043", "Q37610", "Q9344", "Q92771"]

FACT_PROPERTIES = {
    "P569": "date_of_birth",
    "P19": "birthplace",
    "P106": "occupation",
    "P101": "field_of_work",
    "P69": "educated_at",
    "P108": "employer",
    "P800": "notable_work",
    "P166": "award_received",
    "P27": "citizenship",
}

# Setup robust session
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

def extract_label(labels, fallback_qid):
    return labels.get('en', {}).get('value') or \
           labels.get('en-gb', {}).get('value') or \
           labels.get('mul', {}).get('value') or \
           labels.get('de', {}).get('value') or \
           labels.get('fr', {}).get('value') or \
           fallback_qid

def fetch_primary(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        resp = session.get(url, headers={'User-Agent': 'NLP_Agentic/1.0'}, timeout=15)
        if resp.status_code == 200:
            return qid, resp.json().get("entities", {}).get(qid, {})
    except Exception as e:
        print(f"Failed to fetch {qid}: {e}")
    return qid, {}

def main():
    print("Collecting 60 perfect scientists dynamically from Wikidata...")
    raw_data = {}
    
    # 1. Fetch the 60 scientists
    all_qids = high_qids + medium_qids + low_qids
    print(f"Fetching facts for {len(all_qids)} entities via CDN-cached EntityData...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_primary, q) for q in all_qids]
        for f in concurrent.futures.as_completed(futures):
            qid, data = f.result()
            if data:
                raw_data[qid] = data
            time.sleep(0.1)
            
    print(f"Successfully downloaded {len(raw_data)} out of {len(all_qids)} primary entities.")

    # 2. Extract facts and identify referenced Q-IDs
    results = []
    qids_to_resolve = set()
    
    for tier_name, qids in [("High", high_qids), ("Medium", medium_qids), ("Low", low_qids)]:
        for qid in qids:
            ent = raw_data.get(qid)
            if not ent:
                continue
                
            labels = ent.get("labels", {})
            label = extract_label(labels, qid)
            sitelinks = len(ent.get("sitelinks", {}))
            claims = ent.get("claims", {})
            
            facts = {}
            for prop, key in FACT_PROPERTIES.items():
                if prop not in claims:
                    continue
                values = []
                for claim in claims[prop]:
                    mainsnak = claim.get("mainsnak", {})
                    if mainsnak.get("snaktype") != "value": continue
                    datavalue = mainsnak.get("datavalue", {})
                    vtype = datavalue.get("type")
                    if vtype == "time":
                        raw = datavalue["value"]["time"]
                        values.append(raw.lstrip("+").split("T")[0].split("-")[0])
                    elif vtype == "wikibase-entityid":
                        item_qid = datavalue["value"]["id"]
                        values.append(item_qid)
                        qids_to_resolve.add(item_qid)
                    else:
                        values.append(str(datavalue.get("value")))
                if values:
                    facts[key] = values
                    
            results.append({
                "id": qid,
                "name": label,
                "sitelinks": sitelinks,
                "facts": facts,
                "tier": tier_name
            })

    # 3. Resolve all referenced Q-IDs into human-readable text
    to_resolve_list = list(qids_to_resolve)
    print(f"Resolving {len(to_resolve_list)} referenced labels...")
    label_map = {}
    
    for i in range(0, len(to_resolve_list), 50):
        batch = to_resolve_list[i:i + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "en|en-gb|mul|de|fr|es",
            "format": "json",
        }
        try:
            resp = session.get("https://www.wikidata.org/w/api.php", params=params, headers={"User-Agent": "NLP_Agentic/1.0"}, timeout=30)
            if resp.status_code == 200:
                resolved_data = resp.json().get("entities", {})
                for r_qid, r_ent in resolved_data.items():
                    r_labels = r_ent.get("labels", {})
                    label_map[r_qid] = extract_label(r_labels, r_qid)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
        time.sleep(1) # Be extremely polite to the API
        
    # 4. Inject resolved labels into final facts
    for item in results:
        for key, values in item["facts"].items():
            resolved = [label_map.get(v, v) if str(v).startswith("Q") else v for v in values]
            item["facts"][key] = ", ".join(resolved)

    # 5. Save output
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "entities.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved {len(results)} fully resolved entities to {out_path}")

if __name__ == "__main__":
    main()
