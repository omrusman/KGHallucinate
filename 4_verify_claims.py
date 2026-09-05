import json
import os
import re
import torch
from transformers import pipeline
from tqdm import tqdm

EXCLUSIVE_FIELDS = {"date_of_birth", "birthplace"}

LIST_FIELDS = {"occupation", "field_of_work", "educated_at", "employer",
                "notable_work", "award_received", "citizenship"}

FIELD_TEMPLATES = {
    "date_of_birth": "{name} was born in the year {value}.",
    "birthplace": "{name} was born in {value}.",
    "occupation": "{name}'s occupation is {value}.",
    "field_of_work": "{name} works in the field of {value}.",
    "educated_at": "{name} was educated at {value}.",
    "employer": "{name} worked at or was employed by {value}.",
    "notable_work": "{name} is known for {value}.",
    "award_received": "{name} received the following award or honor: {value}.",
    "citizenship": "{name} was a citizen of {value}.",
}

MAX_ITEMS_PER_FIELD = 50
BATCH_SIZE = 32
ENTAIL_THRESHOLD = 0.6
CONTRA_THRESHOLD = 0.6
MAX_PREMISE_CHARS = 400

BIRTH_KEYWORDS = {"born", "birth", "borne"}

COMMON_CAPITALIZED_WORDS = {
    "he", "she", "his", "her", "it", "they", "their", "the", "this", "these",
    "was", "were", "is", "are", "in", "on", "at", "a", "an", "also", "with",
    "from", "born", "held", "worked", "studied", "received", "known",
    "developed", "professor", "professorships",
}


def extract_candidate_entities(claim, exclude_words=frozenset()):
    words = claim.split()
    candidates = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^\w\u00C0-\u017F]", "", w)
        if len(clean) < 4:
            continue
        if i == 0:
            continue
        if clean.lower() in COMMON_CAPITALIZED_WORDS:
            continue
        if clean.lower() in exclude_words:
            continue
        if clean[0].isupper():
            candidates.append(clean)
    return candidates


def substring_match(claim, field_items, entity_name):
    name_words = {w.lower() for w in re.split(r"[^\w\u00C0-\u017F]", entity_name) if w}
    candidates = extract_candidate_entities(claim, exclude_words=name_words)
    if not candidates:
        return None
    for field, item_value in field_items:
        item_lower = item_value.lower()
        for cand in candidates:
            if cand.lower() in item_lower:
                return field, item_value
    return None


def build_fact_data(entity):
    name = entity["name"]
    facts = entity.get("facts", {})
    sentences = []
    list_field_raw_items = []

    for field, template in FIELD_TEMPLATES.items():
        value = facts.get(field)
        if not value:
            continue

        kind = "exclusive" if field in EXCLUSIVE_FIELDS else "list"

        whole_sentence = template.format(name=name, value=value)
        if len(whole_sentence) > MAX_PREMISE_CHARS:
            whole_sentence = whole_sentence[:MAX_PREMISE_CHARS].rsplit(" ", 1)[0] + "."
        sentences.append((field, kind, whole_sentence))

        items = [v.strip() for v in value.split(",") if v.strip()]
        if len(items) > 1:
            for item in items[:MAX_ITEMS_PER_FIELD]:
                item_sentence = template.format(name=name, value=item)
                if len(item_sentence) > MAX_PREMISE_CHARS:
                    item_sentence = item_sentence[:MAX_PREMISE_CHARS].rsplit(" ", 1)[0] + "."
                sentences.append((field, kind, item_sentence))
                if kind == "list":
                    list_field_raw_items.append((field, item))
        elif kind == "list" and items:
            list_field_raw_items.append((field, items[0]))

    return sentences, list_field_raw_items


def classify_batch(classifier, inputs):
    try:
        return classifier(inputs, batch_size=len(inputs), truncation=True)
    except TypeError:
        pass
    except Exception as e:
        print(f"    batch call failed ({e}); falling back to per-item classification")

    results = []
    for inp in inputs:
        try:
            r = classifier([inp])[0]
        except Exception as e:
            print(f"    single-item classification failed: {e}")
            r = {"label": "NEUTRAL", "score": 0.0}
        results.append(r)
    return results


def main():
    if not os.path.exists("data/claims.json"):
        print("data/claims.json not found! Please run 3_decompose_claims.py first.")
        return

    with open("data/claims.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loading roberta-large-mnli...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    classifier = pipeline("text-classification", model="roberta-large-mnli", device=device)
    if device == -1:
        torch.set_num_threads(os.cpu_count())

    print(f"Loaded {len(data)} entities.")

    tasks = []
    n_substring_hits = 0

    for e_idx, entity in enumerate(data):
        fact_sentences, list_field_raw_items = build_fact_data(entity)
        entity["verified_claims"] = {}

        for model, claims in entity.get("decomposed_claims", {}).items():
            entity["verified_claims"][model] = [None] * len(claims)

            for c_idx, claim in enumerate(claims):
                if not fact_sentences:
                    entity["verified_claims"][model][c_idx] = {
                        "claim": claim, "label": "Unverifiable", "score": 0.0,
                        "matched_fact_field": None, "matched_fact_sentence": None,
                        "method": "none",
                    }
                    continue

                match = substring_match(claim, list_field_raw_items, entity["name"])
                if match is not None:
                    field, item_value = match
                    n_substring_hits += 1
                    entity["verified_claims"][model][c_idx] = {
                        "claim": claim, "label": "Supported", "score": 1.0,
                        "matched_fact_field": field,
                        "matched_fact_sentence": f"[substring match] {item_value}",
                        "method": "substring",
                    }
                    continue

                claim_lower = claim.lower()
                for field, kind, premise in fact_sentences:
                    if kind == "exclusive" and not any(kw in claim_lower for kw in BIRTH_KEYWORDS):
                        continue
                    tasks.append({
                        "entity_idx": e_idx, "model": model, "claim_idx": c_idx,
                        "claim": claim, "field": field, "kind": kind, "premise": premise,
                    })

    print(f"Substring fast-path resolved {n_substring_hits} claims directly (no NLI needed).")
    print(f"Built {len(tasks)} (claim, fact) comparison pairs for the remaining claims. "
          f"Running batched NLI classification...")

    best = {}
    n_batch_errors = 0

    for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc="Verifying (batched)"):
        batch = tasks[i:i + BATCH_SIZE]
        inputs = [{"text": t["premise"], "text_pair": t["claim"]} for t in batch]

        results = classify_batch(classifier, inputs)
        if results is None or len(results) != len(batch):
            n_batch_errors += 1
            continue

        for t, result in zip(batch, results):
            key = (t["entity_idx"], t["model"], t["claim_idx"])
            if key not in best:
                best[key] = {"entail": (0.0, None, None), "contra_exclusive": (0.0, None, None)}

            label = result["label"]
            score = result["score"]

            if label == "ENTAILMENT" and score > best[key]["entail"][0]:
                best[key]["entail"] = (score, t["field"], t["premise"])

            if label == "CONTRADICTION" and t["kind"] == "exclusive" \
                    and score > best[key]["contra_exclusive"][0]:
                best[key]["contra_exclusive"] = (score, t["field"], t["premise"])

    if n_batch_errors:
        print(f"WARNING: {n_batch_errors} batches had issues (see messages above).")

    n_missing = 0
    for e_idx, entity in enumerate(data):
        for model, claims in entity.get("decomposed_claims", {}).items():
            for c_idx, claim in enumerate(claims):
                if entity["verified_claims"][model][c_idx] is not None:
                    continue

                key = (e_idx, model, c_idx)
                b = best.get(key)
                if b is None:
                    n_missing += 1
                    entity["verified_claims"][model][c_idx] = {
                        "claim": claim, "label": "Unverifiable", "score": 0.0,
                        "matched_fact_field": None, "matched_fact_sentence": None,
                        "method": "nli",
                    }
                    continue

                contra_score, contra_field, contra_premise = b["contra_exclusive"]
                entail_score, entail_field, entail_premise = b["entail"]

                if contra_score >= CONTRA_THRESHOLD:
                    entity["verified_claims"][model][c_idx] = {
                        "claim": claim, "label": "Contradicted (Hallucination)",
                        "score": contra_score, "matched_fact_field": contra_field,
                        "matched_fact_sentence": contra_premise, "method": "nli",
                    }
                elif entail_score >= ENTAIL_THRESHOLD:
                    entity["verified_claims"][model][c_idx] = {
                        "claim": claim, "label": "Supported",
                        "score": entail_score, "matched_fact_field": entail_field,
                        "matched_fact_sentence": entail_premise, "method": "nli",
                    }
                else:
                    entity["verified_claims"][model][c_idx] = {
                        "claim": claim, "label": "Unverifiable",
                        "score": max(contra_score, entail_score),
                        "matched_fact_field": None, "matched_fact_sentence": None,
                        "method": "nli",
                    }

    if n_missing:
        print(f"NOTE: {n_missing} claims had no successful NLI comparisons at all.")

    os.makedirs("data", exist_ok=True)
    with open("data/verified.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("Verification complete. Results saved to data/verified.json")


if __name__ == "__main__":
    main()
