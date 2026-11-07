"""Warranty V4: auditable keyword and context-based candidate mining.

Run from any working directory:
    python src/nlp/warranty_semantic_v4.py

Requires Python standard library only. Supply Train_subset_50k.csv inside
`train subset.zip` in repo/data, repo parent/data, or ~/Downloads.

IMPORTANT: An extracted candidate is not a verified warranty mention.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_warranty"
OUT.mkdir(exist_ok=True)

ZIP_PATHS = [
    ROOT / "data" / "train subset.zip",
    ROOT.parent / "data" / "train subset.zip",
    ROOT / "train subset.zip",
    Path.home() / "Downloads" / "train subset.zip",
]
zip_path = next((p for p in ZIP_PATHS if p.is_file()), None)
if zip_path is None:
    raise FileNotFoundError(
        "Place 'train subset.zip' in project/data or ~/Downloads. "
        f"Searched: {[str(p) for p in ZIP_PATHS]}"
    )

with zipfile.ZipFile(zip_path) as z:
    dataset = next((f for f in z.namelist() if f.lower().endswith(".csv")), None)
    if not dataset:
        raise ValueError("No CSV file inside train subset.zip")
    with z.open(dataset) as stream:
        reviews = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))

texts = [
    (r.get("full_text") or " ".join(filter(None, [r.get("summary"), r.get("reviewText_clean")])))
    for r in reviews
]
if len(reviews) != 50000:
    raise AssertionError(f"Expected 50000 reviews, got {len(reviews)}")

CORE = re.compile(
    r"\b(?:warranty|guarantee|warranties|warrantee|warrantees|"
    r"warrenty|protection\s+plan)\b", re.I
)
CORE_IDS = {i for i, t in enumerate(texts) if CORE.search(t)}

# Direct terms still require human verification; RMA may refer to normal returns.
DIRECT_PATTERNS = {
    "rma_process": r"\brma\b",
    "squaretrade_provider": r"\bsquare\s*trade\b",
    "spelling_variants": (
        r"\b(?:warenty|waranty|warrany|waranties|warrantey|"
        r"garantee|guaranty|guaranties|guarentee|guarrantee)\b"
    ),
    "related_plan_terms": (
        r"\b(?:extended\s+(?:protection|service\s+plan|repair\s+plan)|"
        r"service\s+contract|replacement\s+plan|protection\s+plans|"
        r"product\s+protection\s+plan|insurance\s+protection|"
        r"insurance\s+policy\s+against\s+(?:unexpected\s+)?failure)\b"
    ),
    "return_authorization": r"\breturn\s+(?:merchandise\s+|material\s+)?authorization\b",
    "data_recovery_protection": (
        r"\b(?:data\s+recovery\s+(?:plan|policy)|"
        r"rescue\s+polic(?:y|ies|y.s))\b"
    ),
}

# Candidate-only rules. These intentionally err on the side of discovery.
# Words such as 'coverage' and 'replacement' are not matched alone.
CONTEXT_PATTERNS = {
    "free_replacement": (
        r"\b(?:free\s+replacement|replacement\s+(?:free\s+of\s+charge|"
        r"at\s+no\s+charge)|free\s+(?:repair|service))\b"
    ),
    "company_repairs_or_replaces": (
        r"\b(?:manufacturer|company|seller|vendor|customer\s+service|"
        r"customer\s+support|technical\s+support)\b.{0,150}?\b"
        r"(?:replaced|repair(?:ed)?|sent\s+(?:me|us)\s+(?:a\s+)?"
        r"(?:replacement|new\s+(?:one|unit|device|product))|"
        r"shipped\s+(?:me|us)\s+(?:a\s+)?"
        r"(?:replacement|new\s+(?:one|unit|device|product)))\b"
    ),
    "repair_then_service_provider": (
        r"\b(?:replaced|repaired|replacement|sent\s+(?:me|us)\s+(?:a\s+)?"
        r"(?:replacement|new\s+(?:one|unit|device|product)))\b.{0,150}?"
        r"\b(?:manufacturer|company|seller|vendor|customer\s+service|"
        r"customer\s+support|technical\s+support)\b"
    ),
    "stands_behind_product": (
        r"\b(?:stands?|stood)\s+behind\s+(?:their|the|our|this|its|my|your)"
        r"\s+(?:product|products|brand|items)\b"
    ),
    "coverage_and_repairs": (
        r"\b(?:not\s+covered|covered\s+by|cover\s+(?:the\s+)?"
        r"(?:repair|replacement)|coverage\s+(?:for|against)\s+"
        r"(?:repair|replacement|damage))\b"
    ),
    "sent_replacement": (
        r"\b(?:sent|sending|shipped|shipping|send)\s+(?:me|us)\s+"
        r"(?:a\s+)?(?:replacement|new\s+(?:one|unit|device|product|pair))\b"
    ),
    "replacement_no_questions": (
        r"\b(?:no\s+questions\s+asked|without\s+(?:any\s+)?questions)\b"
        r".{0,90}?\b(?:replaced|replacement|new\s+one)\b"
    ),
}

compiled = [("direct", k, re.compile(p, re.I | re.S)) for k, p in DIRECT_PATTERNS.items()]
compiled += [("context", k, re.compile(p, re.I | re.S)) for k, p in CONTEXT_PATTERNS.items()]

candidates = []
for i, t in enumerate(texts):
    if i in CORE_IDS:
        continue
    found = [(tier, name, pattern.search(t)) for tier, name, pattern in compiled]
    found = [(tier, name, match) for tier, name, match in found if match]
    if not found:
        continue
    # Direct matches take precedence over contextual ones.
    chosen_tier, chosen_rule, chosen_match = found[0]
    start = max(0, chosen_match.start() - 160)
    end = min(len(t), chosen_match.end() + 220)
    r = reviews[i]
    candidates.append({
        "review_id": i,
        "tier": chosen_tier,
        "primary_rule": chosen_rule,
        "all_matching_rules": ";".join(x[1] for x in found),
        "matched_text": chosen_match.group(0),
        "asin": (r.get("asin") or "").strip(),
        "overall": r.get("overall", ""),
        "context": t[start:end].replace("\n", " "),
        "full_text": t,
        "human_label": "",  # 1 related, 0 unrelated, blank unverified
        "review_notes": "",
    })

CANDIDATE_IDS = {r["review_id"] for r in candidates}
BROAD_IDS = CORE_IDS | CANDIDATE_IDS


def write_csv(name: str, data: list[dict]):
    path = OUT / name
    if not data:
        raise ValueError(f"No records for {name}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return path


def product_stats(ids):
    groups = defaultdict(list)
    for i in ids:
        asin = (reviews[i].get("asin") or "").strip()
        try:
            rating = float(reviews[i]["overall"])
        except (ValueError, TypeError, KeyError):
            continue
        if asin:
            groups[asin].append(rating)
    result = [
        {"asin": asin, "warranty_reviews": len(ratings),
         "avg_rating": round(sum(ratings) / len(ratings), 2)}
        for asin, ratings in groups.items()
    ]
    return sorted(result, key=lambda r: (-r["warranty_reviews"], r["asin"]))

core_reviews = []
for i in sorted(CORE_IDS):
    r = reviews[i]
    core_reviews.append({
        "review_id": i, "asin": (r.get("asin") or "").strip(),
        "overall": r.get("overall", ""),
        "matched_text": CORE.search(texts[i]).group(0), "full_text": texts[i]
    })

write_csv("warranty_v4_core_matches.csv", core_reviews)
write_csv("warranty_v4_candidates.csv", candidates)
core_products = product_stats(CORE_IDS)
broad_products = product_stats(BROAD_IDS)
write_csv("warranty_v4_core_products.csv", core_products)
write_csv("warranty_v4_EXPLORATORY_products.csv", broad_products)

# Review priority batch; not gold labels.
direct = [r for r in candidates if r["tier"] == "direct"]
context = [r for r in candidates if r["tier"] == "context"]
rng = random.Random(42)
priority = direct + rng.sample(context, min(80, len(context)))
write_csv("warranty_v4_priority_review.csv", priority)

# A uniformly random sample from *unmatched* records to discover blind spots.
remaining_ids = sorted(set(range(len(reviews))) - BROAD_IDS)
random_ids = rng.sample(remaining_ids, min(150, len(remaining_ids)))
random_audit = [
    {"review_id": i, "asin": reviews[i].get("asin", ""),
     "overall": reviews[i].get("overall", ""),
     "full_text": texts[i], "human_label": "", "review_notes": ""}
    for i in random_ids
]
write_csv("warranty_v4_random_unmatched_audit.csv", random_audit)

summary = {
    "dataset_rows": len(reviews),
    "core_rule_matches": len(CORE_IDS),
    "core_products": len(core_products),
    "new_unverified_candidates": len(CANDIDATE_IDS),
    "candidate_direct_rule_count": len(direct),
    "candidate_context_rule_count": len(context),
    "exploratory_broad_matches": len(BROAD_IDS),
    "exploratory_broad_products": len(broad_products),
    "core_rate_pct": round(len(CORE_IDS) / len(reviews) * 100, 3),
    "exploratory_rate_pct": round(len(BROAD_IDS) / len(reviews) * 100, 3),
    "candidate_rules": dict(Counter(r["primary_rule"] for r in candidates)),
    "priority_review_size": len(priority),
    "random_unmatched_audit_size": len(random_audit),
    "warnings": [
        "Core is an unverified lexical count, not a gold-standard true prevalence.",
        "All new candidates require manual validation before inclusion as positives.",
        "RMA can be a regular return, and customer service can mean retail service.",
        "Broad product statistics are exploratory; not to be used as verified warranty satisfaction.",
        "Mean overall is general product rating within matches, not aspect-specific sentiment.",
    ],
}
(OUT / "warranty_v4_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

# Save an exact, verifiable smoke-test result for quick team QA.
assert not CORE_IDS.intersection(CANDIDATE_IDS)
assert len(BROAD_IDS) == len(CORE_IDS) + len(CANDIDATE_IDS)
assert len(core_reviews) == len(CORE_IDS)
print("=== WARRANTY V4 (UNVERIFIED CANDIDATE EXPANSION) ===")
for k in ["dataset_rows", "core_rule_matches", "core_products",
          "new_unverified_candidates", "candidate_direct_rule_count",
          "candidate_context_rule_count", "exploratory_broad_matches",
          "exploratory_broad_products", "priority_review_size",
          "random_unmatched_audit_size"]:
    print(f"{k}: {summary[k]}")
print("Saved to:", OUT)
