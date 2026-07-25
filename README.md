# Amazon Video Games Hybrid Recommender

A portfolio-grade Top-K recommendation system built from **814,586**
time-stamped Amazon interactions and rich metadata for the complete
**25,612-item** recommendation catalog.

The project treats recommendation as a ranking and retrieval problem—not a
rating-prediction exercise. It combines collaborative preferences with product
content, evaluates future-item ranking with a temporal split, and exposes the
result through a production-style API.

## Business Problem

A useful recommender has to answer three different product questions:

1. **Known user:** what should this person see next?
2. **New user with one product signal:** what is similar to this game or
   accessory?
3. **Completely new user:** what is a safe default when no preference signal
   exists?

This repository implements a distinct strategy for each case:

| Scenario | Strategy |
|---|---|
| Known user | BPR collaborative ranking + metadata profile |
| New user with a seed item | TF-IDF content similarity |
| No user or item signal | popularity fallback |

## Verified Data

| Property | Value |
|---|---:|
| Interactions | 814,586 |
| Users | 94,762 |
| Recommendation items | 25,612 |
| Matrix density | 0.0336% |
| Median interactions per user | 6 |
| Metadata records | 137,269 |
| Catalog items matched to metadata | **25,612 / 25,612 (100%)** |
| Matched items with usable text | **25,612 (100%)** |
| Matched items with price | 17,135 (66.9%) |
| Date range | Oct 1999–Sep 2023 |

Raw data is intentionally excluded from Git. The project expects Amazon Reviews
2023 interaction data and `meta_Video_Games.jsonl`, joined by `parent_asin`.

## Measured Ranking Results

Metrics use 10,000 eligible test users. Each held-out future positive is ranked
against 100 unseen negative candidates.

| Model | Hit Rate@10 | NDCG@10 | MRR | Coverage@20 |
|---|---:|---:|---:|---:|
| Popularity | 0.5037 | 0.3088 | 0.2670 | 0.2846 |
| BPR-MF | 0.6364 | 0.4267 | 0.3760 | 0.7951 |
| **Hybrid (60% BPR + 40% content)** | **0.6428** | **0.4288** | **0.3763** | **0.7999** |
| Hybrid lift over BPR | **+1.0%** | **+0.5%** | **+0.1%** | **+0.6%** |

Hybrid weights were searched over `0.6, 0.7, 0.8, 0.9` on 10,000 validation
users. Alpha `0.6` produced the highest validation `NDCG@10` and was then
evaluated once on 10,000 test users. This separation prevents reporting a
hand-picked test result. The complete alpha search and final metrics are
checked into `artifacts/hybrid_metrics.json`.

## System Design

```text
Interactions ── temporal split ── BPR-MF ───────────────┐
                                                        ├─ weighted Top-K ranker
Metadata ── text assembly ── TF-IDF item vectors ───────┘
                                  │
                                  ├─ user content profile
                                  └─ item-to-item cold start
```

Metadata text combines title, category, store, features, and description.
Sparse TF-IDF vectors keep training and inference inspectable and CPU-friendly.
The collaborative/content weight is tuned using validation `NDCG@10`.

## Leakage-Safe Evaluation

For every user with at least three positive interactions:

- all but the latest two positives form training history;
- the second-latest positive becomes validation;
- the latest positive becomes test;
- ratings of four or five are treated as positive preference;
- already-seen training items are excluded from candidates.

The main selection metric is `NDCG@10`, supported by Hit Rate@5/10/20, MRR,
and catalog coverage. The evaluation measures whether a future relevant item
appears near the top of the list.

## API

The API requires the generated metadata-aware `hybrid_model.pt` artifact.
Because that artifact is reproducible from the public code and source datasets,
it is not required in Git. Build it before starting the service:

```bash
python scripts/evaluate_hybrid.py \
  --data "/path/to/Video_Games.csv.gz" \
  --metadata "/path/to/meta_Video_Games.jsonl" \
  --model artifacts/bpr_model.pt \
  --output artifacts/hybrid_model.pt
```

Then start the API:

```bash
MODEL_PATH=artifacts/hybrid_model.pt \
  uvicorn recsys.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET /health
GET /recommend/{user_id}?k=10
GET /similar/{parent_asin}?k=10
```

Responses include product title, store, category, rating, price, and image URL
when available—not just opaque item IDs.

### Interactive Demo

**[Open the free public demo →](https://kojunghsu.github.io/amazon-video-games-hybrid-recommender/)**

For the full local API-backed experience, first generate
`artifacts/hybrid_model.pt` as shown above, then run:

```bash
MODEL_PATH=artifacts/hybrid_model.pt \
  uvicorn recsys.api:app --host 0.0.0.0 --port 8000
```

The responsive browser demo supports known-user recommendations, popularity
cold start, and metadata-based similar items. It displays the measured ranking
metrics alongside product cards and requires no separate frontend build.

The GitHub Pages version uses cached recommendations generated from real
positive interactions and product metadata, so recruiters can explore it
instantly without downloading the full model. Rebuild its data with
`scripts/build_static_demo.py`.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Train and benchmark collaborative ranking.
python -m recsys.train \
  --data "/path/to/Video_Games.csv.gz" \
  --config config.json \
  --artifacts artifacts

# 2. Build metadata features, tune alpha on validation, evaluate on test.
python scripts/evaluate_hybrid.py \
  --data "/path/to/Video_Games.csv.gz" \
  --metadata "/path/to/meta_Video_Games.jsonl" \
  --model artifacts/bpr_model.pt \
  --output artifacts/hybrid_model.pt

# 3. Verify deterministic unit tests.
pytest
```

## Repository Structure

```text
.
├── config.json
├── Dockerfile
├── scripts/
│   ├── audit_dataset.py
│   └── evaluate_hybrid.py
├── src/recsys/
│   ├── data.py             # loading and temporal split
│   ├── bpr.py              # pairwise collaborative ranker
│   ├── metadata.py         # streaming metadata ingestion
│   ├── hybrid.py           # TF-IDF profiles and blended ranker
│   ├── evaluate.py         # sampled Top-K evaluation
│   ├── service.py          # online recommendation behavior
│   └── api.py              # FastAPI surface
└── tests/
```

## Engineering Decisions

- **BPR over rating RMSE:** optimizes the order of positive versus negative
  items, which is closer to the product objective.
- **Sparse TF-IDF over a transformer:** every catalog item has structured text,
  but interaction histories are short. TF-IDF provides an explainable and
  economical content baseline before adding semantic embeddings.
- **Validation-selected blending:** avoids choosing the hybrid weight from test
  outcomes.
- **Fallback hierarchy:** the service returns a useful result even when a user
  or collaborative embedding is unavailable.
- **No dense user-item matrix:** negative sampling and sparse content vectors
  keep memory proportional to observed data.

## Honest Limitations

- Ratings are preference proxies; the dataset has no impressions, clicks,
  carts, or purchases.
- Sampled evaluation should eventually be complemented by full-catalog ranking
  on a smaller stratified user cohort.
- Offline ranking lift does not establish online conversion lift.
- Price is missing for roughly one-third of matched catalog items.
- TF-IDF understands shared words, not deeper semantic equivalence; a sentence
  embedding retrieval experiment is the next justified model comparison.
