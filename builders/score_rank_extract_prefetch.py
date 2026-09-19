"""Warm the on-disk page cache for every groupC builder (datasets-server rate-limits hard)."""

import sys

from score_rank_extract_common import rows_many

JOBS = [
    ("nyu-mll/glue", "stsb", "validation", 1500),
    ("Yelp/yelp_review_full", "yelp_review_full", "test", 2000),
    ("nvidia/HelpSteer2", "default", "validation", 1000),
    ("microsoft/ms_marco", "v1.1", "validation", 800),
    ("rajpurkar/squad", "plain_text", "validation", 3000),
    ("tner/conll2003", "conll2003", "test", 2500),
    ("BeIR/nfcorpus", "queries", "queries", 3300),
    ("BeIR/nfcorpus-qrels", "default", "test", 3000),
    ("BeIR/nfcorpus", "corpus", "corpus", 3700),
]

for ds, cfg, split, n in JOBS:
    got = rows_many(ds, cfg, split, n)
    print("cached", ds, cfg, split, len(got), flush=True)
sys.stdout.write("PREFETCH DONE\n")
