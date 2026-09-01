# results/

The Books numbers the paper reports, in the smallest form a reader can check them in.

```
results/
├── paper_results.json       mean / std over the four seeds
└── books/seed_{seed}.json   selected epoch and test metrics of a single run
```

```json
{ "seed": 2026, "best_epoch": 111, "HR@10": 0.1512, "NDCG@10": 0.0931, "MRR": 0.0836 }
```

- **Model selection** — `best_epoch` is the epoch with the highest *validation* NDCG@10, taken from
  `train_log.json`. Test is evaluated exactly once, on that checkpoint. Training used the shared
  budget of 200 epochs with patience 20; every run stopped early.
- **Evaluation** — full-catalog ranking with the items a user has already interacted with masked, on
  the canonical evaluation set of 19,748 users (`src/test.py --leak_drop`, Section 8.2). That set is
  pinned in `src/config.py` by size and by a fingerprint of every instance's user, target, and
  history, and `test.py` refuses to evaluate anything else.
- **std** — sample standard deviation over the seeds (ddof=1).
- The pipeline statistics this directory also receives at run time (`sample_stats_books.json`,
  `card_stats_books.json`, written by stages [2] and [4]) are regenerable and not tracked.

Regenerate:

```bash
python scripts/collect_results.py                 # from runs/books_*_seed*/test_result.json
python scripts/collect_results.py --from_results  # re-aggregate from the seed files above
```

The paper also evaluates Beauty & Personal Care and Video Games; this repository distributes neither
their data artifacts nor their result files (Section 1).
