#!/usr/bin/env python3
import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.config import DATASETS, canonical_n  # noqa: E402

DISPLAY = {"books": "Books", "beauty": "Beauty", "video_games": "Video Games"}
METRICS = ["HR@10", "NDCG@10", "MRR"]

def read_run(d: Path, result_name: str) -> dict:
    res = json.load(open(d / result_name))
    row = {
        "seed": res["args"]["seed"],
        "best_epoch": res["best_epoch"],
        **{m: round(res["test"][m], 6) for m in METRICS},
    }
    return row, res["test"]["n_eval"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["books"],
                    help="datasets to aggregate (default: books, the released domain).")
    ap.add_argument("--from_results", action="store_true",
                    help="aggregate from results/{dataset}/seed_*.json instead of runs/, so that "
                         "paper_results.json always matches the seed files shipped beside it.")
    ap.add_argument("--runs_dir", default=str(REPO / "runs"))
    ap.add_argument("--out_dir", default=str(REPO / "results"))
    ap.add_argument("--result_name", default="test_result.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029])
    args = ap.parse_args()

    runs_dir, out_dir = Path(args.runs_dir), Path(args.out_dir)
    paper_p = out_dir / "paper_results.json"
    paper = json.load(open(paper_p)) if paper_p.exists() else {}
    paper.update({
        "model": "SCOPE",
        "seeds": args.seeds,
        "model_selection": "epoch with the highest validation NDCG@10; "
                           "test is evaluated once, on that checkpoint",
        "std": "sample standard deviation over the seeds (ddof=1)",
    })

    collected = []
    for ds in args.datasets:
        if args.from_results:
            rows = sorted((json.load(open(f)) for f in (out_dir / ds).glob("seed_*.json")),
                          key=lambda r: r["seed"])
            if not rows:
                print(f"[skip] {ds}: no {out_dir/ds}/seed_*.json")
                continue
            paper[DISPLAY.get(ds, ds)] = {
                "n_seeds": len(rows),
                "seeds": [r["seed"] for r in rows],
                "best_epochs": [r["best_epoch"] for r in rows],
                **{m: {"mean": round(st.mean(v := [r[m] for r in rows]), 6),
                       "std": round(st.stdev(v) if len(v) > 1 else 0.0, 6)} for m in METRICS},
            }
            collected.append(ds)
            print(f"[ok] {ds}: {len(rows)} seeds (aggregated from results/{ds}/seed_*.json)")
            continue
        dirs = sorted(p for s in args.seeds
                      for p in runs_dir.glob(f"{ds}_*seed{s}") if (p / args.result_name).exists())
        if not dirs:
            print(f"[skip] {ds}: no run has {args.result_name} under {runs_dir}")
            continue
        pairs = [read_run(d, args.result_name) for d in dirs]
        rows = sorted((r for r, _ in pairs), key=lambda r: r["seed"])

        n_eval = {n for _, n in pairs}
        if len(n_eval) > 1:
            print(f"[warn] {ds}: seeds disagree on the evaluated population: {sorted(n_eval)}")
        exp = canonical_n(ds)
        if exp is not None and n_eval != {exp}:
            print(f"[warn] {ds}: n_eval {sorted(n_eval)} != canonical_n {exp} — "
                  f"these are not the canonical-population numbers (was --leak_drop used?)")

        (out_dir / ds).mkdir(parents=True, exist_ok=True)
        for r in rows:
            (out_dir / ds / f"seed_{r['seed']}.json").write_text(json.dumps(r, indent=2) + "\n")

        paper[DISPLAY.get(ds, ds)] = {
            "n_seeds": len(rows),
            "seeds": [r["seed"] for r in rows],
            "best_epochs": [r["best_epoch"] for r in rows],
            **{m: {"mean": round(st.mean(v := [r[m] for r in rows]), 6),
                   "std": round(st.stdev(v) if len(v) > 1 else 0.0, 6)} for m in METRICS},
        }
        collected.append(ds)
        print(f"[ok] {ds}: {len(rows)} seeds -> {out_dir.name}/{ds}/")

    if not collected:
        print(f"[stop] nothing collected — {paper_p} left untouched")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_p.write_text(json.dumps(paper, indent=2) + "\n")
    print(f"[ok] {paper_p}")

if __name__ == "__main__":
    main()
