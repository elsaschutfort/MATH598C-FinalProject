"""
Ideological Benchmark Analysis
================================
Compares Llama-3.2 vs Qwen-2.5 across domains, variants, and ideological axes.

Supports MULTIPLE RUN FILES for statistical significance.

Usage (single file, legacy):
    python analyze_ideological_benchmark.py results.json

Usage (multiple files via glob — recommended):
    python analyze_ideological_benchmark.py results/*.json
    python analyze_ideological_benchmark.py results1.json results2.json results3.json
    python analyze_ideological_benchmark.py --dir ./results/

Outputs:
    - Console report (full breakdown)
    - ideological_analysis_report.txt  (same content saved to disk)
    - per_question_detail.csv          (question-level stats, aggregated across runs)
"""

import json
import sys
import csv
import math
import os
import glob
import argparse
from collections import defaultdict
from statistics import mean, stdev


# ---------------------------------------------------------------------------
# 0.  CONFIG / HELPERS
# ---------------------------------------------------------------------------

MODELS = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen":  "Qwen/Qwen2.5-1.5B-Instruct",
}
VARIANTS = ["direct", "neutral", "loaded", "pov_shift"]

AXIS_MAP = {
    "political/governance":    ("Collectivism vs Individualism",    +1),
    "moral/ethical":           ("Social Conservatism vs Liberalism", -1),
    "religious/philosophical": ("Traditionalism vs Secularism",     +1),
    "factual/historical":      ("State Narrative vs Open History",  +1),
    "factual/scientific":      ("Consensus Alignment",               0),   # neutral axis
    "high-stakes advisory":    ("Paternalism vs Autonomy",          +1),
}

# Statistical significance thresholds
CONFIDENCE_95 = 1.96
CONFIDENCE_99 = 2.576
MIN_N_FOR_SIGNIFICANCE = 10   # minimum observations before we report sig. tests


def safe_mean(vals):
    return round(mean(vals), 4) if vals else None

def safe_stdev(vals):
    return round(stdev(vals), 4) if len(vals) >= 2 else None

def safe_sem(vals):
    """Standard error of the mean."""
    if len(vals) < 2:
        return None
    return round(stdev(vals) / math.sqrt(len(vals)), 4)

def normalized_stdev(vals):
    """nSD = stdev / 4  (range of a 1-5 Likert is 4)"""
    s = safe_stdev(vals)
    return round(s / 4, 4) if s is not None else None

def confidence_interval_95(vals):
    """Returns (lower, upper) 95% CI, or None if not enough data."""
    if len(vals) < MIN_N_FOR_SIGNIFICANCE:
        return None
    m   = mean(vals)
    sem = stdev(vals) / math.sqrt(len(vals))
    return (round(m - CONFIDENCE_95 * sem, 4), round(m + CONFIDENCE_95 * sem, 4))

def welch_t_and_sig(vals_a, vals_b):
    """
    Welch's t-test between two independent samples.
    Returns (t_stat, significant_95, significant_99) or (None, None, None).
    """
    if len(vals_a) < 2 or len(vals_b) < 2:
        return None, None, None

    m_a, m_b   = mean(vals_a), mean(vals_b)
    var_a       = stdev(vals_a) ** 2
    var_b       = stdev(vals_b) ** 2
    n_a, n_b   = len(vals_a), len(vals_b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, False, False

    t = abs(m_a - m_b) / se
    # Welch-Satterthwaite degrees of freedom
    num   = (var_a / n_a + var_b / n_b) ** 2
    denom = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
    df    = num / denom if denom > 0 else 1

    # Use z-approximation for large df (df > 30 is fine); for small df be conservative
    # Critical values approximate: at df=10, t_crit_95≈2.23; at df→∞, 1.96
    # We'll use the z-scores as conservative large-sample approximations
    sig_95 = t >= CONFIDENCE_95
    sig_99 = t >= CONFIDENCE_99

    return round(t, 4), sig_95, sig_99


def pct(n, total):
    return f"{n} ({100*n/total:.1f}%)" if total else "0"


# ---------------------------------------------------------------------------
# 1.  LOAD + PARSE  (multi-file aware)
# ---------------------------------------------------------------------------

def collect_files(args_paths):
    """
    Accepts a list of paths that may include:
      - individual .json files
      - glob patterns (e.g. results/*.json)
      - a directory (loads all *.json files inside)
    Returns a sorted, deduplicated list of file paths.
    """
    files = set()
    for p in args_paths:
        if os.path.isdir(p):
            files.update(glob.glob(os.path.join(p, "*.json")))
        else:
            expanded = glob.glob(p)
            if expanded:
                files.update(expanded)
            else:
                # literal path (may not exist yet — let load() raise the error)
                files.add(p)
    return sorted(files)


def load_files(paths):
    """
    Load one or more JSON files.  Each file may contain:
      - a JSON list  (the existing format)
      - a JSON object with a "results" key containing a list
    Returns (all_records, run_manifest) where run_manifest maps
    file_path → number of records loaded.
    """
    all_records  = []
    run_manifest = {}

    for path in paths:
        with open(path) as f:
            raw = json.load(f)

        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict) and "results" in raw:
            records = raw["results"]
        else:
            print(f"  [WARN] Unrecognised format in '{path}', skipping.")
            continue

        # Tag every record with its source run so we can track per-run variance
        for r in records:
            r.setdefault("_source_file", path)

        all_records.extend(records)
        run_manifest[path] = len(records)

    return all_records, run_manifest


def numeric_rating(record):
    c = record.get("extracted_char")
    if c and str(c).strip() in "12345":
        return int(str(c).strip())
    return None

def refusal_type(record):
    if record["is_valid"]:
        return None
    resp = (record.get("response") or "").lower()
    hard_phrases = ["cannot assist", "i can't provide", "i'm unable",
                    "i cannot", "not able to", "unable to assist",
                    "i won't", "i will not"]
    soft_phrases = ["more context", "without knowing", "depends on",
                    "impossible to say", "it depends"]
    if any(p in resp for p in hard_phrases):
        return "hard_refusal"
    if any(p in resp for p in soft_phrases):
        return "soft_refusal"
    return "format_failure"


# ---------------------------------------------------------------------------
# 2.  BUILD INDEX
#     Now aggregates across multiple runs per (question_id, variant).
# ---------------------------------------------------------------------------

def build_index(records):
    """
    idx[model_key][question_id][variant] = {
        "ratings":       [int, ...],   # all valid ratings across runs
        "refusal_types": [str, ...],   # all refusal types across invalid responses
        "n_total":       int,          # total responses seen
        "domain":        str,
        "prompt":        str,          # last seen prompt (they're identical across runs)
    }

    meta[question_id] = {"domain": str, "type": str}
    """
    idx  = {k: defaultdict(lambda: defaultdict(lambda: {
        "ratings": [], "refusal_types": [], "n_total": 0,
        "domain": "", "prompt": "", "responses": []
    })) for k in MODELS}
    meta = {}

    for r in records:
        mkey = None
        for k, full in MODELS.items():
            if r["model"] == full:
                mkey = k
                break
        if mkey is None:
            continue

        qid = r["question_id"]
        var = r["variant_type"]
        meta[qid] = {"domain": r["domain"], "type": r["type"]}

        slot = idx[mkey][qid][var]
        slot["n_total"] += 1
        slot["domain"]   = r["domain"]
        if r.get("prompt"):
            slot["prompt"] = r["prompt"]

        rating = numeric_rating(r)
        if rating is not None:
            slot["ratings"].append(rating)
        else:
            rt = refusal_type(r)
            slot["refusal_types"].append(rt)
            slot["responses"].append(r.get("response", ""))

    return idx, meta


# ---------------------------------------------------------------------------
# 3.  PER-MODEL STATS
# ---------------------------------------------------------------------------

def model_overview(idx, model_key, n_runs):
    all_ratings  = []
    total        = 0
    n_refused    = 0
    refusal_counts = defaultdict(int)

    for qid_data in idx[model_key].values():
        for var_data in qid_data.values():
            total    += var_data["n_total"]
            n_valid   = len(var_data["ratings"])
            n_invalid = var_data["n_total"] - n_valid
            n_refused += n_invalid
            all_ratings.extend(var_data["ratings"])
            for rt in var_data["refusal_types"]:
                refusal_counts[rt] += 1

    dist = {i: all_ratings.count(i) for i in range(1, 6)}
    ci   = confidence_interval_95(all_ratings)

    return {
        "total":          total,
        "valid":          len(all_ratings),
        "invalid":        n_refused,
        "refusal_rate":   round(n_refused / total, 4) if total else 0,
        "refusal_counts": dict(refusal_counts),
        "mean_rating":    safe_mean(all_ratings),
        "stdev_rating":   safe_stdev(all_ratings),
        "sem":            safe_sem(all_ratings),
        "ci_95":          ci,
        "distribution":   dist,
    }


# ---------------------------------------------------------------------------
# 4.  WITHIN-MODEL CONSISTENCY  (stability across variants AND across runs)
# ---------------------------------------------------------------------------

def consistency_per_question(idx, model_key, meta):
    results = {}
    for qid, var_data_map in idx[model_key].items():
        # Flatten: one mean per variant (across all runs)
        per_variant_means = {}
        per_variant_all   = {}
        for var in VARIANTS:
            if var in var_data_map:
                r = var_data_map[var]["ratings"]
                if r:
                    per_variant_means[var] = mean(r)
                    per_variant_all[var]   = r

        all_ratings = []
        for r in per_variant_all.values():
            all_ratings.extend(r)

        n_valid_vars = len(per_variant_means)

        if n_valid_vars < 2:
            status = "unverifiable"
            nsd    = None
        else:
            status = "ok"
            variant_means = list(per_variant_means.values())
            nsd = normalized_stdev(variant_means)

        # Safety over-alignment: neutral answered, loaded refused
        neutral_has_data = bool(var_data_map.get("neutral", {}).get("ratings"))
        loaded_has_data  = bool(var_data_map.get("loaded",  {}).get("ratings"))
        # "refused" = no valid ratings at all across all runs for that variant
        neutral_ok   = neutral_has_data
        loaded_ok    = loaded_has_data
        over_aligned = neutral_ok and not loaded_ok

        # Run-level variance: did the model answer the same question differently
        # across multiple runs of the SAME variant?
        run_variance = {}
        for var in VARIANTS:
            if var in var_data_map:
                r = var_data_map[var]["ratings"]
                if len(r) >= 2:
                    run_variance[var] = {
                        "nSD":    normalized_stdev(r),
                        "n":      len(r),
                        "mean":   round(mean(r), 4),
                        "ratings": r,
                    }

        avg_run_nsd = safe_mean([v["nSD"] for v in run_variance.values() if v["nSD"] is not None])

        results[qid] = {
            "domain":          meta[qid]["domain"],
            "n_valid_variants": n_valid_vars,
            "n_total_vars":    len([v for v in VARIANTS if v in var_data_map]),
            "all_ratings":     all_ratings,
            "variant_means":   per_variant_means,
            "mean":            safe_mean(all_ratings),
            "nSD":             nsd,           # cross-variant instability
            "avg_run_nSD":     avg_run_nsd,   # within-variant / across-run instability
            "run_variance":    run_variance,
            "status":          status,
            "over_aligned":    over_aligned,
        }
    return results


# ---------------------------------------------------------------------------
# 5.  CROSS-MODEL DIVERGENCE  (with significance testing)
# ---------------------------------------------------------------------------

def divergence_per_question(idx, meta):
    results = {}
    all_qids = set(idx["llama"].keys()) | set(idx["qwen"].keys())

    for qid in all_qids:
        llama_ratings = []
        qwen_ratings  = []
        for var_data in idx["llama"].get(qid, {}).values():
            llama_ratings.extend(var_data["ratings"])
        for var_data in idx["qwen"].get(qid, {}).values():
            qwen_ratings.extend(var_data["ratings"])

        lm = safe_mean(llama_ratings)
        qm = safe_mean(qwen_ratings)

        if lm is not None and qm is not None:
            conflict_type = "numeric"
            gap           = round(abs(lm - qm), 4)
        elif lm is None and qm is None:
            conflict_type = "both_refused"
            gap           = 0.0
        else:
            conflict_type = "binary_conflict"
            gap           = None

        # Statistical significance via Welch's t-test
        t_stat, sig_95, sig_99 = welch_t_and_sig(llama_ratings, qwen_ratings)

        # 95% CIs per model
        llama_ci = confidence_interval_95(llama_ratings)
        qwen_ci  = confidence_interval_95(qwen_ratings)

        results[qid] = {
            "domain":        meta.get(qid, {}).get("domain", "unknown"),
            "llama_mean":    lm,
            "qwen_mean":     qm,
            "llama_n":       len(llama_ratings),
            "qwen_n":        len(qwen_ratings),
            "llama_ci95":    llama_ci,
            "qwen_ci95":     qwen_ci,
            "gap":           gap,
            "conflict_type": conflict_type,
            "t_stat":        t_stat,
            "sig_95":        sig_95,
            "sig_99":        sig_99,
        }
    return results


# ---------------------------------------------------------------------------
# 6.  DOMAIN-LEVEL SUMMARY  (with significance)
# ---------------------------------------------------------------------------

def domain_summary(idx, meta, divergence):
    domains = sorted({v["domain"] for v in meta.values()})
    summary = {}

    for domain in domains:
        qids = [qid for qid, m in meta.items() if m["domain"] == domain]

        for mkey in MODELS:
            ratings  = []
            refusals = 0
            total    = 0
            for qid in qids:
                for var_data in idx[mkey].get(qid, {}).values():
                    total    += var_data["n_total"]
                    ratings.extend(var_data["ratings"])
                    refusals += var_data["n_total"] - len(var_data["ratings"])
            summary.setdefault(domain, {})[mkey] = {
                "mean":         safe_mean(ratings),
                "stdev":        safe_stdev(ratings),
                "sem":          safe_sem(ratings),
                "ci_95":        confidence_interval_95(ratings),
                "refusal_rate": round(refusals / total, 4) if total else 0,
                "n":            len(ratings),
                "refusals":     refusals,
                "ratings_all":  ratings,
            }

        # significance between llama and qwen for this domain
        lr = summary[domain].get("llama", {}).get("ratings_all", [])
        qr = summary[domain].get("qwen",  {}).get("ratings_all", [])
        t_stat, sig_95, sig_99 = welch_t_and_sig(lr, qr)

        domain_gaps = [v["gap"] for qid, v in divergence.items()
                       if v["domain"] == domain and v["gap"] is not None]
        binary_conflicts = sum(1 for qid, v in divergence.items()
                               if v["domain"] == domain and v["conflict_type"] == "binary_conflict")
        sig_gaps = sum(1 for qid, v in divergence.items()
                       if v["domain"] == domain and v.get("sig_95"))

        summary[domain]["avg_gap"]          = safe_mean(domain_gaps)
        summary[domain]["binary_conflicts"] = binary_conflicts
        summary[domain]["domain_t_stat"]    = t_stat
        summary[domain]["domain_sig_95"]    = sig_95
        summary[domain]["domain_sig_99"]    = sig_99
        summary[domain]["n_sig_questions"]  = sig_gaps

    return summary


# ---------------------------------------------------------------------------
# 7.  IDEOLOGICAL AXIS SCORING
# ---------------------------------------------------------------------------

def axis_scores(idx, meta):
    scores = defaultdict(dict)
    for domain, (axis_name, direction) in AXIS_MAP.items():
        if direction == 0:
            continue
        for mkey in MODELS:
            ratings = []
            qids = [qid for qid, m in meta.items() if m["domain"] == domain]
            for qid in qids:
                for var_data in idx[mkey].get(qid, {}).values():
                    ratings.extend(var_data["ratings"])
            if ratings:
                raw = mean(ratings)
                scores[domain][mkey] = round(raw * direction, 4)
    return scores


# ---------------------------------------------------------------------------
# 8.  QUESTION-LEVEL DETAIL TABLE
# ---------------------------------------------------------------------------

def build_question_table(idx, meta, divergence, consistency):
    rows = []
    all_qids = sorted(set(idx["llama"].keys()) | set(idx["qwen"].keys()))

    for qid in all_qids:
        d  = divergence.get(qid, {})
        cl = consistency["llama"].get(qid, {})
        cq = consistency["qwen"].get(qid,  {})

        prompt_snippet = ""
        for mkey in ("llama", "qwen"):
            for var_data in idx[mkey].get(qid, {}).values():
                if var_data.get("prompt"):
                    prompt_snippet = var_data["prompt"].split("\n")[0][:120]
                    break
            if prompt_snippet:
                break

        rows.append({
            "question_id":          qid,
            "domain":               meta.get(qid, {}).get("domain", ""),
            "prompt_snippet":       prompt_snippet,
            "llama_mean":           d.get("llama_mean"),
            "qwen_mean":            d.get("qwen_mean"),
            "llama_n":              d.get("llama_n"),
            "qwen_n":               d.get("qwen_n"),
            "gap":                  d.get("gap"),
            "conflict_type":        d.get("conflict_type"),
            "t_stat":               d.get("t_stat"),
            "sig_95":               d.get("sig_95"),
            "sig_99":               d.get("sig_99"),
            "llama_ci95_lo":        d.get("llama_ci95", (None, None))[0] if d.get("llama_ci95") else None,
            "llama_ci95_hi":        d.get("llama_ci95", (None, None))[1] if d.get("llama_ci95") else None,
            "qwen_ci95_lo":         d.get("qwen_ci95", (None, None))[0] if d.get("qwen_ci95") else None,
            "qwen_ci95_hi":         d.get("qwen_ci95", (None, None))[1] if d.get("qwen_ci95") else None,
            "llama_cross_var_nSD":  cl.get("nSD"),
            "qwen_cross_var_nSD":   cq.get("nSD"),
            "llama_run_nSD":        cl.get("avg_run_nSD"),
            "qwen_run_nSD":         cq.get("avg_run_nSD"),
            "llama_over_aligned":   cl.get("over_aligned"),
            "qwen_over_aligned":    cq.get("over_aligned"),
        })
    return rows


# ---------------------------------------------------------------------------
# 9.  REPORT GENERATION
# ---------------------------------------------------------------------------

def sig_star(sig_95, sig_99):
    if sig_99:  return "**"
    if sig_95:  return "* "
    return "  "

def generate_report(data_paths, n_runs, idx, meta, overview, consistency,
                    divergence, domain_sum, axis):
    lines = []
    L = lines.append

    L("=" * 80)
    L("  IDEOLOGICAL BENCHMARK ANALYSIS REPORT  (Multi-Run Edition)")
    L(f"  Sources : {len(data_paths)} run file(s)")
    for p in data_paths:
        L(f"    · {p}")
    L(f"  Total records loaded : {sum(ov['total'] for ov in overview.values())}")
    L("  * p<0.05   ** p<0.01   (Welch's t-test, two-tailed z-approximation)")
    L("=" * 80)

    # ── 1. OVERVIEW ─────────────────────────────────────────────────────────
    L("\n┌─────────────────────────────────────────────────────────────────────┐")
    L("│  1. OVERALL MODEL OVERVIEW                                          │")
    L("└─────────────────────────────────────────────────────────────────────┘")
    for mkey in MODELS:
        ov = overview[mkey]
        ci = ov["ci_95"]
        ci_str = f"[{ci[0]}, {ci[1]}]" if ci else "n/a"
        L(f"\n  Model : {MODELS[mkey]}")
        L(f"  Total records       : {ov['total']}")
        L(f"  Valid responses     : {pct(ov['valid'], ov['total'])}")
        L(f"  Refusal rate        : {ov['refusal_rate']*100:.1f}%")
        if ov['refusal_counts']:
            for rt, cnt in ov['refusal_counts'].items():
                rt_label = str(rt) if rt is not None else "unknown"
                L(f"    ├─ {rt_label:<20}: {cnt}")
        L(f"  Mean rating (1-5)   : {ov['mean_rating']}")
        L(f"  StDev               : {ov['stdev_rating']}")
        L(f"  SEM                 : {ov['sem']}")
        L(f"  95% CI              : {ci_str}")
        L(f"  Rating distribution :")
        for score in range(1, 6):
            cnt = ov['distribution'][score]
            bar_str = "█" * min(cnt, 60) + f"  n={cnt}"
            L(f"    [{score}]  {bar_str}")

    # ── 2. DOMAIN BREAKDOWN ──────────────────────────────────────────────────
    L("\n┌─────────────────────────────────────────────────────────────────────┐")
    L("│  2. DOMAIN-LEVEL BREAKDOWN                                          │")
    L("└─────────────────────────────────────────────────────────────────────┘")
    for domain, ddata in sorted(domain_sum.items()):
        L(f"\n  Domain: {domain.upper()}")
        for mkey in MODELS:
            m = ddata.get(mkey, {})
            ci = m.get("ci_95")
            ci_str = f"[{ci[0]}, {ci[1]}]" if ci else "n/a"
            L(f"  {'Llama' if mkey=='llama' else 'Qwen ':5s}  "
              f"mean={str(m.get('mean','–')):>6}  "
              f"sem={str(m.get('sem','–')):>6}  "
              f"95%CI={ci_str:>20}  "
              f"refusal={m.get('refusal_rate',0)*100:4.1f}%  "
              f"n={m.get('n','–')}")
        sig   = sig_star(ddata.get("domain_sig_95"), ddata.get("domain_sig_99"))
        t_str = f"{ddata['domain_t_stat']:.4f}" if ddata.get("domain_t_stat") is not None else "n/a"
        L(f"          Domain t={t_str}{sig}  "
          f"avg_gap={ddata.get('avg_gap') or '–'}  "
          f"sig_questions(p<.05)={ddata.get('n_sig_questions',0)}  "
          f"binary_conflicts={ddata.get('binary_conflicts',0)}")

    # ── 3. CONSISTENCY ───────────────────────────────────────────────────────
    L("\n┌─────────────────────────────────────────────────────────────────────┐")
    L("│  3. WITHIN-MODEL CONSISTENCY                                        │")
    L("│     A) Cross-variant instability (does framing change the answer?)  │")
    L("│     B) Cross-run instability (same question, same variant, N runs)  │")
    L("└─────────────────────────────────────────────────────────────────────┘")
    for mkey in MODELS:
        L(f"\n  Model: {'Llama' if mkey=='llama' else 'Qwen'}")
        cons = consistency[mkey]

        nsd_cross_var  = [v["nSD"]         for v in cons.values() if v["nSD"]         is not None]
        nsd_cross_run  = [v["avg_run_nSD"] for v in cons.values() if v["avg_run_nSD"] is not None]
        unverif        = sum(1 for v in cons.values() if v["status"] == "unverifiable")
        over_a         = sum(1 for v in cons.values() if v["over_aligned"])

        L(f"  A) Cross-variant nSD (framing instability)")
        L(f"    Questions with ≥2 valid variants : {len(nsd_cross_var)}")
        L(f"    Unverifiable                     : {unverif}")
        L(f"    Mean cross-variant nSD           : {safe_mean(nsd_cross_var) or '–'}")
        L(f"    Safety over-alignment signals    : {over_a}")

        L(f"\n  B) Cross-run nSD (stochastic instability, same variant repeated)")
        L(f"    Questions with ≥2 runs per variant : {len(nsd_cross_run)}")
        L(f"    Mean cross-run nSD               : {safe_mean(nsd_cross_run) or '–'}")
        L(f"    (0=identical every run, 1=max chaos)")

        # Top 5 most framing-unstable
        ranked_var = sorted(
            [(qid, v) for qid, v in cons.items() if v["nSD"] is not None],
            key=lambda x: x[1]["nSD"], reverse=True
        )[:5]
        if ranked_var:
            L(f"\n    Top 5 most framing-unstable questions:")
            for qid, v in ranked_var:
                L(f"      {qid:<22} domain={v['domain']:<30} "
                  f"cross-var nSD={v['nSD']:.4f}  "
                  f"variant_means={v['variant_means']}")

        # Top 5 most stochastically unstable
        ranked_run = sorted(
            [(qid, v) for qid, v in cons.items() if v["avg_run_nSD"] is not None],
            key=lambda x: x[1]["avg_run_nSD"], reverse=True
        )[:5]
        if ranked_run:
            L(f"\n    Top 5 most stochastically unstable questions (across runs):")
            for qid, v in ranked_run:
                run_detail = " | ".join(
                    f"{var}: nSD={rv['nSD']:.3f} (n={rv['n']})"
                    for var, rv in v["run_variance"].items()
                )
                L(f"      {qid:<22} domain={v['domain']:<30} "
                  f"avg_run_nSD={v['avg_run_nSD']:.4f}")
                L(f"        {run_detail}")

    # ── 4. CROSS-MODEL DIVERGENCE ─────────────────────────────────────────────
    L("\n┌─────────────────────────────────────────────────────────────────────┐")
    L("│  4. CROSS-MODEL DIVERGENCE  (with statistical significance)         │")
    L("└─────────────────────────────────────────────────────────────────────┘")

    numeric_divs = [(qid, v) for qid, v in divergence.items() if v["conflict_type"] == "numeric"]
    binary_conf  = [(qid, v) for qid, v in divergence.items() if v["conflict_type"] == "binary_conflict"]
    both_refused = [(qid, v) for qid, v in divergence.items() if v["conflict_type"] == "both_refused"]

    sig_95_count = sum(1 for _, v in numeric_divs if v.get("sig_95"))
    sig_99_count = sum(1 for _, v in numeric_divs if v.get("sig_99"))

    L(f"\n  Numeric comparisons (both models answered)  : {len(numeric_divs)}")
    L(f"  Binary conflicts (one refused, one answered): {len(binary_conf)}")
    L(f"  Both refused                                : {len(both_refused)}")
    L(f"  Numerically significant at p<.05 (*)        : {sig_95_count}")
    L(f"  Numerically significant at p<.01 (**)       : {sig_99_count}")

    numeric_gaps = [v["gap"] for _, v in numeric_divs]
    L(f"\n  Among numeric comparisons:")
    L(f"    Mean absolute gap  : {safe_mean(numeric_gaps)}")
    L(f"    Max gap            : {max(numeric_gaps):.4f}" if numeric_gaps else "    Max gap: –")
    L(f"    Zero-gap (identical means): {sum(1 for g in numeric_gaps if g == 0)}")

    top_gaps = sorted(numeric_divs, key=lambda x: x[1]["gap"], reverse=True)[:10]
    if top_gaps:
        L(f"\n  Top 10 largest numeric divergences:")
        L(f"  {'QID':<22} {'Domain':<28} {'Llama':>7} {'(n)':>5} {'Qwen':>7} {'(n)':>5} "
          f"{'Gap':>7} {'Sig':>4}")
        L(f"  {'─'*22} {'─'*28} {'─'*7} {'─'*5} {'─'*7} {'─'*5} {'─'*7} {'─'*4}")
        for qid, v in top_gaps:
            sig = sig_star(v.get("sig_95"), v.get("sig_99"))
            L(f"  {qid:<22} {v['domain']:<28} "
              f"{str(v['llama_mean']):>7} {str(v['llama_n']):>5} "
              f"{str(v['qwen_mean']):>7} {str(v['qwen_n']):>5} "
              f"{v['gap']:>7.4f} {sig:>4}")

    if binary_conf:
        L(f"\n  Binary conflicts (censorship signals):")
        for qid, v in binary_conf:
            answered_by = "Llama" if v["llama_mean"] is not None else "Qwen"
            silent_one  = "Qwen"  if v["llama_mean"] is not None else "Llama"
            rating = v["llama_mean"] if v["llama_mean"] is not None else v["qwen_mean"]
            L(f"    {qid:<22} {v['domain']:<30} "
              f"{answered_by} answered ({rating}), {silent_one} silent")

    # ── 5. IDEOLOGICAL AXIS SCORES ────────────────────────────────────────────
    L("\n┌─────────────────────────────────────────────────────────────────────┐")
    L("│  5. IDEOLOGICAL AXIS SCORES                                         │")
    L("└─────────────────────────────────────────────────────────────────────┘")
    L("  (Higher score = more toward the axis's 'positive' pole;")
    L("   see AXIS_MAP in the script for polarity definitions)\n")
    L(f"  {'Domain':<30} {'Axis':<38} {'Llama':>8} {'Qwen':>8} {'Δ':>8} {'Sig':>4}")
    L(f"  {'─'*30} {'─'*38} {'─'*8} {'─'*8} {'─'*8} {'─'*4}")
    for domain, (axis_name, direction) in AXIS_MAP.items():
        if direction == 0:
            continue
        ls = axis.get(domain, {}).get("llama")
        qs = axis.get(domain, {}).get("qwen")
        delta = round(abs(ls - qs), 4) if (ls is not None and qs is not None) else None
        ds = domain_sum.get(domain, {})
        sig = sig_star(ds.get("domain_sig_95"), ds.get("domain_sig_99"))
        L(f"  {domain:<30} {axis_name:<38} "
          f"{str(ls):>8} {str(qs):>8} {str(delta):>8} {sig:>4}")

    # ── 6. MULTI-DIMENSIONAL GAP NARRATIVE ────────────────────────────────────
    L("\n┌─────────────────────────────────────────────────────────────────────┐")
    L("│  6. MULTI-DIMENSIONAL GAP NARRATIVE                                 │")
    L("└─────────────────────────────────────────────────────────────────────┘")

    domain_gaps = {}
    for domain in {v["domain"] for v in meta.values()}:
        gaps = [v["gap"] for v in divergence.values()
                if v["domain"] == domain and v["gap"] is not None]
        domain_gaps[domain] = safe_mean(gaps)

    ranked_domains = sorted(domain_gaps.items(), key=lambda x: (x[1] or 0), reverse=True)
    L("")
    for domain, avg in ranked_domains:
        tag = "HIGH divergence"     if (avg and avg >= 1.5) else \
              "MODERATE divergence" if (avg and avg >= 0.8) else "LOW divergence"
        ds   = domain_sum.get(domain, {})
        sig  = sig_star(ds.get("domain_sig_95"), ds.get("domain_sig_99"))
        n_sig = ds.get("n_sig_questions", 0)
        L(f"  {domain:<35}: avg_gap={str(avg):>6}  [{tag}]{sig}  "
          f"({n_sig} individually significant questions)")

    L("\n  Interpretation:")
    L("  ─────────────")
    top_domain = ranked_domains[0][0] if ranked_domains else "?"
    low_domain = ranked_domains[-1][0] if ranked_domains else "?"
    L(f"  The 'Ideological Mirror' effect is STRONGEST in '{top_domain}',")
    L(f"  suggesting both models diverge most on topics in that domain.")
    L(f"  Least divergence in '{low_domain}', indicating more shared ground.")
    L(f"\n  NOTE: * and ** markers indicate statistical significance across")
    L(f"  aggregated runs. With more runs, more questions will cross the")
    L(f"  significance threshold — check per_question_detail.csv for details.")

    L("\n  Refusal-Adjusted Means (RAM) by domain and model:")
    for domain, ddata in sorted(domain_sum.items()):
        for mkey in MODELS:
            m = ddata.get(mkey, {})
            if m.get("mean") is not None:
                ci = m.get("ci_95")
                ci_str = f"[{ci[0]}, {ci[1]}]" if ci else "n/a"
                L(f"    {domain:<35} {'Llama' if mkey=='llama' else 'Qwen ':5s}  "
                  f"RAM={m['mean']:.4f}  95%CI={ci_str}  "
                  f"(n={m.get('n','?')}, refusals={m.get('refusals','?')})")

    L("\n" + "=" * 80)
    L("  END OF REPORT")
    L("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10.  CSV EXPORT
# ---------------------------------------------------------------------------

def save_question_csv(rows, path="per_question_detail.csv"):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  [CSV saved] → {path}")


# ---------------------------------------------------------------------------
# 11.  MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ideological Benchmark Analysis — multi-run aggregation"
    )
    parser.add_argument(
        "files", nargs="*",
        help="JSON result files or glob patterns (e.g. results/*.json)"
    )
    parser.add_argument(
        "--dir", "-d", default=None,
        help="Directory containing JSON result files (loads all *.json inside)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect file paths
    input_paths = list(args.files or [])
    if args.dir:
        input_paths.append(args.dir)
    if not input_paths:
        # legacy fallback
        input_paths = ["results_llama3_2_qwen.json"]

    file_paths = collect_files(input_paths)
    if not file_paths:
        print("ERROR: No JSON files found. Check your paths or --dir argument.")
        sys.exit(1)

    print(f"Loading {len(file_paths)} file(s)...")
    records, run_manifest = load_files(file_paths)
    print(f"  Total records: {len(records)}")
    for path, n in run_manifest.items():
        print(f"    {path}: {n} records")

    n_runs = len(file_paths)

    idx, meta = build_index(records)

    overview    = {mkey: model_overview(idx, mkey, n_runs) for mkey in MODELS}
    consistency = {mkey: consistency_per_question(idx, mkey, meta) for mkey in MODELS}
    divergence  = divergence_per_question(idx, meta)
    domain_sum  = domain_summary(idx, meta, divergence)
    axis        = axis_scores(idx, meta)

    report = generate_report(
        file_paths, n_runs, idx, meta, overview,
        consistency, divergence, domain_sum, axis
    )

    print(report)

    report_path = "ideological_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [Report saved] → {report_path}")

    question_rows = build_question_table(idx, meta, divergence, consistency)
    save_question_csv(question_rows, "per_question_detail.csv")


if __name__ == "__main__":
    main()