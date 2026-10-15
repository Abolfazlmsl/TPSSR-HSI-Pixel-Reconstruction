#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Final TPSSR metrics for one or many held-out acquisitions.

The same trained model is evaluated separately on every configured test core.
A combined cross-core summary is then written under results/<experiment>/combined/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"

from _launcher_common import env_for
from experiment_config import (
    add_experiment_args,
    describe,
    experiment_cli_args,
    resolve_experiment,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Final TPSSR test outputs.")
    add_experiment_args(p)
    p.add_argument(
        "--visuals",
        type=int,
        default=3,
        help="Total representative visuals across the configured test acquisitions.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--routing-mode", choices=["hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23"], default="hybrid_v2_region_2expert_v23")
    p.add_argument("--spectral-guard-mae", type=float, default=0.08)
    p.add_argument("--spectral-guard-q80", type=float, default=0.14)
    p.add_argument("--spectral-guard-confidence", type=float, default=0.50)
    p.add_argument("--deep-batch", type=int, default=16)
    p.add_argument("--region-batch", type=int, default=32)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmean(rows, key):
    values = []
    for r in rows:
        try:
            x = float(r[key])
        except Exception:
            continue
        if np.isfinite(x):
            values.append(x)
    return float(np.mean(values)) if values else float("nan")


def distribute_visuals(total, n_tests):
    total = max(0, int(total))
    result = [0] * max(1, int(n_tests))
    for i in range(total):
        result[i % len(result)] += 1
    return result


def _write_v22_comparison_and_audit(spec, outdir, all_rows):
    """Compare strictly same mask identity and protocol, no hidden calibration."""
    previous = {}
    old_folder = 'final_tpssr_hybrid_v2_region_2expert_v21'
    for core in spec.tests:
        prior_file = ROOT / 'results' / spec.experiment_name / core / old_folder / 'test_metrics.csv'
        if not prior_file.is_file():
            continue
        for item in read_csv(prior_file):
            if item.get('status') == 'ok':
                key = (core, item['mask_file'])
                if key in previous:
                    raise RuntimeError('Duplicate V2.1 comparison key: ' + str(key))
                previous[key] = item
    paired = []
    for item in all_rows:
        key = (item['dataset'], item['mask_file'])
        old = previous.get(key)
        if old is None:
            continue
        for k in ('scenario', 'severity', 'repeat', 'eval_protocol'):
            if str(item.get(k, '')) != str(old.get(k, '')):
                raise RuntimeError('V2.2/V2.1 mask protocol mismatch: ' + str(key))
        a, b = float(item['rmse']), float(old['rmse'])
        rec = {k: item.get(k, '') for k in ('dataset', 'mask_file', 'scenario', 'severity', 'repeat')}
        rec.update(v22_rmse=a, v21_rmse=b, delta_rmse=a-b,
                   winner='v22' if a < b-1e-12 else 'v21' if b < a-1e-12 else 'tie',
                   v22_mae=item.get('mae', ''), v21_mae=old.get('mae', ''),
                   v22_sam_deg=item.get('sam_deg', ''), v21_sam_deg=old.get('sam_deg', ''),
                   v22_ssim=item.get('ssim', ''), v21_ssim=old.get('ssim', ''),
                   v22_coverage=item.get('coverage', ''), v21_coverage=old.get('coverage', ''),
                   v22_region_fraction=item.get('final_region_fraction', ''),
                   v21_region_fraction=old.get('final_region_fraction', ''),
                   v22_long_gap_region_pixels=item.get('v22_region_long_gap_pixels', ''),
                   v22_returned_to_hybrid=item.get('v22_region_returned_to_hybrid_pixels', ''))
        paired.append(rec)
    if paired:
        write_csv(outdir / 'PAIRED_V22_V21.csv', paired)
        groups = {}
        for r in paired:
            groups.setdefault((r['dataset'], r['scenario'], int(r['severity'])), []).append(r)
        summary_rows = []
        for (ds, sc, severity), rows in sorted(groups.items()):
            summary_rows.append({
                'dataset': ds, 'scenario': sc, 'severity': severity, 'masks': len(rows),
                'v22_mean_rmse': fmean(rows, 'v22_rmse'),
                'v21_mean_rmse': fmean(rows, 'v21_rmse'),
                'delta_mean_rmse': fmean(rows, 'delta_rmse'),
                'v22_wins': sum(r['winner'] == 'v22' for r in rows),
                'v21_wins': sum(r['winner'] == 'v21' for r in rows),
                'ties': sum(r['winner'] == 'tie' for r in rows),
            })
        write_csv(outdir / 'PAIRED_V22_V21_SCENARIO_SEVERITY.csv', summary_rows)
    # Read both worker copies and provide a hard audit trail.
    workers = []
    for core in spec.tests:
        worker_file = ROOT / 'results' / spec.experiment_name / core / 'final_tpssr_hybrid_v2_region_2expert_v22' / 'worker.rank0.json'
        if worker_file.is_file():
            workers.append(json.loads(worker_file.read_text(encoding='utf-8')).get('v22_policy', {}))
    text = [
        'TPSSR Hybrid V2 / Region V2.2 — development-only true-mask gap purity guard',
        'Frozen Hybrid V2 and Region. No Deep-v3.1 or learned V6 Router.',
        'V2.2 NEVER adds new Region pixels beyond V2.1; it can only reject a V2.1 gap-Region proposal.',
        'No test truth, scenario label, severity or acquisition identifier enters inference decisions.',
        'WARNING: The proposed guard followed retrospective analysis of V2.1 on SWIR5/SWIR10.',
        'SWIR5/SWIR10 ARE NOT AN UNTOUCHED TEST for V2.2; interpret strictly as exploratory.',
        'Region was trained on SWIR4/SWIR9 as well, so those are not independent validation for Region.',
        f'policy_per_worker={json.dumps(workers, sort_keys=True)}',
        f'v22_success={len(all_rows)}; paired_v21={len(paired)}',
        f'v22_wins={sum(x["winner"] == "v22" for x in paired)}; v21_wins={sum(x["winner"] == "v21" for x in paired)}; ties={sum(x["winner"] == "tie" for x in paired)}',
        f'v22_gap_region_eval_voxels_sum={sum(int(float(x.get("v22_longgap_region_eval_voxels",0))) for x in all_rows)}',
        f'v22_gap_region_pixels_sum={sum(int(float(x.get("v22_region_long_gap_pixels",0))) for x in all_rows)}',
        f'returned_to_hybrid_pixels_sum={sum(int(float(x.get("v22_region_returned_to_hybrid_pixels",0))) for x in all_rows)}',
        f'rejected_non_dominant_sum={sum(int(float(x.get("v22_rejected_non_dominant_pixels",0))) for x in all_rows)}',
        f'rejected_fragmented_target_sum={sum(int(float(x.get("v22_rejected_fragmented_pixels",0))) for x in all_rows)}',
    ]
    (outdir / 'V22_POLICY_AUDIT.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')


def combined_summary(spec, routing_mode):
    folder = {"baseline": "final_tpssr", "spectral_safe": "final_tpssr_spectral_safe_stage1_v2", "joint_v3": "final_tpssr_joint_v3", "stage1_rank_v4": "final_tpssr_stage1_rank_v4", "spectral_region": "final_tpssr_spectral_region", "hard60_v5": "final_tpssr_hard60_v5", "hard60_v6": "final_tpssr_hard60_v6", "hard60_v6_hybrid_v2_fixed": "final_tpssr_hard60_v6_hybrid_v2_fixed", "hybrid_v2_region_2expert": "final_tpssr_hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21": "final_tpssr_hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22": "final_tpssr_hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23": "final_tpssr_hybrid_v2_region_2expert_v23"}[routing_mode]
    all_rows = []
    dataset_rows = []
    for test_name in spec.tests:
        test_dir = ROOT / "results" / spec.experiment_name / test_name / folder
        metrics = test_dir / "test_metrics.csv"
        summary_path = test_dir / "summary.json"
        if not metrics.exists():
            raise FileNotFoundError(
                f"Missing Final TPSSR metrics for {test_name}: {metrics}"
            )
        rows = read_csv(metrics)
        ok = [r for r in rows if r.get("status") == "ok"]
        for row in ok:
            row = dict(row)
            row.setdefault("dataset", test_name)
            all_rows.append(row)
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            dataset_rows.append({
                "dataset": test_name,
                "masks": int(s.get("n_ok", len(ok))),
                "rmse": s.get("mean_rmse"),
                "mae": s.get("mean_mae"),
                "r2": s.get("mean_r2"),
                "sam_deg": s.get("mean_sam_deg"),
                "ssim": s.get("mean_ssim"),
                "coverage": s.get("mean_coverage"),
                "spectral_fraction": s.get("mean_final_spectral_fraction"),
                "hybrid_fraction": s.get("mean_final_spectral_fraction") if routing_mode in ("hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23") else "",
                "deep_fraction": s.get("mean_final_deep_fraction"),
                "region_fraction": s.get("mean_final_region_fraction"),
            })

    outdir = ROOT / "results" / spec.experiment_name / "combined" / folder
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "all_test_metrics.csv", all_rows)
    write_csv(outdir / "dataset_summary.csv", dataset_rows)
    if routing_mode == "hard60_v6_hybrid_v2_fixed":
        comparisons = []
        for row in all_rows:
            rmse = float(row['rmse']); ref = float(row['reference_v6_rmse'])
            comparisons.append({
                'dataset':row['dataset'],'mask_file':row['mask_file'],
                'scenario':row['scenario'],'severity':row['severity'],'repeat':row['repeat'],
                'v6_fixed_route_rmse':ref,'hybrid_v2_fixed_route_rmse':rmse,
                'delta_rmse':rmse-ref,
                'winner':'hybrid_v2' if rmse<ref-1e-12 else 'v6' if ref<rmse-1e-12 else 'tie',
                'hybrid_changed_voxels':row.get('hybrid_changed_voxels',''),
                'hybrid_fallback_voxels':row.get('hybrid_fallback_voxels',''),
            })
        write_csv(outdir / 'PAIRED_V6_DIFFERENCES.csv',comparisons)
        route_audit=[]
        for test_name in spec.tests:
            old_file=ROOT/'results'/spec.experiment_name/test_name/'final_tpssr_hard60_v6'/'component_decisions.csv'
            new_file=ROOT/'results'/spec.experiment_name/test_name/'final_tpssr_hard60_v6_hybrid_v2_fixed'/'component_decisions.csv'
            if not old_file.exists():
                route_audit.append(f'{test_name}: V6 component_decisions.csv not found; direct route equality not checked')
                continue
            if not new_file.exists():
                route_audit.append(f'{test_name}: new component decisions missing')
                continue
            oldrows=read_csv(old_file); newrows=read_csv(new_file)
            def route_key(r):
                return (r['mask_file'],int(r['component']))
            od={route_key(r):(int(r['stage1']),int(r['final'])) for r in oldrows}
            nd={route_key(r):(int(r['stage1']),int(r['final'])) for r in newrows}
            overlap = od.keys() & nd.keys()
            mismatches=sum(od[k]!=nd[k] for k in overlap)
            missing_new=len(nd.keys()-od.keys())
            missing_old=len(od.keys()-nd.keys())
            partial = len(nd)<len(od)
            audit_line=(f'{test_name}: V6 components={len(od)} Hybrid-run components={len(nd)} '
                        f'compared={len(overlap)} route_mismatches={mismatches} '
                        f'new_keys_not_in_v6={missing_new} v6_keys_not_in_new={missing_old}'
                        + (' [PARTIAL SMOKE RUN]' if partial else ''))
            route_audit.append(audit_line)
            if mismatches or missing_new or (missing_old and not partial):
                raise RuntimeError('Frozen V6 routing audit FAILED: '+audit_line)

        (outdir/'V6_FIXED_ROUTE_AUDIT.txt').write_text('\n'.join(route_audit)+'\n',encoding='utf-8')

    if routing_mode == "hybrid_v2_region_2expert":
        # Paired at the exact 200-mask identifiers and unchanged eval protocol.
        # RMSE values are per-mask (not aggregated voxel-wise).  The route
        # decisions are intentionally DIFFERENT from the three-Expert baseline.
        old_folder = "final_tpssr_hard60_v6_hybrid_v2_fixed"
        previous = {}
        for core in spec.tests:
            old_csv = (ROOT / "results" / spec.experiment_name / core
                       / old_folder / "test_metrics.csv")
            if not old_csv.is_file():
                continue
            for item in read_csv(old_csv):
                if item.get("status") == "ok":
                    previous[(core, item["mask_file"])] = item
        paired = []
        for item in all_rows:
            key = (item["dataset"], item["mask_file"])
            prior = previous.get(key)
            if prior is None:
                continue
            if (str(prior.get("scenario", "")) != str(item.get("scenario", ""))
                or str(prior.get("severity", "")) != str(item.get("severity", ""))
                or str(prior.get("repeat", "")) != str(item.get("repeat", ""))
                or str(prior.get("eval_protocol", "")) != str(item.get("eval_protocol", ""))):
                raise RuntimeError("Three-vs-two Expert pair metadata mismatch: " + str(key))
            a, b = float(item["rmse"]), float(prior["rmse"])
            paired.append({
                "dataset": item["dataset"], "mask_file": item["mask_file"],
                "scenario": item["scenario"], "severity": item["severity"],
                "repeat": item["repeat"],
                "two_expert_rmse": a, "three_expert_rmse": b,
                "rmse_delta_two_minus_three": a-b,
                "winner": "2_expert" if a < b-1e-12 else "3_expert" if b < a-1e-12 else "tie",
                "two_expert_sam_deg": item.get("sam_deg", ""),
                "three_expert_sam_deg": prior.get("sam_deg", ""),
                "two_expert_ssim": item.get("ssim", ""),
                "three_expert_ssim": prior.get("ssim", ""),
                "two_expert_region_fraction": item.get("final_region_fraction", ""),
                "three_expert_region_fraction": prior.get("final_region_fraction", ""),
            })
        if paired:
            write_csv(outdir / "PAIRED_2EXPERT_VS_3EXPERT.csv", paired)
        win2 = sum(x["winner"] == "2_expert" for x in paired)
        win3 = sum(x["winner"] == "3_expert" for x in paired)
        (outdir / "TWO_EXPERT_AUDIT.txt").write_text(
            "Hybrid V2 + Region: no Deep or learned V6 Router used.\n"
            "Eligibility: split 2D missing mask into 4-connected components "
            "of pixels with >=2 / <2 observed finite bands.\n"
            "Hybrid has >=2 observed; Region has <2 observed.\n"
            "Route labels: 0=Hybrid V2; 1=unused; 2=Region.\n"
            "This is a controlled two-Expert feasibility ablation, not a learned router.\n"
            f"Successful two-Expert masks: {len(all_rows)}; "
            f"paired against previous three-Expert Hybrid run: {len(paired)}; "
            f"two wins={win2}, three wins={win3}, ties={len(paired)-win2-win3}.\n"
            "No paired comparison is possible for missing old masks.\n",
            encoding="utf-8",
        )

    if routing_mode == "hybrid_v2_region_2expert_v22":
        _write_v22_comparison_and_audit(spec, outdir, all_rows)

    if routing_mode == "hybrid_v2_region_2expert_v21":
        # Pair only exact dataset/mask/scenario/severity/repeat/protocol identities.
        # This is an exploratory retrospective comparison, NOT new independent validation.
        baseline_folder = "final_tpssr_hybrid_v2_region_2expert"
        previous = {}
        for core in spec.tests:
            old_csv = (ROOT / "results" / spec.experiment_name / core
                       / baseline_folder / "test_metrics.csv")
            if not old_csv.is_file():
                continue
            for item in read_csv(old_csv):
                if item.get("status") == "ok":
                    previous[(core, item["mask_file"])] = item
        paired = []
        for item in all_rows:
            key = (item["dataset"], item["mask_file"])
            prior = previous.get(key)
            if prior is None:
                continue
            for k in ("scenario", "severity", "repeat", "eval_protocol"):
                if str(item.get(k, "")) != str(prior.get(k, "")):
                    raise RuntimeError("V2.0-vs-V2.1 comparison protocol mismatch: " + str(key))
            a, b = float(item["rmse"]), float(prior["rmse"])
            paired.append({
                "dataset": item["dataset"], "mask_file": item["mask_file"],
                "scenario": item["scenario"], "severity": item["severity"],
                "repeat": item["repeat"],
                "v21_rmse": a, "v20_rmse": b, "delta_rmse_v21_minus_v20": a-b,
                "winner": "v21" if a < b-1e-12 else "v20" if b < a-1e-12 else "tie",
                "v21_mae": item.get("mae", ""), "v20_mae": prior.get("mae", ""),
                "v21_r2": item.get("r2", ""), "v20_r2": prior.get("r2", ""),
                "v21_sam_deg": item.get("sam_deg", ""), "v20_sam_deg": prior.get("sam_deg", ""),
                "v21_ssim": item.get("ssim", ""), "v20_ssim": prior.get("ssim", ""),
                "v21_coverage": item.get("coverage", ""), "v20_coverage": prior.get("coverage", ""),
                "v21_region_fraction": item.get("final_region_fraction", ""),
                "v20_region_fraction": prior.get("final_region_fraction", ""),
                "v21_longgap_region_eval_voxels": item.get("v21_region_long_gap_eval_voxels", ""),
                "v21_longgap_region_pixels": item.get("v21_region_long_gap_pixels", ""),
            })
        if paired:
            write_csv(outdir / "PAIRED_V21_V20.csv", paired)
            severity_pairs = []
            for (dataset, scenario, severity) in sorted({
                (x["dataset"], x["scenario"], int(x["severity"])) for x in paired
            }):
                rr = [x for x in paired if x["dataset"] == dataset and
                      x["scenario"] == scenario and int(x["severity"]) == severity]
                severity_pairs.append({
                    "dataset": dataset, "scenario": scenario, "severity": severity,
                    "masks": len(rr),
                    "v20_mean_rmse": fmean(rr, "v20_rmse"),
                    "v21_mean_rmse": fmean(rr, "v21_rmse"),
                    "delta_mean_rmse": fmean(rr, "delta_rmse_v21_minus_v20"),
                    "v21_wins": sum(x["winner"] == "v21" for x in rr),
                    "v20_wins": sum(x["winner"] == "v20" for x in rr),
                })
            write_csv(outdir / "PAIRED_V21_V20_SCENARIO_SEVERITY.csv", severity_pairs)
        from_json = []
        for core in spec.tests:
            worker = ROOT / "results" / spec.experiment_name / core / "final_tpssr_hybrid_v2_region_2expert_v21" / "worker.rank0.json"
            if worker.is_file():
                from_json.append(json.loads(worker.read_text(encoding="utf-8")).get("v21_policy", {}))
        from_summary = "\n".join([
            "TPSSR Hybrid V2 + Region V2.1 — frozen geometric routing policy",
            "V2.1 uses NO Deep-v3.1 or learned V6 Router; no expert retraining.",
            "No masked target values, labels, scenario names, or test identities enter routing.",
            "These fixed geometry thresholds were NOT calibrated on SWIR5/SWIR10.",
            "SWIR4/SWIR9 are NOT independent validation for the Region expert, which was trained on those cores.",
            "Comparison on SWIR5/SWIR10 is DEVELOPMENT/EXPLORATORY, not an untouched final test.",
            "Frozen policy per worker: " + json.dumps(from_json, sort_keys=True),
            f"Successful masks={len(all_rows)}, paired to V2.0={len(paired)}, "
            f"V2.1 wins={sum(x['winner']=='v21' for x in paired)}, "
            f"V2.0 wins={sum(x['winner']=='v20' for x in paired)}, "
            f"ties={sum(x['winner']=='tie' for x in paired)}.",
            f"Long-gap Region sampled eval voxels summed (can repeat across masks)="
            f"{sum(int(float(r.get('v21_region_long_gap_eval_voxels',0))) for r in all_rows)}.",
            f"Long-gap Region pixels summed over masks="
            f"{sum(int(float(r.get('v21_region_long_gap_pixels',0))) for r in all_rows)}.",
        ])
        (outdir / "V21_POLICY_AUDIT.txt").write_text(from_summary + "\n", encoding="utf-8")

    summary = {
        "experiment": spec.experiment_name,
        "train_datasets": list(spec.train),
        "test_datasets": list(spec.tests),
        "n_test_datasets": len(spec.tests),
        "n_ok_masks": len(all_rows),
        "mean_rmse": fmean(all_rows, "rmse"),
        "mean_mae": fmean(all_rows, "mae"),
        "mean_r2": fmean(all_rows, "r2"),
        "mean_sam_deg": fmean(all_rows, "sam_deg"),
        "mean_ssim": fmean(all_rows, "ssim"),
        "mean_coverage": fmean(all_rows, "coverage"),
        "mean_final_spectral_fraction": fmean(all_rows, "final_spectral_fraction"),
        "mean_final_deep_fraction": fmean(all_rows, "final_deep_fraction"),
        "mean_final_region_fraction": fmean(all_rows, "final_region_fraction"),
    }
    (outdir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    lines = [
        "FINAL TPSSR COMBINED HELD-OUT EVALUATION",
        "=" * 76,
        f"Experiment: {spec.experiment_name}",
        f"Train: {', '.join(spec.train)}",
        f"Test: {', '.join(spec.tests)}",
        f"Successful masks across all test acquisitions: {len(all_rows)}",
        f"RMSE={summary['mean_rmse']:.6f}",
        f"MAE={summary['mean_mae']:.6f}",
        f"R2={summary['mean_r2']:.6f}",
        f"SAM={summary['mean_sam_deg']:.4f} deg",
        f"SSIM={summary['mean_ssim']:.4f}",
        f"Coverage={summary['mean_coverage']:.4f}",
    ]
    (outdir / "COMBINED_SUMMARY.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join(lines))
    print("Saved combined results:", outdir)


def run_all_tests(args, spec):
    print(describe(spec))
    allocations = distribute_visuals(args.visuals, len(spec.tests))
    for i, (test_name, n_visuals) in enumerate(zip(spec.tests, allocations), start=1):
        print(
            f"\n{'='*90}\nEVALUATE TEST DATASET {i}/{len(spec.tests)}: {test_name}\n{'='*90}"
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *experiment_cli_args(spec, active_test=test_name),
            "--visuals",
            str(n_visuals),
            "--gpu",
            str(args.gpu),
            "--deep-batch",
            str(args.deep_batch),
            "--region-batch",
            str(args.region_batch),
            "--crop-size",
            str(args.crop_size),
            "--routing-mode", args.routing_mode,
            "--spectral-guard-mae", str(args.spectral_guard_mae),
            "--spectral-guard-q80", str(args.spectral_guard_q80),
            "--spectral-guard-confidence", str(args.spectral_guard_confidence),
            "--single",
        ]
        print("RUN:", " ".join(map(str, cmd)), flush=True)
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env_for(spec, test_name),
            check=True,
        )
    combined_summary(spec, args.routing_mode)


ARGS = parse_args()
SPEC = resolve_experiment(ARGS)
if not ARGS.single:
    run_all_tests(ARGS, SPEC)
    raise SystemExit(0)

DATASET = SPEC.active_test
os.environ.update(env_for(SPEC, DATASET))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import torch
import config
from src.evaluation_utils import load_dataset, split_blocks, list_mask_files, load_test_mask
from src.final_pipeline import load_pipeline, predict_mask

OUTDIR = Path(config.RESULTS_DIR) / ({"baseline": "final_tpssr", "spectral_safe": "final_tpssr_spectral_safe_stage1_v2", "joint_v3": "final_tpssr_joint_v3", "stage1_rank_v4": "final_tpssr_stage1_rank_v4", "spectral_region": "final_tpssr_spectral_region", "hard60_v5": "final_tpssr_hard60_v5", "hard60_v6": "final_tpssr_hard60_v6", "hard60_v6_hybrid_v2_fixed": "final_tpssr_hard60_v6_hybrid_v2_fixed", "hybrid_v2_region_2expert": "final_tpssr_hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21": "final_tpssr_hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22": "final_tpssr_hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23": "final_tpssr_hybrid_v2_region_2expert_v23"}[ARGS.routing_mode])
METRICS = OUTDIR / "test_metrics.csv"
VISDIR = OUTDIR / "visuals"


def aggregate(rows, key_name):
    out = []
    keys = sorted({str(r[key_name]) for r in rows})
    for key in keys:
        rr = [r for r in rows if str(r[key_name]) == key]
        out.append({
            key_name: key,
            "masks": len(rr),
            "rmse": fmean(rr, "rmse"),
            "mae": fmean(rr, "mae"),
            "r2": fmean(rr, "r2"),
            "sam_deg": fmean(rr, "sam_deg"),
            "ssim": fmean(rr, "ssim"),
            "coverage": fmean(rr, "coverage"),
            "spectral_fraction": fmean(rr, "final_spectral_fraction"),
            "deep_fraction": fmean(rr, "final_deep_fraction"),
            "region_fraction": fmean(rr, "final_region_fraction"),
        })
    return out


def find_case(mask_files, scenario, severity, repeat):
    for path in mask_files:
        _mask, meta = load_test_mask(path)
        if (
            str(meta["scenario"]) == scenario
            and int(meta["severity"]) == int(severity)
            and int(meta["repeat"]) == int(repeat)
        ):
            return path
    return None


def representative_band(mask):
    counts = np.count_nonzero(mask, axis=(0, 1))
    if not np.any(counts > 0):
        return int(mask.shape[2] // 2)
    maximum = int(counts.max())
    candidates = np.flatnonzero(counts == maximum)
    center = (mask.shape[2] - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - center))])


def crop_bounds(mask2d, crop_size):
    ys, xs = np.where(mask2d)
    h, w = mask2d.shape
    size = int(min(crop_size, h, w))
    if len(ys) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = max(0, min(cy - size // 2, h - size))
    x0 = max(0, min(cx - size // 2, w - size))
    return y0, y0 + size, x0, x0 + size


def export_visuals(n_visuals):
    if n_visuals <= 0:
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("Visual reconstruction export requires CUDA.")

    import matplotlib.pyplot as plt

    torch.cuda.set_device(int(ARGS.gpu))
    device = torch.device(f"cuda:{int(ARGS.gpu)}")
    pipeline = load_pipeline(
        device,
        deep_batch=ARGS.deep_batch,
        region_batch=ARGS.region_batch,
        verify_region_batch=True,
        routing_mode=ARGS.routing_mode,
        spectral_guard_mae=ARGS.spectral_guard_mae,
        spectral_guard_q80=ARGS.spectral_guard_q80,
        spectral_guard_confidence=ARGS.spectral_guard_confidence,
    )
    cube, _, split, wavelengths = load_dataset(DATASET)
    _, _, test_cube = split_blocks(cube, split)
    wavelengths = np.asarray(wavelengths)
    mask_files = list_mask_files()
    VISDIR.mkdir(parents=True, exist_ok=True)

    candidate_cases = [
        ("whole_spectrum_pixels", 5, 1),
        ("spatio_spectral_block", 5, 1),
        ("mixed_saturation", 5, 1),
        ("consecutive_spectral_bands", 5, 1),
        ("spatial_patch", 5, 1),
    ]
    requested = candidate_cases[: int(n_visuals)]

    saved = []
    for number, (scenario, severity, repeat) in enumerate(requested, start=1):
        path = find_case(mask_files, scenario, severity, repeat)
        if path is None:
            continue
        mask, meta = load_test_mask(path)
        mask = np.asarray(mask, dtype=bool)
        band = representative_band(mask)
        y0, y1, x0, x1 = crop_bounds(mask[:, :, band], ARGS.crop_size)
        local = mask[y0:y1, x0:x1, band]
        yx = np.argwhere(local)
        if len(yx) == 0:
            continue
        coords = np.column_stack([
            yx[:, 0] + y0,
            yx[:, 1] + x0,
            np.full(len(yx), band, dtype=np.int64),
        ]).astype(np.int64)

        print(
            f"[{DATASET}] Visual {number}/{n_visuals}: {scenario} "
            f"S{severity} R{repeat} | band {band} | {len(coords)} masked pixels"
        )
        out = predict_mask(
            pipeline,
            test_cube,
            mask,
            coords,
            preview_progress=f"{DATASET} Visual {number}/{n_visuals} previews",
            region_progress=f"{DATASET} Visual {number}/{n_visuals} Region",
        )
        original = np.asarray(test_cube[y0:y1, x0:x1, band], dtype=np.float32).copy()
        corrupted = original.copy()
        corrupted[local] = np.nan
        reconstructed = original.copy()
        for (y, x, _b), value in zip(coords, out["prediction"]):
            reconstructed[int(y) - y0, int(x) - x0] = float(value)

        finite = original[np.isfinite(original)]
        if finite.size:
            vmin, vmax = np.quantile(finite, [0.01, 0.99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0

        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("white")
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        for ax, image, title in zip(
            axes,
            [original, corrupted, reconstructed],
            ["Original", "Corrupted", "Final TPSSR"],
        ):
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title)
            ax.axis("off")
        wl = float(wavelengths[band]) if band < len(wavelengths) else float("nan")
        route = out["final_route"]
        fig.suptitle(
            f"{DATASET} | {scenario} | Severity {severity} | Repeat {repeat} | "
            f"band {band} ({wl:.1f} nm) | "
            f"routes S/D/R={np.mean(route==0):.2f}/{np.mean(route==1):.2f}/{np.mean(route==2):.2f}"
        )
        fig.tight_layout()
        outpath = VISDIR / (
            f"{number:02d}_{DATASET}_{scenario}_S{severity}_R{repeat}_band{band}.png"
        )
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(outpath))
    return saved


def main_single():
    if not METRICS.exists():
        raise SystemExit(
            f"Test metrics not found: {METRICS}\nRun test.py for {DATASET} first."
        )
    rows = read_csv(METRICS)
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No successful test rows were found.")

    write_csv(OUTDIR / "scenario_summary.csv", aggregate(ok, "scenario"))
    write_csv(OUTDIR / "severity_summary.csv", aggregate(ok, "severity"))

    worker0 = {}
    p0 = OUTDIR / "worker.rank0.json"
    if p0.exists():
        worker0 = json.loads(p0.read_text(encoding="utf-8"))

    summary = {
        "architecture": (
            "Two experts: Spectral (>=2 observed bands at every component pixel) "
            "else frozen Region (V27 Stage-A + V13 TopologyFirst V1); Deep unused"
            if ARGS.routing_mode == "spectral_region" else
            "Two experts ONLY: Frozen Hybrid V2 BAND gate / Frozen Region. "
            "Mask-based eligibility per 4-connected spectral-availability component "
            "(Hybrid if >=2 observed bands; else Region); no Deep or V6 router"
            if ARGS.routing_mode == "hybrid_v2_region_2expert" else
            "Two experts ONLY: Frozen Hybrid V2 BAND / Frozen Region, plus "
            "v2.1 frozen long-gap and 8-neighbor observed-band support policy; "
            "NO Deep or V6 router"
            if ARGS.routing_mode == "hybrid_v2_region_2expert_v21" else
            "Two experts ONLY: Frozen Hybrid V2 BAND / Frozen Region; "
            "V2.2 true-mask long-gap purity guard (development heuristic); "
            "NO Deep or V6 Router"
            if ARGS.routing_mode == "hybrid_v2_region_2expert_v22" else
            "Two experts ONLY: Frozen Hybrid V2 BAND / Frozen Region; "
            "V2.3 additional long-gap Region + isolated full-spectrum local median; "
            "NO Deep or learned Router"
            if ARGS.routing_mode == "hybrid_v2_region_2expert_v23" else
            "V6 frozen router with Hybrid V2 BAND replacing Spectral-selected OUTPUT ONLY; "
            "Deep-v3.1 and Region unchanged"
            if ARGS.routing_mode == "hard60_v6_hybrid_v2_fixed" else
            "Stage1(Spectral vs Deep-v3.1) -> hard-case gate -> "
            "Frozen V27 Stage-A -> V13 TopologyFirst V1"
        ),
        "experiment": config.EXPERIMENT_NAME,
        "routing_mode": ARGS.routing_mode,
        "train_datasets": list(config.TRAIN_DATASET_FOLDERS),
        "test_datasets": list(config.TEST_DATASET_FOLDERS),
        "dataset": DATASET,
        "n_requested": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "mean_rmse": fmean(ok, "rmse"),
        "mean_mae": fmean(ok, "mae"),
        "mean_r2": fmean(ok, "r2"),
        "mean_sam_deg": fmean(ok, "sam_deg"),
        "mean_ssim": fmean(ok, "ssim"),
        "mean_coverage": fmean(ok, "coverage"),
        "mean_final_spectral_fraction": fmean(ok, "final_spectral_fraction"),
        "mean_final_deep_fraction": fmean(ok, "final_deep_fraction"),
        "mean_final_region_fraction": fmean(ok, "final_region_fraction"),
        "mean_runtime_sec_per_mask": fmean(ok, "runtime_sec"),
        **worker0,
    }

    visual_paths = export_visuals(int(ARGS.visuals))
    summary["visuals"] = visual_paths
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    lines = [
        "FINAL TPSSR EVALUATION",
        "=" * 72,
        summary["architecture"],
        f"Experiment: {config.EXPERIMENT_NAME}",
        f"Dataset: {DATASET}",
        f"Masks: {summary['n_ok']}/{summary['n_requested']} successful",
        f"RMSE={summary['mean_rmse']:.6f}",
        f"MAE={summary['mean_mae']:.6f}",
        f"R2={summary['mean_r2']:.6f}",
        f"SAM={summary['mean_sam_deg']:.4f} deg",
        f"SSIM={summary['mean_ssim']:.4f}",
        f"Coverage={summary['mean_coverage']:.4f}",
        "",
        "Mean final route fractions:",
        f"  {'Hybrid V2' if ARGS.routing_mode in ('hybrid_v2_region_2expert', 'hybrid_v2_region_2expert_v21', 'hybrid_v2_region_2expert_v22') else 'Spectral'}={summary['mean_final_spectral_fraction']:.4f}",
        f"  {'Deep (disabled)' if ARGS.routing_mode in ('hybrid_v2_region_2expert', 'hybrid_v2_region_2expert_v21', 'hybrid_v2_region_2expert_v22') else 'Deep-v3.1'}={summary['mean_final_deep_fraction']:.4f}",
        f"  Final Region={summary['mean_final_region_fraction']:.4f}",
        "",
        f"Gate threshold: {summary.get('gate_threshold', float('nan'))}",
        f"Visuals exported for this dataset: {len(visual_paths)}",
    ]
    (OUTDIR / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main_single()
