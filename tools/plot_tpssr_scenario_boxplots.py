#!/usr/bin/env python3
"""Make one journal-ready per-mask RMSE boxplot for each TPSSR damage scenario.

The source workbook is TPSSR_FINAL_ALL_RESULTS.xlsx (sheet: All_Masks).
Optionally replace its TPSSR rows with newer per-mask test_metrics.csv exports.
Only masks with complete reconstruction coverage are plotted.

Requirements: Python 3.9+, numpy, matplotlib. No Excel reader dependency.
"""

from __future__ import annotations

import argparse
import csv
import math
import posixpath
import re
import statistics
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Times New Roman is used where installed (e.g., Windows). Tinos is a
# metric-compatible serif fallback for machines without Times New Roman.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Tinos", "Liberation Serif", "DejaVu Serif"],
    "svg.fonttype": "none",  # Keep labels editable in exported SVG figures.
    "axes.unicode_minus": False,
})

METHOD_COLORS = {
    "TPSSR": "#4E79A7",
    "Spectral": "#76B7B2",
    "Spatial": "#B9A4D7",
    "Spatio-spectral": "#59A5D8",
    "PCA": "#EDB86C",
    "RF": "#8FBC8F",
    "DDS2M": "#E18D84",
    "PFGIN": "#A4A8C8",
}
MEAN_COLOR = "#1F7A4D"

XML_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_PKGREL = "http://schemas.openxmlformats.org/package/2006/relationships"

METHODS = (
    "TPSSR", "Spectral", "Spatial", "Spatio-spectral",
    "PCA", "RF", "DDS2M", "PFGIN",
)
SCENARIOS = (
    "isolated_voxels", "single_detector_line", "multiple_detector_lines",
    "partial_vertical_lines", "spatial_patch", "consecutive_spectral_bands",
    "spatio_spectral_block", "whole_spectrum_pixels", "band_wide_saturation",
    "mixed_saturation",
)
SCENARIO_TITLES = {
    "isolated_voxels": "Isolated Voxels",
    "single_detector_line": "Single Detector Line",
    "multiple_detector_lines": "Multiple Detector Lines",
    "partial_vertical_lines": "Partial Vertical Lines",
    "spatial_patch": "Spatial Patch",
    "consecutive_spectral_bands": "Consecutive Spectral Bands",
    "spatio_spectral_block": "Spatio-spectral Block",
    "whole_spectrum_pixels": "Whole-spectrum Pixels",
    "band_wide_saturation": "Band-wide Saturation",
    "mixed_saturation": "Mixed Saturation",
}


def args_from_command_line():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--excel", type=Path, default=Path("TPSSR_FINAL_ALL_RESULTS.xlsx"))
    parser.add_argument("--split", default="SWIR_8x2", help="Exact Split value in All_Masks")
    parser.add_argument("--tpssr-csv", type=Path, nargs="*", default=[], metavar="CSV",
                        help="Replace old workbook TPSSR rows with current per-mask CSV files")
    parser.add_argument("--outdir", type=Path, default=Path("figures/scenario_boxplots"))
    parser.add_argument("--baselines-only", action="store_true", help="Draw seven benchmarks without TPSSR")
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--svg-only", action="store_true", help="Skip PNG exports")
    return parser.parse_args()


def clean_rel_path(path: str) -> str:
    path = path.replace("\\", "/")
    return path.lstrip("/") if path.startswith("/") else posixpath.normpath(posixpath.join("xl", path))


def xlsx_rows(path: Path, sheet_name: str):
    """Read worksheet cells using the standard library, preserving sparse columns."""
    with zipfile.ZipFile(path) as archive:
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        relroot = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relpaths = {r.get("Id"): clean_rel_path(r.get("Target"))
                    for r in relroot.findall(f"{{{XML_PKGREL}}}Relationship")}
        sheet_id = None
        for sheet in book.findall(f".//{{{XML_MAIN}}}sheet"):
            if sheet.get("name") == sheet_name:
                sheet_id = sheet.get(f"{{{XML_REL}}}id")
                break
        if sheet_id is None:
            raise ValueError(f"Worksheet '{sheet_name}' was not found in {path}")
        sheet_path = relpaths.get(sheet_id)
        if not sheet_path or sheet_path not in archive.namelist():
            raise ValueError(f"Unable to locate worksheet '{sheet_name}' in the XLSX archive")

        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{XML_MAIN}}}si"):
                strings.append("".join(t.text or "" for t in item.iter(f"{{{XML_MAIN}}}t")))

        root = ET.fromstring(archive.read(sheet_path))
        for row in root.findall(f".//{{{XML_MAIN}}}sheetData/{{{XML_MAIN}}}row"):
            cells = {}
            for cell in row.findall(f"{{{XML_MAIN}}}c"):
                col = re.match(r"[A-Z]+", cell.get("r", ""))
                if col is None:
                    continue
                col_number = 0
                for letter in col.group(0):
                    col_number = 26 * col_number + ord(letter) - ord("A") + 1
                value = cell.find(f"{{{XML_MAIN}}}v")
                inline = cell.find(f"{{{XML_MAIN}}}is")
                if cell.get("t") == "s" and value is not None:
                    content = strings[int(value.text)]
                elif inline is not None:
                    content = "".join(t.text or "" for t in inline.iter(f"{{{XML_MAIN}}}t"))
                elif value is not None:
                    content = value.text or ""
                else:
                    content = ""
                cells[col_number - 1] = content
            yield int(row.get("r", "0")), cells


def workbook_records(path: Path, split: str):
    header = None
    records = {}
    for number, cells in xlsx_rows(path, "All_Masks"):
        if number == 4:
            header = {index: value for index, value in cells.items()}
            required = {"Split", "Dataset", "Method", "Scenario", "Severity", "Repeat",
                        "Mask_filename", "RMSE", "Coverage", "Full_coverage", "Status"}
            if not required <= set(header.values()):
                raise ValueError(f"All_Masks columns missing: {sorted(required - set(header.values()))}")
            continue
        if number < 5 or header is None:
            continue
        record = {name: cells.get(index, "") for index, name in header.items()}
        if record.get("Split") != split:
            continue
        method = record.get("Method")
        if method not in METHODS:
            continue
        key = (record["Dataset"], record["Scenario"], int(float(record["Severity"])),
               int(float(record["Repeat"])), Path(record["Mask_filename"]).name)
        if (method, key) in records:
            raise ValueError(f"Duplicate workbook mask: {method} / {key}")
        records[(method, key)] = record
    if not records:
        raise ValueError(f"No per-mask rows for split {split!r} in {path}")
    return records


def is_full_coverage(record):
    try:
        rmse = float(record["RMSE"])
        coverage = float(record["Coverage"])
    except (ValueError, TypeError, KeyError):
        return False
    status = str(record.get("Status", "")).strip().lower()
    full_flag = str(record.get("Full_coverage", "")).strip().lower()
    return (math.isfinite(rmse) and rmse >= 0
            and coverage >= 1 - 1e-9
            and full_flag in ("1", "true", "yes")
            and status in ("ok", "success", "complete"))


def merge_latest_tpssr(records, csv_paths):
    if not csv_paths:
        return 0
    existing_keys = {key for method, key in records if method == "TPSSR"}
    replaced = set()
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            for item in rows:
                key = (item["dataset"], item["scenario"], int(item["severity"]),
                       int(item["repeat"]), Path(item["mask_file"]).name)
                if key not in existing_keys:
                    raise ValueError(f"Latest CSV mask does not match the workbook: {key} in {path}")
                if key in replaced:
                    raise ValueError(f"Duplicate TPSSR mask across input CSV files: {key}")
                original = records[("TPSSR", key)]
                records[("TPSSR", key)] = {
                    **original, "RMSE": item.get("rmse", ""),
                    "Coverage": item.get("coverage", ""),
                    "Full_coverage": "1" if item.get("coverage") and float(item["coverage"]) >= 1 - 1e-9 else "0",
                    "Status": item.get("status", ""),
                }
                replaced.add(key)
    print(f"[TPSSR] Replaced {len(replaced)} workbook masks with current CSV measurements")
    if len(replaced) != len(existing_keys):
        raise ValueError(f"Incomplete TPSSR replacement: {len(replaced)}/{len(existing_keys)} masks. "
                         "Pass all relevant test_metrics CSV files, or omit --tpssr-csv.")
    return len(replaced)


def collect_by_scenario(records, methods):
    grouped = defaultdict(lambda: defaultdict(list))
    total = defaultdict(lambda: defaultdict(int))
    for (method, key), row in records.items():
        if method not in methods:
            continue
        scenario = key[1]
        total[scenario][method] += 1
        if is_full_coverage(row):
            grouped[scenario][method].append(float(row["RMSE"]))
    return grouped, total


def plot_scenario(scenario, values, counts, methods, options, destination):
    title = SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title())
    fig, ax = plt.subplots(figsize=(15.0, 7.5), constrained_layout=True)
    ax.set_title(title, fontsize=40, fontweight="bold", pad=22)

    all_values = [v for method in methods for v in values.get(method, [])]
    if not all_values:
        raise ValueError(f"No fully covered masks in {scenario}; cannot draw an RMSE boxplot")

    mean_handle = None
    for i, method in enumerate(methods, start=1):
        data = values.get(method, [])
        if not data:
            ax.text(i, .05, "N/A", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=30)
            continue

        box = ax.boxplot(
            data,
            positions=[i],
            widths=0.63,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            showfliers=False,  # Hide outlier markers, not outliers in the statistics.
            whis=1.5,
            boxprops={"edgecolor": "#374151", "linewidth": 1.7},
            whiskerprops={"color": "#374151", "linewidth": 1.6},
            capprops={"color": "#374151", "linewidth": 1.6},
            medianprops={"linewidth": 0, "color": "none"},
            meanprops={"color": MEAN_COLOR, "linestyle": "--", "linewidth": 3.0},
        )
        box["boxes"][0].set_facecolor(METHOD_COLORS[method])
        box["boxes"][0].set_alpha(0.82)
        if mean_handle is None:
            mean_handle = box["means"][0]


    ax.set_xlim(.42, len(methods) + .58)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, fontsize=22)
    ax.tick_params(axis="x", pad=12, length=0)
    ax.tick_params(axis="y", labelsize=30, length=5, width=1.1)
    ax.set_ylabel("RMSE", fontsize=30, labelpad=14)
    ax.set_yscale(options.yscale)
    if options.yscale == "linear":
        ax.set_ylim(bottom=0)
    elif min(all_values) <= 0:
        warnings.warn(f"Log scale requires RMSE > 0 in {scenario}; using linear scale instead")
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D7DCE2", alpha=0.6, linewidth=0.9)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    if mean_handle is not None:
        ax.legend(handles=[mean_handle], labels=["Mean RMSE"],
                  loc="upper left", frameon=False, fontsize=30)

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{scenario}_rmse_boxplot"
    # fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    if not options.svg_only:
        fig.savefig(path.with_suffix(".png"), dpi=options.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    options = args_from_command_line()
    methods = METHODS[1:] if options.baselines_only else METHODS
    records = workbook_records(options.excel, options.split)
    merge_latest_tpssr(records, options.tpssr_csv)
    grouped, totals = collect_by_scenario(records, methods)
    options.outdir.mkdir(parents=True, exist_ok=True)
    summary_file = options.outdir / "SCENARIO_BOXPLOT_STATISTICS.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "scenario", "method", "full_masks", "available_masks",
                         "mean_RMSE", "median_RMSE", "min_RMSE", "max_RMSE", "q1_RMSE", "q3_RMSE"])
        for i, scenario in enumerate(SCENARIOS, start=1):
            if scenario not in totals:
                print(f"[SKIP] {scenario}: no data in requested split")
                continue
            values, counts = grouped[scenario], totals[scenario]
            path = plot_scenario(scenario, values, counts, methods, options, options.outdir)
            for method in methods:
                data = values.get(method, [])
                if data:
                    q1, q3 = np.percentile(data, [25, 75])
                    writer.writerow([options.split, scenario, method, len(data), counts.get(method, 0),
                                     statistics.fmean(data), statistics.median(data), min(data), max(data), q1, q3])
                else:
                    writer.writerow([options.split, scenario, method, 0, counts.get(method, 0),
                                     "", "", "", "", "", ""])
            print(f"[{i:02d}/10] {path.name}.png/.svg")
    print(f"\nDone. Figures: {options.outdir.resolve()}")
    print(f"Statistics: {summary_file.resolve()}")


if __name__ == "__main__":
    main()
