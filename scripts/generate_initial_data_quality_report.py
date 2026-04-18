#!/usr/bin/env python3
"""Generate the initial data understanding report and appendix tables.

The workflow intentionally uses only the Python standard library so it can run
from a clean checkout without dependency installation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "initial_data"
REPORT_DIR = ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
REPORT_PATH = REPORT_DIR / "initial_data_quality_report.md"

UNIQUE_CAP = 250_000
TOP_VALUE_CAP = 10_000
TOP_VALUES_PER_COLUMN = 10

EXPECTED_LOGICAL_ROWS = {
    "AffectedPersons__20260418_072029_UTC.csv": 16,
    "AllocationFlow__20260418_071755_UTC.csv": 198,
    "AllocationsByOrgType__20260418_071803_UTC.csv": 18,
    "AllocationsTimeline__20260418_071808_UTC.csv": 49,
    "Allocations__20260418_071733_UTC.csv": 33,
    "CBPFcontributions__20260418_071813_UTC.csv": 145,
    "ContributionTrends__20260418_071819_UTC.csv": 381,
    "ContributionsFlow__20260418_071825_UTC.csv": 145,
    "GenderAgeMarker__20260418_072032_UTC.csv": 35,
    "SectorsOverview__20260418_071833_UTC.csv": 97,
    "cod_population_admin0.csv": 6_722,
    "cod_population_admin1.csv": 91_471,
    "cod_population_admin2.csv": 1_001_583,
    "cod_population_admin3.csv": 241_962,
    "cod_population_admin4.csv": 17_465,
    "fts_incoming_funding_global.csv": 7_990,
    "fts_internal_funding_global.csv": 675,
    "fts_outgoing_funding_global.csv": 3_199,
    "fts_requirements_funding_cluster_global.csv": 8_030,
    "fts_requirements_funding_covid_global.csv": 152,
    "fts_requirements_funding_global.csv": 3_805,
    "fts_requirements_funding_globalcluster_global.csv": 10_630,
    "hpc_hno_2024.csv": 387_819,
    "hpc_hno_2025.csv": 318_259,
    "hpc_hno_2026.csv": 134,
    "humanitarian-response-plans.csv": 910,
}

SOURCE_URLS = {
    "HNO": "https://data.humdata.org/dataset/global-hpc-hno",
    "HRP": "https://data.humdata.org/dataset/humanitarian-response-plans",
    "COD-PS": "https://data.humdata.org/dataset/cod-ps-global",
    "FTS": "https://data.humdata.org/dataset/global-requirements-and-funding-data",
    "CBPF": "https://cbpf.data.unocha.org/",
}

SOURCE_NOTES = [
    {
        "source": "HNO / HPC",
        "url": "https://hdx-hapi.readthedocs.io/en/latest/data_usage_guides/affected_people/",
        "note": (
            "HNO data represents people by sector/status. Source guidance warns not "
            "to sum PIN across sectors or population statuses because the same "
            "people can appear in multiple groups."
        ),
    },
    {
        "source": "FTS",
        "url": "https://hdx-hapi.readthedocs.io/en/latest/data_usage_guides/coordination_and_context/",
        "note": (
            "FTS funding data is reported by donors and recipient organizations; "
            "timepoints are not regular and appeal-linked funding is the most "
            "directly comparable to HRP requirements."
        ),
    },
    {
        "source": "COD-PS",
        "url": "https://knowledge.base.unocha.org/wiki/spaces/imtoolbox/pages/2491252951/COD-PS%2BStandards%2Band%2BProcess",
        "note": (
            "COD-PS follows a best-available-data principle. Reference year, "
            "p-code consistency, sex/age breakdown, and admin-level coherence are "
            "core quality dimensions."
        ),
    },
    {
        "source": "HRP",
        "url": "https://knowledge.base.unocha.org/wiki/spaces/imtoolbox/pages/42046871/Humanitarian%2BResponse%2BPlan%2BHRP",
        "note": (
            "HRPs build on HNO evidence and communicate strategic objectives, "
            "cluster plans, and resource mobilization needs."
        ),
    },
    {
        "source": "CBPF",
        "url": "https://cbpf.data.unocha.org/",
        "note": (
            "The CBPF Data Hub provides contribution and allocation views for "
            "country-based pooled funds, including allocations, sectors, targeted "
            "and reached people, and gender/age marker views."
        ),
    },
]

NUMERIC_KEYWORDS = (
    "amount",
    "allocation",
    "budget",
    "contribution",
    "exchange",
    "funding",
    "in need",
    "people",
    "percent",
    "percentage",
    "population",
    "project",
    "reached",
    "requirements",
    "targeted",
)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%Y %I:%M:%S %p",
    "%Y/%m/%d",
)


def csv_paths() -> list[Path]:
    return sorted(DATA_DIR.glob("*.csv"))


def source_family(path: Path) -> str:
    name = path.name
    if name.startswith("hpc_hno_"):
        return "HNO"
    if name == "humanitarian-response-plans.csv":
        return "HRP"
    if name.startswith("cod_population_"):
        return "COD-PS"
    if name.startswith("fts_"):
        return "FTS"
    return "CBPF"


def is_hxl_row(row: list[str]) -> bool:
    non_empty = [cell.strip() for cell in row if cell.strip()]
    if not non_empty:
        return False
    hxl_cells = sum(1 for cell in non_empty if cell.startswith("#"))
    return hxl_cells / len(non_empty) >= 0.7


def normalized_row(row: list[str], width: int) -> list[str]:
    if len(row) < width:
        return row + [""] * (width - len(row))
    if len(row) > width:
        return row[: width - 1] + [" | ".join(row[width - 1 :])]
    return row


def iter_records(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        hxl_tags = None
        for index, raw_row in enumerate(reader):
            row = normalized_row(raw_row, len(header))
            if index == 0 and is_hxl_row(row):
                hxl_tags = row
                continue
            yield header, hxl_tags, dict(zip(header, row)), row


def read_header(path: Path) -> tuple[list[str], list[str] | None]:
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        try:
            first = normalized_row(next(reader), len(header))
        except StopIteration:
            return header, None
        return header, first if is_hxl_row(first) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_value(value: object) -> str:
    return "" if value is None else str(value).strip()


def parse_number(value: object) -> float | None:
    text = clean_value(value)
    if not text:
        return None
    text = text.replace(",", "").replace("$", "").replace("%", "")
    if text in {"-", "nan", "NaN", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value: object) -> datetime | None:
    text = clean_value(value)
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def is_iso3(value: object) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}", clean_value(value)))


def split_codes(value: object) -> list[str]:
    text = clean_value(value)
    if not text or text.startswith("#"):
        return []
    parts = re.split(r"[|,;]", text)
    return [part.strip() for part in parts if part.strip() and part.strip() != "|"]


def normalize_label(value: object) -> str:
    text = clean_value(value).lower()
    text = re.sub(r"^\(closed\)\s*", "", text)
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def format_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if abs(value) >= 1:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def pct(part: int | float, whole: int | float) -> float:
    return 0.0 if not whole else round(100 * part / whole, 4)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(prob: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = prob * (len(ordered) - 1)
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return ordered[lower]
        weight = pos - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "p01": q(0.01),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p99": q(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def should_review_numeric(column: str) -> bool:
    lower = column.lower()
    return any(keyword in lower for keyword in NUMERIC_KEYWORDS)


def infer_type(stats: dict[str, object]) -> str:
    non_empty = int(stats["non_empty_count"])
    if non_empty == 0:
        return "empty"
    numeric_count = int(stats["numeric_count"])
    date_count = int(stats["date_count"])
    iso3_count = int(stats["iso3_count"])
    integer_count = int(stats["integer_count"])
    if numeric_count / non_empty >= 0.95:
        return "integer" if integer_count == numeric_count else "decimal"
    if date_count / non_empty >= 0.95:
        return "date"
    if iso3_count / non_empty >= 0.95:
        return "iso3_code"
    return "string"


def init_column_stats() -> dict[str, object]:
    return {
        "blank_count": 0,
        "non_empty_count": 0,
        "unique_values": set(),
        "unique_capped": False,
        "top_values": Counter(),
        "top_other_count": 0,
        "numeric_count": 0,
        "integer_count": 0,
        "numeric_values": [],
        "date_count": 0,
        "iso3_count": 0,
        "examples": [],
    }


def add_quality_flag(
    flags: list[dict[str, object]],
    file_name: str,
    check_type: str,
    column: str,
    status: str,
    affected_count: int,
    total_count: int,
    detail: str,
) -> None:
    flags.append(
        {
            "file": file_name,
            "check_type": check_type,
            "column": column,
            "status": status,
            "affected_count": affected_count,
            "total_count": total_count,
            "affected_pct": pct(affected_count, total_count),
            "detail": detail,
        }
    )


def profile_file(path: Path) -> dict[str, object]:
    header, hxl_tags = read_header(path)
    col_stats = {column: init_column_stats() for column in header}
    row_count = 0
    seen_rows: set[str] = set()
    duplicate_rows = 0
    year_values: set[str] = set()
    country_codes: set[str] = set()
    malformed_rows = 0

    hno_year_match = re.search(r"hpc_hno_(\d{4})\.csv$", path.name)
    if hno_year_match:
        year_values.add(hno_year_match.group(1))

    logical_checks = Counter()
    negative_counts = Counter()
    date_parse_failures = Counter()
    iso_parse_failures = Counter()
    percent_oob = Counter()
    cod_required_missing = Counter()
    cod_population_non_integer = 0

    for _, _, record, row in iter_records(path):
        row_count += 1
        if len(row) != len(header):
            malformed_rows += 1
        row_hash = hashlib.sha1("\x1f".join(row).encode("utf-8", errors="replace")).hexdigest()
        if row_hash in seen_rows:
            duplicate_rows += 1
        else:
            seen_rows.add(row_hash)

        for column in header:
            value = clean_value(record.get(column))
            stats = col_stats[column]
            if not value:
                stats["blank_count"] = int(stats["blank_count"]) + 1
                continue

            stats["non_empty_count"] = int(stats["non_empty_count"]) + 1
            if not stats["unique_capped"]:
                unique_values = stats["unique_values"]
                assert isinstance(unique_values, set)
                unique_values.add(value)
                if len(unique_values) > UNIQUE_CAP:
                    stats["unique_capped"] = True
                    stats["unique_values"] = set()
            top_values = stats["top_values"]
            assert isinstance(top_values, Counter)
            if value in top_values or len(top_values) < TOP_VALUE_CAP:
                top_values[value] += 1
            else:
                stats["top_other_count"] = int(stats["top_other_count"]) + 1
            examples = stats["examples"]
            assert isinstance(examples, list)
            if len(examples) < 5 and value not in examples:
                examples.append(value)

            number = parse_number(value)
            if number is not None:
                stats["numeric_count"] = int(stats["numeric_count"]) + 1
                numeric_values = stats["numeric_values"]
                assert isinstance(numeric_values, list)
                numeric_values.append(number)
                if float(number).is_integer():
                    stats["integer_count"] = int(stats["integer_count"]) + 1
                if should_review_numeric(column) and number < 0:
                    negative_counts[column] += 1

            if parse_date(value) is not None:
                stats["date_count"] = int(stats["date_count"]) + 1
            if is_iso3(value):
                stats["iso3_count"] = int(stats["iso3_count"]) + 1

        for year_column in ("Year", "year", "years", "budgetYear", "Reference_year"):
            if year_column in record:
                for candidate in split_codes(record.get(year_column)):
                    if re.fullmatch(r"\d{4}", candidate):
                        year_values.add(candidate)
                value = clean_value(record.get(year_column))
                if re.fullmatch(r"\d{4}", value):
                    year_values.add(value)
        for date_column in ("startDate", "date", "AllocationHCLastProjectApprovalDate"):
            if date_column in record:
                parsed = parse_date(record.get(date_column))
                if parsed:
                    year_values.add(str(parsed.year))

        for country_column in ("Country ISO3", "ISO3", "countryCode"):
            if country_column in record and is_iso3(record[country_column]):
                country_codes.add(clean_value(record[country_column]))
        for country_column in ("locations", "destLocations", "srcLocations"):
            if country_column in record:
                country_codes.update(code for code in split_codes(record[country_column]) if is_iso3(code))

        if {"Targeted", "In Need"}.issubset(record):
            targeted = parse_number(record["Targeted"])
            in_need = parse_number(record["In Need"])
            if targeted is not None and in_need is not None and targeted > in_need:
                logical_checks["targeted_gt_in_need"] += 1
        if {"Reached", "Targeted"}.issubset(record):
            reached = parse_number(record["Reached"])
            targeted = parse_number(record["Targeted"])
            if reached is not None and targeted is not None and reached > targeted:
                logical_checks["reached_gt_targeted"] += 1
        if {"funding", "requirements"}.issubset(record):
            funding = parse_number(record["funding"])
            requirements = parse_number(record["requirements"])
            if funding is not None and requirements is not None and requirements > 0 and funding > requirements:
                logical_checks["funding_gt_requirements"] += 1
        if "Population" in record and source_family(path) == "COD-PS":
            population = parse_number(record["Population"])
            if population is not None and not float(population).is_integer():
                cod_population_non_integer += 1

        for column in header:
            value = clean_value(record.get(column))
            lower = column.lower()
            if value and "date" in lower and parse_date(value) is None:
                date_parse_failures[column] += 1
            if column in {"Country ISO3", "ISO3", "countryCode"} and value and not is_iso3(value):
                iso_parse_failures[column] += 1
            if value and ("percent" in lower or "percentage" in lower):
                number = parse_number(value)
                if number is not None and (number < 0 or number > 100):
                    percent_oob[column] += 1

        if source_family(path) == "COD-PS":
            level_match = re.search(r"admin(\d)", path.name)
            admin_level = int(level_match.group(1)) if level_match else 0
            required = ["ISO3", "Country", "Population_group", "Gender", "Age_range", "Population", "Reference_year"]
            if admin_level >= 1:
                required.extend(["ADM1_PCODE", "ADM1_NAME"])
            if admin_level >= 2:
                required.extend(["ADM2_PCODE", "ADM2_NAME"])
            if admin_level >= 3:
                required.extend(["ADM3_PCODE", "ADM3_NAME"])
            if admin_level >= 4:
                required.extend(["ADM4_PCODE", "ADM4_NAME"])
            for column in required:
                if column in record and not clean_value(record[column]):
                    cod_required_missing[column] += 1

    column_rows = []
    missing_rows = []
    distribution_rows = []
    numeric_rows = []
    quality_flags = []

    for column in header:
        stats = col_stats[column]
        non_empty = int(stats["non_empty_count"])
        blank = int(stats["blank_count"])
        unique_values = stats["unique_values"]
        assert isinstance(unique_values, set)
        unique_count = f">={UNIQUE_CAP}" if stats["unique_capped"] else str(len(unique_values))
        inferred = infer_type(stats)
        examples = stats["examples"]
        assert isinstance(examples, list)
        column_rows.append(
            {
                "file": path.name,
                "source_family": source_family(path),
                "column": column,
                "inferred_type": inferred,
                "non_empty_count": non_empty,
                "blank_count": blank,
                "blank_pct": pct(blank, row_count),
                "unique_count": unique_count,
                "unique_count_capped": bool(stats["unique_capped"]),
                "examples": " | ".join(examples),
            }
        )
        missing_rows.append(
            {
                "file": path.name,
                "source_family": source_family(path),
                "column": column,
                "row_count": row_count,
                "blank_count": blank,
                "blank_pct": pct(blank, row_count),
                "non_empty_count": non_empty,
            }
        )

        top_values = stats["top_values"]
        assert isinstance(top_values, Counter)
        for rank, (value, count) in enumerate(top_values.most_common(TOP_VALUES_PER_COLUMN), start=1):
            distribution_rows.append(
                {
                    "file": path.name,
                    "column": column,
                    "distribution_type": "top_value",
                    "metric": str(rank),
                    "value": value,
                    "count": count,
                    "pct_non_empty": pct(count, non_empty),
                }
            )

        numeric_values = stats["numeric_values"]
        assert isinstance(numeric_values, list)
        if numeric_values and (inferred in {"integer", "decimal"} or should_review_numeric(column)):
            q = quantiles(numeric_values)
            numeric_row = {
                "file": path.name,
                "source_family": source_family(path),
                "column": column,
                "numeric_count": len(numeric_values),
                "non_empty_count": non_empty,
                "numeric_parse_pct": pct(len(numeric_values), non_empty),
            }
            for metric in ("min", "p01", "p25", "p50", "p75", "p99", "max", "mean"):
                numeric_row[metric] = format_number(q.get(metric))
                distribution_rows.append(
                    {
                        "file": path.name,
                        "column": column,
                        "distribution_type": "numeric_summary",
                        "metric": metric,
                        "value": format_number(q.get(metric)),
                        "count": len(numeric_values),
                        "pct_non_empty": pct(len(numeric_values), non_empty),
                    }
                )
            numeric_rows.append(numeric_row)

        if row_count and blank / row_count >= 0.5:
            add_quality_flag(
                quality_flags,
                path.name,
                "missingness",
                column,
                "warning",
                blank,
                row_count,
                "Column is blank in at least half of logical rows. Review whether this is structural or data loss.",
            )

    add_quality_flag(
        quality_flags,
        path.name,
        "row_structure",
        "",
        "ok" if malformed_rows == 0 else "warning",
        malformed_rows,
        row_count,
        "Rows have the same number of parsed fields as the header." if malformed_rows == 0 else "Some rows had unexpected width.",
    )
    add_quality_flag(
        quality_flags,
        path.name,
        "duplicate_rows",
        "",
        "ok" if duplicate_rows == 0 else "warning",
        duplicate_rows,
        row_count,
        "Exact duplicate logical rows.",
    )

    for column, count in negative_counts.items():
        add_quality_flag(quality_flags, path.name, "negative_numeric_value", column, "needs_domain_review", count, row_count, "Negative values in a measure column.")
    for column, count in date_parse_failures.items():
        add_quality_flag(quality_flags, path.name, "date_parse", column, "warning", count, row_count, "Non-empty date-like values that do not parse with expected formats.")
    for column, count in iso_parse_failures.items():
        add_quality_flag(quality_flags, path.name, "iso3_parse", column, "warning", count, row_count, "Non-empty country-code values that are not ISO3-shaped.")
    for column, count in percent_oob.items():
        add_quality_flag(quality_flags, path.name, "percentage_range", column, "needs_domain_review", count, row_count, "Percentage-like values outside 0..100.")
    for check_name, count in logical_checks.items():
        add_quality_flag(quality_flags, path.name, check_name, "", "needs_domain_review", count, row_count, "Logical comparison flagged by the profiling rules.")
    for column, count in cod_required_missing.items():
        add_quality_flag(quality_flags, path.name, "cod_required_missing", column, "warning", count, row_count, "COD-PS required/expected analysis field is blank.")
    if cod_population_non_integer:
        add_quality_flag(quality_flags, path.name, "cod_population_integer", "Population", "warning", cod_population_non_integer, row_count, "COD-PS population values should be integer-like.")

    file_hash = sha256_file(path)
    file_row = {
        "file": path.name,
        "source_family": source_family(path),
        "source_url": SOURCE_URLS[source_family(path)],
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
        "sha256": file_hash,
        "logical_rows": row_count,
        "columns": len(header),
        "hxl_metadata_row": bool(hxl_tags),
        "years": " | ".join(sorted(year_values)),
        "country_code_count": len(country_codes),
        "country_code_examples": " | ".join(sorted(country_codes)[:20]),
        "duplicate_logical_rows": duplicate_rows,
        "column_names": " | ".join(header),
    }
    if hxl_tags:
        file_row["hxl_tags"] = " | ".join(f"{column}={tag}" for column, tag in zip(header, hxl_tags))
    else:
        file_row["hxl_tags"] = ""

    return {
        "file": file_row,
        "columns": column_rows,
        "missing": missing_rows,
        "distributions": distribution_rows,
        "numeric": numeric_rows,
        "quality_flags": quality_flags,
    }


def collect_values(file_names: Iterable[str], column: str, split: bool = False, normalizer=None) -> set[str]:
    values: set[str] = set()
    for file_name in file_names:
        path = DATA_DIR / file_name
        if not path.exists():
            continue
        for _, _, record, _ in iter_records(path):
            if column not in record:
                continue
            raw_values = split_codes(record[column]) if split else [clean_value(record[column])]
            for value in raw_values:
                if not value or value.startswith("#"):
                    continue
                values.add(normalizer(value) if normalizer else value)
    return values


def collect_any_columns(file_names: Iterable[str], columns: Iterable[str], split: bool = False, normalizer=None) -> set[str]:
    result: set[str] = set()
    for column in columns:
        result.update(collect_values(file_names, column, split=split, normalizer=normalizer))
    return result


def relationship_row(
    relationship: str,
    source_label: str,
    target_label: str,
    source_values: set[str],
    target_values: set[str],
    notes: str,
) -> dict[str, object]:
    source = {value for value in source_values if value}
    target = {value for value in target_values if value}
    matched = source & target
    unmatched = source - target
    return {
        "relationship": relationship,
        "source": source_label,
        "target": target_label,
        "source_count": len(source),
        "target_count": len(target),
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "coverage_pct": pct(len(matched), len(source)),
        "sample_unmatched": " | ".join(sorted(unmatched)[:25]),
        "notes": notes,
    }


def build_cross_reference_coverage() -> list[dict[str, object]]:
    hno_files = ["hpc_hno_2024.csv", "hpc_hno_2025.csv", "hpc_hno_2026.csv"]
    cod_files = [f"cod_population_admin{i}.csv" for i in range(5)]
    fts_requirement_files = [
        "fts_requirements_funding_global.csv",
        "fts_requirements_funding_cluster_global.csv",
        "fts_requirements_funding_globalcluster_global.csv",
        "fts_requirements_funding_covid_global.csv",
    ]
    fts_flow_files = [
        "fts_incoming_funding_global.csv",
        "fts_internal_funding_global.csv",
        "fts_outgoing_funding_global.csv",
    ]
    cbpf_files = [
        "AffectedPersons__20260418_072029_UTC.csv",
        "AllocationFlow__20260418_071755_UTC.csv",
        "AllocationsByOrgType__20260418_071803_UTC.csv",
        "Allocations__20260418_071733_UTC.csv",
        "CBPFcontributions__20260418_071813_UTC.csv",
        "ContributionTrends__20260418_071819_UTC.csv",
        "ContributionsFlow__20260418_071825_UTC.csv",
        "GenderAgeMarker__20260418_072032_UTC.csv",
        "SectorsOverview__20260418_071833_UTC.csv",
    ]

    hno_countries = collect_values(hno_files, "Country ISO3")
    cod_countries = collect_values(cod_files, "ISO3")
    fts_countries = collect_values(fts_requirement_files, "countryCode")
    fts_location_codes = collect_any_columns(fts_flow_files, ["destLocations", "srcLocations"], split=True)
    hrp_locations = collect_values(["humanitarian-response-plans.csv"], "locations", split=True)

    rows = [
        relationship_row("hno_country_to_cod_iso3", "HNO Country ISO3", "COD ISO3", hno_countries, cod_countries, "Country-level needs can be checked against COD population availability."),
        relationship_row("hno_country_to_fts_country_code", "HNO Country ISO3", "FTS countryCode", hno_countries, fts_countries, "Country-level needs can be compared to appeal funding where codes overlap."),
        relationship_row("hno_country_to_hrp_locations", "HNO Country ISO3", "HRP locations", hno_countries, hrp_locations, "HNO countries can be linked to HRP plans by ISO3 location lists."),
        relationship_row("hrp_locations_to_cod_iso3", "HRP locations", "COD ISO3", hrp_locations, cod_countries, "HRP plan countries can be checked against COD population availability."),
        relationship_row("fts_flow_locations_to_hrp_locations", "FTS flow locations", "HRP locations", fts_location_codes, hrp_locations, "FTS flow rows may contain multi-country location lists."),
    ]

    for level in (1, 2, 3):
        hno_pcodes = collect_values(hno_files, f"Admin {level} PCode")
        cod_pcodes = collect_any_columns(
            [f"cod_population_admin{i}.csv" for i in range(level, 5)],
            [f"ADM{level}_PCODE"],
        )
        rows.append(
            relationship_row(
                f"hno_admin{level}_pcode_to_cod_admin{level}_pcode",
                f"HNO Admin {level} PCode",
                f"COD ADM{level}_PCODE",
                hno_pcodes,
                cod_pcodes,
                "Admin p-code matching is strongest for 2024/2025 HNO because 2026 has no admin fields.",
            )
        )

    hrp_codes = collect_values(["humanitarian-response-plans.csv"], "code")
    hrp_ids = collect_values(["humanitarian-response-plans.csv"], "internalId")
    fts_codes = collect_any_columns(fts_requirement_files, ["code"]) | collect_values(fts_flow_files, "destPlanCode")
    fts_ids = collect_any_columns(fts_requirement_files, ["id"]) | collect_values(fts_flow_files, "destPlanId")
    rows.extend(
        [
            relationship_row("hrp_code_to_fts_plan_code", "HRP code", "FTS code/destPlanCode", hrp_codes, fts_codes, "Best plan-level join candidate by public plan/appeal code."),
            relationship_row("hrp_internal_id_to_fts_plan_id", "HRP internalId", "FTS id/destPlanId", hrp_ids, fts_ids, "Best plan-level join candidate by internal numeric identifier."),
        ]
    )

    hno_sector_labels = set()
    for file_name in hno_files:
        for _, _, record, _ in iter_records(DATA_DIR / file_name):
            cluster = clean_value(record.get("Cluster"))
            description = clean_value(record.get("Description"))
            if cluster and cluster != "ALL":
                hno_sector_labels.add(normalize_label(description or cluster))
    fts_sectors = collect_any_columns(fts_requirement_files, ["cluster"], normalizer=normalize_label)
    cbpf_sectors = collect_values(["SectorsOverview__20260418_071833_UTC.csv"], "Cluster", normalizer=normalize_label)
    rows.extend(
        [
            relationship_row("hno_sector_label_to_fts_cluster", "HNO sector descriptions", "FTS cluster names", hno_sector_labels, fts_sectors, "Uses normalized labels, not authoritative sector IDs."),
            relationship_row("hno_sector_label_to_cbpf_cluster", "HNO sector descriptions", "CBPF sector names", hno_sector_labels, cbpf_sectors, "Uses normalized labels, not authoritative sector IDs."),
        ]
    )

    fund_columns = ["CBPF Name", "Fund", "PooledFund", "PooledFundName"]
    fund_names = collect_any_columns(cbpf_files, fund_columns)
    regional_or_closed = {
        fund
        for fund in fund_names
        if any(token in fund.lower() for token in ("regional", "rhpf", "rhp", "envelope", "(closed)"))
    }
    rows.append(
        {
            "relationship": "cbpf_fund_names_need_country_lookup",
            "source": "CBPF fund name columns",
            "target": "country ISO3",
            "source_count": len(fund_names),
            "target_count": "",
            "matched_count": "",
            "unmatched_count": len(fund_names),
            "coverage_pct": "",
            "sample_unmatched": " | ".join(sorted(regional_or_closed)[:25]),
            "notes": f"No direct ISO3 key is present in the CBPF exports. {len(regional_or_closed)} fund names look regional or closed and need a fund-country lookup before country joins.",
        }
    )

    return rows


def build_dataset_catalog(file_inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    by_file = {row["file"]: row for row in file_inventory}

    def years(*names: str) -> str:
        observed: set[str] = set()
        for name in names:
            observed.update(str(by_file[name]["years"]).split(" | ") if by_file[name]["years"] else [])
        return " | ".join(sorted(value for value in observed if value))

    return [
        {
            "source_family": "HNO",
            "files": "hpc_hno_2024.csv | hpc_hno_2025.csv | hpc_hno_2026.csv",
            "unit_of_observation": "Country/admin-sector-category caseload row; wide measures for population statuses.",
            "time_coverage": years("hpc_hno_2024.csv", "hpc_hno_2025.csv", "hpc_hno_2026.csv"),
            "geography": "Country ISO3; admin p-codes in 2024/2025 only.",
            "core_measures": "Population, In Need, Targeted, Affected, Reached.",
            "identifiers": "Country ISO3, admin p-codes, Cluster, Description, Category.",
            "known_caveats": "Do not sum people across sectors/statuses; 2026 schema has no admin columns; category is freeform/blank-heavy.",
        },
        {
            "source_family": "HRP",
            "files": "humanitarian-response-plans.csv",
            "unit_of_observation": "Humanitarian response plan or appeal version.",
            "time_coverage": years("humanitarian-response-plans.csv"),
            "geography": "One or more ISO3 codes in locations.",
            "core_measures": "Original and revised requirements in USD.",
            "identifiers": "code, internalId, locations, years, categories.",
            "known_caveats": "Contains multi-country plans and a HXL metadata row; not a realized funding table.",
        },
        {
            "source_family": "COD-PS",
            "files": "cod_population_admin0.csv through cod_population_admin4.csv",
            "unit_of_observation": "Population group, sex, age, and administrative unit row.",
            "time_coverage": years(*(f"cod_population_admin{i}.csv" for i in range(5))),
            "geography": "ISO3 plus ADM1-ADM4 p-codes where available.",
            "core_measures": "Population by reference year, sex, age range, and population group.",
            "identifiers": "ISO3, country, admin p-codes, age/sex/population group.",
            "known_caveats": "Reference years vary by country/admin level; admin coverage becomes narrower at deeper levels.",
        },
        {
            "source_family": "FTS",
            "files": "fts_requirements_* and fts_*_funding_global.csv",
            "unit_of_observation": "Appeal requirement/funding row, cluster row, or reported funding flow.",
            "time_coverage": years(
                "fts_requirements_funding_global.csv",
                "fts_requirements_funding_cluster_global.csv",
                "fts_requirements_funding_globalcluster_global.csv",
                "fts_requirements_funding_covid_global.csv",
                "fts_incoming_funding_global.csv",
                "fts_internal_funding_global.csv",
                "fts_outgoing_funding_global.csv",
            ),
            "geography": "countryCode and multi-code source/destination location fields.",
            "core_measures": "Requirements, funding, percent funded, flow amount USD, original amount/currency.",
            "identifiers": "id, code, countryCode, clusterCode, flow id/refCode, destPlanCode/destPlanId.",
            "known_caveats": "Some funding is not linked to a plan; flow location fields can represent multiple countries.",
        },
        {
            "source_family": "CBPF",
            "files": "Allocation, contribution, sector, affected persons, and marker exports.",
            "unit_of_observation": "Fund-year, donor-fund-year, allocation, organization type, sector, or marker summary row.",
            "time_coverage": years(
                "AffectedPersons__20260418_072029_UTC.csv",
                "AllocationFlow__20260418_071755_UTC.csv",
                "AllocationsByOrgType__20260418_071803_UTC.csv",
                "Allocations__20260418_071733_UTC.csv",
                "CBPFcontributions__20260418_071813_UTC.csv",
                "ContributionTrends__20260418_071819_UTC.csv",
                "ContributionsFlow__20260418_071825_UTC.csv",
                "GenderAgeMarker__20260418_072032_UTC.csv",
                "SectorsOverview__20260418_071833_UTC.csv",
            ),
            "geography": "Fund names, not direct ISO3 country keys.",
            "core_measures": "Allocations, contributions, direct/net funding, targeted/reached people, beneficiaries, project counts.",
            "identifiers": "Year, fund/fund name, donor, partner type, allocation type, cluster, marker code.",
            "known_caveats": "Needs fund-country mapping for country joins; one contribution file is an exact duplicate of another.",
        },
    ]


def build_can_cannot_say_matrix() -> list[dict[str, str]]:
    return [
        {
            "topic": "HNO needs and response caseloads",
            "can_say": "For available countries/sectors/years, describe reported population, people in need, targeted, affected, and reached values.",
            "cannot_say": "Cannot infer unique people across sectors/statuses or causal effect of assistance.",
            "primary_datasets": "hpc_hno_2024.csv | hpc_hno_2025.csv | hpc_hno_2026.csv",
            "quality_notes": "HNO 2024/2025 include admin fields; 2026 is country-sector only in this snapshot.",
        },
        {
            "topic": "HRP requirements",
            "can_say": "Describe original and revised USD requirements by response plan, year, category, and location list.",
            "cannot_say": "Cannot say which requirements were funded unless joined to FTS; cannot treat regional plans as single-country plans without parsing locations.",
            "primary_datasets": "humanitarian-response-plans.csv",
            "quality_notes": "Includes HXL metadata row and multi-country location lists.",
        },
        {
            "topic": "Funding against requirements",
            "can_say": "Compare reported FTS requirements, funding, and percent funded by appeal/country/year/cluster where identifiers exist.",
            "cannot_say": "Cannot guarantee complete real-time funding; cannot compare unlinked flows to HRP requirements without plan linkage.",
            "primary_datasets": "fts_requirements_* | fts_*_funding_global.csv",
            "quality_notes": "Some rows are 'Not specified' with blank plan identifiers; flow locations can contain multiple countries.",
        },
        {
            "topic": "Population baselines",
            "can_say": "Use COD-PS as population baselines by country/admin level, sex, age range, population group, and reference year.",
            "cannot_say": "Cannot assume every value is current-year or that all countries have the same admin depth.",
            "primary_datasets": "cod_population_admin0.csv through cod_population_admin4.csv",
            "quality_notes": "Reference years and admin depth vary; deeper admin files cover fewer countries.",
        },
        {
            "topic": "CBPF pooled fund allocations and contributions",
            "can_say": "Describe allocations, contribution trends, sectors, partner types, targeted/reached people, and gender/age marker summaries by fund/year.",
            "cannot_say": "Cannot reliably join to country ISO3 or HRP without a fund-country mapping; cannot use duplicated contribution exports as independent evidence.",
            "primary_datasets": "CBPF exports",
            "quality_notes": "Fund names include regional and closed funds; contribution flow and contribution exports are exact duplicates.",
        },
        {
            "topic": "Cross-dataset joins",
            "can_say": "Join strongest on ISO3, admin p-codes, HRP/FTS plan codes or IDs, and normalized sector labels for exploratory checks.",
            "cannot_say": "Cannot treat name-only joins as authoritative; cannot resolve CBPF country coverage from fund names alone.",
            "primary_datasets": "All source families",
            "quality_notes": "The cross-reference coverage table lists matched/unmatched values and sample gaps.",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    subset = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in subset:
        values = []
        for column in columns:
            value = str(row.get(column, ""))
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_family_counts(file_inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in file_inventory:
        family = str(row["source_family"])
        group = grouped.setdefault(
            family,
            {"source_family": family, "files": 0, "logical_rows": 0, "size_mb": 0.0, "countries_observed": 0},
        )
        group["files"] = int(group["files"]) + 1
        group["logical_rows"] = int(group["logical_rows"]) + int(row["logical_rows"])
        group["size_mb"] = float(group["size_mb"]) + float(row["size_mb"])
        group["countries_observed"] = max(int(group["countries_observed"]), int(row["country_code_count"]))
    for group in grouped.values():
        group["size_mb"] = round(float(group["size_mb"]), 2)
    return sorted(grouped.values(), key=lambda row: str(row["source_family"]))


def build_duplicate_checks(file_inventory: list[dict[str, object]], profile_results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in file_inventory:
        by_hash[str(row["sha256"])].append(str(row["file"]))
    for digest, files in sorted(by_hash.items()):
        if len(files) > 1:
            rows.append(
                {
                    "check_type": "exact_file_hash_duplicate",
                    "scope": "file",
                    "item_a": files[0],
                    "item_b": " | ".join(files[1:]),
                    "status": "warning",
                    "detail": f"{len(files)} files share sha256 {digest}.",
                }
            )
    for result in profile_results:
        file_row = result["file"]
        duplicates = int(file_row["duplicate_logical_rows"])
        rows.append(
            {
                "check_type": "exact_row_duplicate",
                "scope": "logical rows",
                "item_a": file_row["file"],
                "item_b": "",
                "status": "ok" if duplicates == 0 else "warning",
                "detail": f"{duplicates} duplicate rows among {file_row['logical_rows']} logical rows.",
            }
        )
    return rows


def build_report(
    file_inventory: list[dict[str, object]],
    dataset_catalog: list[dict[str, object]],
    missingness_profile: list[dict[str, object]],
    duplicate_checks: list[dict[str, object]],
    cross_reference: list[dict[str, object]],
    can_cannot: list[dict[str, object]],
    quality_flags: list[dict[str, object]],
) -> str:
    total_rows = sum(int(row["logical_rows"]) for row in file_inventory)
    total_mb = sum(float(row["size_mb"]) for row in file_inventory)
    family_counts = summarize_family_counts(file_inventory)
    high_missing = [
        row
        for row in missingness_profile
        if float(row["blank_pct"]) >= 90 and int(row["row_count"]) > 0
    ]
    high_missing = sorted(high_missing, key=lambda row: (-float(row["blank_pct"]), row["file"], row["column"]))[:20]
    duplicate_file_rows = [row for row in duplicate_checks if row["check_type"] == "exact_file_hash_duplicate"]
    top_quality = [
        row
        for row in quality_flags
        if row["status"] in {"warning", "needs_domain_review"} and int(row["affected_count"]) > 0
    ]
    top_quality = sorted(top_quality, key=lambda row: (-int(row["affected_count"]), row["file"], row["check_type"]))[:20]

    report = [
        "# Initial Data Quality Report",
        "",
        "Generated from the checked-in CSV snapshot in `initial_data`. Source documentation was used only to interpret fields, caveats, and expected joins; the workflow does not re-download live source data.",
        "",
        "## Executive Summary",
        "",
        f"- Profiled {len(file_inventory)} CSV files, {total_rows:,} logical rows, and {total_mb:.2f} MB of local data.",
        "- The collection covers five source families: HNO/HPC needs, HRP plan requirements, COD-PS population, FTS requirements/funding, and CBPF pooled fund exports.",
        "- HNO 2024/2025 and HRP include a second-row HXL metadata row; the profiling excludes it from logical row counts.",
        "- `CBPFcontributions__20260418_071813_UTC.csv` and `ContributionsFlow__20260418_071825_UTC.csv` are exact file duplicates and should not be treated as independent evidence.",
        "- The strongest cross-dataset keys are ISO3 country codes, admin p-codes, and HRP/FTS plan codes/IDs. CBPF exports need a separate fund-country lookup before country-level joins.",
        "",
        "## Source Family Inventory",
        "",
        markdown_table(family_counts, ["source_family", "files", "logical_rows", "size_mb", "countries_observed"]),
        "",
        "## Which Dataset Contains What",
        "",
        markdown_table(
            dataset_catalog,
            [
                "source_family",
                "files",
                "unit_of_observation",
                "time_coverage",
                "geography",
                "core_measures",
                "identifiers",
                "known_caveats",
            ],
        ),
        "",
        "## Data Quality Highlights",
        "",
        "- Completeness is intentionally uneven: COD admin depth narrows from 139 countries at admin0 to one country at admin4, while HNO 2026 is country-sector only in this snapshot.",
        "- High-blank fields often represent structural absence rather than accidental loss, for example lower admin columns in shallower COD files and project/emergency fields in aggregate FTS flow exports.",
        "- Several logical checks are domain-review flags, not automatic errors. Examples include targeted values exceeding in-need values, reached values exceeding targeted values, and funding exceeding requirements.",
        "- FTS location fields and HRP locations can contain multiple country codes in one row; they must be exploded before country-level analysis.",
        "- Source guidance says HNO people-in-need values should not be summed across sectors or statuses because people may appear in multiple groups.",
        "",
        "### Exact Duplicate Checks",
        "",
        markdown_table(duplicate_file_rows, ["check_type", "item_a", "item_b", "status", "detail"]) if duplicate_file_rows else "No exact file duplicates detected.",
        "",
        "### Largest Quality Flags",
        "",
        markdown_table(top_quality, ["file", "check_type", "column", "status", "affected_count", "total_count", "affected_pct", "detail"], limit=20),
        "",
        "### Columns With At Least 90% Missing Values",
        "",
        markdown_table(high_missing, ["file", "column", "row_count", "blank_count", "blank_pct"], limit=20),
        "",
        "## Cross-Dataset Connections",
        "",
        markdown_table(cross_reference, ["relationship", "source", "target", "source_count", "target_count", "matched_count", "unmatched_count", "coverage_pct", "sample_unmatched", "notes"]),
        "",
        "## What We Can And Cannot Say",
        "",
        markdown_table(can_cannot, ["topic", "can_say", "cannot_say", "primary_datasets", "quality_notes"]),
        "",
        "## Reasonable Questions To Ask Next",
        "",
        "- Which countries have HNO needs, HRP requirements, FTS funding, and COD population coverage in the same year?",
        "- Where do HNO country/admin p-codes fail to match COD-PS p-codes, and are mismatches due to old boundaries, missing COD levels, or source inconsistency?",
        "- How do HRP revised requirements compare with FTS funding and percent funded by country, year, and cluster?",
        "- Which sectors have comparable labels across HNO, FTS, and CBPF, and where do sector naming conventions block analysis?",
        "- Which CBPF funds can be mapped cleanly to one country, and which regional/closed funds need manual treatment?",
        "- Which population denominators are current enough to support per-capita or share-of-population indicators?",
        "",
        "## Appendix Tables",
        "",
        "- `reports/tables/file_inventory.csv`",
        "- `reports/tables/dataset_catalog.csv`",
        "- `reports/tables/column_profile.csv`",
        "- `reports/tables/missingness_profile.csv`",
        "- `reports/tables/value_distributions.csv`",
        "- `reports/tables/numeric_profile.csv`",
        "- `reports/tables/duplicate_checks.csv`",
        "- `reports/tables/quality_flags.csv`",
        "- `reports/tables/cross_reference_coverage.csv`",
        "- `reports/tables/can_cannot_say_matrix.csv`",
        "- `reports/tables/validation_results.csv`",
        "",
        "## Source Notes",
        "",
    ]
    for note in SOURCE_NOTES:
        report.append(f"- [{note['source']}]({note['url']}): {note['note']}")
    report.append("")
    return "\n".join(report)


def run_validations(
    file_inventory: list[dict[str, object]],
    duplicate_checks: list[dict[str, object]],
    cross_reference: list[dict[str, object]],
    before_hashes: dict[str, str],
) -> list[dict[str, object]]:
    rows = []

    def add(name: str, status: str, detail: str) -> None:
        rows.append({"validation": name, "status": status, "detail": detail})

    actual_files = {row["file"] for row in file_inventory}
    expected_files = set(EXPECTED_LOGICAL_ROWS)
    add(
        "expected_file_count",
        "pass" if actual_files == expected_files else "fail",
        f"Expected {len(expected_files)} files and found {len(actual_files)}.",
    )
    by_file = {row["file"]: row for row in file_inventory}
    bad_counts = [
        f"{file}: expected {expected}, found {by_file.get(file, {}).get('logical_rows')}"
        for file, expected in EXPECTED_LOGICAL_ROWS.items()
        if str(by_file.get(file, {}).get("logical_rows")) != str(expected)
    ]
    add("snapshot_row_counts", "pass" if not bad_counts else "fail", "; ".join(bad_counts) if bad_counts else "All logical row counts match the checked-in snapshot.")

    duplicate_details = " ".join(str(row["detail"]) + " " + str(row["item_a"]) + " " + str(row["item_b"]) for row in duplicate_checks)
    add(
        "cbpf_duplicate_contribution_files",
        "pass" if "CBPFcontributions__20260418_071813_UTC.csv" in duplicate_details and "ContributionsFlow__20260418_071825_UTC.csv" in duplicate_details else "fail",
        "CBPF contribution duplicate pair detected by sha256.",
    )

    hxl_files = {row["file"] for row in file_inventory if row["hxl_metadata_row"]}
    expected_hxl = {"hpc_hno_2024.csv", "hpc_hno_2025.csv", "humanitarian-response-plans.csv"}
    add(
        "hxl_metadata_detection",
        "pass" if expected_hxl.issubset(hxl_files) else "fail",
        f"Detected HXL metadata rows in: {' | '.join(sorted(hxl_files))}.",
    )

    cross_by_name = {row["relationship"]: row for row in cross_reference}
    for relationship in (
        "hno_country_to_cod_iso3",
        "hno_admin1_pcode_to_cod_admin1_pcode",
        "hrp_code_to_fts_plan_code",
        "hno_sector_label_to_fts_cluster",
    ):
        row = cross_by_name.get(relationship, {})
        matched = int(row.get("matched_count") or 0)
        add(
            relationship,
            "pass" if matched > 0 else "fail",
            f"Matched {matched} values.",
        )

    after_hashes = {path.name: sha256_file(path) for path in csv_paths()}
    add(
        "initial_data_unchanged",
        "pass" if before_hashes == after_hashes else "fail",
        "Input CSV sha256 values match before and after report generation.",
    )

    return rows


def main() -> int:
    if not DATA_DIR.exists():
        print(f"Missing data directory: {DATA_DIR}", file=sys.stderr)
        return 1
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    paths = csv_paths()
    before_hashes = {path.name: sha256_file(path) for path in paths}

    profile_results = []
    for path in paths:
        print(f"Profiling {path.name}...", flush=True)
        profile_results.append(profile_file(path))

    file_inventory = [result["file"] for result in profile_results]
    column_profile = [row for result in profile_results for row in result["columns"]]
    missingness_profile = [row for result in profile_results for row in result["missing"]]
    value_distributions = [row for result in profile_results for row in result["distributions"]]
    numeric_profile = [row for result in profile_results for row in result["numeric"]]
    quality_flags = [row for result in profile_results for row in result["quality_flags"]]
    duplicate_checks = build_duplicate_checks(file_inventory, profile_results)
    cross_reference = build_cross_reference_coverage()
    dataset_catalog = build_dataset_catalog(file_inventory)
    can_cannot = build_can_cannot_say_matrix()
    validation_results = run_validations(file_inventory, duplicate_checks, cross_reference, before_hashes)

    write_csv(TABLE_DIR / "file_inventory.csv", file_inventory)
    write_csv(TABLE_DIR / "dataset_catalog.csv", dataset_catalog)
    write_csv(TABLE_DIR / "column_profile.csv", column_profile)
    write_csv(TABLE_DIR / "missingness_profile.csv", missingness_profile)
    write_csv(TABLE_DIR / "value_distributions.csv", value_distributions)
    write_csv(TABLE_DIR / "numeric_profile.csv", numeric_profile)
    write_csv(TABLE_DIR / "duplicate_checks.csv", duplicate_checks)
    write_csv(TABLE_DIR / "quality_flags.csv", quality_flags)
    write_csv(TABLE_DIR / "cross_reference_coverage.csv", cross_reference)
    write_csv(TABLE_DIR / "can_cannot_say_matrix.csv", can_cannot)
    write_csv(TABLE_DIR / "validation_results.csv", validation_results)

    REPORT_PATH.write_text(
        build_report(
            file_inventory,
            dataset_catalog,
            missingness_profile,
            duplicate_checks,
            cross_reference,
            can_cannot,
            quality_flags,
        ),
        encoding="utf-8",
    )

    failed = [row for row in validation_results if row["status"] != "pass"]
    if failed:
        for row in failed:
            print(f"VALIDATION FAILED: {row['validation']}: {row['detail']}", file=sys.stderr)
        return 2

    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}", flush=True)
    print(f"Wrote appendix tables under {TABLE_DIR.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
