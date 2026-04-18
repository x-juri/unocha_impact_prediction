"""
scripts/plot_correlation.py

Merges GDELT Crisis News Intensity with CERF funding data (from the notebook)
and produces a scatter plot: crisis media coverage vs. CERF allocation.

Requires:
    data/gdelt_coverage.csv   (from fetch_gdelt_coverage.py)
    data/funding_vulnerability.csv  (from notebook ETL)

Outputs:
    reports/gdelt_cerf_scatter.png

Usage:
    source .venv/bin/activate
    python scripts/plot_correlation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
GDELT_PATH = Path("data/gdelt_coverage.csv")
CERF_PATH = Path("data/funding_vulnerability.csv")
OUT_PNG = Path("reports/gdelt_cerf_scatter.png")

# ── Continent colour palette ──────────────────────────────────────────────────
CONTINENT_COLOURS: dict[str, str] = {
    "Africa": "#E07B39",
    "Asia": "#5B8DB8",
    "Americas": "#6BAE75",
    "Europe": "#9B6BB5",
    "Oceania": "#C4A84F",
}
DEFAULT_COLOUR = "#888888"

# Countries to always label (high-profile or analytically interesting)
ALWAYS_LABEL = {
    "Yemen", "Syria", "Afghanistan", "South Sudan", "Somalia",
    "Haiti", "Myanmar", "Niger", "Nigeria", "Ethiopia",
    "Democratic Republic of the Congo", "Sudan",
}


def load_cerf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Aggregate to country level across all rounds
    agg = (
        df.groupby(["ISO3", "countryName", "continentName"], as_index=False)
        .agg(
            total_cerf_usd=("totalAmountApproved", "sum"),
            num_projects=("projectID", "count"),
            num_rounds=("Round", "nunique"),
        )
        .rename(columns={"ISO3": "ISO3"})
    )
    return agg


def load_gdelt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Aggregate CNI to country level
    agg = (
        df.groupby("ISO3", as_index=False)
        .agg(
            crisis_article_volume=("crisis_article_volume", "sum"),
            neg_event_count=("neg_event_count", "sum"),
            avg_tone=("avg_tone", "mean"),
        )
    )
    return agg


def add_regression_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> float:
    """Fit log-log regression, draw line, return Pearson r on log values."""
    mask = (x >= 0) & (y >= 0)
    coeffs = np.polyfit(x[mask], y[mask], 1)
    r = float(np.corrcoef(x[mask], y[mask])[0, 1])

    x_line = np.linspace(x[mask].min(), x[mask].max(), 200)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, color="#CC3333", linewidth=1.5,
            linestyle="--", alpha=0.7, label=f"Linear OLS fit  (r = {r:.2f})")
    return r


def main() -> None:
    if not GDELT_PATH.exists():
        raise FileNotFoundError(
            f"{GDELT_PATH} not found. Run scripts/fetch_gdelt_coverage.py first."
        )

    cerf = load_cerf(CERF_PATH)
    gdelt = load_gdelt(GDELT_PATH)

    merged = cerf.merge(gdelt, on="ISO3", how="inner")
    print(f"Matched {len(merged)} countries out of {len(cerf)} CERF / {len(gdelt)} GDELT")
    if merged.empty:
        print("No overlapping countries — check ISO3 codes.")
        return

    x = merged["crisis_article_volume"].values.astype(float)
    y = merged["total_cerf_usd"].values.astype(float)
    colours = [
        CONTINENT_COLOURS.get(c, DEFAULT_COLOUR)
        for c in merged["continentName"]
    ]

    # Dot size proportional to number of projects
    sizes = 60 + merged["num_projects"] * 2.5

    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(x, y, c=colours, s=sizes, alpha=0.82, edgecolors="white",
                    linewidths=0.6, zorder=3)

    add_regression_line(ax, x, y)

    # ── Labels ────────────────────────────────────────────────────────────────
    for _, row in merged.iterrows():
        if row["countryName"] in ALWAYS_LABEL or row["total_cerf_usd"] > 3e7:
            ax.annotate(
                row["countryName"],
                xy=(row["crisis_article_volume"], row["total_cerf_usd"]),
                xytext=(6, 3),
                textcoords="offset points",
                fontsize=7.5,
                color="#333333",
                clip_on=True,
            )

    # ── Axes ──────────────────────────────────────────────────────────────────
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Crisis News Intensity (sampled GDELT article volume, linear scale)", fontsize=11)
    ax.set_ylabel("Total CERF UFE Funding 2022-2025 (USD, linear scale)", fontsize=11)
    ax.set_title(
        "Crisis Media Coverage vs. CERF Underfunded Emergency Allocations (2022–2025)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    # ── Legend: continents ────────────────────────────────────────────────────
    continent_patches = [
        mpatches.Patch(color=col, label=cont)
        for cont, col in CONTINENT_COLOURS.items()
        if cont in merged["continentName"].values
    ]
    size_legend_handles = [
        plt.scatter([], [], s=60 + n * 2.5, color="#888888", alpha=0.7,
                    edgecolors="white", label=f"{n} projects")
        for n in [5, 15, 30]
    ]
    first_legend = ax.legend(
        handles=continent_patches,
        title="Continent",
        loc="upper left",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.85,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=size_legend_handles + [
            plt.Line2D([0], [0], color="#CC3333", linestyle="--", linewidth=1.5,
                       label="Linear OLS fit")
        ],
        title="Dot size = # projects",
        loc="lower right",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.85,
    )

    # ── Footnote ──────────────────────────────────────────────────────────────
    fig.text(
        0.01, 0.01,
        "GDELT: sampled 1st & 15th of each month 2022-2025 | events with GoldsteinScale < 0 only | "
        "CERF: UFE window, project-level data from HDX",
        fontsize=7, color="#666666",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {OUT_PNG}")

    # Print top/bottom countries by funding-per-CNI ratio (under/over-represented)
    merged["usd_per_crisis_article"] = merged["total_cerf_usd"] / merged["crisis_article_volume"]
    print("\nTop 5 countries with MOST funding relative to media coverage (under-reported crises):")
    print(
        merged.nlargest(5, "usd_per_crisis_article")[
            ["countryName", "total_cerf_usd", "crisis_article_volume", "usd_per_crisis_article"]
        ].to_string(index=False)
    )
    print("\nTop 5 countries with LEAST funding relative to media coverage (media-heavy, less funded):")
    print(
        merged.nsmallest(5, "usd_per_crisis_article")[
            ["countryName", "total_cerf_usd", "crisis_article_volume", "usd_per_crisis_article"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
