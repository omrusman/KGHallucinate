import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

MODEL_DISPLAY_NAMES = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "qwen/qwen-2.5-7b-instruct": "Qwen-2.5-7B",
    "mistralai/ministral-8b-2512": "Ministral-8B",
}

LABEL_COLORS = {
    "Supported": "#4C9A6A",
    "Contradicted (Hallucination)": "#C0392B",
    "Unverifiable": "#B0B0B0",
}

TIER_COLORS = {
    "High": "#2D3A6B",
    "Medium": "#7B5AA6",
    "Low": "#E08A5F",
}

TIER_ORDER = ["High", "Medium", "Low"]
LABEL_ORDER = ["Supported", "Contradicted (Hallucination)", "Unverifiable"]

# Printed widths on the poster, in cm. Each figure is built at its true
# printed size, so one point in the figure equals one point on the poster
# and the text in both figures appears at the same size.
FIG1_CM = 30.0
FIG2_CM = 15.0
CM_PER_INCH = 2.54

INPUT_PATH = "data/verified.json"
FIGURES_DIR = os.path.join("results", "figures")
RESULTS_DIR = "results"


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 4_verify_claims.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(data):
    models = list(MODEL_DISPLAY_NAMES.keys())
    tier_model_counts = {m: {t: Counter() for t in TIER_ORDER} for m in models}
    overall_counts = {m: Counter() for m in models}

    for entity in data:
        tier = entity["tier"]
        if tier not in TIER_ORDER:
            continue
        for model, vclaims in entity.get("verified_claims", {}).items():
            if model not in tier_model_counts:
                continue
            for c in vclaims:
                tier_model_counts[model][tier][c["label"]] += 1
                overall_counts[model][c["label"]] += 1

    return models, tier_model_counts, overall_counts


def rate(counter, label):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return 100 * counter.get(label, 0) / total


def set_style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.9,
    })


def make_fig1(models, tier_model_counts, out_path):
    width_in = FIG1_CM / CM_PER_INCH
    fig, axes = plt.subplots(1, 3, figsize=(width_in, width_in * 0.30), sharey=True)
    fig.suptitle("Hallucination Rate by Entity Popularity Tier", fontsize=15, y=1.04)

    all_rates = [
        rate(tier_model_counts[m][t], "Contradicted (Hallucination)")
        for m in models for t in TIER_ORDER
    ]
    y_max = max(4.2, max(all_rates) * 1.28) if all_rates else 4.2

    for ax, model in zip(axes, models):
        rates = [rate(tier_model_counts[model][t], "Contradicted (Hallucination)") for t in TIER_ORDER]
        colors = [TIER_COLORS[t] for t in TIER_ORDER]
        bars = ax.bar(TIER_ORDER, rates, color=colors, width=0.7)
        ax.set_title(MODEL_DISPLAY_NAMES[model], pad=7)
        if model == models[0]:
            ax.set_ylabel("Hallucination rate (%)")
        ax.set_ylim(0, y_max)
        ax.grid(axis="x", visible=False)
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.025,
                    f"{r:.1f}%", ha="center", va="bottom", fontsize=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}  (place at {FIG1_CM:.1f} cm wide)")


def make_fig2(models, overall_counts, out_path):
    width_in = FIG2_CM / CM_PER_INCH
    # Match fig1's printed height so the two can sit side by side at full
    # size. fig1 is FIG1_CM wide with a 0.30 aspect, so its height is
    # FIG1_CM * 0.30; fig2 uses that same height at its own width.
    height_cm = FIG1_CM * 0.30
    fig, ax = plt.subplots(figsize=(width_in, height_cm / CM_PER_INCH))

    model_labels = [MODEL_DISPLAY_NAMES[m] for m in models]
    bottom = [0.0] * len(models)

    # Short legend labels: the full wording appears in the poster text.
    short_labels = {
        "Supported": "Supported",
        "Contradicted (Hallucination)": "Contradicted",
        "Unverifiable": "Unverifiable",
    }

    for label in LABEL_ORDER:
        values = [rate(overall_counts[m], label) for m in models]
        ax.bar(model_labels, values, bottom=bottom, label=short_labels[label],
               color=LABEL_COLORS[label], width=0.62)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylabel("Share of claims (%)")
    ax.set_title("Verification Outcomes", fontsize=14, pad=7)
    ax.set_ylim(0, 104)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, fontsize=10, columnspacing=1.2, handlelength=1.2)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}  (place at {FIG2_CM:.1f} cm wide)")


def print_summary(models, tier_model_counts):
    print()
    print("=== Summary statistics ===")
    for m in models:
        for t in TIER_ORDER:
            c = tier_model_counts[m][t]
            total = sum(c.values())
            if total == 0:
                continue
            print(f"{MODEL_DISPLAY_NAMES[m]:15s} {t:7s} n={total:4d}  "
                  f"Hallucination={rate(c, 'Contradicted (Hallucination)'):.1f}%  "
                  f"Supported={rate(c, 'Supported'):.1f}%  "
                  f"Unverifiable={rate(c, 'Unverifiable'):.1f}%")
        print()


def export_excel(data, out_path):
    records = []
    for ent in data:
        name = ent.get("name")
        tier = ent.get("tier", "Unknown")
        sitelinks = ent.get("sitelinks")
        for model, claims in ent.get("verified_claims", {}).items():
            for claim_data in claims:
                records.append({
                    "Entity": name,
                    "Tier": tier,
                    "Sitelinks": sitelinks,
                    "Model": MODEL_DISPLAY_NAMES.get(model, model),
                    "Claim": claim_data.get("claim"),
                    "Label": claim_data.get("label"),
                    "Method": claim_data.get("method"),
                    "Score": claim_data.get("score", 0.0),
                    "Matched Fact": claim_data.get("matched_fact_sentence"),
                })

    df = pd.DataFrame(records)

    overall_stats = df.groupby("Model")["Label"].value_counts().unstack(fill_value=0)
    overall_stats["Total Claims"] = overall_stats.sum(axis=1)
    if "Contradicted (Hallucination)" in overall_stats.columns:
        overall_stats["Hallucination Rate (%)"] = (
            overall_stats["Contradicted (Hallucination)"] / overall_stats["Total Claims"] * 100
        ).round(2)

    tier_stats = df.groupby(["Model", "Tier"])["Label"].value_counts().unstack(fill_value=0)
    tier_stats["Total Claims"] = tier_stats.sum(axis=1)
    if "Contradicted (Hallucination)" in tier_stats.columns:
        tier_stats["Hallucination Rate (%)"] = (
            tier_stats["Contradicted (Hallucination)"] / tier_stats["Total Claims"] * 100
        ).round(2)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        overall_stats.to_excel(writer, sheet_name="Overall_Stats")
        tier_stats.to_excel(writer, sheet_name="Tier_Stats")
        df.to_excel(writer, sheet_name="Raw_Data", index=False)

    print(f"Saved {out_path}")


def main():
    data = load_data(INPUT_PATH)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_style()

    models, tier_model_counts, overall_counts = aggregate(data)

    make_fig1(models, tier_model_counts, os.path.join(FIGURES_DIR, "fig1_hallucination_by_tier.png"))
    make_fig2(models, overall_counts, os.path.join(FIGURES_DIR, "fig2_label_breakdown.png"))

    print_summary(models, tier_model_counts)
    try:
        export_excel(data, os.path.join(RESULTS_DIR, "hallucination_results.xlsx"))
    except PermissionError:
        print("\nWARNING: Could not save hallucination_results.xlsx because it is open in another program.")


if __name__ == "__main__":
    main()
