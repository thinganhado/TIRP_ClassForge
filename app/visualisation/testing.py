# testing.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import text
import sys
import os

# Add app directory to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app import create_app, db

# Set seaborn style
sns.set(style="whitegrid")

# Start Flask app context
app = create_app()

with app.app_context():
    
    # Load dummy metrics from comparison_metrics1
    df = pd.read_sql(text("SELECT * FROM comparison_metrics1"), db.engine)

    # --- GPA Comparison (Split: GA vs Random) ---
    gpa_df = df[df["metric_type"] == "GPA"]
    df_gpa = gpa_df.copy()

    # Split into GA and Random
    gpa_ga = df_gpa[df_gpa["allocation_type"] == "GA"].sort_values("class_id")
    gpa_rand = df_gpa[df_gpa["allocation_type"] == "Random"].sort_values("class_id")

    # Calculate means
    mean_ga = gpa_ga["metric_value"].mean()
    mean_rand = gpa_rand["metric_value"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # GA Chart
    sns.barplot(
        ax=axes[0],
        data=gpa_ga,
        x="class_id", y="metric_value",
        color="#131d35"
    )
    axes[0].axhline(mean_ga, color="#FF4C4C", linestyle="--", linewidth=2)
    axes[0].set_title("GPA - Genetic Algorithm", fontsize=15)
    axes[0].set_xlabel("Class ID", fontsize=12)
    axes[0].set_ylabel("Average GPA", fontsize=12)
    axes[0].set_ylim(0, max(df_gpa["metric_value"]) + 5)

    # Random Chart
    sns.barplot(
        ax=axes[1],
        data=gpa_rand,
        x="class_id", y="metric_value",
        color="#40B0DF"
    )
    axes[1].axhline(mean_rand, color="#FF4C4C", linestyle="--", linewidth=2)
    axes[1].set_title("GPA - Random Allocation", fontsize=15)
    axes[1].set_xlabel("Class ID", fontsize=12)
    axes[1].set_ylabel("")  # Hide duplicate Y label
    axes[1].set_ylim(0, max(df_gpa["metric_value"]) + 5)

    plt.suptitle("Average GPA per Class by Allocation Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Conflict Comparison ---
    # --- Conflict Comparison (Dumbbell Plot) ---
    conflict_df = df[df["metric_type"] == "Conflicts"]

    # Pivot to have one row per class with GA and Random as columns
    pivot_df = conflict_df.pivot(index="class_id", columns="allocation_type", values="metric_value").sort_index()

    # Plot dumbbell chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw lines between GA and Random values
    for idx, row in pivot_df.iterrows():
        ax.plot([row["GA"], row["Random"]], [idx, idx], color="gray", linewidth=2, zorder=1)

    # Plot GA points
    ax.scatter(pivot_df["GA"], pivot_df.index, color="#131d35", s=100, label="Genetic Algorithm", zorder=2)

    # Plot Random points
    ax.scatter(pivot_df["Random"], pivot_df.index, color="#40B0DF", s=100, label="Random Allocation", zorder=3)

    # Labels and legend
    ax.set_title("Number of Conflicts per Class", fontsize=16)
    ax.set_xlabel("Conflict Count", fontsize=14)
    ax.set_ylabel("Class ID", fontsize=14)
    ax.set_yticks(pivot_df.index)
    ax.set_yticklabels([f"Class {i}" for i in pivot_df.index])
    ax.invert_yaxis()  # Optional: highest class on top
    ax.legend(title="Allocation Type", fontsize=12, title_fontsize=13)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


    ###########

    # Load wellbeing data
    df = pd.read_sql(text("SELECT * FROM comparison_metrics1 WHERE metric_type = 'Wellbeing'"), db.engine)
    df = df[df["metric_label"].isin(["High Wellbeing", "Low Wellbeing"])]

    # Pivot by allocation and class
    pivot = df.pivot_table(index=["allocation_type", "class_id"], 
                           columns="metric_label", values="metric_value").reset_index()
    pivot.sort_values(by=["allocation_type", "class_id"], inplace=True)

    # Set allocation color scheme
    colors = {
        "GA": {"Low": "#fcbad3", "High": "#131d35"},       # pink low, dark blue high
        "Random": {"Low": "#ffd6a5", "High": "#40B0DF"}    # orange low, cyan high
    }

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    spacing = 0.15
    allocations = ["GA", "Random"]
    class_ids = sorted(pivot["class_id"].unique())

    # Draw bars
    for i, class_id in enumerate(class_ids):
        for j, alloc in enumerate(allocations):
            row = pivot[(pivot["class_id"] == class_id) & (pivot["allocation_type"] == alloc)]
            if not row.empty:
                base_x = i + (j - 0.5) * (bar_width + spacing)
                low_val = row["Low Wellbeing"].values[0]
                high_val = row["High Wellbeing"].values[0]

                ax.bar(base_x, low_val, width=bar_width, color=colors[alloc]["Low"])
                ax.bar(base_x, high_val, bottom=low_val, width=bar_width, color=colors[alloc]["High"])

                # Labels
                ax.text(base_x, low_val / 2, f"{low_val:.2f}", ha="center", va="center", fontsize=8)
                ax.text(base_x, low_val + high_val / 2, f"{high_val:.2f}", ha="center", va="center", fontsize=8)

    # Styling
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels([f"Class {i}" for i in class_ids])
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Wellbeing Distribution by Class and Allocation", fontsize=14)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#131d35", label="High Wellbeing (GA)"),
        Patch(facecolor="#fcbad3", label="Low Wellbeing (GA)"),
        Patch(facecolor="#40B0DF", label="High Wellbeing (Random)"),
        Patch(facecolor="#ffd6a5", label="Low Wellbeing (Random)")
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()
