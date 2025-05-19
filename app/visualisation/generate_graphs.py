import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend to avoid Tkinter thread errors
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
from sqlalchemy import text
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app import db

sns.set(style="whitegrid")

def _to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_graphs():
    df = pd.read_sql(text("SELECT * FROM comparison_metrics1"), db.engine)

    def generate_gpa_chart(df):
        gpa_df = df[df["metric_type"] == "GPA"]
        gpa_ga = gpa_df[gpa_df["allocation_type"] == "GA"].sort_values("class_id")
        gpa_rand = gpa_df[gpa_df["allocation_type"] == "Random"].sort_values("class_id")

        mean_ga = gpa_ga["metric_value"].mean()
        mean_rand = gpa_rand["metric_value"].mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

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
        axes[0].set_ylim(0, max(gpa_df["metric_value"]) + 5)

        sns.barplot(
            ax=axes[1],
            data=gpa_rand,
            x="class_id", y="metric_value",
            color="#40B0DF"
        )
        axes[1].axhline(mean_rand, color="#FF4C4C", linestyle="--", linewidth=2)
        axes[1].set_title("GPA - Random Allocation", fontsize=15)
        axes[1].set_xlabel("Class ID", fontsize=12)
        axes[1].set_ylabel("")
        axes[1].set_ylim(0, max(gpa_df["metric_value"]) + 5)

        plt.suptitle("Average GPA per Class by Allocation Type", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return _to_base64(fig)

    def generate_conflict_chart(df):
        conflict_df = df[df["metric_type"] == "Conflicts"]
        pivot_df = conflict_df.pivot(index="class_id", columns="allocation_type", values="metric_value").sort_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, row in pivot_df.iterrows():
            ax.plot([row["GA"], row["Random"]], [idx, idx], color="gray", linewidth=2, zorder=1)
        ax.scatter(pivot_df["GA"], pivot_df.index, color="#131d35", s=100, label="Genetic Algorithm", zorder=2)
        ax.scatter(pivot_df["Random"], pivot_df.index, color="#40B0DF", s=100, label="Random Allocation", zorder=3)

        ax.set_title("Number of Conflicts per Class", fontsize=16)
        ax.set_xlabel("Conflict Count", fontsize=14)
        ax.set_ylabel("Class ID", fontsize=14)
        ax.set_yticks(pivot_df.index)
        ax.set_yticklabels([f"Class {i}" for i in pivot_df.index])
        ax.invert_yaxis()
        ax.legend(title="Allocation Type", fontsize=12, title_fontsize=13)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        return _to_base64(fig)

    gpa_chart = generate_gpa_chart(df)
    conflict_chart = generate_conflict_chart(df)

    return {
        "gpa_chart_img": gpa_chart,
        "conflict_chart_img": conflict_chart
    }
