# testing.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import text
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app import create_app, db

# Set global seaborn style
sns.set(style="whitegrid")

# Create Flask app context for DB access
app = create_app()

with app.app_context():
    # Load metrics
    df = pd.read_sql(text("SELECT * FROM comparison_metrics"), db.engine)

    # --- GPA Comparison (Side-by-Side) ---
    gpa_df = df[df["metric_type"] == "GPA"]
    df_gpa = gpa_df.copy()
    df_gpa["allocation_type"] = df_gpa["allocation_type"].replace({"GA": "Genetic Algorithm", "Random": "Random Allocation"})

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_gpa,
        x="class_id", y="metric_value", hue="allocation_type",
        palette=["#add8e6", "#fcbad3"], errorbar=None
    )
    plt.title("Average GPA per Class", fontsize=16)
    plt.xlabel("Class ID", fontsize=14)
    plt.ylabel("Average GPA", fontsize=14)
    plt.legend(title="Allocation Type", fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.show()

# --- Wellbeing Distribution (GA only, stacked bar chart) ---
    wb_df = df[(df["metric_type"] == "Wellbeing") & (df["allocation_type"] == "GA")]

    if wb_df.empty:
        print("⚠️ No wellbeing data found for GA.")
    else:
        # Ensure metric_value is numeric
        wb_df["metric_value"] = pd.to_numeric(wb_df["metric_value"], errors="coerce")

        # Pivot to get: class_id as rows, wellbeing labels as columns
        pivot_wb = wb_df.pivot(index="class_id", columns="metric_label", values="metric_value").fillna(0)

        # Reorder wellbeing levels for aesthetic
        desired_order = ["Low", "Medium", "High"]
        pivot_wb = pivot_wb[[col for col in desired_order if col in pivot_wb.columns]]

        # Plot stacked bar chart with pastel palette
        pivot_wb.plot(
            kind="bar", stacked=True, figsize=(10, 6),
            color=["#ffd6a5", "#b5ead7", "#c7ceea"]
        )
        plt.title("Wellbeing Distribution (GA Allocation)", fontsize=16)
        plt.xlabel("Class ID", fontsize=14)
        plt.ylabel("Proportion of Wellbeing Labels", fontsize=14)
        plt.legend(title="Wellbeing Label", fontsize=12, title_fontsize=13)
        plt.tight_layout()
        plt.show()

    # --- Conflict Comparison (GA vs Random) ---
    conflict_df = df[df["metric_type"] == "Conflicts"]
    df_conf = conflict_df.copy()
    df_conf["allocation_type"] = df_conf["allocation_type"].replace({"GA": "Genetic Algorithm", "Random": "Random Allocation"})

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_conf,
        x="class_id", y="metric_value", hue="allocation_type",
        palette=["#add8e6", "#fcbad3"], errorbar=None
    )
    plt.title("Number of Conflicts per Class", fontsize=16)
    plt.xlabel("Class ID", fontsize=14)
    plt.ylabel("Conflict Count", fontsize=14)
    plt.legend(title="Allocation Type", fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.show()