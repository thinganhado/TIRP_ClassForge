import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from sqlalchemy import text
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app import db


def _to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_graphs():
    # --- Load comparison metrics from database ---
    query = "SELECT * FROM comparison_metrics"
    df = pd.read_sql(text(query), db.engine)

    # --- GPA Comparison Chart ---
    def generate_gpa_chart(df):
        gpa_df = df[df["metric_type"] == "GPA"]
        pivot = gpa_df.pivot(index="class_id", columns="allocation_type", values="metric_value").fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.35
        x = pivot.index
        ax.bar(x - width/2, pivot["Random"], width=width, label="Random", color="#FFD53D")
        ax.bar(x + width/2, pivot["GA"], width=width, label="GA", color="#40B0DF")
        ax.set_title("Average GPA per Class")
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Average GPA")
        ax.legend()
        plt.tight_layout()
        return _to_base64(fig)

    # --- Wellbeing Distribution Chart (GA only) ---
    def generate_wellbeing_chart(df):
        wb_df = df[(df["metric_type"] == "Wellbeing") & (df["allocation_type"] == "GA")]
        pivot = wb_df.pivot(index="class_id", columns="metric_label", values="metric_value").fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1")
        ax.set_title("Wellbeing Distribution (GA Allocation)")
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Proportion")
        plt.tight_layout()
        return _to_base64(fig)

    # --- Conflict Chart (bar chart GA vs Random) ---
    def generate_conflict_chart(df):
        conflict_df = df[df["metric_type"] == "Conflicts"]
        pivot = conflict_df.pivot(index="class_id", columns="allocation_type", values="metric_value").fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.35
        x = pivot.index
        ax.bar(x - width/2, pivot["Random"], width=width, label="Random", color="#FFD53D")
        ax.bar(x + width/2, pivot["GA"], width=width, label="GA", color="#40B0DF")
        ax.set_title("Conflict Count per Class")
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Number of Conflicts")
        ax.legend()
        plt.tight_layout()
        return _to_base64(fig)

    # --- Generate All Charts ---
    gpa_chart = generate_gpa_chart(df)
    wellbeing_chart = generate_wellbeing_chart(df)
    conflict_chart = generate_conflict_chart(df)

    return {
        "gpa_chart_img": gpa_chart,
        "wellbeing_chart_img": wellbeing_chart,
        "conflict_chart_img": conflict_chart,
    }