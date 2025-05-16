# insights.py - Prompts 1 & 2: Setup and Load All Data

import pandas as pd
import torch
import os
import json
from sqlalchemy import create_engine, text
import sys
# Ensure parent dir is in path to allow app import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app import create_app, db

# --- Initialize Flask App Context ---
app = create_app()

# --- Load Student Survey and Relationship Data ---
student_data = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="responses")
participant_data = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="participants")

# Load relationship data
friends = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_0_Friends")
influence = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_1_Influential")
meaningful = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_2_Feedback")
Moretime = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_3_MoreTime")
Advice = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_4_Advice")
Disrespect = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_5_Disrespect")
Activity = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_affiliation_0_SchoolActivit")

# Remove self-loops
for df in [friends, influence, meaningful, Moretime, Advice, Disrespect, Activity]:
    df.drop(df[df['Source'] == df['Target']].index, inplace=True)

# Clean participant data
participant_data.columns = participant_data.columns.str.strip()
participant_data['Perc_Academic'] = participant_data['Perc_Academic'].fillna(participant_data['Perc_Academic'].mean())

# --- Load External Model Outputs (now ID-based) ---
student_scores_df = pd.read_excel("R-GCN_files/student_scores_with_ids.xlsx")
friendship_df = pd.read_excel("R-GCN_files/friendship_scores_with_ids.xlsx")

# --- Load GPA and Wellbeing Data ---
gpa_df = pd.read_csv("ha_outputs/gpa_predictions_with_bins.csv")
gpa_df.rename(columns={"Student_ID": "Participant-ID"}, inplace=True)
global_mean_gpa = gpa_df["Predicted_GPA"].mean()

wellbeing_df = pd.read_csv("Clustering/output/wellbeing_classification_results.csv")

# --- Load Bully-Victim Assignments ---
bully_df = pd.read_csv("ha_outputs/community_bully_assignments.csv")
bully_df.rename(columns={"Student_ID": "Participant-ID"}, inplace=True)
bully_to_victims = {}
for _, row in bully_df.iterrows():
    if row["Is_Bully"] == 0:
        bully_to_victims.setdefault(row["Primary_Bully_ID"], []).append(row["Participant-ID"])

print("✅ Setup complete. All files loaded successfully.")

with app.app_context():
    # Load GA allocation
    ga_allocation = db.session.execute(text("SELECT class_id, student_id FROM allocations")).fetchall()
    df_ga = pd.DataFrame(ga_allocation, columns=["class_id", "Participant-ID"])

    # Load random allocation
    random_allocation = db.session.execute(text("SELECT class_id, participant_id FROM random_allo")).fetchall()
    df_random = pd.DataFrame(random_allocation, columns=["class_id", "Participant-ID"])

    print("GA and random allocations loaded successfully.")

    # Ensure 'Participant-ID' is of consistent type across all relevant DataFrames
    df_ga["Participant-ID"] = df_ga["Participant-ID"].astype(str)
    df_random["Participant-ID"] = df_random["Participant-ID"].astype(str)
    participant_data["Participant-ID"] = participant_data["Participant-ID"].astype(str)
    gpa_df["Participant-ID"] = gpa_df["Participant-ID"].astype(str)
    student_scores_df["Participant-ID"] = student_scores_df["Participant-ID"].astype(str)
    wellbeing_df["Participant-ID"] = wellbeing_df["Participant-ID"].astype(str)

    # Merge with participant and score data
    df_ga = df_ga.merge(participant_data, on="Participant-ID", how="left")
    df_ga = df_ga.merge(gpa_df, on="Participant-ID", how="left")
    df_ga = df_ga.merge(student_scores_df, on="Participant-ID", how="left")
    df_ga = df_ga.merge(wellbeing_df[["Participant-ID", "wellbeing_label"]], on="Participant-ID", how="left")

    df_random = df_random.merge(participant_data, on="Participant-ID", how="left")
    df_random = df_random.merge(gpa_df, on="Participant-ID", how="left")
    df_random = df_random.merge(student_scores_df, on="Participant-ID", how="left")
    df_random = df_random.merge(wellbeing_df[["Participant-ID", "wellbeing_label"]], on="Participant-ID", how="left")

    # --- Prompt 3: Quantitative Metrics ---

    def compute_gpa_balance(df, label):
        gpa_by_class = df.groupby("class_id")["Predicted_GPA"].mean()
        std_dev = gpa_by_class.std()
        print(f"📊 {label} GPA Std Dev: {std_dev:.4f}")
        return std_dev

    def compute_friendship_metrics(df, friendship_df, label):
        avg_friends_list = []
        std_friends_per_class = []

        for class_id in df["class_id"].unique():
            class_ids = df[df["class_id"] == class_id]["Participant-ID"].tolist()
            friend_counts = []

            for student in class_ids:
                is_friend = (
                    (friendship_df["Participant-ID 1"] == student) & (friendship_df["Participant-ID 2"].isin(class_ids))
                ) | (
                    (friendship_df["Participant-ID 2"] == student) & (friendship_df["Participant-ID 1"].isin(class_ids))
                )
                num_friends = len(friendship_df[is_friend])
                friend_counts.append(num_friends)

            avg_friends = sum(friend_counts) / len(friend_counts) if friend_counts else 0
            std_dev_friends = pd.Series(friend_counts).std() if friend_counts else 0

            avg_friends_list.append(avg_friends)
            std_friends_per_class.append(std_dev_friends)

        overall_avg_friends = sum(avg_friends_list) / len(avg_friends_list) if avg_friends_list else 0
        overall_std_dev_friends = sum(std_friends_per_class) / len(std_friends_per_class) if std_friends_per_class else 0

        print(f"🤝 {label} - Avg Friends: {overall_avg_friends:.4f}, Std Dev: {overall_std_dev_friends:.4f}")
        return overall_avg_friends, overall_std_dev_friends

    def compute_std_balance(df_alloc, score_df, score_column, label):
        merged = pd.merge(df_alloc, score_df, on="Participant-ID")
        avg_scores_per_class = merged.groupby("class_id")[score_column].mean()
        std_dev = avg_scores_per_class.std()
        print(f"{label} Std Dev of {score_column}: {std_dev:.4f}")
        return std_dev

    def compute_wellbeing_std(df, label):
        prop_by_class = df.groupby("class_id")["wellbeing_label"].value_counts(normalize=True).unstack().fillna(0)
        std_dev = prop_by_class.std().mean()
        print(f"❤️‍🩹 {label} Wellbeing Std Dev: {std_dev:.4f}")
        return std_dev

    def count_bully_victim_conflicts(df, bully_dict):
        conflict_count = 0
        for class_id, group in df.groupby("class_id"):
            class_pids = set(group["Participant-ID"])
            for bully, victims in bully_dict.items():
                if bully in class_pids:
                    overlap = class_pids.intersection(set(victims))
                    conflict_count += len(overlap)
        return conflict_count

    # --- Run All Metrics ---
    gpa_std_ga = compute_gpa_balance(df_ga, "GA Allocation")
    gpa_std_rand = compute_gpa_balance(df_random, "Random Allocation")

    avg_friends_ga, std_friend_ga = compute_friendship_metrics(df_ga, friendship_df, "GA Allocation")
    avg_friends_rand, std_friend_rand = compute_friendship_metrics(df_random, friendship_df, "Random Allocation")

    influence_std_ga = compute_std_balance(df_ga, student_scores_df, "influential_score", "GA Allocation")
    influence_std_rand = compute_std_balance(df_random, student_scores_df, "influential_score", "Random Allocation")

    isolation_std_ga = compute_std_balance(df_ga, student_scores_df, "isolated_score", "GA Allocation")
    isolation_std_rand = compute_std_balance(df_random, student_scores_df, "isolated_score", "Random Allocation")

    std_wellbeing_ga = compute_wellbeing_std(df_ga, "GA Allocation")
    std_wellbeing_rand = compute_wellbeing_std(df_random, "Random Allocation")

    bully_conflicts_ga = count_bully_victim_conflicts(df_ga, bully_to_victims)
    bully_conflicts_rand = count_bully_victim_conflicts(df_random, bully_to_victims)

    # --- Prompt 4–6: Observations ---

    def generate_friendship_observation(df, friendship_df, label):
        min_avg = float('inf')
        weakest_class = None

        for class_id, group in df.groupby("class_id"):
            class_ids = group["Participant-ID"].tolist()
            friend_counts = []
            for student in class_ids:
                is_friend = (
                    (friendship_df["Participant1-ID"] == student) & (friendship_df["Participant2-ID"].isin(class_ids))
                ) | (
                    (friendship_df["Participant2-ID"] == student) & (friendship_df["Participant1-ID"].isin(class_ids))
                )
                friend_counts.append(len(friendship_df[is_friend]))
            avg_friends = sum(friend_counts) / len(friend_counts) if friend_counts else 0
            if avg_friends < min_avg:
                min_avg = avg_friends
                weakest_class = class_id

        return f"{label}: Class {weakest_class} has the lowest average friendships ({min_avg:.2f}), suggesting weaker social ties."

    def generate_influence_observation(df, label):
        influ_per_class = df.groupby("class_id")["influential_score"].mean()
        highest_class = influ_per_class.idxmax()
        max_score = influ_per_class.max()
        return f"{label}: Class {highest_class} has the highest average influence score ({max_score:.2f}), indicating a highly influential group."

    def generate_isolation_observation(df, label):
        isolation_per_class = df.groupby("class_id")["isolated_score"].mean()
        highest_class = isolation_per_class.idxmax()
        max_score = isolation_per_class.max()
        return f"{label}: Class {highest_class} has the highest average isolation score ({max_score:.2f}), indicating possible social detachment."

    def generate_gpa_observation(df, label):
        gpa_per_class = df.groupby("class_id")["Predicted_GPA"].mean()
        lowest_class = gpa_per_class.idxmin()
        highest_class = gpa_per_class.idxmax()
        range_diff = gpa_per_class.max() - gpa_per_class.min()

        if range_diff > 0.5:
            return f"{label}: Class {highest_class} has the highest average GPA ({gpa_per_class[highest_class]:.2f}), while Class {lowest_class} has the lowest ({gpa_per_class[lowest_class]:.2f})."
        else:
            return f"{label}: GPA levels are relatively balanced across classes."

    def generate_conflict_observation(df, bully_dict, label):
        conflict_counts = {}
        for class_id, group in df.groupby("class_id"):
            class_pids = set(group["Participant-ID"])
            count = 0
            for bully, victims in bully_dict.items():
                if bully in class_pids:
                    count += len(set(victims) & class_pids)
            conflict_counts[class_id] = count
        max_class = max(conflict_counts, key=conflict_counts.get, default=None)
        max_conflicts = conflict_counts.get(max_class, 0)
        if max_conflicts > 0:
            return f"{label}: Class {max_class} has the most bully-victim overlaps ({max_conflicts} pairs), which may require intervention."
        else:
            return f"{label}: No significant bully-victim overlaps detected."

    # --- Generate Observations ---
    insight_friendship_ga = generate_friendship_observation(df_ga, friendship_df, "GA Allocation")
    insight_friendship_rand = generate_friendship_observation(df_random, friendship_df, "Random Allocation")

    insight_influence_ga = generate_influence_observation(df_ga, "GA Allocation")
    insight_influence_rand = generate_influence_observation(df_random, "Random Allocation")

    insight_isolation_ga = generate_isolation_observation(df_ga, "GA Allocation")
    insight_isolation_rand = generate_isolation_observation(df_random, "Random Allocation")

    insight_gpa_ga = generate_gpa_observation(df_ga, "GA Allocation")
    insight_gpa_rand = generate_gpa_observation(df_random, "Random Allocation")

    insight_bully_ga = generate_conflict_observation(df_ga, bully_to_victims, "GA Allocation")
    insight_bully_rand = generate_conflict_observation(df_random, bully_to_victims, "Random Allocation")

    # --- Save to Database ---
    db.session.execute(text("DELETE FROM allocation_insights"))
    db.session.execute(text("DELETE FROM allocation_observations"))

    db.session.execute(text("""
        INSERT INTO allocation_insights (
            allocation_type,
            avg_friends_per_student,
            friendship_std_dev,
            influence_std_dev,
            isolation_std_dev,
            gpa_std_dev,
            bully_conflict_count,
            wellbeing_std_dev
        ) VALUES (:type, :avg_friends, :friend_std, :influ_std, :isol_std, :gpa_std, :bully_conf, :wellbeing_std)
    """), {
        "type": "GA",
        "avg_friends": avg_friends_ga,
        "friend_std": std_friend_ga,
        "influ_std": influence_std_ga,
        "isol_std": isolation_std_ga,
        "gpa_std": gpa_std_ga,
        "bully_conf": bully_conflicts_ga,
        "wellbeing_std": std_wellbeing_ga
    })

    db.session.execute(text("""
        INSERT INTO allocation_insights (
            allocation_type,
            avg_friends_per_student,
            friendship_std_dev,
            influence_std_dev,
            isolation_std_dev,
            gpa_std_dev,
            bully_conflict_count,
            wellbeing_std_dev
        ) VALUES (:type, :avg_friends, :friend_std, :influ_std, :isol_std, :gpa_std, :bully_conf, :wellbeing_std)
    """), {
        "type": "Random",
        "avg_friends": avg_friends_rand,
        "friend_std": std_friend_rand,
        "influ_std": influence_std_rand,
        "isol_std": isolation_std_rand,
        "gpa_std": gpa_std_rand,
        "bully_conf": bully_conflicts_rand,
        "wellbeing_std": std_wellbeing_rand
    })

    db.session.execute(text("""
        INSERT INTO allocation_observations (
            allocation_type,
            friendship_observation,
            influence_observation,
            isolation_observation,
            gpa_observation,
            conflict_observation
        ) VALUES (:type, :friend_obs, :influ_obs, :isol_obs, :gpa_obs, :conflict_obs)
    """), {
        "type": "GA",
        "friend_obs": insight_friendship_ga,
        "influ_obs": insight_influence_ga,
        "isol_obs": insight_isolation_ga,
        "gpa_obs": insight_gpa_ga,
        "conflict_obs": insight_bully_ga
    })

    db.session.execute(text("""
        INSERT INTO allocation_observations (
            allocation_type,
            friendship_observation,
            influence_observation,
            isolation_observation,
            gpa_observation,
            conflict_observation
        ) VALUES (:type, :friend_obs, :influ_obs, :isol_obs, :gpa_obs, :conflict_obs)
    """), {
        "type": "Random",
        "friend_obs": insight_friendship_rand,
        "influ_obs": insight_influence_rand,
        "isol_obs": insight_isolation_rand,
        "gpa_obs": insight_gpa_rand,
        "conflict_obs": insight_bully_rand
    })

    db.session.commit()
    print("✅ All insights and observations saved to the database.")