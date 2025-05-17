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
    
    def compute_friendship_metrics(df, friends_df, label):
        # Ensure consistent types
        df["Participant-ID"] = df["Participant-ID"].astype(str)
        friends_df["Source"] = friends_df["Source"].astype(str)
        friends_df["Target"] = friends_df["Target"].astype(str)

        avg_friends_list = []
        std_friends_per_class = []

        for class_id in df["class_id"].unique():
            class_ids = df[df["class_id"] == class_id]["Participant-ID"].tolist()
            class_set = set(class_ids)
            friend_counts = []

            for student in class_ids:
                # Count friendships where both ends are in the same class
                num_friends = len(friends_df[
                    ((friends_df["Source"] == student) & (friends_df["Target"].isin(class_set))) |
                    ((friends_df["Target"] == student) & (friends_df["Source"].isin(class_set)))
                ])
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
        merged = df_alloc
        avg_scores_per_class = merged.groupby("class_id")[score_column].mean()
        std_dev = avg_scores_per_class.std()
        print(f"{label} Std Dev of {score_column}: {std_dev:.4f}")
        return std_dev

    def compute_wellbeing_std(df, label):
        prop_by_class = df.groupby("class_id")["wellbeing_label"].value_counts(normalize=True).unstack().fillna(0)
        std_dev = prop_by_class.std().mean()
        print(f"❤️‍🩹 {label} Wellbeing Std Dev: {std_dev:.4f}")
        return std_dev

    def count_disrespect_conflicts(df, disrespect_df):
        df["Participant-ID"] = df["Participant-ID"].astype(str)
        disrespect_df["Source"] = disrespect_df["Source"].astype(str)
        disrespect_df["Target"] = disrespect_df["Target"].astype(str)

        conflict_count = 0

        for class_id, group in df.groupby("class_id"):
            class_ids = set(group["Participant-ID"])

            # Filter disrespect edges where both students are in this class
            in_class_conflicts = disrespect_df[
                (disrespect_df["Source"].isin(class_ids)) & 
                (disrespect_df["Target"].isin(class_ids))
            ]

            conflict_count += len(in_class_conflicts)

        return conflict_count

    # --- Run All Metrics ---
    gpa_std_ga = compute_gpa_balance(df_ga, "GA Allocation")
    gpa_std_rand = compute_gpa_balance(df_random, "Random Allocation")

    avg_friends_ga, std_friend_ga = compute_friendship_metrics(df_ga, friends, "GA Allocation")
    avg_friends_rand, std_friend_rand = compute_friendship_metrics(df_random, friends, "Random Allocation")

    influence_std_ga = compute_std_balance(df_ga, student_scores_df, "influential_score", "GA Allocation")
    influence_std_rand = compute_std_balance(df_random, student_scores_df, "influential_score", "Random Allocation")

    isolation_std_ga = compute_std_balance(df_ga, student_scores_df, "isolated_score", "GA Allocation")
    isolation_std_rand = compute_std_balance(df_random, student_scores_df, "isolated_score", "Random Allocation")

    std_wellbeing_ga = compute_wellbeing_std(df_ga, "GA Allocation")
    std_wellbeing_rand = compute_wellbeing_std(df_random, "Random Allocation")

    bully_conflicts_ga = count_disrespect_conflicts(df_ga, Disrespect)
    bully_conflicts_rand = count_disrespect_conflicts(df_random, Disrespect)

    # --- New Observations: 5 Custom Insights ---

    def generate_friendship_coverage(df, friends_df):
        friends_df["Source"] = friends_df["Source"].astype(str)
        friends_df["Target"] = friends_df["Target"].astype(str)
        df["Participant-ID"] = df["Participant-ID"].astype(str)

        students = df["Participant-ID"].tolist()
        count_with_friends = 0

        class_groups = df.groupby("class_id")

        for class_id, group in class_groups:
            class_ids = set(group["Participant-ID"])
            for student in class_ids:
                is_friend = (
                    ((friends_df["Source"] == student) & (friends_df["Target"].isin(class_ids))) |
                    ((friends_df["Target"] == student) & (friends_df["Source"].isin(class_ids)))
                )
                if not friends_df[is_friend].empty:
                    count_with_friends += 1

        percentage = (count_with_friends / len(df)) * 100 if len(df) > 0 else 0
        return f"{percentage:.1f}% of students have at least one friend in their class."

    def generate_friendship_balance(std_dev_friends):
        return f"Friendship balance across classes shows a standard deviation of {std_dev_friends:.2f} in within-class friend counts."

    def generate_gpa_fairness(df):
        gpa_by_class = df.groupby("class_id")["Predicted_GPA"].mean()
        std_dev = gpa_by_class.std()
        min_gpa = gpa_by_class.min()
        max_gpa = gpa_by_class.max()
        return f"GPA fairness: Std Dev = {std_dev:.2f}, Min Class GPA = {min_gpa:.2f}, Max Class GPA = {max_gpa:.2f}"

    def generate_high_risk_summary(df, bully_dict):
        df["Participant-ID"] = df["Participant-ID"].astype(str)
        risk_scores = {}

        for class_id, group in df.groupby("class_id"):
            pids = set(group["Participant-ID"])

            isolation = group["isolated_score"].mean()
            influence = group["influential_score"].mean()

            conflict_count = 0
            for bully, victims in bully_dict.items():
                if bully in pids:
                    conflict_count += len(pids.intersection(set(victims)))

            # Simple scoring: more isolation, more conflicts, less influence
            risk_score = isolation + conflict_count - influence
            risk_scores[class_id] = risk_score

        worst_class = max(risk_scores, key=risk_scores.get)
        return f"Class {worst_class} shows the most high-risk social dynamics (combined isolation, low influence, and bullying overlaps)."

    def generate_wellbeing_imbalance(df):
        prop_by_class = df.groupby("class_id")["wellbeing_label"].value_counts(normalize=True).unstack().fillna(0)
        std_dev = prop_by_class.std().mean()
        return f"Wellbeing label imbalance across classes: Std Dev = {std_dev:.2f}"

    # --- Generate Observations ---
    # Generate for GA Allocation
    obs_friend_cov_ga = generate_friendship_coverage(df_ga, friends)
    obs_friend_bal_ga = generate_friendship_balance(std_friend_ga)
    obs_gpa_fairness_ga = generate_gpa_fairness(df_ga)
    obs_high_risk_ga = generate_high_risk_summary(df_ga, bully_to_victims)
    obs_wellbeing_ga = generate_wellbeing_imbalance(df_ga)

    # Generate for Random Allocation
    obs_friend_cov_rand = generate_friendship_coverage(df_random, friends)
    obs_friend_bal_rand = generate_friendship_balance(std_friend_rand)
    obs_gpa_fairness_rand = generate_gpa_fairness(df_random)
    obs_high_risk_rand = generate_high_risk_summary(df_random, bully_to_victims)
    obs_wellbeing_rand = generate_wellbeing_imbalance(df_random)

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

    # Insert GA Allocation Observations
    db.session.execute(text("""
        INSERT INTO allocation_observations (
            allocation_type,
            friendship_coverage,
            friendship_balance,
            gpa_fairness,
            high_risk_dynamics,
            wellbeing_imbalance
        ) VALUES (:type, :friend_cov, :friend_bal, :gpa_fair, :risk, :wellbeing)
    """), {
        "type": "GA",
        "friend_cov": obs_friend_cov_ga,
        "friend_bal": obs_friend_bal_ga,
        "gpa_fair": obs_gpa_fairness_ga,
        "risk": obs_high_risk_ga,
        "wellbeing": obs_wellbeing_ga
    })

    # Insert Random Allocation Observations
    db.session.execute(text("""
        INSERT INTO allocation_observations (
            allocation_type,
            friendship_coverage,
            friendship_balance,
            gpa_fairness,
            high_risk_dynamics,
            wellbeing_imbalance
        ) VALUES (:type, :friend_cov, :friend_bal, :gpa_fair, :risk, :wellbeing)
    """), {
        "type": "Random",
        "friend_cov": obs_friend_cov_rand,
        "friend_bal": obs_friend_bal_rand,
        "gpa_fair": obs_gpa_fairness_rand,
        "risk": obs_high_risk_rand,
        "wellbeing": obs_wellbeing_rand
    })


    db.session.commit()
    print("✅ All insights and observations saved to the database.")