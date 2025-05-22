"""## Data Pre-Processing"""

import random
from random import randint
from random import shuffle
import copy
import numpy as np
import pandas as pd
import torch
import os
import json
from sqlalchemy import create_engine, text
import sys
import os

# Ensure parent dir is in path to allow app impo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app import create_app, db

app = create_app()

# Load soft constraint config
if os.path.exists("soft_constraints_config.json"):
    with open("soft_constraints_config.json", "r") as f:
        weights_config = json.load(f)
else:
    weights_config = {}

# --- Scale all weights except integer thresholds ---
scaled_weights = {}
for k, v in weights_config.items():
    if k == "min_friends_required":  # keep this as integer threshold
        scaled_weights[k] = int(v)
    else:
        scaled_weights[k] = (float(v) / 100) * 2  # maps 0–100 → 0.0–2.0

weights = scaled_weights

# ALL DB reads 
with app.app_context():
    participant_data     = pd.read_sql("SELECT * FROM participants", db.engine)
    student_data         = pd.read_sql("SELECT * FROM responses", db.engine)
    friends              = pd.read_sql("SELECT * FROM net_friends", db.engine)
    influence            = pd.read_sql("SELECT * FROM net_influential", db.engine)
    meaningful           = pd.read_sql("SELECT * FROM net_feedback", db.engine)
    Moretime             = pd.read_sql("SELECT * FROM net_moretime", db.engine)
    Advice               = pd.read_sql("SELECT * FROM net_advice", db.engine)
    Disrespect           = pd.read_sql("SELECT * FROM net_disrespect", db.engine)
    friendship_df        = pd.read_sql("SELECT * FROM friendship_scores", db.engine)
    student_scores_df    = pd.read_sql("SELECT * FROM student_scores", db.engine)
    gpa_df               = pd.read_sql("SELECT * FROM gpa_predictions_with_bins", db.engine)
    bully_df             = pd.read_sql("SELECT * FROM community_bully_assignments", db.engine)
    wellbeing_df         = pd.read_sql("SELECT * FROM wellbeing_classification_results", db.engine)
    start_df             = pd.read_sql("SELECT student_index, class_assigned FROM seed_allocations", db.engine)

# rename columns AFTER the block
gpa_df.rename(columns={"Student_ID": "Participant-ID"}, inplace=True)
bully_df.rename(columns={"Student_ID": "Participant-ID"}, inplace=True)
participant_data["student_id"] = (
        participant_data["student_id"]
        .astype(str)           # 32481  →  '32481'
        .str.strip()        
)
student_ids = student_data["student_id"].values

# Remove self-loops from the DataFrame
friends = friends[friends["source_student_id"] != friends["target_student_id"]]
influence = influence[influence["source_student_id"] != influence["target_student_id"]]
meaningful = meaningful[meaningful["source_student_id"] != meaningful["target_student_id"]]
Moretime = Moretime[Moretime["source_student_id"] != Moretime["target_student_id"]]
Advice = Advice[Advice["source_student_id"] != Advice["target_student_id"]]
Disrespect = Disrespect[Disrespect["source_student_id"] != Disrespect["target_student_id"]]

# Participant_data cleaning
participant_data.columns = participant_data.columns.str.strip()  # Fix spaces
participant_data['Perc_Academic'] = participant_data['perc_academic'].fillna(participant_data['perc_academic'].mean())

# --- Load starting point ---
# hetero_graph = torch.load("R-GCN_files/hetero_graph.pt", weights_only=False)
global_mean_gpa = gpa_df["Predicted_GPA"].mean()       # Global mean GPA

# --- Create ID mappings ---
id_to_index = {pid: idx for idx, pid in enumerate(participant_data["student_id"])}
index_to_id = {v: k for k, v in id_to_index.items()}

wellbeing_df["student_index"] = wellbeing_df["Participant_ID"].map(id_to_index)
wellbeing_df = wellbeing_df.dropna(subset=["student_index"])
wellbeing_df["student_index"] = wellbeing_df["student_index"].astype(int)

# Build mapping from bully to their victims using the Primary_Bully_ID field
bully_to_victims = {}
for _, row in bully_df.iterrows():
    bully_id = row["Primary_Bully_ID"]
    student_id = row["Participant-ID"]
    if row["Is_Bully"] == 0:  # Only victims
        bully_to_victims.setdefault(bully_id, []).append(student_id)

# --- Add Participant-ID columns to R-GCN Outputs ---
student_scores_df["Participant-ID"] = student_scores_df["student_index"].map(index_to_id)
friendship_df["Participant1-ID"] = friendship_df["student1"].map(index_to_id)
friendship_df["Participant2-ID"] = friendship_df["student2"].map(index_to_id)

# --- Build starting individual ---
start_individual = start_df.sort_values("student_index")["class_assigned"].tolist()

# --- Build friendship lookup ---
friendship_lookup = {}
for _, row in friendship_df.iterrows():
    i, j, score = row['student1'], row['student2'], row['friendship_score']
    friendship_lookup[(i, j)] = score
    friendship_lookup[(j, i)] = score 

def fitness(individual, weights=None):
    # --- Default weights if none provided ---
    default_weights = {
            "min_friends_required": 1,
            "friend_inclusion_weight": 1.0,
            "friend_balance_weight": 1.0,
            "influence_std_weight": 1.0,
            "isolation_std_weight": 1.0,
            "gpa_penalty_weight": 2.0,
            "bully_penalty_weight": 2.0,
            "wellbeing_penalty_weight": 1.0
        }
    if weights is None:
        weights = default_weights
    else:
        # Fill missing weights with defaults
        for key in default_weights:
            weights.setdefault(key, default_weights[key])

    num_classes = 6
    class_students = {i: [] for i in range(num_classes)}

    for student_idx, class_id in enumerate(individual):
        class_students[class_id].append(student_idx)

    # --- Friend Coverage ---
    friend_coverage_per_class = []
    friendship_counts = []

    for class_id, students in class_students.items():
        has_friend_count = 0
        friend_pair_count = 0

        for i in range(len(students)):
            student_idx = students[i]
            friends_found = sum(
                1 for other in students if other != student_idx
                and friendship_lookup.get((student_idx, other), 0) > 0.3
            )
            if friends_found >= weights["min_friends_required"]:
                has_friend_count += 1

            for j in range(i + 1, len(students)):
                pair = (students[i], students[j])
                if pair in friendship_lookup:
                    friend_pair_count += 1

        friend_coverage_per_class.append(has_friend_count / len(students) if students else 0)
        friendship_counts.append(friend_pair_count)

    avg_friend_coverage = np.mean(friend_coverage_per_class)
    friend_coverage_imbalance = np.std(friend_coverage_per_class)

    friend_inclusion_score = (
    weights["friend_inclusion_weight"] * avg_friend_coverage
    - weights["friend_balance_weight"] * friend_coverage_imbalance
    )

    # --- R-GCN Scores ---
    influential_avgs, isolated_avgs, gpa_avgs = [], [], []

    for class_id, students in class_students.items():
        if students:
            influential_avgs.append(student_scores_df.loc[students, "influential_score"].mean())
            isolated_avgs.append(student_scores_df.loc[students, "isolated_score"].mean())

            class_ids = [index_to_id[i] for i in students]
            matched_gpa = gpa_df[gpa_df["Participant-ID"].isin(class_ids)]["Predicted_GPA"]
            if not matched_gpa.empty:
                gpa_avgs.append(matched_gpa.mean())

    influence_std = pd.Series(influential_avgs).std()
    isolated_std = pd.Series(isolated_avgs).std()

    global_gpa_std = gpa_df["Predicted_GPA"].std()

    gpa_penalty = (sum(((avg - global_mean_gpa) / global_gpa_std) ** 2 for avg in gpa_avgs) / num_classes if gpa_avgs else 0)

    # --- Bully-Victim Overlap ---
    total_bully_victim_overlap = 0
    for class_id, students in class_students.items():
        class_pids = [index_to_id[i] for i in students]
        class_pid_set = set(class_pids)
        for bully_pid in class_pid_set:
            victims = bully_to_victims.get(bully_pid, [])
            overlap = len(set(victims) & class_pid_set)
            total_bully_victim_overlap += overlap

    avg_bully_victim_overlap = total_bully_victim_overlap / num_classes
    bully_penalty = avg_bully_victim_overlap ** 2

    # --- Wellbeing Balance ---
    low_counts, high_counts = [], []
    for class_id, students in class_students.items():
        class_pids = [index_to_id[i] for i in students]
        class_labels = wellbeing_df[wellbeing_df["Participant_ID"].isin(class_pids)]["wellbeing_label"]
        low_counts.append((class_labels == "low").sum())
        high_counts.append((class_labels == "high").sum())

    low_std = np.std(low_counts)
    high_std = np.std(high_counts)
    wellbeing_penalty = low_std + high_std

    # --- Final Fitness ---
    fitness_score = (
        + friend_inclusion_score
        - weights["influence_std_weight"] * influence_std
        - weights["isolation_std_weight"] * isolated_std
        - weights["gpa_penalty_weight"] * gpa_penalty
        - weights["bully_penalty_weight"] * avg_bully_victim_overlap
        - weights["wellbeing_penalty_weight"] * wellbeing_penalty
        - weights["bully_penalty_weight"] * bully_penalty
    )

    return fitness_score

# --- Repair Function ---
def repair(individual, num_classes=6, max_size=30, max_diff=5):

    # Step 1: Count students in each class
    class_to_students = {i: [] for i in range(num_classes)}
    for sid, cid in enumerate(individual):
        class_to_students[cid].append(sid)

    # Step 2: Enforce class size ≤ max_size
    for cid in range(num_classes):
        while len(class_to_students[cid]) > max_size:
            sid = random.choice(class_to_students[cid])
            # Find a class with available space
            target_cid = min((k for k in range(num_classes) if len(class_to_students[k]) < max_size),
                            key=lambda k: len(class_to_students[k]), default=None)
            if target_cid is not None and target_cid != cid:
                individual[sid] = target_cid
                class_to_students[cid].remove(sid)
                class_to_students[target_cid].append(sid)

    # Step 3: Balance class sizes (≤ max_diff)
    def get_max_min_class():
        sizes = [(cid, len(students)) for cid, students in class_to_students.items()]
        sizes.sort(key=lambda x: x[1])
        return sizes[-1][0], sizes[0][0]

    max_cid, min_cid = get_max_min_class()
    while len(class_to_students[max_cid]) - len(class_to_students[min_cid]) > max_diff:
        sid = random.choice(class_to_students[max_cid])
        individual[sid] = min_cid
        class_to_students[max_cid].remove(sid)
        class_to_students[min_cid].append(sid)
        max_cid, min_cid = get_max_min_class()

    return individual

# --- Crossover Function ---
def crossover(parent1, parent2):
    """Uniform crossover: for each student, randomly pick parent1 or parent2's assignment."""
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

# --- Mutation Function ---
def mutate(individual, num_swaps=2):
    """Randomly swap the classes of num_swaps pairs of students."""
    individual = individual.copy()  # Don't mutate original
    for _ in range(num_swaps):
        i, j = random.sample(range(len(individual)), 2)  # Randomly pick two students
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# --- Tournament Selection ---
def tournament_selection(population, fitnesses, tournament_size=3):
    """Select individuals via tournament selection."""
    selected = []
    for _ in range(len(population)):
        competitors = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(competitors, key=lambda x: x[1])  # Highest fitness wins
        selected.append(copy.deepcopy(winner[0]))
    return selected

# === Genetic Algorithm for Classroom Allocation ===
# === PART 3: Main Evolution Loop and Final Output ===

# --- Parameters ---
population_size = 50
max_generations = 100
num_elites = 2
num_swaps_per_mutation = 2
tournament_size = 3

# --- Optional: define or receive weights ---
# This can later be overridden by frontend input
# weights = None  or a dictionary of custom weights

# --- Initialize Population ---
def initialize_population(start_individual, population_size):
    population = [start_individual]  # Start with seed

    for _ in range(population_size - 1):
        new_individual = mutate(start_individual, num_swaps=num_swaps_per_mutation)
        population.append(new_individual)

    return population

population = initialize_population(start_individual, population_size)

# --- Main Evolution Loop ---
best_fitness_over_time = []

for generation in range(max_generations):
    # 1. Evaluate fitness
    fitnesses = [fitness(ind, weights) for ind in population]

    # 2. Elite Preservation
    elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:num_elites]
    elites = [population[i] for i in elite_indices]

    # 3. Selection
    selected = tournament_selection(population, fitnesses, tournament_size)

    # 4. Crossover
    children = []
    for i in range(0, len(selected) - 1, 2):
        parent1 = selected[i]
        parent2 = selected[i + 1]
        child = crossover(parent1, parent2)
        children.append(child)

    # 5. Mutation + Repair
    children = [mutate(child, num_swaps=num_swaps_per_mutation) for child in children]
    children = [repair(child, num_classes=6, max_size=30, max_diff=5) for child in children]

    # 6. Form Next Generation
    population = elites + children

    # 7. Record Best Fitness
    best_fitness = max(fitnesses)
    best_fitness_over_time.append(best_fitness)

    print(f"Generation {generation+1}/{max_generations} - Best Fitness: {best_fitness:.4f}")

# --- Get Best Individual ---
final_fitnesses = [fitness(ind, weights) for ind in population]
best_individual = population[final_fitnesses.index(max(final_fitnesses))]

with app.app_context():
    # Clear existing records
    db.session.execute(text("DELETE FROM allocations"))

    for i, class_id in enumerate(best_individual):
        participant_id = index_to_id[i]
        predicted_gpa = gpa_df.loc[gpa_df['Participant-ID'] == participant_id, 'Predicted_GPA'].values[0] if participant_id in gpa_df['Participant-ID'].values else None
        influential_score = student_scores_df.loc[i, "influential_score"]
        isolated_score = student_scores_df.loc[i, "isolated_score"]
        wellbeing_label = wellbeing_df.set_index("Participant_ID").reindex([participant_id])["wellbeing_label"].values[0]

        db.session.execute(
            text("INSERT INTO allocations (class_id, student_id) VALUES (:class_id, :student_id)"),
            {"class_id": int(class_id), "student_id": str(participant_id)}
        )

    db.session.commit()
    
    print("Final class allocations inserted directly into the database.")

if __name__ == "__main__":
    print("Starting class allocation algorithm...")
    print("Using optimization parameters from soft_constraints_config.json")
    # The script will execute from top to bottom when run
    print("Class allocation completed successfully!")
    print("Student allocations have been saved to the database.")
    # Return successful exit code
    exit(0)