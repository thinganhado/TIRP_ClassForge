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

# Ensure parent dir is in path to allow app import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app import create_app, db
from app.database.spec_endpoint import HardConstraint

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

# Load student survey responses (features)
student_data = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="responses")

participant_data = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="participants")

participant_data["Participant-ID"] = (
        participant_data["Participant-ID"]
        .astype(str)           # 32481  →  '32481'
        .str.strip()           # drop accidental spaces
)

# Load student relationships (edges)
friends = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_0_Friends")

influence = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_1_Influential")

meaningful = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_2_Feedback")

Moretime = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_3_MoreTime")

Advice = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_4_Advice")

Disrespect = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_5_Disrespect")

Activity = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="net_affiliation_0_SchoolActivit")

student_ids = student_data["Participant-ID"].values

# Remove self-loops from the DataFrame
friends = friends[friends["Source"] != friends["Target"]]
influence = influence[influence["Source"] != influence["Target"]]
meaningful = meaningful[meaningful["Source"] != meaningful["Target"]]
Moretime = Moretime[Moretime["Source"] != Moretime["Target"]]
Advice = Advice[Advice["Source"] != Advice["Target"]]
Disrespect = Disrespect[Disrespect["Source"] != Disrespect["Target"]]
Activity = Activity[Activity["Source"] != Activity["Target"]]

# Participant_data cleaning
participant_data.columns = participant_data.columns.str.strip()  # Fix spaces
participant_data['Perc_Academic'] = participant_data['Perc_Academic'].fillna(participant_data['Perc_Academic'].mean())


# --- Load starting point ---
hetero_graph = torch.load("R-GCN_files/hetero_graph.pt", weights_only=False)
friendship_df = pd.read_excel("R-GCN_files/friendship_scores.xlsx")
student_scores_df = pd.read_excel("R-GCN_files/student_scores.xlsx")


# --- Load predicted GPA file (🆕)
gpa_df = pd.read_csv("ha_outputs/gpa_predictions_with_bins.csv")  # Must include 'Participant-ID' and 'Predicted_GPA'
gpa_df.rename(columns={"Student_ID": "Participant-ID"}, inplace=True)
global_mean_gpa = gpa_df["Predicted_GPA"].mean()       # Global mean GPA

# --- Create ID mappings ---
id_to_index = {pid: idx for idx, pid in enumerate(participant_data["Participant-ID"])}
index_to_id = {v: k for k, v in id_to_index.items()}

# --- Convert hard_constraints to index ---
def _idx_sets_from_db():
    """Fetch latest hard-constraint record and convert the IDs to indices
       understood by the GA.  Returns (pairs_idx, moves_idx).
    """
    with app.app_context():
        latest = (
            HardConstraint
            .query
            .order_by(HardConstraint.id.desc())
            .first()
        )

    if not latest:                       # no constraints stored yet
        return [], []

    separate_ids = [[str(s) for s in grp] for grp in latest.separate_pairs]
    move_ids     = [{"sid": str(m["sid"]), "cls": int(m["cls"])}
                    for m in latest.forced_moves]

    # ---- convert to index space ----
    pairs_idx = [tuple(id_to_index[sid] for sid in grp) for grp in separate_ids]
    moves_idx = [(id_to_index[m["sid"]], m["cls"])       for m in move_ids]

    return pairs_idx, moves_idx

SEPARATE_IDX, MOVE_IDX = _idx_sets_from_db()

# ─── DEBUG 1 ───────────────────────────────────────────────
print("DBG-1  separate_idx", SEPARATE_IDX[:3], "…", file=sys.stderr)
print("DBG-1  move_idx     ", MOVE_IDX[:3],      file=sys.stderr)
assert all(isinstance(i, int) for grp in SEPARATE_IDX for i in grp), "pair still str"
assert all(isinstance(m[0], int) and isinstance(m[1], int) for m in MOVE_IDX), "move still str"
# ──────────────────────────────────────────────────────────

# ---------------------------------------------------------
def _enforce_hard(individual, num_classes=6):
    """
    Return a *copy* of ``individual`` with
    1. every move-request applied, and
    2. every separate-group split across different classes.
    Capacity is *not* handled here – size balancing is left to the
    main `repair()` logic.
    """
    ind = individual.copy()

    # ----- 1. force moves -----
    for sid, target_cls in MOVE_IDX:
        ind[sid] = target_cls

    # build class → [students] dict once
    bin_by_class = {c: [] for c in range(num_classes)}
    for sid, cid in enumerate(ind):
        bin_by_class[cid].append(sid)

    # ----- 2. split separate groups -----
    for group in SEPARATE_IDX:
        for cid in range(num_classes):
            members = [s for s in bin_by_class[cid] if s in group]
            while len(members) > 1:             # clash found
                sid = members.pop()             # take one of them
                # pick the first class that has no other member of the group
                dest = next(k for k in range(num_classes)
                            if all(x not in group for x in bin_by_class[k]))
                ind[sid] = dest
                bin_by_class[cid].remove(sid)
                bin_by_class[dest].append(sid)

    return ind
# ---------------------------------------------------------

# Loading Community Detection Files
# Re-load the community_bully_assignments file
bully_df = pd.read_csv("ha_outputs/community_bully_assignments.csv")
bully_df.rename(columns={"Student_ID": "Participant-ID"}, inplace=True)

# --- Load Wellbeing Classification Results
wellbeing_df = pd.read_csv("Clustering/output/wellbeing_classification_results.csv")

# Map Participant-ID to student_index
wellbeing_df["student_index"] = wellbeing_df["Participant-ID"].map(id_to_index)

# Drop any rows where student_index couldn't be matched
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
start_df = pd.read_excel("student_data/seed_allocations.xlsx")
start_individual = start_df.sort_values("student_index")["class_assigned"].tolist()

# ─── DEBUG 2  (put this **after** start_individual is defined) ─────────
if SEPARATE_IDX or MOVE_IDX:
    probe    = start_individual.copy()
    enforced = _enforce_hard(probe)

    # every forced-move still correct?
    for sid, target in MOVE_IDX:
        assert enforced[sid] == target, f"forced move NOT applied for {sid}"

    # every separate group split?
    for grp in SEPARATE_IDX:
        assert len({enforced[s] for s in grp}) == len(grp), \
               f"group still collides {grp}"

    print("DBG-2  _enforce_hard OK", file=sys.stderr)
# ───────────────────────────────────────────────────────────────

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
        "friend_inclusion_weight": 1.5,
        "friend_balance_weight": 1.0,
        "influence_std_weight": 1.5,
        "isolation_std_weight": 1.5,
        "gpa_penalty_weight": 1.5,
        "bully_penalty_weight": 2.0,
        "wellbeing_penalty_weight": 1.5
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

    gpa_penalty = sum((avg - global_mean_gpa) ** 2 for avg in gpa_avgs) / num_classes if gpa_avgs else 0

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

    # --- Wellbeing Balance ---
    low_counts, high_counts = [], []
    for class_id, students in class_students.items():
        class_pids = [index_to_id[i] for i in students]
        class_labels = wellbeing_df[wellbeing_df["Participant-ID"].isin(class_pids)]["wellbeing_label"]
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
    )

    return fitness_score

# --- Repair Function ---
def repair(individual, num_classes=6, max_size=30, max_diff=5):
    # --- hard rules first ---
    individual = _enforce_hard(individual, num_classes)

        # ─── DEBUG 4 (post–hard) ──────────────────────────────
    if SEPARATE_IDX or MOVE_IDX:
        # stop immediately if something went wrong
        for sid, target in MOVE_IDX:
            if individual[sid] != target:
                raise RuntimeError(f"DBG-4 forced move lost! sid {sid}")
        for grp in SEPARATE_IDX:
            if len({individual[s] for s in grp}) != len(grp):
                raise RuntimeError(f"DBG-4 separation lost! grp {grp}")
    # ───────────────────────────────────────────────────────

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
    seed = _enforce_hard(start_individual)
    population = [seed]

        # ─── DEBUG 3 (first seed) ──────────────────────────────
    if SEPARATE_IDX or MOVE_IDX:
        print("DBG-3  seed sample", seed[:30], file=sys.stderr)
    # ───────────────────────────────────────────────────────

    for _ in range(population_size - 1):
        child = mutate(seed, num_swaps=num_swaps_per_mutation)
        child = _enforce_hard(child)          # keep mutation legal
        population.append(child)

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

    if generation == 0:          # only once is enough
        for child in children[:3]:
            repair(child)        # will raise if broken
        print("DBG-5  gen-0 children OK", file=sys.stderr)

    # 5. Mutation + Repair
    children = [mutate(child, num_swaps=num_swaps_per_mutation) for child in children]
    children = [repair(child, num_classes=6, max_size=30, max_diff=5) for child in children]

    # 6. Form Next Generation
    population = elites + children

    # 7. Record Best Fitness
    best_fitness = max(fitnesses)
    best_fitness_over_time.append(best_fitness)

# --- Get Best Individual ---
final_fitnesses = [fitness(ind, weights) for ind in population]
best_individual = population[final_fitnesses.index(max(final_fitnesses))]

# --- Create Final Output DataFrame ---
app = create_app()

with app.app_context():
    # Clear existing records
    db.session.execute(text("DELETE FROM allocations"))

    for i, class_id in enumerate(best_individual):
        participant_id = index_to_id[i]
        predicted_gpa = gpa_df.loc[gpa_df['Participant-ID'] == participant_id, 'Predicted_GPA'].values[0] if participant_id in gpa_df['Participant-ID'].values else None
        influential_score = student_scores_df.loc[i, "influential_score"]
        isolated_score = student_scores_df.loc[i, "isolated_score"]
        wellbeing_label = wellbeing_df.set_index("Participant-ID").reindex([participant_id])["wellbeing_label"].values[0]

        db.session.execute(
            text("INSERT INTO allocations (class_id, student_id) VALUES (:class_id, :student_id)"),
            {"class_id": int(class_id), "student_id": str(participant_id)}
        )

    db.session.commit()
    
    print("Final class allocations inserted directly into the database.")

    random_allocations = []

    # Build a list of participant IDs from your GA allocation loop
    participant_ids = [index_to_id[i] for i in range(len(best_individual))]

    # Shuffle participant IDs to randomize order
    shuffle(participant_ids)

    num_classes = 6
    min_students_per_class = 25
    total_required = num_classes * min_students_per_class

    if len(participant_ids) < total_required:
        raise ValueError("Not enough students to guarantee at least 25 per class.")

    # Step 1: Assign minimum 25 students per class
    balanced_allocations = []
    for class_id in range(num_classes):
        for _ in range(min_students_per_class):
            pid = participant_ids.pop()
            balanced_allocations.append((pid, class_id))

    # Step 2: Assign remaining students randomly but evenly
    for i, pid in enumerate(participant_ids):
        class_id = i % num_classes
        balanced_allocations.append((pid, class_id))

    # Optional: Clear old random allocations
    db.session.execute(text("DELETE FROM random_allo"))

    # Insert into database
    for pid, class_id in balanced_allocations:
        db.session.execute(
            text("INSERT INTO random_allo (participant_id, class_id) VALUES (:pid, :cid)"),
            {"pid": str(pid), "cid": int(class_id)}
        )

    db.session.commit()
    print("Random allocation inserted into the random_allo table.")

if __name__ == "__main__":
    print("Starting class allocation algorithm...")
    print("Using optimization parameters from soft_constraints_config.json")
    # The script will execute from top to bottom when run
    print("Class allocation completed successfully!")
    print("Student allocations have been saved to the database.")
    # Return successful exit code
    exit(0)