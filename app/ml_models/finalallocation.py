"""## Data Pre-Processing"""

import random
import copy
import numpy as np
import pandas as pd
import torch

# Load student survey responses (features)
student_data = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="responses")

participant_data = pd.read_excel("student_data/Student Survey - Jan.xlsx", sheet_name="participants")

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
global_mean_gpa = gpa_df["Predicted_GPA"].mean()       # 🆕 Global mean GPA

# --- Create ID mappings ---
id_to_index = {pid: idx for idx, pid in enumerate(participant_data["Participant-ID"])}
index_to_id = {v: k for k, v in id_to_index.items()}

# Loading Community Detection Files
# Re-load the community_bully_assignments file
bully_df = pd.read_csv("ha_outputs/community_bully_assignments.csv")
bully_df.rename(columns={"Student_ID": "Participant-ID"}, inplace=True)

# --- Load Wellbeing Classification Results (🆕)
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

# Display a few entries to confirm structure
list(bully_to_victims.items())[:3]

# --- Add Participant-ID columns to R-GCN Outputs ---
student_scores_df["Participant-ID"] = student_scores_df["student_index"].map(index_to_id)
friendship_df["Participant1-ID"] = friendship_df["student1"].map(index_to_id)
friendship_df["Participant2-ID"] = friendship_df["student2"].map(index_to_id)

# --- Build starting individual ---
start_df = pd.read_excel("student_data/seed_allocations.xlsx")
start_individual = start_df.sort_values("student_index")["class_assigned"].tolist()

# --- Build friendship lookup ---
friendship_lookup = {}
for _, row in friendship_df.iterrows():
    i, j, score = row['student1'], row['student2'], row['friendship_score']
    friendship_lookup[(i, j)] = score
    friendship_lookup[(j, i)] = score  # Symmetric

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
    friendship_balance_penalty = np.std(friendship_counts)

    friend_inclusion_score = (
        weights["friend_inclusion_weight"] * avg_friend_coverage -
        weights["friend_balance_weight"] * friend_coverage_imbalance
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
weights = None  # or a dictionary of custom weights

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

# --- Get Best Individual ---
final_fitnesses = [fitness(ind, weights) for ind in population]
best_individual = population[final_fitnesses.index(max(final_fitnesses))]

# --- Create Final Output DataFrame ---
results = []

for i, class_id in enumerate(best_individual):
    results.append({
        "student_index": i,
        "participant_id": index_to_id[i],
        "final_class_assigned": class_id,
        "Predicted_GPA": gpa_df.loc[gpa_df['Participant-ID'] == index_to_id[i], 'Predicted_GPA'].values[0] if index_to_id[i] in gpa_df['Participant-ID'].values else None,
        "influential_score": student_scores_df.loc[i, "influential_score"],
        "isolated_score": student_scores_df.loc[i, "isolated_score"],
        "wellbeing_label": wellbeing_df.set_index("Participant-ID").reindex([index_to_id[i] for i in best_individual])["wellbeing_label"].values
    })

final_df = pd.DataFrame(results)

# --- Save Final Output ---
final_df.to_excel("final_class_allocations_ga.xlsx", index=False)
print("✅ Final class allocations saved to 'final_class_allocations_ga.xlsx'")