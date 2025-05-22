# 🧠 ClassForge: AI-Powered Classroom Allocation System

ClassForge (TIRP) is a research-driven application that intelligently allocates students into classrooms to optimize academic performance, social well-being, and inclusivity. It combines **graph machine learning**, **multi-objective optimization**, and **customizable admin controls** to support school administrators in forming balanced and supportive student groups.

Project is accessible via http://207.211.144.204/

---

## 🏗️ Project Stack

| Layer      | Technology               |
|------------|---------------------------|
| Backend    | Flask (Python)            |
| Frontend   | Power BI & D3 Visuals     |
| ML Models  | PyTorch Geometric (R-GAT, R-GCN, BERT) |
| Database   | MySQL (AWS RDS)           |
| Hosting    | Cloud Linux Server        |
| Graph Tools| NetworkX                  |

---

## 🧠 Key Features

### 📊 1. Data-Driven Allocation Engine

Allocates students into balanced classes using:
- **GPA predictions** (via R-GAT)
- **Well-being scores** (via R-GCN regression)
- **Friendship networks** modeled with NetworkX
- **Bullying/conflict detection** via Leicht-Newman modularity
- **`finalallocation.py`**: GA script and logic

---

### 🧮 2. Multi-Objective Genetic Algorithm (MOGA)

Optimizes classroom assignments with respect to:
- Academic balance (GPA variance)
- Social cohesion (friendship density)
- Well-being
- Conflict and isolation mitigation

---

### 🤝 3. GNN-Based Social Modeling

- **R-GAT** for GPA prediction
- **Leicht-Newman Community Detection** for Bullying/conflict detection
- **R-GCN** for:
  - **Regression**: Well-being prediction
  - **Classification**:
    - Isolated students
    - Influential students
    - Link prediction

> All classification outputs are **soft scores** (probabilities between 0 and 1)

---

### 🧩 4. Customisation Interface

#### ✅ Manual Customisation
- Add/remove students or change class sizes
- Adjust constraint weights (e.g., prioritize well-being over GPA)
- View and compare historical allocation results

#### 🤖 AI Assistant (NLP Interface)
- Built using **BERT-based intent classification** and rule-based recommendation
- Accepts natural language input for:
  - Modifying weights
  - Locking students together/apart
  - "What if" scenario testing
- Updates allocation and provides explainable feedback

---

## 🔍 Visualisation & Insights

- **Power BI Dashboards** for GPA spread, well-being, and class comparisons
- **NetworkX Graphs** to reveal hidden subgroups and conflict zones
- **`insights.py`** compares AI allocation with random allocation to quantify improvements in:
  - GPA distribution
  - Friendship preservation
  - Isolation risk

---

## 📁 Folder Structure

```bash
TIRP/
├── .github/workflows/                # CI/CD configs (e.g., deploy.yml)
├── .vscode/                          # Editor settings
├── app/
│   ├── cache/                        # Cached artifacts
│   ├── database/                     # Database session and connection
│   ├── ml_models/                    # Core ML logic and scripts
│   │   ├── Clustering/               # Community detection modules
│   │   ├── R-GCN_files/              # GNN regression model files
│   │   ├── student_data/            # Processed input data
│   │   ├── finalallocation.py       # GA optimization logic
│   │   ├── apply_specifications.py  # Manual override for constraints
│   │   ├── r_gcnmodels.py           # Well-being regression model
│   │   ├── soft_constraints_config.json  # Custom weights and rules
│   │   ├── insights.py              # Comparison script (AI vs random)
│   │   └── [BERT model files]       # NLP assistant models
│   ├── models/
│   │   └── trained_models/          # BERT classifier & generator (intent & recommendation)
│   ├── static/                       # JS, CSS, chatbot assets
│   ├── templates/                    # HTML templates (Flask)
│   ├── visualisation/                # NetworkX outputs & Power BI exports
│   ├── __init__.py                   # App entry
│   ├── config.py                     # DB credentials and settings
│   └── routes.py                     # Flask routing
├── run.py                            # Flask app runner
├── requirements.txt                  # Python dependencies
├── tirp_db_connect.session.sql       # DB session script
└── .gitignore                        # Ignored files
