# AI-Powered Student Classroom Allocation System

This project intelligently allocates students into balanced classrooms using machine learning, social network analysis, and optimization. It aims to improve student outcomes by considering both academic and social well-being.

## Objectives
- Balance academic performance between classrooms
- Maximize student well-being and social connectivity
- Avoid bullying and isolation
- Allow user-defined constraints and preferences

## Key Features

### 1. Machine Learning (GNNs)
- Predict student **GPA**, **well-being**, and **social influence**
- Model based on directed social networks using **Graph Attention Networks** (GAT) and **HeteroGNNs**
- Framework: `PyTorch Geometric`

### 2. Optimization with Genetic Algorithm
- Multi-objective GA (NSGA-II) assigns students to classes
- Hard constraints: max students per class
- Soft objectives: minimize academic gap, maximize friendship ties, spread well-being

### 3. Interactive Visualization
- **Power BI Dashboards** for performance summaries
- **NetworkX Graphs** showing student friendships and allocations
- **Flask Web Interface** to visualize, interact with, and customize allocations

### 4. Database Integration
- Student and class info stored via **Supabase (PostgreSQL)**

### 5. Planned: NLP Assistant
- Uses LLM to help school staff interpret network structures
- Natural language explanations of clusters and recommendations

## Tech Stack
- Python (PyTorch, Scikit-learn, NetworkX, DEAP, Flask)
- Power BI for visual dashboards
- Supabase for backend database
- Jupyter for experimentation
