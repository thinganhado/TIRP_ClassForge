# ClassForge - AI-Powered Class Allocation System

ClassForge is an intelligent class allocation system designed to optimize student placement based on multiple factors including academic performance, wellbeing, social dynamics, and friendship networks.

## AI Assistant Feature

The AI Assistant (Customisation Agent) uses advanced Natural Language Processing to understand user requests and translate them into specific algorithm parameters. It implements a multi-level approach with BERT models and graceful degradation to simpler methods when necessary.

### Key Features

- **Natural Language Input**: Users can express their allocation priorities in plain English
- **Smart Parameter Adjustment**: Automatically adjusts optimization weights and parameters
- **Recommendation Engine**: Provides context-aware suggestions based on student data
- **Constraint Validation**: Ensures requested constraints are feasible and reasonable
- **Integration with Core System**: Changes made through the AI are applied to the allocation engine

## Social Network Analysis (SNA) Capabilities

The system now includes powerful Social Network Analysis features to better understand and optimize student social dynamics:

### SNA Features

- **Network Structure Analysis**: Calculate metrics including density, clustering coefficients and connectivity
- **Centrality Measures**: Identify influential students based on degree, betweenness, and closeness centrality
- **Community Detection**: Identify natural friendship groups and clusters
- **Isolation Detection**: Find potentially socially isolated students
- **Social Recommendations**: Get data-driven recommendations for class allocation based on social patterns

### SNA API Endpoints

- **POST /api/network/analyze**: Analyze overall network structure and identify key metrics
- **POST /api/network/isolated**: Identify students at risk of social isolation
- **POST /api/network/communities**: Detect and analyze friendship communities
- **GET /api/network/recommendations**: Get network-based optimization recommendations

### Example SNA Insights

```
- "Some students have high betweenness centrality, acting as bridges between different social groups."
- "The network has 3 separate groups with no connections between them."
- "High clustering coefficient indicates strong friend groups forming tight-knit communities."
```

## Technical Implementation

The AI Assistant uses a transfer learning approach:

1. **Base Model**: Pre-trained language models including BERT and DistilBERT
2. **Fine-tuning**: Model fine-tuned on teacher comments and constraint recommendations
3. **Last Layer Training**: The final layer specifically trained on class optimization data
4. **CSV Training Data**: Teacher comments used to train context-aware responses

The Social Network Analysis uses:
1. **NetworkX**: Graph algorithms for social network calculations
2. **Community Detection**: Louvain and modularity-based algorithms
3. **Centrality Metrics**: Degree, closeness, and betweenness centrality

## Configuration Parameters

The AI Assistant can modify the following parameters:

```json
{
  "gpa_penalty_weight": 30,
  "wellbeing_penalty_weight": 50,
  "bully_penalty_weight": 60,
  "influence_std_weight": 60,
  "isolated_std_weight": 60,
  "min_friends_required": 1,
  "friend_inclusion_weight": 50,
  "friendship_balance_weight": 60,
  "prioritize_academic": 5,
  "prioritize_wellbeing": 4,
  "prioritize_bullying": 3,
  "prioritize_social_influence": 2,
  "prioritize_friendship": 1
}
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python run.py` (from the TIRP directory)
3. Access the web interface at `http://localhost:5000`

## Using the Social Network Analysis Features

To use the SNA features, you can:

1. **Through the Web Interface**: Use the AI Assistant chat and ask questions about social networks, communities, or isolated students

2. **Through the API**: Send relationship data to the API endpoints in the following format:
```json
{
  "relationships": [
    {"student1": "student_id_1", "student2": "student_id_2", "strength": 0.8},
    {"student1": "student_id_1", "student2": "student_id_3", "strength": 0.5},
    ...
  ]
}
```

## Documentation

For more detailed technical information, see `AI_ASSISTANT_DOCUMENTATION.md`
