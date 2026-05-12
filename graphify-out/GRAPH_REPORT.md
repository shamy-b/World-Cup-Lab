# Graph Report - .  (2026-05-12)

## Corpus Check
- Corpus is ~7,967 words - fits in a single context window. You may not need a graph.

## Summary
- 118 nodes · 181 edges · 14 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 70 edges (avg confidence: 0.59)
- Token cost: 1,500 input · 500 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Predictive Modeling|Predictive Modeling]]
- [[_COMMUNITY_Data Ingestion & Normalization|Data Ingestion & Normalization]]
- [[_COMMUNITY_Deep Learning & Sequence Modeling|Deep Learning & Sequence Modeling]]
- [[_COMMUNITY_Application & Simulation UI|Application & Simulation UI]]
- [[_COMMUNITY_Elo Rating Logic|Elo Rating Logic]]
- [[_COMMUNITY_Core Platform Vision|Core Platform Vision]]
- [[_COMMUNITY_Dynamic Ratings|Dynamic Ratings]]
- [[_COMMUNITY_Embeddings|Embeddings]]
- [[_COMMUNITY_ML Methodology|ML Methodology]]
- [[_COMMUNITY_Statistical Analytics|Statistical Analytics]]
- [[_COMMUNITY_Graph Learning|Graph Learning]]
- [[_COMMUNITY_Explainable AI|Explainable AI]]
- [[_COMMUNITY_Web Backend|Web Backend]]
- [[_COMMUNITY_Analytics Dashboard|Analytics Dashboard]]

## God Nodes (most connected - your core abstractions)
1. `ChronosIngestor` - 34 edges
2. `EloSystem` - 23 edges
3. `FeatureEngineer` - 21 edges
4. `ChronosPredictor` - 12 edges
5. `WorldCupSimulator` - 11 edges
6. `FootballSequenceDataset` - 10 edges
7. `FootballLSTM` - 8 edges
8. `train_dl_model()` - 8 edges
9. `TeamSequencer` - 8 edges
10. `TeamNormalizer` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Dynamic Elo Rating System for Chronos.     Includes tournament weighting, margin` --uses--> `ChronosIngestor`  [INFERRED]
  C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\elo.py → C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\ingestion.py
- `Returns the K-factor for a tournament. Falls back to keyword matching.` --uses--> `ChronosIngestor`  [INFERRED]
  C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\elo.py → C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\ingestion.py
- `Iterates through match history and generates Elo features.` --uses--> `ChronosIngestor`  [INFERRED]
  C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\elo.py → C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\ingestion.py
- `Advanced Feature Engineering Engine for Chronos.     Generates rolling statistic` --uses--> `ChronosIngestor`  [INFERRED]
  C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\features.py → C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\ingestion.py
- `Master function: generates all features in one pass.` --uses--> `ChronosIngestor`  [INFERRED]
  C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\features.py → C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\src\ingestion.py

## Communities

### Community 0 - "Predictive Modeling"
Cohesion: 0.13
Nodes (11): ChronosPredictor, Chronological split: train < 2018, validation 2018-2021, test >= 2022., Optuna hyperparameter search for XGBoost., Optuna hyperparameter search for LightGBM., Full training pipeline with ensemble., Advanced ML Engine for Chronos.     Ensemble of XGBoost + LightGBM with Optuna h, Full pipeline: ingest -> features -> elo -> feature selection., FeatureEngineer (+3 more)

### Community 1 - "Data Ingestion & Normalization"
Cohesion: 0.11
Nodes (10): Returns prestige score with keyword fallback., ChronosIngestor, Loads all CSV files from the dataset directory., Standardizes datatypes and validates core constraints., Generates unique Match IDs following the blueprint: YEAR_HOME_AWAY_HASH., Data ingestion engine for the Chronos Football Intelligence Platform.     Handle, Maps all historical names to their current 'Successor' names.         This is us, Returns all former names for a given modern team. (+2 more)

### Community 2 - "Deep Learning & Sequence Modeling"
Cohesion: 0.15
Nodes (9): Dataset, FootballLSTM, Research-grade LSTM for Latent Football Knowledge Extraction., train_dl_model(), FootballSequenceDataset, Prepares sequential data for LSTM/Transformer models., Groups matches by team and creates rolling sequence windows., PyTorch Dataset for football match sequences.     Converts historical matches in (+1 more)

### Community 3 - "Application & Simulation UI"
Cohesion: 0.19
Nodes (5): load_engine(), Precalculates all possible matchups for lightning-fast Monte Carlo., Extracts the most recent feature state for every team., Predicts probability of team1 advancing against team2 at a neutral venue., WorldCupSimulator

### Community 4 - "Elo Rating Logic"
Cohesion: 0.24
Nodes (4): EloSystem, Iterates through match history and generates Elo features., Dynamic Elo Rating System for Chronos.     Includes tournament weighting, margin, Returns the K-factor for a tournament. Falls back to keyword matching.

### Community 5 - "Core Platform Vision"
Cohesion: 0.25
Nodes (8): Chronos Engine, Data Ingestion System, Feature Engineering Engine, Historical Normalization Engine, International Football Intelligence Platform, Monte Carlo Simulations, Simulation Engine, Python Analytics Stack

### Community 6 - "Dynamic Ratings"
Cohesion: 1.0
Nodes (2): Dynamic Rating Systems, Elo Rating System

### Community 7 - "Embeddings"
Cohesion: 1.0
Nodes (2): Deep Learning System, Team Embeddings

### Community 8 - "ML Methodology"
Cohesion: 1.0
Nodes (2): Chronological Splitting Rule, Machine Learning System

### Community 15 - "Statistical Analytics"
Cohesion: 1.0
Nodes (1): Statistical Analytics Engine

### Community 16 - "Graph Learning"
Cohesion: 1.0
Nodes (1): Graph Learning System

### Community 17 - "Explainable AI"
Cohesion: 1.0
Nodes (1): Explainable AI Engine

### Community 18 - "Web Backend"
Cohesion: 1.0
Nodes (1): Django REST Backend

### Community 19 - "Analytics Dashboard"
Cohesion: 1.0
Nodes (1): Interactive Analytics Dashboard

## Knowledge Gaps
- **24 isolated node(s):** `Data ingestion engine for the Chronos Football Intelligence Platform.     Handle`, `Loads all CSV files from the dataset directory.`, `Standardizes datatypes and validates core constraints.`, `Generates unique Match IDs following the blueprint: YEAR_HOME_AWAY_HASH.`, `PyTorch Dataset for football match sequences.     Converts historical matches in` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Dynamic Ratings`** (2 nodes): `Dynamic Rating Systems`, `Elo Rating System`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Embeddings`** (2 nodes): `Deep Learning System`, `Team Embeddings`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ML Methodology`** (2 nodes): `Chronological Splitting Rule`, `Machine Learning System`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Statistical Analytics`** (1 nodes): `Statistical Analytics Engine`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Graph Learning`** (1 nodes): `Graph Learning System`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Explainable AI`** (1 nodes): `Explainable AI Engine`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Web Backend`** (1 nodes): `Django REST Backend`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Analytics Dashboard`** (1 nodes): `Interactive Analytics Dashboard`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ChronosIngestor` connect `Data Ingestion & Normalization` to `Predictive Modeling`, `Application & Simulation UI`, `Elo Rating Logic`?**
  _High betweenness centrality (0.297) - this node is a cross-community bridge._
- **Why does `train_dl_model()` connect `Deep Learning & Sequence Modeling` to `Predictive Modeling`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Why does `EloSystem` connect `Elo Rating Logic` to `Predictive Modeling`, `Data Ingestion & Normalization`, `Application & Simulation UI`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `ChronosIngestor` (e.g. with `ChronosPredictor` and `Advanced ML Engine for Chronos.     Ensemble of XGBoost + LightGBM with Optuna h`) actually correct?**
  _`ChronosIngestor` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `EloSystem` (e.g. with `ChronosPredictor` and `Advanced ML Engine for Chronos.     Ensemble of XGBoost + LightGBM with Optuna h`) actually correct?**
  _`EloSystem` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `FeatureEngineer` (e.g. with `ChronosPredictor` and `Advanced ML Engine for Chronos.     Ensemble of XGBoost + LightGBM with Optuna h`) actually correct?**
  _`FeatureEngineer` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ChronosPredictor` (e.g. with `ChronosIngestor` and `FeatureEngineer`) actually correct?**
  _`ChronosPredictor` has 3 INFERRED edges - model-reasoned connections that need verification._