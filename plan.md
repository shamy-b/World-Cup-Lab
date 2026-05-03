# 🏆 Master Project Plan: "Global Pitch Analytics" 
*(An Enterprise-Grade Football Intelligence Platform)*

## 1. Project Overview
To build a truly **massive, portfolio-defining project**, we are expanding "Chrono-Pitch" into a comprehensive **Sports Analytics Platform**. This platform will combine advanced Machine Learning (Elo, Poisson, XGBoost), rigorous Data Engineering (geopolitical normalization), and a multi-page interactive web application.

The final product will have three core features:
1. **The Match Simulator ("Chrono-Pitch"):** Simulate hypothetical matchups across eras using a custom time-decaying Bayesian Elo model.
2. **Era Dominance & Geospatial Flow:** A global visualization of football's spread and the rise/fall of empires (e.g., the Hungarian Golden Team vs. modern Spain).
3. **The Player "Clutch" Engine:** Deep analytics isolating the true value of individual goalscorers and penalty shootout psychology.

---

## 2. Professional Directory Structure
To support a massive codebase, we will use a production-grade structure:

```text
C:\Users\Asus\OneDrive\Desktop\FIFA World Cup from 1872 to 2026\
│
├── dataset/                     # Raw CSV data
│
├── src/                         # Core Python Packages
│   ├── __init__.py
│   ├── data/
│   │   └── etl_pipeline.py      # Object-oriented data cleaning and joining
│   ├── features/
│   │   └── geopolitics.py       # Maps USSR -> Russia, etc. based on year
│   ├── models/
│   │   ├── elo_system.py        # Advanced time-decaying Elo algorithm
│   │   ├── poisson_sim.py       # Monte Carlo match simulator
│   │   └── clutch_metrics.py    # Win Probability Added (WPA) for players
│   └── visualization/
│       └── plot_utils.py        # Reusable Plotly/Altair chart generators
│
├── tests/                       # Unit tests for our math models
│   └── test_elo.py
│
├── app/                         # Streamlit Multi-Page App
│   ├── Home.py                  # Landing dashboard
│   └── pages/
│       ├── 1_🎮_Match_Simulator.py
│       ├── 2_🌍_Era_Dominance.py
│       └── 3_🎯_Player_Clutch_Factor.py
│
├── notebooks/                   # Jupyter notebooks for model prototyping
├── requirements.txt             # Project dependencies
└── plan.md                      # This master document
```

---

## 3. Implementation Phases

### Phase 1: Enterprise Data Engineering & ETL
**Goal:** Build an automated pipeline that ingests raw CSVs, normalizes geopolitical borders, and engineers advanced ML features.
- **Geopolitical Engine:** Script a system that understands the timeline of `former_names.csv`. If a user selects 1980, "Germany" is split into West/East. If 2020, they are merged.
- **Tournament Weights:** Calculate dynamic weights for matches (e.g., World Cup Final is exponentially heavier than a friendly).
- **Home Advantage Index:** Calculate distance traveled and neutral venue impact on match outcomes.

### Phase 2: The Core Mathematical Engine (Advanced Stats)
**Goal:** Build the statistical backend that evaluates team and player strength.
- **Bivariate Poisson Regression:** Build a model that evaluates the interaction between Team A's Attack Rating and Team B's Defense Rating, calculating Expected Goals (xG).
- **Time-Decay Elo Algorithm:** Implement a custom Elo rating system that updates over 49,000 matches. Include a Margin of Victory (MoV) multiplier.
- **Player Clutch Factor (WPA):** Iterate through `goalscorers.csv` and `shootouts.csv`. Calculate how much each specific goal shifted the team's probability of winning the match. 

### Phase 3: The Multi-Page Streamlit Application
**Goal:** Build a beautiful, responsive frontend to interact with our models.
- **Page 1 (Simulator):** Users select any two teams and eras. The backend runs 10,000 Monte Carlo simulations using the Poisson Engine and returns exact scoreline probabilities and Win/Loss pie charts.
- **Page 2 (Era Dominance):** A dynamic timeline and geospatial map (using Plotly Geo) showing which countries held the highest Elo ratings in every decade since 1872.
- **Page 3 (Player Stats):** A dashboard proving exactly who the most clutch goalscorers in history are based on the Win Probability Added metric.

### Phase 4: Polish & Deployment Ready
- Write unit tests for the Elo calculation to ensure mathematical accuracy.
- Apply a custom premium dark-mode CSS theme to Streamlit.
- Optimize Pandas DataFrames (convert to Parquet) so the app loads instantly.

---

## 4. Next Immediate Steps
1. Re-scaffold the directory structure to match the new `src/` and `app/` architecture.
2. Install dependencies.
3. Write `src/data/etl_pipeline.py` to process the 4 datasets into a clean Parquet file.
