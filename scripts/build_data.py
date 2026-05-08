"""
Phase 1 + Phase 2: Data Engineering & ELO Engine
Reads raw CSVs -> cleans -> engineers features -> computes ELO for all 49K matches.
Run this script ONCE to generate all processed parquet files used by the Streamlit app.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "dataset"
OUT  = RAW / "processed"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFEDERATION MAP  (add teams as needed; covers all major footballing nations)
# ─────────────────────────────────────────────────────────────────────────────
CONFEDERATION = {
    # UEFA
    "England":"UEFA","Scotland":"UEFA","Wales":"UEFA","Northern Ireland":"UEFA",
    "Republic of Ireland":"UEFA","France":"UEFA","Germany":"UEFA","Italy":"UEFA",
    "Spain":"UEFA","Portugal":"UEFA","Netherlands":"UEFA","Belgium":"UEFA",
    "Switzerland":"UEFA","Austria":"UEFA","Sweden":"UEFA","Norway":"UEFA",
    "Denmark":"UEFA","Finland":"UEFA","Poland":"UEFA","Czech Republic":"UEFA",
    "Czechoslovakia":"UEFA","Slovakia":"UEFA","Hungary":"UEFA","Romania":"UEFA",
    "Bulgaria":"UEFA","Yugoslavia":"UEFA","Serbia":"UEFA","Croatia":"UEFA",
    "Slovenia":"UEFA","Bosnia and Herzegovina":"UEFA","Montenegro":"UEFA",
    "North Macedonia":"UEFA","Albania":"UEFA","Greece":"UEFA","Turkey":"UEFA",
    "Russia":"UEFA","Ukraine":"UEFA","Belarus":"UEFA","Moldova":"UEFA",
    "Lithuania":"UEFA","Latvia":"UEFA","Estonia":"UEFA","Iceland":"UEFA",
    "Luxembourg":"UEFA","Cyprus":"UEFA","Malta":"UEFA","Liechtenstein":"UEFA",
    "Andorra":"UEFA","San Marino":"UEFA","Kosovo":"UEFA","Georgia":"UEFA",
    "Armenia":"UEFA","Azerbaijan":"UEFA","Kazakhstan":"UEFA","Israel":"UEFA",
    # CONMEBOL
    "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Uruguay":"CONMEBOL",
    "Chile":"CONMEBOL","Paraguay":"CONMEBOL","Bolivia":"CONMEBOL",
    "Peru":"CONMEBOL","Colombia":"CONMEBOL","Ecuador":"CONMEBOL",
    "Venezuela":"CONMEBOL",
    # CONCACAF
    "United States":"CONCACAF","Mexico":"CONCACAF","Canada":"CONCACAF",
    "Costa Rica":"CONCACAF","Honduras":"CONCACAF","El Salvador":"CONCACAF",
    "Guatemala":"CONCACAF","Panama":"CONCACAF","Cuba":"CONCACAF",
    "Jamaica":"CONCACAF","Trinidad and Tobago":"CONCACAF","Haiti":"CONCACAF",
    # CAF
    "Brazil":"CONMEBOL",  # already above — example to show structure
    "Nigeria":"CAF","Ghana":"CAF","Cameroon":"CAF","Ivory Coast":"CAF",
    "Senegal":"CAF","Morocco":"CAF","Egypt":"CAF","Algeria":"CAF",
    "Tunisia":"CAF","South Africa":"CAF","DR Congo":"CAF","Mali":"CAF",
    "Burkina Faso":"CAF","Guinea":"CAF","Zambia":"CAF","Zimbabwe":"CAF",
    "Kenya":"CAF","Ethiopia":"CAF","Tanzania":"CAF","Uganda":"CAF",
    "Angola":"CAF","Mozambique":"CAF","Malawi":"CAF","Gabon":"CAF",
    "Congo":"CAF","Togo":"CAF","Benin":"CAF","Niger":"CAF","Sudan":"CAF",
    "Libya":"CAF","Rwanda":"CAF","Burundi":"CAF","Eritrea":"CAF",
    # AFC
    "Japan":"AFC","South Korea":"AFC","China PR":"AFC","Australia":"AFC",
    "Iran":"AFC","Saudi Arabia":"AFC","Iraq":"AFC","Qatar":"AFC",
    "United Arab Emirates":"AFC","Kuwait":"AFC","Bahrain":"AFC","Oman":"AFC",
    "Jordan":"AFC","Syria":"AFC","Thailand":"AFC","Vietnam":"AFC",
    "Indonesia":"AFC","Malaysia":"AFC","Philippines":"AFC","India":"AFC",
    "Pakistan":"AFC","Bangladesh":"AFC","Nepal":"AFC","Sri Lanka":"AFC",
    "North Korea":"AFC","Uzbekistan":"AFC","Kazakhstan":"AFC","Tajikistan":"AFC",
    # OFC
    "New Zealand":"OFC","Australia":"OFC",
}


def get_confederation(team: str) -> str:
    return CONFEDERATION.get(team, "Other")


# ─────────────────────────────────────────────────────────────────────────────
# TOURNAMENT WEIGHT MAP
# ─────────────────────────────────────────────────────────────────────────────
def tournament_weight(tournament: str) -> int:
    t = tournament.lower()
    if "fifa world cup" in t or t == "world cup":
        return 5
    if any(x in t for x in ["copa américa", "euro", "african cup", "asian cup",
                              "gold cup", "nations cup", "afcon"]):
        return 4
    if "qualification" in t or "qualifier" in t:
        return 3
    if "nations league" in t:
        return 3
    if "friendly" in t:
        return 1
    return 2


def is_knockout_tournament(tournament: str) -> bool:
    ko_keywords = ["world cup", "copa", "euro", "afcon", "gold cup",
                   "asian cup", "olympic", "nations cup"]
    return any(k in tournament.lower() for k in ko_keywords)


# ─────────────────────────────────────────────────────────────────────────────
# K-FACTOR FOR ELO
# ─────────────────────────────────────────────────────────────────────────────
def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "fifa world cup" in t or t == "world cup":
        return 60.0
    if any(x in t for x in ["copa", "euro", "afcon", "gold cup",
                              "asian cup", "nations cup", "olympic"]):
        return 50.0
    if "qualification" in t or "qualifier" in t or "nations league" in t:
        return 40.0
    return 20.0  # Friendly


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def build_name_map(former_names_path: Path) -> dict:
    """Build a dict: former_name -> current_name, scoped by date ranges."""
    fn = pd.read_csv(former_names_path, parse_dates=["start_date", "end_date"])
    # We want a simple mapping for historical normalization
    mapping = {}
    for _, row in fn.iterrows():
        mapping[row["former"]] = row["current"]
    return mapping


def normalize_team(name: str, name_map: dict) -> str:
    return name_map.get(name, name)


def clean_results(results_path: Path, name_map: dict) -> pd.DataFrame:
    print("  Loading results.csv ...")
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    print("  Normalizing team names ...")
    df["home_team"] = df["home_team"].map(lambda x: normalize_team(x, name_map))
    df["away_team"] = df["away_team"].map(lambda x: normalize_team(x, name_map))

    print("  Engineering features ...")
    df["goal_diff"]    = df["home_score"] - df["away_score"]
    df["total_goals"]  = df["home_score"] + df["away_score"]
    df["year"]         = df["date"].dt.year
    df["decade"]       = (df["year"] // 10) * 10
    df["month"]        = df["date"].dt.month

    df["result"] = np.where(
        df["goal_diff"] > 0, "home_win",
        np.where(df["goal_diff"] < 0, "away_win", "draw")
    )
    df["result_code"] = df["result"].map(
        {"home_win": 2, "draw": 1, "away_win": 0}
    )

    df["tournament_weight"] = df["tournament"].map(tournament_weight)
    df["is_knockout"]       = df["tournament"].map(is_knockout_tournament)
    df["k_factor"]          = df["tournament"].map(k_factor)

    df["confederation_home"] = df["home_team"].map(get_confederation)
    df["confederation_away"] = df["away_team"].map(get_confederation)
    df["confederation_clash"] = (
        df["confederation_home"] != df["confederation_away"]
    ).astype(int)

    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1"])

    print(f"  Results cleaned: {len(df):,} matches, {df['home_team'].nunique()} unique home teams")
    return df


def clean_goalscorers(goalscorers_path: Path, name_map: dict) -> pd.DataFrame:
    print("  Loading goalscorers.csv ...")
    df = pd.read_csv(goalscorers_path, parse_dates=["date"])
    df = df.sort_values(["date", "home_team", "away_team", "minute"]).reset_index(drop=True)

    df["home_team"] = df["home_team"].map(lambda x: normalize_team(x, name_map))
    df["away_team"] = df["away_team"].map(lambda x: normalize_team(x, name_map))
    df["team"]      = df["team"].map(lambda x: normalize_team(x, name_map))

    df["own_goal"] = df["own_goal"].astype(str).str.upper().isin(["TRUE", "1"])
    df["penalty"]  = df["penalty"].astype(str).str.upper().isin(["TRUE", "1"])

    # Goal number within each match
    df["match_goal_number"] = df.groupby(["date", "home_team", "away_team"]).cumcount() + 1

    # Is this goal scored by the home team or away team?
    df["scored_by_home"] = (df["team"] == df["home_team"]).astype(int)

    print(f"  Goalscorers cleaned: {len(df):,} goals, {df['scorer'].nunique():,} unique scorers")
    return df


def clean_shootouts(shootouts_path: Path, name_map: dict) -> pd.DataFrame:
    print("  Loading shootouts.csv ...")
    df = pd.read_csv(shootouts_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["home_team"] = df["home_team"].map(lambda x: normalize_team(x, name_map))
    df["away_team"] = df["away_team"].map(lambda x: normalize_team(x, name_map))
    df["winner"]    = df["winner"].map(lambda x: normalize_team(str(x), name_map))
    print(f"  Shootouts cleaned: {len(df):,} shootouts")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — ELO ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class ELOEngine:
    """
    Computes dynamic ELO ratings for all international teams across 150 years.

    K-factor is match-type-dependent.
    Goal difference modifier dampens blowout inflation.
    """

    STARTING_ELO: float = 1500.0
    SCALE: float = 400.0

    def __init__(self):
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def _get_rating(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.STARTING_ELO
        return self.ratings[team]

    @staticmethod
    def _goal_diff_multiplier(goal_diff: int) -> float:
        """Logarithmic multiplier capped at 1.75 to prevent blowout distortion."""
        raw = np.log(abs(goal_diff) + 1) * 0.4 + 1.0
        return min(raw, 1.75)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / self.SCALE))

    def process_match(self, row: pd.Series) -> dict:
        home = row["home_team"]
        away = row["away_team"]
        goal_diff = int(row["goal_diff"])
        k = float(row["k_factor"])

        elo_h = self._get_rating(home)
        elo_a = self._get_rating(away)

        exp_h = self._expected(elo_h, elo_a)
        exp_a = 1.0 - exp_h

        # Actual scores
        if goal_diff > 0:
            score_h, score_a = 1.0, 0.0
        elif goal_diff < 0:
            score_h, score_a = 0.0, 1.0
        else:
            score_h, score_a = 0.5, 0.5

        gdm = self._goal_diff_multiplier(goal_diff)

        delta_h = k * gdm * (score_h - exp_h)
        delta_a = k * gdm * (score_a - exp_a)

        new_elo_h = elo_h + delta_h
        new_elo_a = elo_a + delta_a

        self.ratings[home] = new_elo_h
        self.ratings[away] = new_elo_a

        return {
            "home_elo_before":  round(elo_h, 2),
            "away_elo_before":  round(elo_a, 2),
            "home_elo_after":   round(new_elo_h, 2),
            "away_elo_after":   round(new_elo_a, 2),
            "elo_diff":         round(elo_h - elo_a, 2),
            "elo_delta_home":   round(delta_h, 2),
            "elo_delta_away":   round(delta_a, 2),
            "exp_home_win":     round(exp_h, 4),
        }

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all matches chronologically and return df with ELO columns."""
        print(f"  Running ELO engine on {len(df):,} matches ...")
        elo_records = []

        for _, row in df.iterrows():
            record = self.process_match(row)
            elo_records.append(record)

        elo_df = pd.DataFrame(elo_records)
        result = pd.concat([df.reset_index(drop=True), elo_df], axis=1)

        # Validate sanity
        top = pd.DataFrame(
            sorted(self.ratings.items(), key=lambda x: -x[1])[:5],
            columns=["team", "current_elo"]
        )
        print("\n  -- Current Top 5 ELO Ratings --")
        print(top.to_string(index=False))

        peak = result.groupby("home_team")["home_elo_after"].max().sort_values(ascending=False).head(5)
        print("\n  -- All-Time Peak ELO (by home_team) --")
        print(peak.to_string())

        return result

    def get_current_ratings(self) -> pd.DataFrame:
        """Return a sorted DataFrame of current ELO ratings."""
        return (
            pd.DataFrame(self.ratings.items(), columns=["team", "elo"])
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SHOOTOUT WIN RATES  (for Monte Carlo simulator)
# ─────────────────────────────────────────────────────────────────────────────
def compute_shootout_stats(shootouts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team shootout win rate with Bayesian smoothing.
    Beta prior: alpha=2, beta=2 → shrinks low-sample teams toward 50%.
    """
    ALPHA, BETA = 2.0, 2.0

    records = []
    teams = pd.unique(shootouts_df[["home_team", "away_team"]].values.ravel())

    for team in teams:
        mask = (shootouts_df["home_team"] == team) | (shootouts_df["away_team"] == team)
        appearances = shootouts_df[mask]
        wins = (appearances["winner"] == team).sum()
        n = len(appearances)
        smoothed_rate = (wins + ALPHA) / (n + ALPHA + BETA)
        records.append({
            "team": team,
            "shootout_appearances": n,
            "shootout_wins": wins,
            "raw_win_rate": wins / n if n > 0 else 0.5,
            "smoothed_win_rate": round(smoothed_rate, 4),
        })

    return pd.DataFrame(records).sort_values("smoothed_win_rate", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  WORLD CUP LAB - Data Engineering Pipeline")
    print("=" * 60)

    # Build name normalisation map
    print("\n[1/5] Building name normalisation map ...")
    name_map = build_name_map(RAW / "former_names.csv")
    print(f"  {len(name_map)} historical -> current mappings loaded")

    # Clean results
    print("\n[2/5] Cleaning results.csv ...")
    results = clean_results(RAW / "results.csv", name_map)
    results.to_parquet(OUT / "results_cleaned.parquet", index=False)
    print(f"  [OK] Saved results_cleaned.parquet ({len(results):,} rows)")

    # Clean goalscorers
    print("\n[3/5] Cleaning goalscorers.csv ...")
    goals = clean_goalscorers(RAW / "goalscorers.csv", name_map)
    goals.to_parquet(OUT / "goalscorers_cleaned.parquet", index=False)
    print(f"  [OK] Saved goalscorers_cleaned.parquet ({len(goals):,} rows)")

    # Clean shootouts + compute stats
    print("\n[4/5] Cleaning shootouts.csv & computing shootout stats ...")
    shootouts = clean_shootouts(RAW / "shootouts.csv", name_map)
    shootout_stats = compute_shootout_stats(shootouts)
    shootouts.to_parquet(OUT / "shootouts_cleaned.parquet", index=False)
    shootout_stats.to_parquet(OUT / "shootout_stats.parquet", index=False)
    print(f"  [OK] Saved shootouts_cleaned.parquet ({len(shootouts):,} rows)")
    print(f"  [OK] Saved shootout_stats.parquet ({len(shootout_stats):,} teams)")

    # Run ELO engine
    print("\n[5/5] Running ELO Engine ...")
    engine = ELOEngine()
    results_with_elo = engine.run(results)
    results_with_elo.to_parquet(OUT / "results_with_elo.parquet", index=False)
    print(f"\n  [OK] Saved results_with_elo.parquet ({len(results_with_elo):,} rows)")

    # Save current ratings snapshot
    current_ratings = engine.get_current_ratings()
    current_ratings.to_parquet(OUT / "current_elo_ratings.parquet", index=False)
    print(f"  [OK] Saved current_elo_ratings.parquet ({len(current_ratings):,} teams)")

    print("\n" + "=" * 60)
    print("  Pipeline complete. All parquet files saved to dataset/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
