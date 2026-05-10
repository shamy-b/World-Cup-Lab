import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Set page config
st.set_page_config(
    page_title="Chronos Football Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean typography and spacing */
    body {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Styling headers */
    h1, h2, h3 {
        font-weight: 600 !important;
        color: #1E293B;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0F172A;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #334155;
        border-color: #334155;
        color: white;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Insert src into path so we can import our modules
sys.path.insert(0, 'src')

from simulation import WorldCupSimulator

# --- CACHING & DATA LOADING ---
@st.cache_resource(show_spinner=True)
def load_simulator():
    """Loads the prediction engine and precalculates feature states."""
    sim = WorldCupSimulator()
    
    default_teams = [
        "Spain", "Argentina", "France", "England", "Netherlands", "Colombia", "Germany", "Brazil",
        "Portugal", "Japan", "Uruguay", "Croatia", "Italy", "Morocco", "Switzerland", "Senegal",
        "United States", "Mexico", "Iran", "South Korea", "Denmark", "Austria", "Ecuador", "Ukraine",
        "Australia", "Peru", "Serbia", "Poland", "Sweden", "Wales", "Hungary", "Ivory Coast"
    ]
    sim.precalculate_matrix(default_teams)
    return sim, default_teams

with st.spinner("Initializing predictive models..."):
    try:
        sim_engine, top_teams = load_simulator()
    except Exception as e:
        st.error(f"Failed to load engine components: {e}")
        st.stop()

# --- SIDEBAR NAV ---
st.sidebar.title("Chronos Analytics")
st.sidebar.markdown("International Football Prediction Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Global Power Index", 
    "Match Probability Analysis", 
    "Tournament Projections"
])

# --- PAGE: GLOBAL POWER INDEX ---
if page == "Global Power Index":
    st.title("Global Power Index")
    st.markdown("Current international rankings derived from time-adjusted Elo models.")
    
    # Extract Elo rankings from engine
    elo_data = [{"Team": team, "Index Score": round(rating, 1)} for team, rating in sim_engine.elo_system.ratings.items()]
    df_elo = pd.DataFrame(elo_data).sort_values("Index Score", ascending=False).reset_index(drop=True)
    df_elo.index += 1 # 1-based ranking
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Top 20 Nations")
        st.dataframe(df_elo.head(20), height=600, use_container_width=True)
        
    with col2:
        st.markdown("### Power Distribution")
        
        top_15 = df_elo.head(15).sort_values('Index Score', ascending=True)
        fig = go.Figure(go.Bar(
            x=top_15['Index Score'],
            y=top_15['Team'],
            orientation='h',
            marker=dict(
                color=top_15['Index Score'],
                colorscale='Blues',
                showscale=False
            )
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                range=[1900, df_elo['Index Score'].max() + 50],
                showgrid=True,
                gridcolor='#E2E8F0',
                title="Index Score"
            ),
            yaxis=dict(title=""),
            margin=dict(l=0, r=0, t=30, b=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE: MATCH PROBABILITY ANALYSIS ---
elif page == "Match Probability Analysis":
    st.title("Match Probability Analysis")
    st.markdown("Assess win/draw/loss probabilities for hypothetical matchups using the ensemble prediction model.")
    st.markdown("---")
    
    all_teams = sorted(list(sim_engine.team_features.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Select Team 1", all_teams, index=all_teams.index('France') if 'France' in all_teams else 0)
    with col2:
        away_team = st.selectbox("Select Team 2", all_teams, index=all_teams.index('Argentina') if 'Argentina' in all_teams else 1)
        
    if st.button("Generate Prediction", type="primary"):
        if home_team == away_team:
            st.warning("Please select two distinct teams.")
        else:
            with st.spinner("Calculating probability distribution..."):
                f1 = sim_engine.team_features[home_team]
                f2 = sim_engine.team_features[away_team]
                
                row1 = {
                    'elo_home': f1['elo'], 'elo_away': f2['elo'], 'elo_diff': f1['elo'] - f2['elo'],
                    'elo_expected_home': 1 / (1 + 10 ** ((f2['elo'] - f1['elo']) / 400)),
                    'tournament_prestige': 1.0, 
                    'is_neutral': 1,
                    'h2h_home_win_rate': 0.5, 'h2h_away_win_rate': 0.5, 'h2h_draw_rate': 0.0, 'h2h_matches': 0
                }
                row1['elo_expected_away'] = 1 - row1['elo_expected_home']
                
                for c in sim_engine.feature_cols:
                    if c.endswith('_home') and not c.startswith('h2h') and not c.startswith('elo'):
                        row1[c] = f1.get(c.replace('_home', ''), 0)
                    elif c.endswith('_away') and not c.startswith('h2h') and not c.startswith('elo'):
                        row1[c] = f2.get(c.replace('_away', ''), 0)
                
                for c in sim_engine.feature_cols:
                    if c not in row1: row1[c] = 0
                    
                X = pd.DataFrame([row1])[sim_engine.feature_cols]
                
                xgb_p = sim_engine.xgb_model.predict_proba(X)
                lgb_p = sim_engine.lgb_model.predict_proba(X)
                meta_X = np.hstack([xgb_p, lgb_p])
                probs = sim_engine.meta_model.predict_proba(meta_X)[0]
                
                p_away, p_draw, p_home = probs[0], probs[1], probs[2]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{home_team} Win", f"{p_home*100:.1f}%")
                c2.metric("Draw", f"{p_draw*100:.1f}%")
                c3.metric(f"{away_team} Win", f"{p_away*100:.1f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_chart, col_stats = st.columns([1, 1])
                
                with col_chart:
                    fig = go.Figure(data=[go.Pie(
                        labels=[f"{home_team} Win", "Draw", f"{away_team} Win"],
                        values=[p_home, p_draw, p_away],
                        hole=.5,
                        marker_colors=['#0F172A', '#94A3B8', '#3B82F6'],
                        textinfo='label+percent',
                        hoverinfo='label+percent'
                    )])
                    fig.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0),
                        showlegend=False,
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_stats:
                    st.markdown("### Underlying Metrics")
                    
                    metrics_df = pd.DataFrame({
                        "Metric": ["Index Score (Elo)", "Recent Form Momentum", "Goal Scoring Volatility"],
                        home_team: [f"{f1['elo']:.1f}", f"{f1.get('momentum_10', 0):.2f}", f"{f1.get('roll_gd_std_10', 0):.2f}"],
                        away_team: [f"{f2['elo']:.1f}", f"{f2.get('momentum_10', 0):.2f}", f"{f2.get('roll_gd_std_10', 0):.2f}"]
                    })
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

# --- PAGE: SIMULATOR ---
elif page == "Tournament Projections":
    st.title("Tournament Projections")
    st.markdown("Execute Monte Carlo simulations to project tournament progression probabilities.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Simulation Parameters")
        iterations = st.selectbox("Iteration Count", [1000, 5000, 10000, 25000], index=0)
        
        run_sim = st.button("Execute Simulation", type="primary", use_container_width=True)
    
    if run_sim:
        with st.spinner("Processing..."):
            sim_engine.precalculate_matrix(top_teams)
            report = sim_engine.run_monte_carlo(top_teams, iterations=iterations)
            
            st.markdown("### Simulation Results")
            
            # Format dataframe for display
            display_df = report.copy()
            for col in ['R16 (%)', 'QF (%)', 'SF (%)', 'Final (%)', 'Win (%)']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(display_df, hide_index=True, use_container_width=True, height=600)
            
            st.markdown("### Highest Probability Outcomes")
            top_winners = report.head(10).sort_values('Win (%)', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=top_winners['Win (%)'],
                y=top_winners['Team'],
                orientation='h',
                marker=dict(
                    color=top_winners['Win (%)'],
                    colorscale='Blues',
                    showscale=False
                ),
                text=top_winners['Win (%)'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Championship Probability (%)", showgrid=True, gridcolor='#E2E8F0'),
                yaxis=dict(title=""),
                margin=dict(l=0, r=0, t=30, b=0),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
