import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys

# Set page config
st.set_page_config(
    page_title="Chronos Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist light mode
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #FAFAFA;
    }

    /* Minimalist White Cards */
    .clean-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
        margin-bottom: 24px;
        transition: box-shadow 0.2s ease;
    }
    .clean-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Typography */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        background-color: transparent;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #64748B;
        font-weight: 500;
        font-size: 1.05rem;
    }
    .stTabs [aria-selected="true"] {
        color: #0F172A !important;
        border-bottom: 2px solid #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# Insert src into path
sys.path.insert(0, 'src')
from simulation import WorldCupSimulator

# --- DATA LOADING ---
@st.cache_resource(show_spinner=False)
def load_engine():
    sim = WorldCupSimulator()
    default_teams = [
        "Spain", "Argentina", "France", "England", "Netherlands", "Colombia", "Germany", "Brazil",
        "Portugal", "Japan", "Uruguay", "Croatia", "Italy", "Morocco", "Switzerland", "Senegal",
        "United States", "Mexico", "Iran", "South Korea", "Denmark", "Austria", "Ecuador", "Ukraine",
        "Australia", "Peru", "Serbia", "Poland", "Sweden", "Wales", "Hungary", "Ivory Coast"
    ]
    sim.precalculate_matrix(default_teams)
    return sim, default_teams

with st.spinner("Initializing Chronos Neural Engine..."):
    sim_engine, top_teams = load_engine()

# --- HEADER ---
st.markdown("<div class='hero-title'>Chronos Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Advanced Machine Learning Platform for International Football</div>", unsafe_allow_html=True)

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["Global Power Index", "Match Predictor", "Tournament Projections"])

# ==========================================
# TAB 1: GLOBAL POWER INDEX
# ==========================================
with tab1:
    elo_data = [{"Nation": team, "Power Index (Elo)": round(rating, 1)} for team, rating in sim_engine.elo_system.ratings.items()]
    df_elo = pd.DataFrame(elo_data).sort_values("Power Index (Elo)", ascending=False).reset_index(drop=True)
    df_elo.index += 1 
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-label">Global #1</div>
            <div class="metric-value">{df_elo.iloc[0]['Nation']}</div>
            <div style="color: #64748B; margin-top: 4px; font-weight: 500;">{df_elo.iloc[0]['Power Index (Elo)']} Rating</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-label">Global #2</div>
            <div class="metric-value">{df_elo.iloc[1]['Nation']}</div>
            <div style="color: #64748B; margin-top: 4px; font-weight: 500;">{df_elo.iloc[1]['Power Index (Elo)']} Rating</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="clean-card">
            <div class="metric-label">Global #3</div>
            <div class="metric-value">{df_elo.iloc[2]['Nation']}</div>
            <div style="color: #64748B; margin-top: 4px; font-weight: 500;">{df_elo.iloc[2]['Power Index (Elo)']} Rating</div>
        </div>
        """, unsafe_allow_html=True)

    col_table, col_chart = st.columns([1, 2])
    
    with col_table:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; color:#0F172A; font-size:1.1rem; font-weight:600;'>Power Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(df_elo.head(25), height=450)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; color:#0F172A; font-size:1.1rem; font-weight:600;'>Top 15 Power Distribution</h3>", unsafe_allow_html=True)
        top_15 = df_elo.head(15).sort_values('Power Index (Elo)', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=top_15['Power Index (Elo)'],
            y=top_15['Nation'],
            orientation='h',
            marker=dict(
                color='#1E293B'
            )
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#475569', family='Inter'),
            xaxis=dict(
                range=[1900, df_elo['Power Index (Elo)'].max() + 30],
                showgrid=True,
                gridcolor='#F1F5F9',
                title=""
            ),
            yaxis=dict(title=""),
            margin=dict(l=0, r=0, t=10, b=0),
            height=410
        )
        st.plotly_chart(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: MATCH PREDICTOR
# ==========================================
with tab2:
    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
    all_teams = sorted(list(sim_engine.team_features.keys()))
    
    col1, col_vs, col2 = st.columns([3, 1, 3])
    with col1:
        home_team = st.selectbox("Select Team 1", all_teams, index=all_teams.index('France') if 'France' in all_teams else 0)
    with col_vs:
        st.markdown("<div style='text-align: center; margin-top: 28px; font-size: 1.2rem; font-weight: 600; color: #94A3B8;'>vs</div>", unsafe_allow_html=True)
    with col2:
        away_team = st.selectbox("Select Team 2", all_teams, index=all_teams.index('Argentina') if 'Argentina' in all_teams else 1)
    
    st.markdown("</div>", unsafe_allow_html=True)

    if home_team != away_team:
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
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="clean-card" style="text-align: center;">
                <div class="metric-label">{home_team} Win</div>
                <div class="metric-value">{p_home*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="clean-card" style="text-align: center; background-color: #F8FAFC;">
                <div class="metric-label">Draw</div>
                <div class="metric-value" style="color: #64748B;">{p_draw*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="clean-card" style="text-align: center;">
                <div class="metric-label">{away_team} Win</div>
                <div class="metric-value">{p_away*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        col_p_chart, col_p_stats = st.columns([1, 1])
        with col_p_chart:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:0; color:#0F172A; font-size:1.1rem; font-weight:600;'>Probability Matrix</h3>", unsafe_allow_html=True)
            fig = go.Figure(data=[go.Pie(
                labels=[f"{home_team} Win", "Draw", f"{away_team} Win"],
                values=[p_home, p_draw, p_away],
                hole=.65,
                marker_colors=['#0F172A', '#E2E8F0', '#94A3B8'],
                textinfo='label+percent',
                textfont=dict(family='Inter', color='#0F172A', size=13)
            )])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=0, r=0),
                showlegend=False,
                height=250
            )
            st.plotly_chart(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_p_stats:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:0; color:#0F172A; font-size:1.1rem; font-weight:600;'>Tactical Overview</h3>", unsafe_allow_html=True)
            
            metrics_df = pd.DataFrame({
                "Metric": ["Elo Index", "Momentum Rating", "Scoring Variance"],
                home_team: [f"{f1['elo']:.1f}", f"{f1.get('momentum_10', 0):.2f}", f"{f1.get('roll_gd_std_10', 0):.2f}"],
                away_team: [f"{f2['elo']:.1f}", f"{f2.get('momentum_10', 0):.2f}", f"{f2.get('roll_gd_std_10', 0):.2f}"]
            })
            st.dataframe(metrics_df, hide_index=True, height=250)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please select two distinct teams.")

# ==========================================
# TAB 3: TOURNAMENT MULTIVERSE
# ==========================================
with tab3:
    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; color:#0F172A; font-size:1.1rem; font-weight:600;'>Simulation Parameters</h3>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 4])
    with c1:
        iterations = st.selectbox("Parallel Simulations", [1000, 5000, 10000], index=0)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_sim = st.button("Execute Simulation", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_sim:
        with st.spinner(f"Computing {iterations} realities..."):
            sim_engine.precalculate_matrix(top_teams)
            report = sim_engine.run_monte_carlo(top_teams, iterations=iterations)
            
            col_chart, col_table = st.columns([2, 3])
            
            with col_chart:
                st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top:0; color:#0F172A; font-size:1.1rem; font-weight:600;'>Championship Probability</h3>", unsafe_allow_html=True)
                top_winners = report.head(10).sort_values('Win (%)', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=top_winners['Win (%)'],
                    y=top_winners['Team'],
                    orientation='h',
                    marker=dict(
                        color='#0F172A'
                    ),
                    text=top_winners['Win (%)'].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside',
                    textfont=dict(color='#0F172A')
                ))
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#475569', family='Inter'),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(title=""),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=450
                )
                st.plotly_chart(fig)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_table:
                st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top:0; color:#0F172A; font-size:1.1rem; font-weight:600;'>Simulation Output Matrix</h3>", unsafe_allow_html=True)
                
                display_df = report.copy()
                for col in ['R16 (%)', 'QF (%)', 'SF (%)', 'Final (%)', 'Win (%)']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(display_df, hide_index=True, height=450)
                st.markdown("</div>", unsafe_allow_html=True)
