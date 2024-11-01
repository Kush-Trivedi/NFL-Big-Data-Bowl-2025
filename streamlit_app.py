import streamlit as st 

st.set_page_config(layout="wide")

st.markdown(
    r"""
    <style>
    .reportview-container {
            margin-top: -3em;
        }
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    r"""
    <style>
    div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
        height: 5rem;
        width: auto;
        display: block;  
        margin-left: auto;  
        margin-right: auto; 
    }

    div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
    div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
        display: flex;
        align-items: center;
        justify-content: center;  /* Center items in the sidebar */
    }

    [data-testid="stSidebarNav"]::before {
        content: "Perfectly Imperfect";
        margin-left: 20px;
        font-size: 30px;
        font-weight: bold;
        position: relative;
    }
    </style>
    """, unsafe_allow_html=True
)

# Introduction Page
introduction_page = st.Page(
    page="views/introduction.py",
    title="NFL Big Data Bowl 2025",
    icon=":material/target:",   
    default=True
)

# Game Predictor Page
game_predictor_page = st.Page(
    page="views/game_predictor.py",
    title="NFL Play Simulator",
    icon=":material/sports_football:"
)

# Offense Page
offense_page = st.Page(
    page="views/offense.py",
    title="Offense",
    icon=":material/directions_run:"
)


# Defense Page
defense_page = st.Page(
    page="views/defense.py",
    title="Defense",
    icon=":material/sports_kabaddi:"
)

# Player Stats Page
player_stats_page = st.Page(
    page="views/player_stats.py",
    title="Player Stats",
    icon=":material/query_stats:"
)

# Team Stats Page
team_stats_page = st.Page(
    page="views/team_stats.py",
    title="Team Stats",
    icon=":material/analytics:"
)

# Navigation Setup
pg = st.navigation(
    {
        "Intoduction": [introduction_page],
        "Playground": [game_predictor_page],
        "Playbook": [offense_page, defense_page],
        "Visualization Stats": [player_stats_page, team_stats_page]
    }
)

st.logo("assets/navbar/big-data-bowl.png")

# Run Navigation
pg.run()
