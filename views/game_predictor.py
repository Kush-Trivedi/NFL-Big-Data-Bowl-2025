import streamlit as st

st.title("Welcome to :orange[Playground Simulator]")
st.subheader("Game Situation")

# Columns for quarter, down, and yards to go
col1, col2, col3 = st.columns(3)

with col1:
    st.selectbox("Quarter", options=[1, 2, 3, 4, 5], key="quarter")

with col2:
    st.selectbox("Down", options=[1, 2, 3, 4], key="down")

with col3:
    st.selectbox("Yards to Go", options=list(range(1, 101)), key="yards")

st.write("#####")
col4, col5 = st.columns(2)

with col4:
    st.subheader("Offense")
    offense_formation = st.selectbox("Offense Formation", options=["Shotgun", "Singleback", "I-Formation"])
    offense_players = st.multiselect(
        "Select 11 Offense Players", 
        options=[f"Offense Player {chr(65 + j)}" for j in range(20)]
    )

    # Placeholder for offense error messages
    error_placeholder_offense = st.empty()
    if len(offense_players) < 11:
        error_placeholder_offense.error("Please select exactly 11 players for the offense.")
    elif len(offense_players) > 11:
        error_placeholder_offense.error("Too many players selected. Please select exactly 11 players for the offense.")
    elif len(offense_players) != len(set(offense_players)):
        error_placeholder_offense.error("Each offense player must be unique!")
    else:
        error_placeholder_offense.empty()

with col5:
    st.subheader("Defense")
    defense_formation = st.selectbox("Defense Formation", options=["Nickle", "Man", "Zone"])
    defense_players = st.multiselect(
        "Select 11 Defense Players", 
        options=[f"Defense Player {chr(65 + j)}" for j in range(20)]
    )

    # Placeholder for defense error messages
    error_placeholder_defense = st.empty()
    if len(defense_players) < 11:
        error_placeholder_defense.error("Please select exactly 11 players for the defense.")
    elif len(defense_players) > 11:
        error_placeholder_defense.error("Too many players selected. Please select exactly 11 players for the defense.")
    elif len(defense_players) != len(set(defense_players)):
        error_placeholder_defense.error("Each defense player must be unique!")
    else:
        error_placeholder_defense.empty()

if offense_players and defense_players:
    duplicate_players = set(offense_players) & set(defense_players)
    if duplicate_players:
        st.error(f"Players cannot be the same between offense and defense! Duplicates: {', '.join(duplicate_players)}")

# Full-width submit button
button_clicked = st.markdown(
    """
    <style>
        .button-container {
            display: flex;                   /* Use flexbox for centering */
            justify-content: center;         /* Center items horizontally */
            margin: 20px 0;                 /* Optional: add some vertical spacing */
        }

        .full-width-button {
            width: 50%;                      /* Button width */
            color: white;                    /* Text color */
            background-color: #76528BFF;    /* Background color */
            padding: 10px;                   /* Padding inside the button */
            font-size: 16px;                 /* Font size */
            text-align: center;              /* Center text */
            border: none;                    /* No border */
            cursor: pointer;                 /* Pointer cursor */
            border-radius: 8px;              /* Rounded corners */
        }

        .full-width-button:hover {
            background-color: #603F83FF;    /* Change color on hover */
        }
    </style>

    <div class="button-container">
        <div onclick="window.submit_button = true;" class="full-width-button">Predict Game Play</div>
    </div>
    """,
    unsafe_allow_html=True
)

if "submit_button" in st.session_state and st.session_state.submit_button:
    if len(offense_players) == 11 and len(defense_players) == 11 and not duplicate_players:
        st.success("Form submitted successfully!")
    else:
        st.error("Please ensure all selections are correct before submitting.")


