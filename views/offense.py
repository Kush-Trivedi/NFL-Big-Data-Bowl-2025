import streamlit as st
from pathlib import Path
import base64
import random
import weasyprint
from streamlit_option_menu import option_menu

border_color = "rgb(40,40,40)"
shadow_color = "rgba(0, 0, 0, 0.1)"
hover_shadow_color = "rgba(0, 0, 0, 0.2)"

# CSS for styling the cards and other components
st.markdown(f"""
<style>
    .card {{
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        min-height: 150px;
        background-color: var(--background-color);
        color: var(--text-color);
        box-shadow: 0 4px 6px {shadow_color}, 0 2px 4px {shadow_color}; 
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }}

    .card:hover {{
        box-shadow: 0 8px 16px {hover_shadow_color}, 0 4px 4px {hover_shadow_color};
        transform: translateY(-2px);
    }}

    .logo-container {{
        width: 30%;
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .logo {{
        max-width: 100%;
        max-height: 200%;
        object-fit: contain;
    }}
    .description {{
        width: 70%;
        padding: 0 10px;
    }}
    .left .logo-container {{
        order: 1;
    }}
    .left .description {{
        order: 2;
    }}
    .right .logo-container {{
        order: 2;
    }}
    .right .description {{
        order: 1;
    }}
    .strength-weakness {{
        margin-top: 10px;
    }}
    .strength, .weakness {{
        margin-bottom: 5px;
    }}
    ul {{
        margin: 0;
        padding-left: 20px;
    }}
  
</style>
""", unsafe_allow_html=True)

st.title(":blue[Offensive] Strategies & Tendency")

# Function to convert images to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Function to generate defensive strength and weakness
def generate_offensive_strength_weakness():
    strength = random.sample([
        "Effective against zone defense",
        "High scoring potential",
        "Utilizes player strengths",
        "Unpredictable for opponents",
        "Creates mismatches",
        "Good floor spacing",
        "Fast-paced execution",
        "Versatile for different lineups"
    ], 5)
    
    weakness = random.sample([
        "Requires precise timing",
        "Can be turnover-prone",
        "Depends on specific player skills",
        "May struggle against man-to-man defense",
        "Time-weaknessuming to master",
        "Limited effectiveness in low-post",
        "Vulnerable to defensive switches",
        "May tire players quickly"
    ], 5)
    
    return strength, weakness


# Function to generate and display cards
def display_offensive_cards(logo_folder, selected):
    counter = 0
    full_content = ""

    for logo_file in logo_folder.glob("*.png"):
        img_base64 = img_to_base64(str(logo_file))
        layout_class = "left" if counter % 2 == 0 else "right"
        team_name = logo_file.stem
        strength, weakness = generate_offensive_strength_weakness()

        # Generate card HTML
        card_html = f"""
        <div class="card clearfix {layout_class}">
            <div class="logo-container">
                <img src="data:image/png;base64,{img_base64}" class="logo">
            </div>
            <div class="description">
                <div class="strength-weakness">
                    <div class="strength">
                        <strong>Strength:</strong>
                        <ul>
                            {"".join(f"<li>{pro}</li>" for pro in strength)}
                        </ul>
                    </div>
                    <div class="weakness">
                        <strong>Weakness:</strong>
                        <ul>
                            {"".join(f"<li>{con}</li>" for con in weakness)}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)
        full_content += card_html

        # Define download PDF function for individual team report
        def download_team_pdf(team_name, strength, weakness, img_base64):
            html_content = f"""
            <html>
            <head>
                <style>
                    @page {{ size: A4; margin: 1cm; }}
                    body {{ font-family: Arial, sans-serif; }}
                    .card {{ border: 2px solid black; border-radius: 5px; padding: 10px; margin-bottom: 20px; page-break-inside: avoid; }}
                    .logo-container {{ width: 30%; float: left; margin-right: 10px; }}
                    .logo {{ max-width: 100%; max-height: 150px; object-fit: contain; }}
                    .description {{ width: 65%; float: right; }}
                    .strength-weakness {{ margin-top: 10px; }}
                    .strength, .weakness {{ margin-bottom: 5px; }}
                    ul {{ margin: 0; padding-left: 20px; }}
                    .clearfix::after {{ content: ""; clear: both; display: table; }}
                </style>
            </head>
            <body>
                <h1>{team_name} Strategies & Tendency for {selected} 2022</h1>
                <div class="card clearfix">
                    <div class="logo-container">
                        <img src="data:image/png;base64,{img_base64}" class="logo">
                    </div>
                    <div class="description">
                        <div class="strength-weakness">
                            <div class="strength">
                                <strong>strength:</strong>
                                <ul>
                                    {"".join(f"<li>{pro}</li>" for pro in strength)}
                                </ul>
                            </div>
                            <div class="weakness">
                                <strong>weakness:</strong>
                                <ul>
                                    {"".join(f"<li>{con}</li>" for con in weakness)}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            return weasyprint.HTML(string=html_content).write_pdf()

        # Generate team-specific PDF
        team_pdf = download_team_pdf(team_name, strength, weakness, img_base64)

        # Create two columns with specified width proportions (80% and 20%)
        col1, col2 = st.columns([45, 55])

        # First column for the description
        with col1:
            # Right-align the text using HTML and CSS
            st.markdown(
                f'<div style="text-align: left; padding: 8px;">'
                f'<p>Download the {team_name} strategy and tactic for {selected} 2022 here:</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Second column for the download button
        with col2:
            st.download_button(
                label="Download",
                data=team_pdf,
                file_name=f"{selected}_{team_name}_report.pdf",
                mime="application/pdf",
                key=f"download_{team_name}"
            )

        counter += 1

    # Define the full report download function after generating all content
    def download_full_pdf(content):
        html_content = f"""
        <html>
        <head>
            <style>
                @page {{ size: A4; margin: 1cm; }}
                body {{ font-family: Arial, sans-serif; }}
                .card {{ border: 2px solid black; border-radius: 5px; padding: 10px; margin-bottom: 20px; page-break-inside: avoid; }}
                .logo-container {{ width: 30%; float: left; margin-right: 10px; }}
                .logo {{ max-width: 100%; max-height: 150px; object-fit: contain; }}
                .description {{ width: 65%; float: right; }}
                .strength-weakness {{ margin-top: 10px; }}
                .strength, .weakness {{ margin-bottom: 5px; }}
                ul {{ margin: 0; padding-left: 20px; }}
                .clearfix::after {{ content: ""; clear: both; display: table; }}
            </style>
        </head>
        <body>
            <h1>{selected} 2022 Strategies & Tendency for all Teams</h1>
            {content}
        </body>
        </html>
        """
        return weasyprint.HTML(string=html_content).write_pdf()

    # Generate the full PDF
    full_pdf = download_full_pdf(full_content)

    # Full report download button at the top
    st.download_button(
        label="Download Full Report",
        data=full_pdf,
        file_name="season_offensive.pdf",
        mime="application/pdf",
        key="download_full_pdf"
    )


# Main option menu for "Season" or "Weekly"
selected = option_menu(
    menu_title=None,
    options=["Season", "Weekly"],
    icons=["view-list", "calendar-week"],
    orientation="horizontal"
)

logo_folder = Path("assets/logo")

if selected == "Season":
    display_offensive_cards(logo_folder, selected)

elif selected == "Weekly":
    week_options = [str(i) for i in range(1, 9)] 
    selected_week = option_menu(
        menu_title=None,
        options=week_options,
        iweakness=["file-emark-word","file-emark-wor","file-emark-wor","file-emark-wor","file-emark-wor","file-emark-wor","file-emark-wor","file-emark-wor"],
        orientation="horizontal"
    )
    display_offensive_cards(logo_folder, f"Week {selected_week}")
