# Import Libraries
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, UnidentifiedImageError


plt.rcParams['figure.dpi'] = 180
plt.rcParams["figure.figsize"] = (25, 17)
colors = sns.color_palette('Set3')
sns.set_theme(rc={
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif'
})


class NFLFieldVertical:
    """
    This class creates a vertical football field visualization, complete with team colors,
    end zones, yard markers, and optional logos. It supports customization of the field’s dimensions
    and the home and visitor team information.

    Attributes:
        width (float): Width of the football field in yards (default 53.3).
        height (float): Height of the football field in yards (default 120).
        home_team (str): Abbreviation of the home team.
        home_team_color (str): Color representing the home team.
        visitor_team (str): Abbreviation of the visiting team.
        visitor_team_color (str): Color representing the visitor team.
        logo_abbr (str): Abbreviation of the team logo to load (optional).
        fig, ax: Matplotlib figure and axis used for plotting the field.
    """

    def __init__(self, width=53.3, height=120, home_team="", home_team_color="", 
                 visitor_team="", visitor_team_color="", logo_abbr=""):
        """Initializes the NFL field with team details, dimensions, and colors."""
        from utils.helpers import Logger
        self.logger = Logger().get_logger()
        self.width = width
        self.height = height
        self.home_team = home_team
        self.home_team_color = home_team_color
        self.visitor_team = visitor_team
        self.visitor_team_color = visitor_team_color

        # Dictionary mapping team abbreviations to their full names.
        self.team_names = {
            'LA': 'Los Angeles Rams', 'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers',
            'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'DET': 'Detroit Lions',
            'HOU': 'Houston Texans', 'MIA': 'Miami Dolphins', 'NYJ': 'New York Jets',
            'WAS': 'Washington Commanders', 'ARI': 'Arizona Cardinals',
            'LAC': 'Los Angeles Chargers', 'MIN': 'Minnesota Vikings', 
            'TEN': 'Tennessee Titans', 'DAL': 'Dallas Cowboys', 'SEA': 'Seattle Seahawks',
            'KC': 'Kansas City Chiefs', 'BAL': 'Baltimore Ravens', 'CLE': 'Cleveland Browns',
            'JAX': 'Jacksonville Jaguars', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
            'PIT': 'Pittsburgh Steelers', 'SF': 'San Francisco 49ers', 'DEN': 'Denver Broncos',
            'LV': 'Las Vegas Raiders', 'GB': 'Green Bay Packers', 'BUF': 'Buffalo Bills',
            'PHI': 'Philadelphia Eagles', 'IND': 'Indianapolis Colts', 'NE': 'New England Patriots',
            'TB': 'Tampa Bay Buccaneers'
        }

        # Create the football field visualization.
        self.fig, self.ax = self.create_pitch()

    def create_pitch(self):
        """
        Creates the football field visualization, including end zones, yard markers, and optional logos.
        Returns:
            fig, ax: Matplotlib figure and axis for further customization or saving.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height + 1)
        ax.axis('off')  # Remove axis labels.

        # Draw the field background.
        background = Rectangle((0, 0), self.width, self.height, linewidth=1, 
                               facecolor='#97BC62FF', edgecolor='black', capstyle='round')
        ax.add_patch(background)

        # Draw horizontal yard lines, alternating between solid and dashed.
        for i in range(21):
            style = '--' if i % 2 != 0 else '-'
            ax.plot([0, self.width], [10 + 5 * i] * 2, c="white", linestyle=style, lw=1, alpha=0.8)

        # Draw yard numbers on both sides of the field.
        for units in range(10, 100, 10):
            units_text = units if units <= 50 else 100 - units
            ax.text(self.width - 7.5, 10 + units - 1.1, units_text, size=18, c="white", weight="bold", alpha=0.8)
            ax.text(7.5, 10 + units - 1.1, units_text, size=18, c="white", weight="bold", alpha=0.8)

        # Draw small tick marks along the sidelines.
        for x in range(20):
            for j in range(1, 5):
                ax.plot([1, 3], [10 + x * 5 + j] * 2, color="white", lw=1, alpha=0.8)
                ax.plot([self.width - 1, self.width - 3], [10 + x * 5 + j] * 2, color="white", lw=1, alpha=0.8)

        # Draw tick marks near the center of the field.
        gap = 5  
        center_x1, center_x2 = (self.width / 2) - gap, (self.width / 2) + gap
        for y in range(20):
            for j in range(1, 5):
                ax.plot([center_x1, center_x1 + 1], [10 + y * 5 + j] * 2, color="white", lw=1, alpha=0.8)
                ax.plot([center_x2, center_x2 - 1], [10 + y * 5 + j] * 2, color="white", lw=1, alpha=0.8)

        # Add team names and end zones.
        visitor_full_name = self.team_names.get(self.visitor_team, self.visitor_team)
        ax.text(self.width / 2, 5.5, self.visitor_team, size=30, c="white", weight="bold", ha='center')
        ax.text(self.width / 2, 1, visitor_full_name, size=30, c="white", weight="bold", ha='center')
        ax.add_patch(Rectangle((0, 0), self.width, 10, ec="black", fc=self.visitor_team_color, lw=1))

        home_full_name = self.team_names.get(self.home_team, self.home_team)
        ax.text(self.width / 2, self.height - 5, self.home_team, size=30, c="white", weight="bold", ha='center')
        ax.text(self.width / 2, self.height - 9, home_full_name, size=30, c="white", weight="bold", ha='center')
        ax.add_patch(Rectangle((0, self.height - 10), self.width, 10, ec="black", fc=self.home_team_color, lw=1))

        # Add the team logo at the center (if available).
        logo_path = os.path.join('assets/logo', f'{self.home_team}.png')
        if os.path.exists(logo_path):
            try:
                self._add_logo(ax, logo_path)
            except (UnidentifiedImageError, OSError) as e:
                self.logger.error(f"Error loading logo '{logo_path}': {e}")
        else:
            self.logger.warning(f"Logo file not found: {logo_path}")

        return fig, ax

    def _add_logo(self, ax, logo_path):
        """Adds the team logo at the center of the field."""
        logo = Image.open(logo_path).rotate(90, expand=True)
        width, height = logo.size
        square_size = max(width, height)
        square_logo = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
        square_logo.paste(logo, ((square_size - width) // 2, (square_size - height) // 2))
        ax.imshow(np.array(square_logo), extent=[(self.width - 10) / 2, (self.width + 10) / 2,
                                                 (self.height - 10) / 2, (self.height + 10) / 2], 
                  aspect='auto', zorder=10)

    def save_pitch(self, folder_path, filename='pitch.png'):
        """
        Saves the football field visualization to the specified folder.
        Args:
            folder_path (str): Directory to save the pitch image.
            filename (str): Name of the saved image file (default 'pitch.png').
        """
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, filename)
            self.fig.savefig(file_path, bbox_inches='tight')
            self.logger.info(f"Pitch saved successfully at: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save pitch: {e}")
        finally:
            plt.close(self.fig)




# import os
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.patches import Rectangle


# plt.rcParams['figure.dpi'] = 180
# plt.rcParams["figure.figsize"] = (25, 17)
# colors = sns.color_palette('Set3')
# sns.set_theme(rc={
#     'axes.facecolor': '#FFFFFF',
#     'figure.facecolor': '#FFFFFF',
#     'font.sans-serif': 'Arial',
#     'font.family': 'sans-serif'
# })


# class NFLFieldVertical:
#     def __init__(self, width=53.3, height=120, home_team="", home_team_color="", visitor_team = "", visitor_team_color="",logo_abbr=""):
#         self.width = width
#         self.height = height
#         self.home_team = home_team
#         self.home_team_color = home_team_color
#         self.visitor_team = visitor_team
#         self.visitor_team_color = visitor_team_color
#         self.team_names = {
#             'LA': 'Los Angeles Rams',
#             'ATL': 'Atlanta Falcons',
#             'CAR': 'Carolina Panthers',
#             'CHI': 'Chicago Bears',
#             'CIN': 'Cincinnati Bengals',
#             'DET': 'Detroit Lions',
#             'HOU': 'Houston Texans',
#             'MIA': 'Miami Dolphins',
#             'NYJ': 'New York Jets',
#             'WAS': 'Washington Commanders',
#             'ARI': 'Arizona Cardinals',
#             'LAC': 'Los Angeles Chargers',
#             'MIN': 'Minnesota Vikings',
#             'TEN': 'Tennessee Titans',
#             'DAL': 'Dallas Cowboys',
#             'SEA': 'Seattle Seahawks',
#             'KC': 'Kansas City Chiefs',
#             'BAL': 'Baltimore Ravens',
#             'CLE': 'Cleveland Browns',
#             'JAX': 'Jacksonville Jaguars',
#             'NO': 'New Orleans Saints',
#             'NYG': 'New York Giants',
#             'PIT': 'Pittsburgh Steelers',
#             'SF': 'San Francisco 49ers',
#             'DEN': 'Denver Broncos',
#             'LV': 'Las Vegas Raiders',
#             'GB': 'Green Bay Packers',
#             'BUF': 'Buffalo Bills',
#             'PHI': 'Philadelphia Eagles',
#             'IND': 'Indianapolis Colts',
#             'NE': 'New England Patriots',
#             'TB': 'Tampa Bay Buccaneers'
#         }
#         self.fig, self.ax = self.create_pitch()

#     def create_pitch(self):
#         fig, ax = plt.subplots()
#         ax.set_xlim(0, self.width)  
#         ax.set_ylim(0, self.height - 1) 
#         ax.axis('off')

#         background = Rectangle((0, 0), self.width, self.height, linewidth=1, facecolor='#97BC62FF', edgecolor='black', capstyle='round')
#         ax.add_patch(background)

#         for i in range(21):
#             if i % 2 != 0:
#                 ax.plot([0, self.width], [10 + 5 * i] * 2, c="white", linestyle='--', lw=1, alpha=0.4)
#             else:
#                 ax.plot([0, self.width], [10 + 5 * i] * 2, c="white", lw=1, alpha=0.8)

#         for units in range(10, 100, 10):
#             units_text = units if units <= 50 else 100 - units
#             ax.text(self.width - 7.5, 10 + units - 1.1, units_text, size=18, c="white", weight="bold", alpha=0.8)
#             ax.text(7.5, 10 + units - 1.1, units_text, size=18, c="white", weight="bold", alpha=0.8)

#         for x in range(20):
#             for j in range(1, 5):
#                 ax.plot([1, 3], [10 + x * 5 + j] * 2, color="white", lw=1, alpha=0.8)
#                 ax.plot([self.width - 1, self.width - 3], [10 + x * 5 + j] * 2, color="white", lw=1, alpha=0.8)
 
#         gap = 5  
#         center_x1 = (self.width / 2) - gap  
#         center_x2 = (self.width / 2) + gap 
#         for y in range(20):
#             for j in range(1, 5):
#                 ax.plot([center_x1, center_x1 + 1], [10 + y * 5 + j] * 2, color="white", lw=1, alpha=0.8)
#                 ax.plot([center_x2, center_x2 - 1], [10 + y * 5 + j] * 2, color="white", lw=1, alpha=0.8)

#         visitor_full_name = self.team_names.get(self.visitor_team, self.visitor_team)  # Get full name
#         ax.text(self.width / 2, 5.5, self.visitor_team, size=30, c="white", weight="bold", ha='center')  
#         ax.text(self.width / 2, 1, visitor_full_name, size=30, c="white", weight="bold", ha='center')  

#         end_zone_top = Rectangle((0, 0), self.width, 10, ec="black", fc=self.visitor_team_color, lw=1)
#         ax.add_patch(end_zone_top)

#         home_full_name = self.team_names.get(self.home_team, self.home_team)  # Get full name
#         ax.text(self.width / 2, self.height - 5, self.home_team, size=30, c="white", weight="bold", ha='center')  
#         ax.text(self.width / 2, self.height - 9, home_full_name, size=30, c="white", weight="bold", ha='center')  

#         end_zone_bottom = Rectangle((0, self.height - 10), self.width, 10, ec="black", fc=self.home_team_color, lw=1)
#         ax.add_patch(end_zone_bottom)

#         logo_path = os.path.join('assets/logo', f'{self.home_team}.png')
#         if os.path.exists(logo_path):
#             logo = Image.open(logo_path)
#             logo = logo.rotate(90, expand=True)
#             width, height = logo.size
#             square_size = max(width, height)
#             square_logo = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
#             paste_x = (square_size - width) // 2
#             paste_y = (square_size - height) // 2
#             square_logo.paste(logo, (paste_x, paste_y))
            
#             logo_array = np.array(square_logo)
#             logo_size = 10  
#             logo_extent = [
#                 (self.width - logo_size) / 2,
#                 (self.width + logo_size) / 2,
#                 (self.height - logo_size) / 2,
#                 (self.height + logo_size) / 2
#             ]
#             ax.imshow(logo_array, extent=logo_extent, aspect='auto', zorder=10)
#         else:
#             print(f"Logo file not found at {logo_path}")


#         return fig, ax

#     def save_pitch(self, folder_path, filename='pitch.png'):
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         file_path = os.path.join(folder_path, filename)
#         self.fig.savefig(file_path, bbox_inches='tight')
#         plt.close(self.fig)
