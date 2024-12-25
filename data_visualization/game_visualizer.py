import os
import time
import imageio
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
from utils.helpers import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from datetime import datetime, timedelta
from utils.helpers import NFLPlotVisualizer, NFLPlotVisualizeUtils
from data_preprocessesing.nfl_data_loader import SingleGamePlayExtractor


plt.rcParams['figure.dpi'] = 180
plt.rcParams["figure.figsize"] = (25, 17)

sns.set_theme(rc={
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif'
})

# Initilize the logger
logger = Logger().get_logger()

class GameVisualizer:    
    def __init__(self, df, pitch_image_path, assets_dir='assets/game-play/'):
        """
        Initializes the GameVisualizer with the provided game data, pitch image path, 
        and optional assets directory.

        Parameters:
        - df (DataFrame): DataFrame containing game play data.
        - pitch_image_path (str): Path to the pitch image used for visualization.
        - assets_dir (str): Directory path for storing generated assets. Default is 'assets/game-play/'.
        """
        self.df = df
        self.utils = NFLPlotVisualizeUtils()
        self.data_processor = SingleGamePlayExtractor(df)
        self.visualizer = NFLPlotVisualizer(pitch_image_path)
        self.assets_dir = assets_dir 
    

    def create_directory(self, gameId, playId):
        """
        Creates a directory structure based on the provided gameId and playId.

        Parameters:
        - gameId (int): The unique ID for the game.
        - playId (int): The unique ID for the play within the game.

        Returns:
        - str: The path to the created directory.
        """
        directory_path = os.path.join(self.assets_dir, f"game_{gameId}", f"play_{playId}")
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def get_game_clock_at(self, timestamp, start_time, game_clock_timedelta):
        """
        Calculates the remaining game clock time at a specific timestamp.

        Parameters:
        - timestamp (str): The current timestamp in ISO format.
        - start_time (datetime): The starting time of the game.
        - game_clock_timedelta (timedelta): The total game clock duration.

        Returns:
        - str: The remaining game time formatted as MM:SS.
        """
        current_time = datetime.fromisoformat(timestamp)
        elapsed_time = current_time - start_time
        remaining_time = game_clock_timedelta - elapsed_time

        if remaining_time.total_seconds() < 0:
            remaining_time = timedelta(0)

        minutes, seconds = divmod(int(remaining_time.total_seconds()), 60)
        return f"{minutes:02}:{seconds:02}"

    def plot_game_in_matplotlib(self, gameId, playId):
        """
        Plots the game play based on the provided gameId and playId using Matplotlib.

        Parameters:
        - gameId (int): The unique ID for the game to be visualized.
        - playId (int): The unique ID for the play within the game.

        Raises:
        - ValueError: If no data is available for the provided gameId and playId.
        """
        home_visitor_team_colors = {
            'LA': "#B3995D", 'ATL': "#A5ACAF", 'CAR': "#0085CA", 'CHI': "#DD4814", 
            'CIN': "#FB4F14", 'DET': "#0076B6", 'HOU': "#03202F", 'MIA': "#008E97", 
            'NYJ': "#203731", 'WAS': "#773141", 'ARI': "#97233F", 'LAC': "#FFB81C", 
            'MIN': "#4F2683", 'TEN': "#0C2340", 'DAL': "#869397", 'SEA': "#69BE28", 
            'KC': "#E31837", 'BAL': "#241773", 'CLE': "#FF3C00", 'JAX': "#006778", 
            'NO': "#D3BC8D", 'NYG': "#0B2265", 'PIT': "#FFB81C", 'SF': "#B3995D", 
            'DEN': "#FB4F14", 'LV': "#000000", 'GB': "#203731", 'BUF': "#00338D", 
            'PHI': "#004C54", 'IND': "#002C5F", 'NE': "#002244", 'TB': "#D50A0A"
        }

        team_colors = {
            'LA': "#B3995D", 'ATL': "#A5ACAF", 'CAR': "#0085CA", 'CHI': "#DD4814", 
            'CIN': "#FB4F14", 'DET': "#0076B6", 'HOU': "#03202F", 'MIA': "#008E97", 
            'NYJ': "#203731", 'WAS': "#773141", 'ARI': "#97233F", 'LAC': "#FFB81C", 
            'MIN': "#4F2683", 'TEN': "#0C2340", 'DAL': "#869397", 'SEA': "#69BE28", 
            'KC': "#E31837", 'BAL': "#241773", 'CLE': "#FF3C00", 'JAX': "#006778", 
            'NO': "#D3BC8D", 'NYG': "#0B2265", 'PIT': "#FFB81C", 'SF': "#B3995D", 
            'DEN': "#FB4F14", 'LV': "#000000", 'GB': "#203731", 'BUF': "#00338D", 
            'PHI': "#004C54", 'IND': "#002C5F", 'NE': "#002244", 'TB': "#D50A0A"
        }
        move_colors = {
            'LA': "#002244", 'ATL': "#000000", 'CAR': "#101820", 'CHI': "#0B162A", 
            'CIN': "#000000", 'DET': "#B0B7BC", 'HOU': "#A71930", 'MIA': "#F58220", 
            'NYJ': "#000000", 'WAS': "#FFB612", 'ARI': "#000000", 'LAC': "#FFB81C", 
            'MIN': "#FFC62F", 'TEN': "#4B92DB", 'DAL': "#041E42", 'SEA': "#A5ACAF", 
            'KC': "#FFB81C", 'BAL': "#000000", 'CLE': "#311D00", 'JAX': "#9F792C", 
            'NO': "#000000", 'NYG': "#A71930", 'PIT': "#101820", 'SF': "#AA0000", 
            'DEN': "#002244", 'LV': "#A5ACAF", 'GB': "#FFB81C", 'BUF': "#C60C30", 
            'PHI': "#A5ACAF", 'IND': "#A5ACAF", 'NE': "#C60C30", 'TB': "#FF7900"
        }
        look_colors = {
            'LA': "#FFFFFF", 'ATL': "#A71930", 'CAR': "#BFC0BF", 'CHI': "#FFFFFF", 
            'CIN': "#FFFFFF", 'DET': "#FFFFFF", 'HOU': "#FFFFFF", 'MIA': "#FFFFFF", 
            'NYJ': "#000000", 'WAS': "#FFFFFF", 'ARI': "#9F792C", 'LAC': "#FFB81C", 
            'MIN': "#FFFFFF", 'TEN': "#FFFFFF", 'DAL': "#FFFFFF", 'SEA': "#FFFFFF", 
            'KC': "#FFFFFF", 'BAL': "#9E7C0C", 'CLE': "#FFFFFF", 'JAX': "#FFFFFF", 
            'NO': "#FFFFFF", 'NYG': "#A5ACAF", 'PIT': "#FFFFFF", 'SF': "#FFFFFF", 
            'DEN': "#FFFFFF", 'LV': "#FFFFFF", 'GB': "#FFFFFF", 'BUF': "#FFFFFF", 
            'PHI': "#FFFFFF", 'IND': "#FFFFFF", 'NE': "#FFFFFF", 'TB': "#FFFFFF"
        }
        ball_snapped = False
        is_clock_active = False
        ball_snap_timestamp = None
        best_receiver_selected = False
        stop_line_plotting = False
        pass_forward_occurred = False
        outcome_marker_position = None
        outcome_text = ""
        best_receiver_widening = None
        best_receiver_beam = None
        # Initialize storage for frame-by-frame scores
        frame_ids = []
        receiver_scores = {} 

        if not ((self.df['gameId'] == gameId) & (self.df['playId'] == playId)).any():
            raise ValueError("No data available for the provided gameId and playId.")
        
        try:
            game_play_df, play_info = self.data_processor.extract_game_play_data(gameId, playId)
        except Exception as e:
            raise ValueError(f"Error extracting game data: {str(e)}")
            
        
        # Extract relevant information from play_info
        first_down_marker = self.data_processor.determine_first_down_marker(play_info)
        unique_frame_ids = play_info['unique_frame_ids']
        defensive_team = play_info['defensive_team']
        possession_team = play_info['possession_team']
        play_description = play_info['play_description']
        offense_formation = play_info['offense_formation']
        line_of_scrimmage = play_info['line_of_scrimmage']
        yard_line_number = play_info['yard_line_number']
        down = play_info['down']
        quarter = play_info['quarter']
        yards_to_go = play_info['yards_to_go']
        pre_snap_home_score = play_info['pre_snap_home_score']
        pre_snap_visitor_score = play_info['pre_snap_visitor_score']
        home_team_abbr = play_info['home_team_abbr']
        visitor_team_abbr = play_info['visitor_team_abbr']
        game_clock = play_info["game_clock"]
        clock_time_list = play_info["time"]
        current_game_clock = game_clock   
        all_unique_events = play_info["event"]     

        # Parse the game clock
        game_clock_minutes, game_clock_seconds = map(int, game_clock.split(':'))
        game_clock_timedelta = timedelta(minutes=game_clock_minutes, seconds=game_clock_seconds)
        clock_time_list = sorted(clock_time_list, key=lambda t: datetime.fromisoformat(t))
   
        # Determine team colors
        home_team_color = home_visitor_team_colors.get(home_team_abbr, "black") 
        visitor_team_color = home_visitor_team_colors.get(visitor_team_abbr, "black") 
        defense_team_color = team_colors.get(defensive_team, "black") 
        offense_team_color = team_colors.get(possession_team, "black")

        # Initialize movement and look colors
        defense_move_color = move_colors.get(defensive_team, "black")
        offense_move_color = move_colors.get(possession_team, "black")
        defense_look_color = look_colors.get(defensive_team, "black")
        offense_look_color = look_colors.get(possession_team, "black")
        
        # Create directory for saving images and videos
        save_directory = self.create_directory(gameId, playId)

        # Initialize the base plot
        self.utils.initialize_base_plot(home_team_abbr, home_team_color, visitor_team_abbr, visitor_team_color)

        # Dictionary to store past positions of selected players for trace
        player_traces = {'TE': {}, 'WR': {}, 'RB': {}}

        frames = []   
        # Iterate through unique frame IDs to plot each frame     
        for frameId in unique_frame_ids:
            frame_data = game_play_df[game_play_df['frameId'] == frameId]
            ball_data = frame_data[frame_data['Team'] == 'football']
            defensive_players_data = frame_data[frame_data['Team'] == defensive_team]
            offense_player_data = frame_data[frame_data['Team'] == possession_team]

            # Extract defense formation
            extract_defense_formation = defensive_players_data['position'].value_counts()
            defense_formation = ', '.join([f"{count}- {position}" for position, count in extract_defense_formation.items()])
            offense_player_data = frame_data[frame_data['Team'] == possession_team]
            events = frame_data['event'].astype(str).unique()   
            frame_types = frame_data['frameType'].astype(str).unique()   
    
            # Extract ball and player positions
            ball_x = ball_data['x'].values[0]
            ball_y = ball_data['y'].values[0]
            offense_x = offense_player_data['x']
            offense_y = offense_player_data['y']
            defensive_x = defensive_players_data['x']
            defensive_y = defensive_players_data['y']

            # Calculate min and max for plot limits
            min_x = min(min(offense_y), min(defensive_y), ball_y) - 5  
            max_x = max(max(offense_y), max(defensive_y), ball_y) + 5
            min_y = min(min(offense_x), min(defensive_x), ball_x) - 5 
            max_y = max(max(offense_x), max(defensive_x), ball_x) + 5

  
            # Initialize the plot for the current frame
            fig, ax = self.visualizer.initialize_plot(line_of_scrimmage, first_down_marker)
            self.visualizer.plot_network_graph(defensive_players_data,node_color=defense_team_color)

            # Check if any of the specified events occurred in this frame
            for event in events:
                if any(event in events for event in ['pass_outcome_caught', 'pass_outcome_incomplete', 'pass_outcome_touchdown']):
                    # Set the position to display the "X" marker and keep it in subsequent frames
                    outcome_marker_position = (ball_y, ball_x)
                    if event == 'pass_outcome_caught':
                        outcome_text = "Pass Caught"
                    elif event == 'pass_outcome_incomplete':
                        outcome_text = "Pass Incomplete"
                    elif event == 'pass_outcome_touchdown':
                        outcome_text = "Pass Touchdown"

            # Plot the "X" marker at the saved position if the event has occurred in this or any previous frame
            if outcome_marker_position is not None:
                ax.plot(outcome_marker_position[0], outcome_marker_position[1], marker='x', color='black', markersize=20,markeredgewidth=3,label=None) 
                ax.text(outcome_marker_position[0], outcome_marker_position[1] - 1.7, outcome_text, color='black', fontsize=20, ha='center', weight='bold')

            # Plot traces of players
            for position in player_traces:
                for player_id, trace in player_traces[position].items():
                    trace_x, trace_y = zip(*trace) 
                    ax.plot(trace_y, trace_x, marker='o', markersize=4, linestyle='-', color='white', linewidth=3, alpha=0.7, label=None)
            
            # Plot defensive players
            for _, player in defensive_players_data.iterrows():
                x1, y1 = player['x'], player['y']
                jersey_number = int(player['jerseyNumber'])
                self.visualizer.add_player_scatter_with_arrows(ax, x1, y1, jersey_number, defense_team_color, defense_move_color, defense_look_color, player['dir'], player['o'],label_prefix=f"{player['position']}: {jersey_number:02} - {player['displayName']}")
                ax.margins(0.1)

            # Extract QB position and defender positions
            qb_data = offense_player_data[offense_player_data['position'] == 'QB']
            if qb_data.empty:
                continue

            qb_x, qb_y = qb_data.iloc[0]['x'], qb_data.iloc[0]['y']
            defender_positions = [(def_row['x'], def_row['y']) for _, def_row in defensive_players_data.iterrows()]

            # Only find after the ball is snaped and finc till the event is not pass_forward
            if 'BEFORE_SNAP' in frame_types:
                ball_snapped = True

            if ball_snapped and not best_receiver_selected:
                best_receiver_widening, best_score_widening, best_receiver_beam, best_score_beam = self.visualizer.find_best_receiver(offense_player_data, qb_data.iloc[0], defender_positions)
                logger.warning(f"Primary (Widening): {best_receiver_widening}, Backup (Beam): {best_receiver_beam}")
                
                if any(event in events for event in ['pass_forward']): # 'pass_outcome_caught', 'pass_outcome_incomplete', 'pass_outcome_touchdown'
                    best_receiver_selected = True


            # Once stop_line_plotting is True, it should stay True for all remaining frames
            if not stop_line_plotting:
                stop_line_plotting = any(event in events for event in ['pass_outcome_caught', 'pass_outcome_incomplete', 'pass_outcome_touchdown'])            
            
            # Store scores for each eligible receiver for both widening and beam methods
            for _, receiver in offense_player_data[offense_player_data['position'].isin(['TE', 'WR', 'RB'])].iterrows():
                receiver_name = receiver['displayName']
                
                # Here, obtain or calculate scores for each receiver
                score_widening = best_score_widening if receiver_name == best_receiver_widening else None
                score_beam = best_score_beam if receiver_name == best_receiver_beam else None
                
                # Initialize score lists for each receiver if not already done
                if receiver_name not in receiver_scores:
                    receiver_scores[receiver_name] = {'widening': [], 'beam': []}
                
                # Append scores for the current frameß
                receiver_scores[receiver_name]['widening'].append(score_widening)
                receiver_scores[receiver_name]['beam'].append(score_beam)
            
            # Plot offensive players
            for _, off_row in offense_player_data.iterrows():
                x1, y1 = off_row['x'], off_row['y']
                x2, y2 = qb_x, qb_y
                position = off_row['position']
        
                offense_jersey_number = int(off_row['jerseyNumber'])
                self.visualizer.add_player_scatter_with_arrows(ax, off_row['x'], off_row['y'], offense_jersey_number, offense_team_color, offense_move_color, offense_look_color, off_row['dir'], off_row['o'],label_prefix=f"{off_row['position']}: {offense_jersey_number:02} - {off_row['displayName']}")

                # Update trace for eligible offensive players
                if position in ['TE', 'WR', 'RB']:
                    player_id = off_row['displayName']
                    if player_id not in player_traces[position]:
                        player_traces[position][player_id] = []  # Initialize trace if not present
                    player_traces[position][player_id].append((x1, y1)) 

                # Plot distance lines only for eligible receivers
                if off_row['position'] in ['TE', 'WR', 'RB'] and not stop_line_plotting:
                    # Calculate distance between QB and the eligible receiver
                    distance = self.utils.calculate_distance(x1, y1, x2, y2)
                    # Use the custom function to add the distance line
                    self.visualizer.add_player_distance_line(ax, distance, x1, x2, y1, y2)
                
                # Highlight best receiver
                if off_row['displayName'] in [best_receiver_widening, best_receiver_beam]:
                    if 'pass_forward' in events:
                        pass_forward_occurred = True

                    if not pass_forward_occurred:
                        if best_receiver_widening == best_receiver_beam and off_row['displayName'] == best_receiver_widening:
                            ax.text(off_row['y'], off_row['x'] + 1, "Best Receiver", color="black", fontsize=14, weight='bold')
                            small_radius = 1
                            x_vals = np.linspace(off_row['x'] - small_radius, off_row['x'] + small_radius, 50)
                            y_vals = np.linspace(off_row['y'] - small_radius, off_row['y'] + small_radius, 50)
                            X, Y = np.meshgrid(x_vals, y_vals)
                            distance_from_center = np.sqrt((X - off_row['x'])**2 + (Y - off_row['y'])**2)
                            Z = np.exp(-distance_from_center**2 / 0.01)
                            Z[distance_from_center > small_radius] = np.nan
                            ax.contourf(Y, X, Z, levels=10, cmap="twilight", alpha=0.5)
                        else:
                            if best_receiver_widening is not None and off_row['displayName'] == best_receiver_widening:
                                ax.text(off_row['y'], off_row['x'] + 1, "Strategic Receiver", color="black", fontsize=14, weight='bold')
                                small_radius = 1
                                x_vals = np.linspace(off_row['x'] - small_radius, off_row['x'] + small_radius, 50)
                                y_vals = np.linspace(off_row['y'] - small_radius, off_row['y'] + small_radius, 50)
                                X, Y = np.meshgrid(x_vals, y_vals)
                                distance_from_center = np.sqrt((X - off_row['x'])**2 + (Y - off_row['y'])**2)
                                Z = np.exp(-distance_from_center**2 / 0.01)
                                Z[distance_from_center > small_radius] = np.nan
                                ax.contourf(Y, X, Z, levels=10, cmap="twilight", alpha=0.5)
                            
                            if best_receiver_beam is not None and off_row['displayName'] == best_receiver_beam:
                                ax.text(off_row['y'], off_row['x'] - 1.2, "Primary Receiver", color="black", fontsize=14, weight='bold')
                                small_radius = 1
                                x_vals = np.linspace(off_row['x'] - small_radius, off_row['x'] + small_radius, 50)
                                y_vals = np.linspace(off_row['y'] - small_radius, off_row['y'] + small_radius, 50)
                                X, Y = np.meshgrid(x_vals, y_vals)
                                distance_from_center = np.sqrt((X - off_row['x'])**2 + (Y - off_row['y'])**2)
                                Z = np.exp(-distance_from_center**2 / 0.01)
                                Z[distance_from_center > small_radius] = np.nan
                                ax.contourf(Y, X, Z, levels=10, cmap="twilight", alpha=0.5)

            # Plot the ball
            ax.add_artist(Ellipse((ball_y, ball_x), 0.5, 0.55, facecolor="#755139FF", ec="k", lw=2)) 
            ax.axes.get_yaxis().set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            ax.set_axis_on()
            ax.axes.get_xaxis().set_visible(True)
            ax.set_xlabel(None)
    
            # Format and set plot title
            words = play_description.split()
            formatted_play_description = '\n'.join(' '.join(words[i:i+20]) for i in range(0, len(words), 20))
            title_str = (
                f'FRAME: {frameId}      ✤ Play Description: {formatted_play_description}\n'
                f'✤ Offense Formation: {offense_formation}     ✤ Defense Formation: {defense_formation}'
            )
            ax.set_title(title_str, x=0.6, y=1, fontweight='bold', fontsize=18)

            # Prepare legends and score display
            down_suffix = self.visualizer.get_down_suffix(down) 
            quarter_suffix = self.visualizer.get_down_suffix(quarter) 
            top_handles = [
                Line2D([0], [0], marker='o', color='w', label=defensive_team, markersize=28, 
                       markerfacecolor=defense_team_color, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label=possession_team, markersize=28, 
                       markerfacecolor=offense_team_color, markeredgecolor='k'),
                Line2D([0], [0], marker='|', color='#00539CFF', label=f'LOS: {yard_line_number}', linestyle='None',
                       markersize=28, markeredgewidth=4),
                Line2D([0], [0], marker='|', color='#FDD20EFF', label=f'Down: {down}', linestyle='None',
                       markersize=28, markeredgewidth=4),
            ]
            top_labels = [
                f'Defense: {defensive_team}', f'Offense: {possession_team}', 
                f'LOS: {yard_line_number} yds', f'Down: {down}'
            ]
            self.visualizer.add_legends(ax, top_handles, top_labels)

            # Add arrows legend
            arrow_handles = [
                Line2D([0], [0], label='Defense Player Moving', marker='>', markersize=20, markeredgecolor="black", 
                       markerfacecolor=defense_move_color, linestyle='-', color="white", linewidth=3),
                Line2D([0], [0], label='Defense Player Facing', marker='>', markersize=20, markeredgecolor='black', 
                       markerfacecolor=defense_look_color, linestyle='-', color="white", linewidth=3),
                Line2D([0], [0], label='Offense Player Moving', marker='>', markersize=20, markeredgecolor="black", 
                       markerfacecolor=offense_move_color, linestyle='-', color="white", linewidth=3),
                Line2D([0], [0], label='Defense Player Facing', marker='>', markersize=20, markeredgecolor='black', 
                markerfacecolor=offense_look_color, linestyle='-', color="white", linewidth=3),
            ]
            arrow_legend = ax.legend(title="Players Direction",handles=arrow_handles, loc='center left', bbox_to_anchor=(1, 0.07), fontsize='x-large',title_fontsize=22,prop={'size': 19})
            ax.add_artist(arrow_legend)

            # Add player legend
            ax.legend(title="Players Roster", loc='center left', bbox_to_anchor=(1, 0.52),fontsize='x-large', title_fontsize=22, prop={'size': 19})

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.axis('off')

            # Add score board       
            if 'ball_snap' in events and not is_clock_active:
                is_clock_active = True
                ball_snap_timestamp = frame_data.iloc[0]['time'] 

            if is_clock_active:
                current_game_clock = self.get_game_clock_at(
                    frame_data.iloc[0]['time'],  
                    datetime.fromisoformat(ball_snap_timestamp),  
                    game_clock_timedelta  
                )

            scoreboard_str = (
                f"{home_team_abbr}: {pre_snap_home_score}   |   "
                f"{visitor_team_abbr}: {pre_snap_visitor_score}   |   "
                f"{down}{down_suffix} & {yards_to_go}   |   "
                f"{quarter}{quarter_suffix}      {current_game_clock}"
            )

            fig.text(0.5, 0.1, scoreboard_str, ha='center', va='top', fontsize=45, fontweight='bold')

            # Append frame_id and corresponding scores
            frame_ids.append(frameId)

            # Create and save the image from frames
            image_filename = f"{gameId}_{playId}_{frameId:04d}.png"
            image_path = os.path.join(save_directory, image_filename)
            self.visualizer.save_plot_to_image(fig, image_path)
            frames.append(Image.open(image_path).copy()) 
            logger.info(f"Frame {frameId} saved at {image_path}")

            # plt.cla() 
            # ax.clear()
            # plt.close(fig)

        # --- New Section: Create Separate Score Plot ---
     
        # Filter out NaN events in game_play_df and create (frameId, event) pairs
        event_frames = [(row['frameId'], row['event']) for _, row in game_play_df.iterrows() if pd.notna(row['event'])]

        # Clean receiver_scores by replacing None with 0 and ensuring all values are floats
        cleaned_receiver_scores = {
            receiver_name: {
                'widening': [float(score) if score is not None else 0.0 for score in scores['widening']],
                'beam': [float(score) if score is not None else 0.0 for score in scores['beam']]
            }
            for receiver_name, scores in receiver_scores.items()
        }

        pass_forward_frame = next(
            (frame for frame, event in event_frames if event == 'pass_forward'),
            max(frame_ids)  # Default to the max frame ID if "pass_forward" is not found
        )

        # Filter frame_ids and receiver scores up to the pass_forward_frame
        filtered_event_frames = [(frame, event) for frame, event in event_frames if frame <= pass_forward_frame]
        filtered_frame_ids = [frame for frame in frame_ids if frame <= pass_forward_frame]
        filtered_receiver_scores = {
            receiver_name: {
                'widening': scores['widening'][:len(filtered_frame_ids)],
                'beam': scores['beam'][:len(filtered_frame_ids)]
            }
            for receiver_name, scores in cleaned_receiver_scores.items()
        }

        # Set max_x to the pass_forward_frame for limiting x-axis range
        max_x = pass_forward_frame

        # Proceed with calculating max_y as before
        max_y_score = max(
            max(scores['widening'] + scores['beam']) for scores in filtered_receiver_scores.values()
        )
        max_y = math.ceil(max_y_score)
        if max_y % 2 != 0:
            max_y += 1  # Make it even if it's odd

       # Define a color palette with unique colors for each receiver
        receiver_names = list(cleaned_receiver_scores.keys())
        num_receivers = len(receiver_names)
        colors = sns.color_palette("pastel", num_receivers)

        # Plot with sns.regplot as before
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


        # Apply regplot for widening scores (first subplot)
        for idx, (receiver_name, scores) in enumerate(filtered_receiver_scores.items()):
            sns.regplot(
                x=filtered_frame_ids,
                y=scores['widening'],
                order=5,
                line_kws={'color': colors[idx], 'linestyle': '-.'},
                ax=ax1,
                scatter=False,
                label=f"{receiver_name}"
            )

        # Add vertical lines at event frames for the widening scores plot
        for frame, event_name in filtered_event_frames:
            ax1.axvline(x=frame, color='black', linestyle='--', alpha=0.6, label=None)
            ax1.text(frame+3, max_y / 2 - 0.6, event_name, color='black', ha='center', va='center', rotation=90, fontsize=10)


        ax1.set_title("Progressive Widening Search")
        ax1.set_xlabel("Frames")
        ax1.set_ylabel("Score")
        ax1.set_xlim(0, max_x)
        ax1.set_ylim(0, max_y)
        ax1.xaxis.set_major_locator(plt.MultipleLocator(20))  # X-axis interval of 10
        ax1.yaxis.set_major_locator(plt.MultipleLocator(2))  # Y-axis interval of 2
        ax1.grid(which='major', color='#CCCCCC', linestyle='--')
        ax1.grid(which='minor', color='#CCCCCC', linestyle=':')

        # Apply regplot for beam scores (second subplot)
        for idx, (receiver_name, scores) in enumerate(filtered_receiver_scores.items()):
            sns.regplot(
                x=filtered_frame_ids,
                y=scores['beam'],
                order=3,
                line_kws={'color': colors[idx], 'linestyle': '-.'},
                ax=ax2,
                scatter=False,
                label=f"{receiver_name}"
            )

        # Add vertical lines at event frames for the beam scores plot
        for frame, event_name in filtered_event_frames:
            ax2.axvline(x=frame, color='black', linestyle='--', alpha=0.6, label=None)
            ax2.text(frame+3, max_y / 2 - 0.6, event_name, color='black', ha='center', va='center', rotation=90, fontsize=10)


        ax2.set_title("Beam Traget Search")
        ax2.set_xlabel("Frames")
        ax2.set_ylabel("Score")
        ax2.set_xlim(0, max_x)
        ax2.set_ylim(0, max_y)
        ax2.xaxis.set_major_locator(plt.MultipleLocator(20))  # X-axis interval of 10
        ax2.yaxis.set_major_locator(plt.MultipleLocator(2))  # Y-axis interval of 2
        ax2.grid(which='major', color='#CCCCCC', linestyle='--')
        ax2.grid(which='minor', color='#CCCCCC', linestyle=':')

    
        # Create a single legend for the entire figure at the bottom
        handles, labels = ax2.get_legend_handles_labels()
        fig2.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.01), frameon=False)
       
        # Save the receiver score plot as a separate PNG file
        score_plot_path = os.path.join(save_directory, f"{gameId}_{playId}_receiver_scores.png")
        plt.tight_layout() 
        plt.savefig(score_plot_path,bbox_inches='tight')
        plt.close(fig2)
        logger.info(f"Receiver score plot saved at {score_plot_path}")

        # Create and save the gif from frames
        gif_path = os.path.join(save_directory, f"{gameId}_{playId}_animation.gif")
        frames[0].save(gif_path, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=120)
        logger.info(f"Video saved at {gif_path}")

        # Create and save the video from frames
        video_path = os.path.join(save_directory, f"{gameId}_{playId}_animation.mp4")
        processed_frames = self.visualizer.process_frames(frames)
        with imageio.get_writer(video_path, format='ffmpeg', fps=6) as writer:
            for resized_frame in processed_frames:
                writer.append_data(np.array(resized_frame))
        logger.info(f"Video saved at {video_path}")
        time.sleep(10)