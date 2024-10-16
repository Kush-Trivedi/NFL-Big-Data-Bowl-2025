import os
import sys
import time
import logging
import numpy as np
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from data_preprocessesing.nfl_field import NFLFieldVertical


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG"    : "\033[94m",
        "INFO"     : "\033[92m",
        "WARNING"  : "\033[93m",
        "ERROR"    : "\033[91m",
        "CRICTICAL": "\033[41m",
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.RESET)}{log_message}{self.RESET}"

class Logger:
    _handlers_added = False

    def __init__(self, name=__name__, level=logging.INFO, stream=sys.stdout):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not Logger._handlers_added:
            self.handler = logging.StreamHandler(stream)
            self.formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.handler.setFormatter(self.formatter)
            self.logger.addHandler(self.handler)
            Logger._handlers_added = True
    
    def get_logger(self):
        """Returns the logger instance."""
        return self.logger
    

class NFLPlotVisualizeUtils:    
    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points.
        
        Parameters:
            x1 (float): X-coordinate of the first point.
            y1 (float): Y-coordinate of the first point.
            x2 (float): X-coordinate of the second point.
            y2 (float): Y-coordinate of the second point.
        
        Returns:
            float: The Euclidean distance between the two points.
        """
        try:
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error calculating distance: {e}")
            return None

    @staticmethod
    def angle_with_x_axis(x, y):
        """
        Calculate the angle of the line connecting the origin to the point (x, y) with respect to the x-axis.
        
        Parameters:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.
        
        Returns:
            float: Angle in degrees with respect to the x-axis.
        """
        try:
            return np.degrees(np.arctan2(y, x))
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error calculating angle with x-axis: {e}")
            return None

    @staticmethod
    def angle_in_32_segments(angle):
        """
        Convert an angle to a segment from 0 to 31 (32 segments total).
        
        Parameters:
            angle (float): Angle in degrees.
        
        Returns:
            int: Segment number corresponding to the angle.
        """
        if np.isnan(angle):
            return 0 
        angle = angle % 360
        return round(angle / 11.25)
    
    @staticmethod
    def assign_direction(angle):
        """
        Assign a compass direction based on the angle.
        
        Parameters:
            angle (float): Angle in degrees.
        
        Returns:
            str: Compass direction (e.g., "North", "Southwest").
        """
        directions = [
            "North", "North by East", "North-Northeast", "Northeast by North",
            "Northeast", "Northeast by East", "East-Northeast", "East by North",
            "East", "East by South", "East-Southeast", "Southeast by East",
            "Southeast", "Southeast by South", "South-Southeast", "South by East",
            "South", "South by West", "South-Southwest", "Southwest by South",
            "Southwest", "Southwest by West", "West-Southwest", "West by South",
            "West", "West by North", "West-Northwest", "Northwest by West",
            "Northwest", "Northwest by North", "North-Northwest", "North by West",
            "North"  
        ]
        
        bounds = [i * 11.25 for i in range(33)]

        for i in range(len(bounds) - 1):
            lower_bound = bounds[i]
            upper_bound = bounds[i + 1]
            if lower_bound <= angle < upper_bound:
                return directions[i]

        return None
    
    @staticmethod
    def calculate_dx_dy_arrow(x, y, angle, speed, multiplier):
        """
        Calculate the change in x and y (dx, dy) for an arrow based on its angle and speed.
        
        Parameters:
            x (float): X-coordinate of the starting point.
            y (float): Y-coordinate of the starting point.
            angle (float): Angle in degrees.
            speed (float): Speed (used for scaling).
            multiplier (float): Multiplier for adjusting arrow length.
        
        Returns:
            tuple: (dx, dy) change in coordinates for the arrow.
        """
        fixed_length = 0.5 * multiplier  # Adjust arrow length by multiplier
        angle_radians = np.radians(angle)
        
        try:
            if angle <= 90:
                dx = np.sin(angle_radians) * fixed_length
                dy = np.cos(angle_radians) * fixed_length
            elif angle <= 180:
                angle_radians = np.radians(angle - 90)
                dx = np.sin(angle_radians) * fixed_length
                dy = -np.cos(angle_radians) * fixed_length
            elif angle <= 270:
                angle_radians = np.radians(angle - 180)
                dx = -np.sin(angle_radians) * fixed_length
                dy = -np.cos(angle_radians) * fixed_length
            else: 
                angle_radians = np.radians(360 - angle)
                dx = -np.sin(angle_radians) * fixed_length
                dy = np.cos(angle_radians) * fixed_length
            return dx, dy
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error calculating dx and dy: {e}")
            return None, None

    @staticmethod
    def calculate_relative_velocity(speed1, speed2, dir1, dir2):
        """
        Calculate the relative velocity between two moving entities based on their speeds and directions.
        
        Parameters:
            speed1 (float): Speed of the first entity.
            speed2 (float): Speed of the second entity.
            dir1 (float): Direction of the first entity in degrees.
            dir2 (float): Direction of the second entity in degrees.
        
        Returns:
            float: Magnitude of the relative velocity.
        """
        try:
            # Convert directions to radians
            theta1 = np.radians(dir1)
            theta2 = np.radians(dir2)

            # Calculate velocity components
            vx1 = speed1 * np.cos(theta1)
            vy1 = speed1 * np.sin(theta1)
            vx2 = speed2 * np.cos(theta2)
            vy2 = speed2 * np.sin(theta2)

            # Calculate relative velocity components
            rel_vx = vx1 - vx2
            rel_vy = vy1 - vy2

            # Calculate magnitude of relative velocity
            return np.sqrt(rel_vx**2 + rel_vy**2)
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error calculating relative velocity: {e}")
            return None

    @staticmethod
    def calculate_fixed_arrow(angle, length=0.38):
        """
        Calculate the change in coordinates (dx, dy) for a fixed-length arrow based on its angle.
        
        Parameters:
            angle (float): Angle in degrees.
            length (float): Length of the arrow.
        
        Returns:
            tuple: (dx, dy) change in coordinates for the arrow.
        """
        try:
            angle_radians = np.radians(angle)  

            dx = np.cos(angle_radians) * length
            dy = np.sin(angle_radians) * length

            return dx, dy
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error calculating fixed arrow: {e}")
            return None, None


    @staticmethod
    def calculate_time_to_contact(distance, rel_velocity):
        """
        Calculate the time to contact based on the distance and relative velocity.
        
        Parameters:
            distance (float): Distance to the target.
            rel_velocity (float): Relative velocity towards the target.
        
        Returns:
            float: Time to contact in seconds, or np.inf if the relative velocity is zero or negative.
        """
        try:
            return distance / rel_velocity if rel_velocity > 0 else np.inf
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error calculating time to contact: {e}")
            return None

    @staticmethod
    def calculate_angle_of_approach(dir1, dir2):
        """
        Calculate the angle of approach between two directions.
        
        Parameters:
            dir1 (float): Direction of the first entity in degrees.
            dir2 (float): Direction of the second entity in degrees.
        
        Returns:
            float: Angle in degrees between the two directions.
        """
        try:
            # Convert directions to radians and calculate vectors
            vector1 = np.array([np.cos(np.radians(dir1)), np.sin(np.radians(dir1))])
            vector2 = np.array([np.cos(np.radians(dir2)), np.sin(np.radians(dir2))])

            # Calculate the angle between the vectors
            angle = np.degrees(np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0]))
            return angle % 360  
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error calculating angle of approach: {e}")
            return None
    
    @staticmethod
    def resize_for_video(image, target_size):
        """
        Resize an image to the specified target size for video presentation.

        Parameters:
            image (PIL.Image.Image): The image to be resized.
            target_size (tuple): A tuple representing the target size (width, height) in pixels.

        Returns:
            PIL.Image.Image: The resized image.
        """
        return image.resize(target_size, Image.LANCZOS)

    @staticmethod
    def initialize_base_plot(home_team_abbr, home_team_color, visitor_team_abbr, visitor_team_color):
        """
        Initialize the base plot for the NFL field with the specified teams and their colors.

        Parameters:
            home_team_abbr (str): The abbreviation of the home team (e.g., "NE" for New England Patriots).
            home_team_color (str): The color code for the home team (e.g., "#FF0000" for red).
            visitor_team_abbr (str): The abbreviation of the visitor team (e.g., "KC" for Kansas City Chiefs).
            visitor_team_color (str): The color code for the visitor team (e.g., "#0000FF" for blue).

        Returns:
            None
        """
        # Store team abbreviations and colors for plotting
        home_team_abbr = home_team_abbr
        home_team_color = home_team_color
        visitor_team_abbr = visitor_team_abbr
        visitor_team_color = visitor_team_color
        
        # Create an NFL field pitch with the specified dimensions and team details
        pitch = NFLFieldVertical(
            width=53.3,  # Width of the field in meters
            height=120,  # Height of the field in meters
            home_team=home_team_abbr,
            home_team_color=home_team_color,
            visitor_team=visitor_team_abbr,
            visitor_team_color=visitor_team_color
        )
        
        # Save the pitch as an image in the specified folder
        pitch.save_pitch(folder_path='assets', filename='vertical_pitch.png')
        
        # Pause the execution for a specified duration to ensure the pitch is saved
        time.sleep(7)
    
    

class NFLPlotVisualizer:    
    def __init__(self, pitch_image_path):
        """
        Initialize the NFLPlotVisualizer with the path to the pitch image.

        Parameters:
            pitch_image_path (str): The file path to the image of the pitch.

        Returns:
            None
        """
        self.pitch_image_path = pitch_image_path
        try:
            self.pitch_img = Image.open(self.pitch_image_path)
        except Exception as e:
            logging.error(f"Error loading image at {self.pitch_image_path}: {e}")
            raise

    def initialize_plot(self, line_of_scrimmage, first_down_marker):
        """
        Create a new plot with the NFL pitch image and draw the line of scrimmage and first down marker.

        Parameters:
            line_of_scrimmage (float): The y-coordinate of the line of scrimmage.
            first_down_marker (float): The y-coordinate of the first down marker.

        Returns:
            tuple: A tuple containing the figure and axes objects for further customization.
        """
        try:
            fig, ax = plt.subplots()
            ax.imshow(self.pitch_img, extent=[0, 53.3, 0, 120], aspect='auto')
            ax.axhline(y=line_of_scrimmage, color='#00539CFF', linestyle='-', linewidth=4)
            ax.axhline(y=first_down_marker, color='#FDD20EFF', linestyle='-', linewidth=4)
            return fig, ax
        except Exception as e:
            logging.error(f"Error initializing plot: {e}")
            raise

    def process_frames(self, frames):
        """
        Resize a list of image frames to ensure they are all the same size for video processing.

        Parameters:
            frames (list): A list of PIL Image objects representing video frames.

        Returns:
            list: A list of resized PIL Image objects.
        """
        try:
            first_width, first_height = frames[0].size
            target_width = (first_width + 15) // 16 * 16
            target_height = (first_height + 15) // 16 * 16
            target_size = (target_width, target_height)
            resized_frames = [NFLPlotVisualizeUtils.resize_for_video(frame, target_size) for frame in frames]
            return resized_frames
        except Exception as e:
            logging.error(f"Error processing frames: {e}")
            raise

    def add_player_scatter(self, ax, x, y, jersey_number, team_color, label_prefix=''):
        """
        Add a scatter plot for a player at the specified coordinates, with the jersey number displayed.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to draw the scatter plot.
            x (float): The x-coordinate of the player.
            y (float): The y-coordinate of the player.
            jersey_number (int): The jersey number of the player.
            team_color (str): The color to use for the player's marker.
            label_prefix (str, optional): Optional prefix for the label. Defaults to ''.

        Returns:
            None
        """
        try:
            ax.scatter(y, x, color=team_color, s=1000, edgecolors='k', label=f"{label_prefix}")
            ax.text(y, x, str(jersey_number), color='white', ha='center', va='center', fontsize=12, weight='bold')
        except Exception as e:
            logging.error(f"Error adding player scatter: {e}")
            raise

    def add_player_distance_line(self, ax, distance, x1, x2, y1, y2):
        """
        Draw a line between two points representing players, varying the line style based on distance.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to draw the line.
            distance (float): The distance between the two points.
            x1 (float): The y-coordinate of the first point.
            x2 (float): The y-coordinate of the second point.
            y1 (float): The x-coordinate of the first point.
            y2 (float): The x-coordinate of the second point.

        Returns:
            None
        """
        try:
            if 5 < distance < 7:
                ax.plot([y1, y2], [x1, x2], color='k', linestyle='-', linewidth=3, alpha=0.9)
            elif distance < 5:
                ax.plot([y1, y2], [x1, x2], color='k', linestyle='--', linewidth=2.5, alpha=0.6)

            if distance < 5 or 5 < distance < 7:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                ax.text(mid_y, mid_x, f'{int(distance)} yd',
                        fontsize=15, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))
        except Exception as e:
            logging.error(f"Error adding player distance line: {e}")
            raise

    def add_player_moving_looking_direction(self, ax, x, y, dx, dy, color):
        """
        Draw an arrow to represent a player's movement direction on the plot.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to draw the arrow.
            x (float): The y-coordinate of the player.
            y (float): The x-coordinate of the player.
            dx (float): The change in x-coordinate (movement).
            dy (float): The change in y-coordinate (movement).
            color (str): The color of the arrow.

        Returns:
            None
        """
        try:
            ax.arrow(y, x, dx, dy, color=color, ec='black', width=0.25, head_width=0.4, head_length=0.25, shape='full', alpha=0.7)
        except Exception as e:
            logging.error(f"Error adding player moving direction arrow: {e}")
            raise

    def add_player_scatter_with_arrows(self, ax, x, y, jersey_number, team_color, move_color, look_color, moving_angle, looking_angle, length=0.32, label_prefix=''):
        """
        Add a scatter plot for a player with arrows indicating movement and looking direction.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to draw the scatter plot and arrows.
            x (float): The y-coordinate of the player.
            y (float): The x-coordinate of the player.
            jersey_number (int): The jersey number of the player.
            team_color (str): The color of the player's marker.
            move_color (str): The color of the movement arrow.
            look_color (str): The color of the looking direction arrow.
            moving_angle (float): The angle of movement in degrees.
            looking_angle (float): The angle of looking direction in degrees.
            length (float, optional): The length of the arrows. Defaults to 0.32.
            label_prefix (str, optional): Optional prefix for the label. Defaults to ''.

        Returns:
            None
        """
        try:
            dx_m, dy_m = NFLPlotVisualizeUtils.calculate_fixed_arrow(moving_angle, length=length)
            dx_l, dy_l = NFLPlotVisualizeUtils.calculate_fixed_arrow(looking_angle, length=length)

            ax.arrow(y, x, dx_l, dy_l, color=look_color, width=0.1, head_width=0.5, head_length=0.5, alpha=0.8, ec='black', overhang=0.1)
            ax.arrow(y, x, dx_m, dy_m, color=move_color, width=0.1, head_width=0.5, head_length=0.5, alpha=0.8, ec='black', overhang=0.1)

            ax.scatter(y, x, color=team_color, s=1100, edgecolors='black', label=f"{label_prefix}")
            ax.text(y, x, str(jersey_number), color='white', ha='center', va='center', fontsize=12, weight='bold')
        except Exception as e:
            logging.error(f"Error adding player scatter with arrows: {e}")
            raise

    def get_down_suffix(self, down):
        """
        Get the ordinal suffix for a given down (e.g., 1st, 2nd, 3rd, 4th).

        Parameters:
            down (int): The down number.

        Returns:
            str: The ordinal suffix for the down number.
        """
        try:
            if down == 1:
                return "st"
            elif down == 2:
                return "nd"
            elif down == 3:
                return "rd"
            else:
                return "th"
        except Exception as e:
            logging.error(f"Error getting down suffix for {down}: {e}")
            raise

    def add_legends(self, ax, top_handles, top_labels):
        """
        Add a legend to the plot.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to add the legend.
            top_handles (list): A list of handles for the legend entries.
            top_labels (list): A list of labels corresponding to the handles.

        Returns:
            None
        """
        try:
            top_legend = ax.legend(
                title="Team and Game Situation",
                handles=top_handles,
                labels=top_labels,
                loc='center left',
                bbox_to_anchor=(1, 0.94),
                prop={'size': 16},
                ncol=2,
                title_fontsize=22
            )
            ax.add_artist(top_legend)
            logging.info("Legend added successfully.")
        except Exception as e:
            logging.error(f"Error adding legend: {e}")
            raise

    def plot_network_graph(self, defensive_players_df, node_color):
        """
        Plot a network graph of defensive players based on their positions, 
        adding edges between players that are within a certain distance.

        Parameters:
            defensive_players_df (DataFrame): A DataFrame containing player information 
                                                with columns 'displayName', 'x', and 'y'.
            node_color (str): The color to use for the nodes in the network graph.

        Returns:
            None
        """
        try:
            # Initialize a new graph
            G = nx.Graph()

            # Add nodes to the graph using player display names and their positions
            for _, row in defensive_players_df.iterrows():
                G.add_node(row['displayName'], pos=(row['y'], row['x']))  # 'y' is x-coordinate and 'x' is y-coordinate

            # Add edges between nodes that are within a distance of 5 units
            for node1 in G.nodes:
                for node2 in G.nodes:
                    if node1 != node2:  # Avoid self-loops
                        x1, y1 = G.nodes[node1]['pos']  # Get position of node1
                        x2, y2 = G.nodes[node2]['pos']  # Get position of node2
                        distance = NFLPlotVisualizeUtils.calculate_distance(x1, y1, x2, y2)  # Calculate the distance
                        if distance < 5:  # Check if within distance threshold
                            G.add_edge(node1, node2, weight=1/distance)  # Add edge with weight based on distance

            # Get positions of nodes for plotting
            pos = nx.get_node_attributes(G, 'pos')

            # Draw the network graph without labels, with specified node size and color
            nx.draw_networkx(G, pos, with_labels=False, node_size=1100, font_size=12, font_weight='bold', node_color=node_color)

            # Identify cycles in the graph for additional visual representation
            cycles = list(nx.simple_cycles(G))  # Find simple cycles in the graph
            nx.draw_networkx_nodes(G, pos, node_size=1100, node_color=node_color, edgecolors="black", alpha=0.2)  # Draw nodes

            # Draw edges of cycles with specific styling
            nx.draw_networkx_edges(G, pos, edgelist=cycles, edge_color='black', width=2, alpha=0.4)

            # Draw a filled polygon around each cycle for visual emphasis
            for cycle in cycles:
                cycle_nodes = cycle + [cycle[0]]  # Close the cycle
                cycle_pos = np.array([pos[node] for node in cycle_nodes])  # Get the positions of the cycle nodes
                polygon = plt.Polygon(cycle_pos, closed=True, fill=True, color='#FFCCCB', ec="black", alpha=0.1)  # Create a polygon
                plt.gca().add_patch(polygon)  # Add the polygon to the current axes

            logging.info("Network graph plotted successfully.")
        except Exception as e:
            logging.error(f"Error plotting network graph: {e}")
            raise

    def save_plot_to_image(self, fig, save_path):
        """
        Save the matplotlib figure to the specified path as a PNG file.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object to be saved.
            save_path (str): The file path where the image will be saved.

        Returns:
            None
        """
        try:
            fig.savefig(save_path, format='png', bbox_inches='tight')  # Save the figure to the specified path
            plt.close(fig)
            logging.info(f"Figure saved to {save_path} successfully.")
        except Exception as e:
            logging.error(f"Error saving figure to {save_path}: {e}")
            raise

        