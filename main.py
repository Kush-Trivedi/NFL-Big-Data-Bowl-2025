from utils.helpers import *
from data_preprocessesing.nfl_data_loader import NFLDataLoader
from data_visualization.game_play_visualizer import GamePlayVisualizer

logger = Logger().get_logger()

class NFLProcessor:
    """
    NFLDataProcessor handles the loading and visualization of NFL data.
    
    This class initializes a data loader, retrieves defense and offense data 
    for specified teams, and visualizes game plays using matplotlib.

    Attributes:
        data_loader (NFLDataLoader): An instance of the NFLDataLoader for loading data.
        logger (Logger): Logger instance for logging events and errors.
        team_names (list): A list of NFL team abbreviations to process.
    """

    def __init__(self):
        # Initialize the data loader
        self.data_loader = NFLDataLoader()
        self.data_loader.load_all_data()

        # List of team names
        self.team_names = [
            'LA', 'ATL', 'CAR', 'CHI', 
            'CIN', 'DET', 'HOU', 'MIA', 
            'NYJ', 'WAS', 'ARI', 'LAC', 
            'MIN', 'TEN', 'DAL', 'SEA', 
            'KC', 'BAL', 'CLE', 'JAX', 'NO', 
            'NYG', 'PIT', 'SF', 'DEN', 'LV', 
            'GB', 'BUF', 'PHI', 'IND', 'NE', 'TB'
        ]  

    def process_teams(self):
        """Process data for each team and print summaries."""
        for team_name in self.team_names:
            try:
                # Get defensive team data
                defense_data = self.data_loader.get_defense_team_data(defense_team=team_name, save=True)
                logger.info(f"Retrieved defense data for {team_name}.")
                print(self.data_loader.basic_summary(defense_data, f"Defense {team_name} Data"))

                # Get offensive team data
                offense_data = self.data_loader.get_possession_team_data(possession_team=team_name, save=True)
                logger.info(f"Retrieved offense data for {team_name}.")
                print(self.data_loader.basic_summary(offense_data, f"Offense {team_name} Data"))

            except ValueError as e:
                # Handle the case where no data is found for the defense team
                logger.error(f"Error for team '{team_name}': {e}")
                print(f"Error for team '{team_name}': {e}")

    def visualize_game_play(self, game_id, play_id, pitch_image_path):
        """
        Visualize the game play for the specified game ID and play ID.

        Args:
            game_id (int): The ID of the game to visualize.
            play_id (int): The ID of the play to visualize.
            pitch_image_path (str): The file path to the pitch image.
        """
        try:
            df = self.data_loader.get_specific_game_play_data(game_id, play_id)
            game_visualizer = GamePlayVisualizer(df, pitch_image_path)
            game_visualizer.plot_game_in_matplotlib(game_id, play_id)
            logger.info(f"Successfully visualized game play for Game ID: {game_id}, Play ID: {play_id}.")
        except Exception as e:
            logger.error(f"Error visualizing game play for Game ID: {game_id}, Play ID: {play_id}: {e}")

# Main
if __name__ == "__main__":
    processor = NFLProcessor()
    processor.process_teams()