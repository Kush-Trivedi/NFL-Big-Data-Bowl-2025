from pathlib import Path
from utils.helpers import *
from data_visualization.game_visualizer import GameVisualizer
from data_visualization.plotly_game_visualizer import PlotlyGameVisualizer
from data_preprocessesing.nfl_data_loader import NFLDataLoader, RouteComboAnalyzer, QBRadarProcessor, PlayerRatingProcessor, PlayGroundSimulatorDataProcessor

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
        

        self.route_analyzer_base_path = Path("assets/offense-data")
        self.offense_player_rating_base_path = Path("assets/offense-data")
        self.defense_player_rating_base_path = Path("assets/defense-data")
        self.qb_radar_base_path = Path("assets/qb_radar")
        self.player_ratings_base_path = Path("assets/player_ratings")
        self.qb_radar_base_path.mkdir(parents=True, exist_ok=True)
        self.player_ratings_base_path.mkdir(parents=True, exist_ok=True)

    def process_teams(self):
        """Process data for each team and print summaries."""
        for team_name in self.team_names:
            try:
                # Get defensive team data
                self.data_loader.get_defense_team_data(defense_team=team_name, save=True)
                logger.info(f"Retrieved defense data for {team_name}.")

                # Get offensive team data
                self.data_loader.get_possession_team_data(possession_team=team_name, save=True)
                logger.info(f"Retrieved offense data for {team_name}.")

            except ValueError as e:
                # Handle the case where no data is found for the defense team
                logger.error(f"Error for team '{team_name}': {e}")

    def route_analyzer(self):
        for team_name in self.team_names:
            try:
                # Define path for each team under the base path
                team_path = os.path.join(self.route_analyzer_base_path, team_name)
                file_path = os.path.join(team_path, f"{team_name}_full_data.csv")
                logger.info(f"Route Analyzer Processing Started for Team: {team_name}")
                
                # Initialize RouteComboAnalyzer
                analyzer = RouteComboAnalyzer(file_path)
                
                # Ensure the main team directory exists
                os.makedirs(team_path, exist_ok=True)
                
                # Create subdirectories for each analysis type within the team folder
                combined_path = os.path.join(team_path, "combined")
                quarter_down_path = os.path.join(team_path, "quarter_down_stats")
                overall_strategy_path = os.path.join(team_path, "overall_strategy")
                success_failure_path = os.path.join(team_path, "success_failure_stats")
                
                for path in [combined_path, quarter_down_path, overall_strategy_path, success_failure_path]:
                    os.makedirs(path, exist_ok=True)
                
                # Process combined route combos and save in combined folder
                combined_df = analyzer.analyze_route_combos()
                combined_df.to_csv(os.path.join(combined_path, "full_route_combos.csv"), index=False)
                
                # Process get_specific_quarter_down (20 files for 5 quarters x 4 downs) in quarter_down_stats folder
                for quarter in range(1, 6):
                    for down in range(1, 5):
                        specific_df = analyzer.get_specific_quarter_down(quarter, down)
                        specific_df.to_csv(os.path.join(quarter_down_path, f"route_combos_q{quarter}_d{down}.csv"), index=False)
                
                # Process overall strategy stats and save in overall_strategy folder
                overall_stats_df = analyzer.get_overall_strategy_stats()
                overall_stats_df.to_csv(os.path.join(overall_strategy_path, "overall_strategy_stats.csv"), index=False)
                
                # Process get_quarter_down_strategy_stats (20 files for 5 quarters x 4 downs) in quarter_down_stats folder
                for quarter in range(1, 6):
                    for down in range(1, 5):
                        quarter_down_stats_df = analyzer.get_quarter_down_strategy_stats(quarter, down)
                        quarter_down_stats_df.to_csv(os.path.join(quarter_down_path, f"quarter_down_strategy_stats_q{quarter}_d{down}.csv"), index=False)
                
                # Process success/failure stats and save in success_failure_stats folder
                success_failure_df = analyzer.calculate_success_failure_stats()
                success_failure_df.to_csv(os.path.join(success_failure_path, "success_failure_stats.csv"), index=False)
                
                # Process success/failure stats for each quarter and down (20 files) in success_failure_stats folder
                for quarter in range(1, 6):
                    for down in range(1, 5):
                        quarter_down_success_failure_df = analyzer.calculate_success_failure_stats_for_quarter_down(quarter, down)
                        quarter_down_success_failure_df.to_csv(os.path.join(success_failure_path, f"success_failure_stats_q{quarter}_d{down}.csv"), index=False)
                
                # Process all quarters and downs success/failure stats and save in success_failure_stats folder
                all_quarters_downs_success_failure_df = analyzer.calculate_success_failure_stats_all_quarters_downs()
                all_quarters_downs_success_failure_df.to_csv(os.path.join(success_failure_path, "all_quarters_downs_success_failure_stats.csv"), index=False)
                
                logger.info(f"Route Analyzer Processing Complete for Team: {team_name}")
            except Exception as e:
                logger.error(f"Error processing Route Analyzer for Team {team_name}: {e}")

    def process_qb_radar(self):
        """Process QB radar data for each team, saving individual and combined files."""
        base_file_path = Path("assets/offense-data/{team_name}/{team_name}_full_data.csv")
        QBRadarProcessor.process_qb_radar_for_teams(
            team_names=self.team_names,
            base_file_path=base_file_path,
            output_base_path=self.qb_radar_base_path
        )
        logger.info("QB radar data processing completed for all teams.")


    def process_player_ratings(self):
        processor = PlayerRatingProcessor(self.offense_player_rating_base_path, self.defense_player_rating_base_path, self.player_ratings_base_path)
        processor.process_offense_player_rating(self.team_names)
        processor.process_defense_player_rating(self.team_names)
        processor.process_player_ratings_by_quarter_and_down(self.team_names)
        processor.concat_player_ratings_by_quarter_and_down(self.team_names)
        processor.process_all_player_ratings()

    
    def process_play_ground_simulator_data(self):
        playground_processor = PlayGroundSimulatorDataProcessor(self.team_names)
        playground_processor.process_team_files()



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
            plotly_game_visualizer = PlotlyGameVisualizer(df)
            plotly_game_visualizer.plot_game_in_plotly(game_id, play_id)
            game_visualizer = GameVisualizer(df, pitch_image_path)
            game_visualizer.plot_game_in_matplotlib(game_id, play_id)
            logger.info(f"Successfully visualized game play for Game ID: {game_id}, Play ID: {play_id}.")
        except Exception as e:
            logger.error(f"Error visualizing game play for Game ID: {game_id}, Play ID: {play_id}: {e}")

# Main
if __name__ == "__main__":
    processor = NFLProcessor()
    # processor.process_teams()
    # processor.process_qb_radar()
    # processor.process_player_ratings()
    # processor.process_play_ground_simulator_data()

        
    # Example of visualizing a specific game play
    game_id = 2022110606
    play_id = 1531

    # create random pitch wil be updated when gameplay will be called
    pitch = NFLFieldVertical(width=53.3, height=120,home_team="ARI", home_team_color="#97233F", visitor_team="KC", visitor_team_color="#97233F")
    pitch.save_pitch(
        folder_path='assets', filename='vertical_pitch.png'
    )

    pitch_image_path = Path("assets/vertical_pitch.png")
    processor.visualize_game_play(game_id, play_id, pitch_image_path)
