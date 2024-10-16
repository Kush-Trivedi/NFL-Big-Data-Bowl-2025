import os
import pandas as pd
from utils.helpers import *

# Initilize the logger
logger = Logger().get_logger()

class NFLDataLoader:
    path = "assets/nfl-big-data-bowl-2025/"
    save_offense_path = "assets/offesnse-data/" 
    save_defense_path = "assets/defense-data/"
    
    def __init__(self):
        """Initializes the NFLDataLoader with empty data attributes."""
        self.games = None
        self.players = None
        self.plays = None
        self.tracking = None
        self.player_play = None
        logger.info("NFLDataLoader initialized.")

    def downcast_memory_usage(self, df, df_name, verbose=True):
        """
        Reduces the memory usage of a DataFrame by downcasting numerical columns.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to downcast.
            df_name (str): Name of the DataFrame for logging purposes.
            verbose (bool): Whether to log the compression percentage.
        
        Returns:
            pd.DataFrame: The downcasted DataFrame.
        """
        try:
            start_mem = df.memory_usage().sum() / 1024**2
            logger.info(f"Starting memory usage for {df_name}: {start_mem:.2f} MB")

            for col in df.columns:
                dtype_name = df[col].dtype.name
                if dtype_name == 'object':
                    logger.debug(f"Column '{col}' is of type object; skipping downcast.")
                    pass
                elif dtype_name == 'bool':
                    df[col] = df[col].astype('int8')
                    logger.debug(f"Column '{col}' downcasted to int8.")
                elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                    logger.debug(f"Column '{col}' downcasted to integer.")
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                    logger.debug(f"Column '{col}' downcasted to float.")

            end_mem = df.memory_usage().sum() / 1024**2
            compression = 100 * (start_mem - end_mem) / start_mem
            if verbose:
                logger.info(f"{df_name}: Compressed by {compression:.1f}%")

            return df

        except Exception as e:
            logger.error(f"Error in downcasting {df_name}: {e}")
            raise

    def load_data(self, file_name):
        """
        Loads a CSV file into a DataFrame.
        
        Parameters:
            file_name (str): The name of the CSV file to load.
        
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            file_path = os.path.join(self.path, file_name)
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_name}")
            return df

        except FileNotFoundError:
            logger.error(f"File {file_name} not found in path {self.path}.")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"File {file_name} is empty.")
            raise
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")
            raise

    def load_all_data(self):
        """
        Loads all required datasets with memory optimization.
        """
        try:
            logger.info("Starting to load all datasets.")
            self.games = self.downcast_memory_usage(self.load_data("games.csv"), "Games Dataset")
            self.players = self.downcast_memory_usage(self.load_data("players.csv"), "Players Dataset")
            self.plays = self.downcast_memory_usage(self.load_data("plays.csv"), "Plays Dataset")
            self.player_play = self.downcast_memory_usage(self.load_data("player_play.csv"), "Player Play Dataset")
            logger.info("All data loaded and downcasted successfully.")

        except Exception as e:
            logger.error(f"Error loading all data: {e}")
            raise

    def load_tracking_for_game_play(self, game_id, play_id):
        """
        Loads tracking data for a specific game and play.
        
        Parameters:
            game_id (int): The ID of the game.
            play_id (int): The ID of the play.
        
        Returns:
            pd.DataFrame: The tracking data for the specified game and play.
        """
        try:
            tracking_files = [f"tracking_week_{week_num}.csv" for week_num in range(1, 10)]
            filtered_chunks = []

            logger.info(f"Loading tracking data for gameId {game_id} and playId {play_id}.")

            for file_name in tracking_files:
                file_path = os.path.join(self.path, file_name)
                logger.debug(f"Processing file {file_path}")
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    filtered_chunk = chunk[(chunk['gameId'] == game_id) & (chunk['playId'] == play_id)]
                    if not filtered_chunk.empty:
                        filtered_chunks.append(filtered_chunk)
                        logger.debug(f"Found matching chunk in {file_name} with {len(filtered_chunk)} records.")

            if filtered_chunks:
                tracking_data = pd.concat(filtered_chunks, ignore_index=True)
                logger.info(f"Tracking data loaded for gameId {game_id} and playId {play_id}.")
                return tracking_data
            else:
                error_msg = f"No tracking data found for gameId {game_id} and playId {play_id}."
                logger.warning(error_msg)
                raise ValueError(error_msg)

        except FileNotFoundError as fnf_error:
            logger.error(f"Tracking file not found: {fnf_error}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("One of the tracking files is empty.")
            raise
        except Exception as e:
            logger.error(f"Error loading tracking data for gameId {game_id} and playId {play_id}: {e}")
            raise

    def get_specific_game_play_data(self, game_id, play_id):
        """
        Retrieves specific game and play data, merged with player and tracking data.
        
        Parameters:
            game_id (int): The ID of the game.
            play_id (int): The ID of the play.
        
        Returns:
            pd.DataFrame: The merged data for the specific game and play.
        """
        try:
            logger.info(f"Retrieving data for gameId {game_id} and playId {play_id}.")
            tracking_data = self.load_tracking_for_game_play(game_id, play_id)
            games_and_play_df = pd.merge(self.games, self.plays, on=['gameId'], how='inner')
            logger.debug("Merged games and plays data.")
            
            plays_with_tracking = pd.merge(tracking_data, games_and_play_df, on=['gameId', 'playId'], how='inner')
            logger.debug("Merged tracking data with games and plays.")
            
            players_plays_with_tracking = pd.merge(
                plays_with_tracking, self.player_play, on=['gameId', 'playId', 'nflId'], how='left'
            )
            logger.debug("Merged player play data.")
            
            merged_data = pd.merge(
                players_plays_with_tracking, self.players, on=['nflId', 'displayName'], how='left'
            )
            logger.debug("Merged players data.")
            
            merged_data['nflId'].fillna(999999, inplace=True)
            merged_data['jerseyNumber'] = merged_data['jerseyNumber'].astype(object)
            merged_data['jerseyNumber'].fillna("", inplace=True)
            merged_data.rename(columns={'club': 'Team'}, inplace=True)
            
            logger.info(f"Data merged for gameId {game_id} and playId {play_id}.")
            return merged_data

        except Exception as e:
            logger.error(f"Error retrieving game play data for gameId {game_id} and playId {play_id}: {e}")
            raise

    def get_possession_team_data(self, possession_team, save=True):
        """
        Retrieves and optionally saves data for the possession team.
        
        Parameters:
            possession_team (str): The name of the possession team.
            save (bool): Whether to save the retrieved data to a CSV file.
        
        Returns:
            pd.DataFrame: The merged data for the possession team.
        """
        try:
            logger.info(f"Retrieving possession team data for '{possession_team}'.")
            games_and_play_df = pd.merge(self.games, self.plays, on=['gameId'], how='inner')
            filtered_plays = games_and_play_df[games_and_play_df['possessionTeam'] == possession_team]

            if filtered_plays.empty:
                error_msg = f"No plays found for possession team '{possession_team}'."
                logger.warning(error_msg)
                raise ValueError(error_msg)

            merged_chunks = []
            tracking_files = [f"tracking_week_{week_num}.csv" for week_num in range(1, 10)]

            for file_name in tracking_files:
                file_path = os.path.join(self.path, file_name)
                logger.debug(f"Processing tracking file {file_path}")
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    plays_with_tracking = pd.merge(chunk, filtered_plays, on=['gameId', 'playId'], how='inner')

                    if plays_with_tracking.empty:
                        logger.debug(f"No matching plays in chunk from {file_name}.")
                        continue 

                    players_plays_with_tracking = pd.merge(
                        plays_with_tracking, self.player_play, on=['gameId', 'playId', 'nflId'], how='left'
                    )
                    logger.debug("Merged player play data.")
                    
                    merged_data = pd.merge(
                        players_plays_with_tracking, self.players, on=['nflId', 'displayName'], how='left'
                    )
                    logger.debug("Merged players data.")
                    
                    merged_data['nflId'].fillna(999999, inplace=True)
                    merged_data['jerseyNumber'] = merged_data['jerseyNumber'].astype(object)
                    merged_data['jerseyNumber'].fillna("", inplace=True)
                    merged_data.rename(columns={'club': 'Team'}, inplace=True)
                    
                    merged_chunks.append(merged_data)
                    logger.debug(f"Appended merged data chunk with {len(merged_data)} records.")

            if merged_chunks:
                full_merged_data = pd.concat(merged_chunks, ignore_index=True)
                logger.info(f"Data retrieved for possession team '{possession_team}'.")
            else:
                error_msg = f"No tracking data found for possession team '{possession_team}'."
                logger.warning(error_msg)
                raise ValueError(error_msg)

            if save:
                try:
                    dir_name = f"{possession_team}_offense_data"
                    os.makedirs(self.save_offense_path, exist_ok=True)
                    file_path = os.path.join(self.save_offense_path, f"{dir_name}.csv")
                    full_merged_data.to_csv(file_path, index=False)
                    logger.info(f"Data for possession team '{possession_team}' saved to '{file_path}'.")
                except Exception as e:
                    logger.error(f"Error saving data for possession team '{possession_team}': {e}")
                    raise

            return full_merged_data

        except Exception as e:
            logger.error(f"Error retrieving possession team data: {e}")
            raise

    def get_defense_team_data(self, defense_team, save=True):
        """
        Retrieves and optionally saves data for the defense team.
        
        Parameters:
            defense_team (str): The name of the defense team.
            save (bool): Whether to save the retrieved data to a CSV file.
        
        Returns:
            pd.DataFrame: The merged data for the defense team.
        """
        try:
            logger.info(f"Retrieving defense team data for '{defense_team}'.")
            games_and_play_df = pd.merge(self.games, self.plays, on=['gameId'], how='inner')
            filtered_plays = games_and_play_df[games_and_play_df['defensiveTeam'] == defense_team]

            if filtered_plays.empty:
                error_msg = f"No plays found for defense team '{defense_team}'."
                logger.warning(error_msg)
                raise ValueError(error_msg)

            merged_chunks = []
            tracking_files = [f"tracking_week_{week_num}.csv" for week_num in range(1, 10)]

            for file_name in tracking_files:
                file_path = os.path.join(self.path, file_name)
                logger.debug(f"Processing tracking file {file_path}")
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    plays_with_tracking = pd.merge(chunk, filtered_plays, on=['gameId', 'playId'], how='inner')

                    if plays_with_tracking.empty:
                        logger.debug(f"No matching plays in chunk from {file_name}.")
                        continue 

                    players_plays_with_tracking = pd.merge(
                        plays_with_tracking, self.player_play, on=['gameId', 'playId', 'nflId'], how='left'
                    )
                    logger.debug("Merged player play data.")
                    
                    merged_data = pd.merge(
                        players_plays_with_tracking, self.players, on=['nflId', 'displayName'], how='left'
                    )
                    logger.debug("Merged players data.")
                    
                    merged_data['nflId'].fillna(999999, inplace=True)  
                    merged_data['jerseyNumber'] = merged_data['jerseyNumber'].astype(object)
                    merged_data['jerseyNumber'].fillna("", inplace=True)
                    merged_data.rename(columns={'club': 'Team'}, inplace=True)
                    
                    merged_chunks.append(merged_data)
                    logger.debug(f"Appended merged data chunk with {len(merged_data)} records.")

            if merged_chunks:
                full_merged_data = pd.concat(merged_chunks, ignore_index=True)
                logger.info(f"Data retrieved for defense team '{defense_team}'.")
            else:
                error_msg = f"No tracking data found for defense team '{defense_team}'."
                logger.warning(error_msg)
                raise ValueError(error_msg)

            if save:
                try:
                    dir_name = f"{defense_team}_defense_data"
                    os.makedirs(self.save_defense_path, exist_ok=True)
                    file_path = os.path.join(self.save_defense_path, f"{dir_name}.csv")
                    full_merged_data.to_csv(file_path, index=False)
                    logger.info(f"Data for defense team '{defense_team}' saved to '{file_path}'.")
                except Exception as e:
                    logger.error(f"Error saving data for defense team '{defense_team}': {e}")
                    raise

            return full_merged_data

        except Exception as e:
            logger.error(f"Error retrieving defense team data: {e}")
            raise

    def get_overall_plays_with_tracking_in_chunks(self, chunk_size=100000):
        """
        Retrieves all plays with tracking data in chunks.
        
        Parameters:
            chunk_size (int): The number of rows per chunk when reading tracking files.
        
        Returns:
            pd.DataFrame: The concatenated merged data for all plays.
        """
        try:
            logger.info("Retrieving overall plays with tracking data in chunks.")
            merged_chunks = []
            tracking_files = [f"tracking_week_{week_num}.csv" for week_num in range(1, 10)]
            
            for file_name in tracking_files:
                file_path = os.path.join(self.path, file_name)
                logger.debug(f"Processing tracking file {file_path}")
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    plays_with_tracking = pd.merge(chunk, self.plays, on=['gameId', 'playId'], how='inner')
                    logger.debug(f"Merged chunk with plays: {len(plays_with_tracking)} records.")
                    
                    players_plays_with_tracking = pd.merge(
                        plays_with_tracking, self.player_play, on=['gameId', 'playId', 'nflId'], how='left'
                    )
                    logger.debug("Merged player play data.")
                    
                    merged_data = pd.merge(
                        players_plays_with_tracking, self.players, on=['nflId', 'displayName'], how='left'
                    )
                    logger.debug("Merged players data.")
                    
                    merged_data['nflId'].fillna(999999, inplace=True)
                    merged_data['jerseyNumber'].fillna("", inplace=True)
                    merged_data.rename(columns={'club': 'Team'}, inplace=True)
                    
                    merged_chunks.append(merged_data)
                    logger.debug(f"Appended merged data chunk with {len(merged_data)} records.")

            if merged_chunks:
                full_merged_data = pd.concat(merged_chunks, ignore_index=True)
                logger.info("Successfully retrieved and merged all plays with tracking data.")
                return full_merged_data
            else:
                logger.warning("No tracking data found across all tracking files.")
                return pd.DataFrame()  # Return empty DataFrame if no data found

        except FileNotFoundError as fnf_error:
            logger.error(f"Tracking file not found: {fnf_error}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("One of the tracking files is empty.")
            raise
        except Exception as e:
            logger.error(f"Error retrieving overall plays with tracking data: {e}")
            raise

    def basic_summary(self, data_frame, data_set_name):
        """
        Generates a basic summary of a DataFrame, including data types, null counts, unique counts, and sample values.
        
        Parameters:
            data_frame (pd.DataFrame): The DataFrame to summarize.
            data_set_name (str): The name of the dataset for logging purposes.
        
        Returns:
            pd.DataFrame: A summary DataFrame.
        """
        try:
            logger.info(f"Generating basic summary for dataset '{data_set_name}'.")
            summary = pd.DataFrame(data_frame.dtypes, columns=['Data Type'])
            summary = summary.reset_index()
            summary = summary.rename(columns={'index': 'Feature'})
            summary['Num of Nulls'] = data_frame.isnull().sum().values
            summary['Num of Unique'] = data_frame.nunique().values
            summary['First Value'] = data_frame.iloc[0].values
            summary['Second Value'] = data_frame.iloc[1].values
            summary['Third Value'] = data_frame.iloc[2].values
            summary['Fourth Value'] = data_frame.iloc[3].values
            logger.info(f"Basic summary generated for dataset '{data_set_name}'.")
            return summary

        except IndexError:
            logger.warning(f"DataFrame '{data_set_name}' does not have enough rows to generate sample values.")
            # Handle cases where the DataFrame has fewer than 4 rows
            summary = pd.DataFrame(data_frame.dtypes, columns=['Data Type'])
            summary = summary.reset_index()
            summary = summary.rename(columns={'index': 'Feature'})
            summary['Num of Nulls'] = data_frame.isnull().sum().values
            summary['Num of Unique'] = data_frame.nunique().values
            for i in range(4):
                column_name = f'Value {i+1}'
                if i < len(data_frame):
                    summary[column_name] = data_frame.iloc[i].values
                else:
                    summary[column_name] = None
            logger.info(f"Basic summary generated for dataset '{data_set_name}' with limited sample values.")
            return summary
        except Exception as e:
            logger.error(f"Error generating basic summary for dataset '{data_set_name}': {e}")
            raise


class SingleGamePlayExtractor:
    """Class to extract and analyze data for a specific game play."""
    
    def __init__(self, df):
        """
        Initialize the extractor with the provided DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame containing game and play data.
        """
        self.df = df
        logging.info("SingleGamePlayExtractor initialized with provided DataFrame.")

    def extract_game_play_data(self, gameId, playId):
        """
        Extract data for a specific game and play based on gameId and playId.

        Parameters:
        gameId (int): ID of the game.
        playId (int): ID of the play.

        Returns:
        tuple: A DataFrame with the specific game-play data and a dictionary of play information.

        Raises:
        ValueError: If no data is available for the provided gameId and playId.
        """
        logging.info(f"Extracting data for gameId: {gameId} and playId: {playId}.")
        game_play_df = self.df[(self.df['gameId'] == gameId) & (self.df['playId'] == playId)]

        if game_play_df.empty:
            logging.error(f"No data found for gameId: {gameId} and playId: {playId}.")
            raise ValueError("No data available for the provided gameId and playId.")

        play_info = {
            'defensive_team': game_play_df.defensiveTeam.values[0],
            'possession_team': game_play_df.possessionTeam.values[0],
            'unique_frame_ids': game_play_df['frameId'].unique(),
            'play_description': game_play_df.playDescription.values[0],
            'offense_formation': game_play_df.offenseFormation.values[0],
            'line_of_scrimmage': game_play_df.absoluteYardlineNumber.values[0],
            'down': game_play_df.down.values[0],
            'quarter': game_play_df.quarter.values[0],
            'play_direction': game_play_df.playDirection.values[0],
            'yards_to_go': game_play_df.yardsToGo.values[0],
            'pre_snap_home_score': game_play_df.preSnapHomeScore.values[0],
            'pre_snap_visitor_score': game_play_df.preSnapVisitorScore.values[0],
            'home_team_abbr': game_play_df.homeTeamAbbr.values[0],
            'visitor_team_abbr': game_play_df.visitorTeamAbbr.values[0],
            'game_lock': game_play_df.gameClock.values[0],
            'time': game_play_df['time'].unique()
        }

        logging.info(f"Successfully extracted data for gameId: {gameId} and playId: {playId}.")
        return game_play_df, play_info

    def determine_first_down_marker(self, play_info):
        """
        Determine the position of the first down marker based on play direction.

        Parameters:
        play_info (dict): Dictionary containing play information.

        Returns:
        int: The position of the first down marker.
        """
        logging.info(f"Calculating first down marker for play with direction: {play_info['play_direction']}.")

        if play_info['play_direction'] == "left":
            marker = play_info['line_of_scrimmage'] - play_info['yards_to_go']
        else:
            marker = play_info['line_of_scrimmage'] + play_info['yards_to_go']

        logging.info(f"First down marker calculated as: {marker}.")
        return marker


