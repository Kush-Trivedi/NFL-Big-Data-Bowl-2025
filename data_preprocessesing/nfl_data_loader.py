import os
import re
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import traceback
from utils.helpers import *
from collections import defaultdict, Counter

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
        self.defense_category_rules = {
            "4-3 Defense": ["DT", "DE", "MLB", "ILB"],
            "3-4 Defense": ["NT", "DE", "ILB", "OLB"],
            "5-2 Defense": ["DT", "DE", "ILB"],
            "4-2-5 Nickel": ["CB", "DT", "DE", "FS", "SS"],
            "3-3-5 Nickel": ["CB", "NT", "DE", "ILB", "OLB"],
            "4-1-6 Dime": ["CB", "NT", "DE", "ILB", "FS", "SS"],
            "3-2-6 Dime": ["CB", "DT", "DE", "FS", "ILB"],
            "3-1-7 Quarter": ["CB", "FS", "NT", "OLB"],
            "5-3 Goal Line": ["DT", "DE", "ILB", "FS"],
            "6-2 Goal Line": ["DT", "DE", "ILB"],
            "Big Nickel": ["CB", "DE", "DT", "FS"],
            "Heavy DE Front": ["DE", "CB", "ILB"],
            "3-3-5 Stack": ["CB", "OLB", "ILB", "DE"],
            "5-1 Front": ["NT", "DT", "FS", "CB", "ILB"],
            "4-4 Defense": ["CB", "DE", "DT", "SS", "FS"],
            "Mixed DB Support": ["CB", "SS", "DE", "DT", "NT"],
            "LB Heavy Defense": ["OLB", "CB", "ILB", "NT"],
            "Interior DL Heavy": ["DT", "ILB", "OLB", "FS", "SS"],
            "Double NT Formation": ["NT", "DE", "SS", "FS", "CB"],
            "Secondary Emphasis with DT": ["CB", "FS", "SS", "DT", "OLB"],
            "Heavy LB Defense": ["OLB", "CB", "ILB"],
            "Hybrid Nickel": ["CB", "SS", "NT", "ILB", "DE"],
            "Mixed Front with Secondary Support": ["CB", "SS", "DE", "DT", "NT"],
            "Heavy Interior": ["DT", "ILB", "OLB", "FS", "SS"],
            "DL Dominant Hybrid": ["DT", "DE", "OLB", "FS", "SS", "CB"],
            "Secondary Emphasis with Multi-Linebackers": ["CB", "FS", "SS", "ILB", "OLB"],
            "Hybrid 3-4 with DB": ["DE", "NT", "CB", "FS", "SS"],
            "Goal Line Variants": ["NT", "DT", "CB", "SS", "FS"],
            "Big Secondary and LB": ["CB", "FS", "OLB", "MLB"],
            "DL Heavy with Minimal Secondary": ["DT", "DE", "CB", "SS", "FS"],
        }

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
            
            merged_data['nflId'] = merged_data['nflId'].fillna(999999)
            merged_data['jerseyNumber'] = merged_data['jerseyNumber'].astype(object).fillna("")
            merged_data.rename(columns={'club': 'Team'}, inplace=True)
            
            logger.info(f"Data merged for gameId {game_id} and playId {play_id}.")
            return merged_data

        except Exception as e:
            logger.error(f"Error retrieving game play data for gameId {game_id} and playId {play_id}: {e}")
            raise

    def name_match(self, display_name, receiver_name):
        """
        Compares a display name and a receiver name to check if they refer to the same person.
        
        The function extracts the first initial and the last name from both names, normalizes them by 
        removing any non-alphabetic characters, and checks if they match. It is useful for matching 
        player names in sports analytics or similar applications.

        Parameters:
        - display_name (str): The name displayed, typically in "First Last" format.
        - receiver_name (str): The name of the receiver, typically in "F.Last" format (first initial followed by last name).

        Returns:
        - bool: True if the names match based on the first initial and last name; False otherwise.
        """

        if pd.isna(display_name) or pd.isna(receiver_name):
            return False
        
        display_parts = display_name.split()
        receiver_parts = receiver_name.split('.')
        
        display_initial = display_parts[0][0].lower()  # First initial in displayName
        receiver_initial = receiver_parts[0][0].lower()  # First initial in receiver_name
        
        # Combine remaining parts and normalize by removing punctuation and converting to lowercase
        display_last = re.sub(r'[^a-zA-Z]', '', ''.join(display_parts[1:])).lower()
        receiver_last = re.sub(r'[^a-zA-Z]', '', ''.join(receiver_parts[1:])).lower()
        
        return display_initial == receiver_initial and display_last == receiver_last
    
    def weekly_offense_pass_analysis(self, df):
        """
        Performs weekly pass analysis and returns relevant DataFrames.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame containing play data.
        
        Returns:
            tuple: A tuple containing two DataFrames: weekly route analysis and weekly passer-receiver analysis.
        """
        # Code 1
        temp_pass = df[['gameId', 'playId','season', 'week', 'play_type', 'passer_player_name', 
                        'receiver_player_name', 'passResult', 'displayName', 'routeRan']].drop_duplicates().reset_index(drop=True)
        route_counts = temp_pass.groupby(['season','week', 'displayName', 'routeRan']).size().reset_index(name='route_count')
        route_counts['total_routes'] = route_counts.groupby(['season','week', 'displayName'])['route_count'].transform('sum')
        route_counts['route %'] = (route_counts['route_count'] / route_counts['total_routes'] * 100).round(2)
        temp_pass['is_match'] = temp_pass.apply(lambda row: self.name_match(row['displayName'], row['receiver_player_name']), axis=1)
        matched_temp_pass = temp_pass[temp_pass['is_match']]
        pass_result_counts = matched_temp_pass.groupby(['season','week', 'displayName', 'routeRan', 'passResult']).size().unstack(fill_value=0).reset_index()

        expected_pass_results = ['C', 'I', 'IN', 'R', 'S']
        for col in expected_pass_results:
            if col not in pass_result_counts:
                pass_result_counts[col] = 0

        pass_result_counts['total_passes'] = pass_result_counts.groupby(['season','week', 'displayName'])[expected_pass_results].transform('sum').sum(axis=1)

        for col in expected_pass_results:
            pass_result_counts[f'{col} %'] = (pass_result_counts[col] / pass_result_counts['total_passes'] * 100).fillna(0).round(2)

        merged_route_pass_result_df = pd.merge(route_counts, pass_result_counts, on=['season','week', 'displayName', 'routeRan'], how='left')
        merged_route_pass_result_df.fillna(0, inplace=True)

        # Code 2
        temp_pass = df[['gameId', 'playId','season', 'week', 'play_type', 'passer_player_name', 'pass_location', 'receiver_player_name', 'passResult']].drop_duplicates().reset_index(drop=True)
        pass_df = temp_pass[temp_pass['play_type'] == 'pass']
        pass_counts = pass_df.groupby(['season','week', 'passer_player_name', 'receiver_player_name', 'pass_location', 'passResult']).size().reset_index(name='pass_count')
        result_counts = pass_df.groupby(['season','week', 'passer_player_name', 'receiver_player_name', 'passResult']).size().unstack(fill_value=0).reset_index()
        merged_pass_receiver_df = pd.merge(pass_counts, result_counts, on=['season','week', 'passer_player_name', 'receiver_player_name'], how='left')
        merged_pass_receiver_df['total_passes'] = merged_pass_receiver_df.groupby(['season','week', 'passer_player_name', 'receiver_player_name'])['pass_count'].transform('sum')

        for result_type in ['C', 'I', 'S', 'IN', 'R']:
            if result_type in merged_pass_receiver_df.columns:
                merged_pass_receiver_df[f'{result_type} %'] = (merged_pass_receiver_df[result_type] / merged_pass_receiver_df['total_passes'] * 100).round(2).fillna(0)
            else:
                merged_pass_receiver_df[f'{result_type} %'] = 0.00

        for location in merged_pass_receiver_df['pass_location'].unique():
            mask = merged_pass_receiver_df['pass_location'] == location
            merged_pass_receiver_df[f'{location} %'] = (merged_pass_receiver_df.loc[mask, 'pass_count'] / merged_pass_receiver_df['total_passes'] * 100).round(2)

        for result_type in ['C', 'I', 'S', 'IN', 'R']:
            for location in merged_pass_receiver_df['pass_location'].unique():
                column_name = f'{result_type} {location} %'
                mask = (merged_pass_receiver_df['passResult'] == result_type) & (merged_pass_receiver_df['pass_location'] == location)
                merged_pass_receiver_df[column_name] = (merged_pass_receiver_df.loc[mask, 'pass_count'] / merged_pass_receiver_df['total_passes'] * 100).round(2).fillna(0)

        merged_pass_receiver_df.fillna(0, inplace=True)

        return merged_route_pass_result_df, merged_pass_receiver_df

    def offense_pass_analysis(self, df):
        """
        Performs pass analysis and returns relevant DataFrames.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame containing play data.
        
        Returns:
            tuple: A tuple containing two DataFrames: route analysis and passer-receiver analysis.
        """
        # Code 1
        temp_pass = df[['gameId', 'playId','play_type', 'passer_player_name', 
                            'receiver_player_name', 'passResult', 
                            'displayName', 'routeRan']].drop_duplicates().reset_index(drop=True)

        route_counts = temp_pass.groupby(['displayName', 'routeRan']).size().reset_index(name='route_count')
        total_routes = route_counts.groupby('displayName')['route_count'].transform('sum')
        route_counts['route %'] = (route_counts['route_count'] / total_routes * 100).round(2)
        temp_pass['is_match'] = temp_pass.apply(lambda row: self.name_match(row['displayName'], row['receiver_player_name']), axis=1)
        matched_temp_pass = temp_pass[temp_pass['is_match']]
        pass_result_counts = matched_temp_pass.groupby(['displayName', 'routeRan', 'passResult']).size().unstack(fill_value=0).reset_index()

        expected_pass_results = ['C', 'I', 'IN', 'R', 'S']
        for col in expected_pass_results:
            if col not in pass_result_counts:
                pass_result_counts[col] = 0

        pass_result_counts['total_passes'] = pass_result_counts.groupby(['displayName'])[expected_pass_results].transform('sum').sum(axis=1)

        for col in expected_pass_results:
            pass_result_counts[f'{col} %'] = (pass_result_counts[col] / pass_result_counts['total_passes'] * 100).fillna(0).round(2)

        merged_route_pass_result_df = pd.merge(route_counts, pass_result_counts, on=['displayName', 'routeRan'], how='left')
        merged_route_pass_result_df.fillna(0, inplace=True)

        # Code 2
        temp_pass = df[['gameId', 'playId','play_type', 'passer_player_name', 'pass_location', 'receiver_player_name', 'passResult']].drop_duplicates().reset_index(drop=True)
        pass_df = temp_pass[temp_pass['play_type'] == 'pass']
        pass_counts = pass_df.groupby(['passer_player_name', 'receiver_player_name', 'pass_location','passResult']).size().reset_index(name='pass_count')
        result_counts = pass_df.groupby(['passer_player_name', 'receiver_player_name', 'passResult']).size().unstack(fill_value=0).reset_index()
        merged_pass_receiver_df = pd.merge(pass_counts, result_counts, on=['passer_player_name', 'receiver_player_name'], how='left')
        merged_pass_receiver_df['total_passes'] = merged_pass_receiver_df.groupby(['passer_player_name', 'receiver_player_name'])['pass_count'].transform('sum')

        for result_type in ['C', 'I', 'S', 'IN', 'R']:
            if result_type in merged_pass_receiver_df.columns:
                merged_pass_receiver_df[f'{result_type} %'] = (merged_pass_receiver_df[result_type] / merged_pass_receiver_df['total_passes'] * 100).round(2).fillna(0)
            else:
                merged_pass_receiver_df[f'{result_type} %'] = 0.00

        for location in merged_pass_receiver_df['pass_location'].unique():
            merged_pass_receiver_df[f'{location} %'] = (merged_pass_receiver_df.loc[merged_pass_receiver_df['pass_location'] == location, 'pass_count'] / merged_pass_receiver_df['total_passes'] * 100).round(2)

        for result_type in ['C', 'I', 'S', 'IN', 'R']:
            for location in merged_pass_receiver_df['pass_location'].unique():
                column_name = f'{result_type} {location} %'
                mask = (merged_pass_receiver_df['passResult'] == result_type) & (merged_pass_receiver_df['pass_location'] == location)
                merged_pass_receiver_df[column_name] = (merged_pass_receiver_df.loc[mask, 'pass_count'] / merged_pass_receiver_df['total_passes'] * 100).round(2).fillna(0)

        merged_pass_receiver_df.fillna(0, inplace=True)

        return merged_route_pass_result_df, merged_pass_receiver_df
    
    def normalize_component(self, component):
        return re.sub(r"[-]", "", component).strip()
     
    def matches_rule(self, formation_components, rule_keywords):
        """Check if the formation components match the rule keywords."""
        keyword_counts = {}
        for keyword in rule_keywords:
            match = re.match(r"(\d+)?\s*(\w+)", keyword)
            count = int(match.group(1)) if match.group(1) else 1
            position = match.group(2)
            keyword_counts[position] = keyword_counts.get(position, 0) + count

        for position, required_count in keyword_counts.items():
            actual_count = sum(1 for comp in formation_components if position in comp)
            if actual_count < required_count:
                return False
        return True
    
    def assign_category(self, formation):
        """Assign a category to the formation with expanded variant checks for 4-3, 3-4, and 5-2 defenses."""
        formation_components = [self.normalize_component(comp) for comp in formation.split(",")]

        # Check specifically for 4-3, 3-4, and 5-2 base defenses with expanded variations
        for base_defense in ["4-3 Defense", "3-4 Defense", "5-2 Defense"]:
            if self.matches_rule(formation_components, self.defense_category_rules[base_defense]):
                # Count the number of defensive backs (DBs), linemen, and linebackers
                db_count = sum(1 for comp in formation_components if 'CB' in comp or 'FS' in comp or 'SS' in comp)
                dl_count = sum(1 for comp in formation_components if 'DT' in comp or 'DE' in comp or 'NT' in comp)
                lb_count = sum(1 for comp in formation_components if 'ILB' in comp or 'OLB' in comp or 'MLB' in comp)

                # Identify variations based on counts
                if db_count >= 5 and lb_count <= 3:
                    return f"{base_defense} Nickel Variant"
                elif db_count >= 6:
                    return f"{base_defense} Dime Variant"
                elif db_count >= 7:
                    return f"{base_defense} Quarter Variant"
                elif dl_count >= 5:
                    return f"{base_defense} Heavy Front"
                elif lb_count >= 4:
                    return f"{base_defense} LB Heavy"
                elif db_count >= 5 and lb_count >= 3:
                    return f"{base_defense} Hybrid Defense"

                # If no specific variation, return the base defense
                return base_defense

        # If not matching base defenses, apply standard category rules
        for category, keywords in self.defense_category_rules.items():
            if self.matches_rule(formation_components, keywords):
                return category

        return "Other"

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
                    
                    merged_data['nflId'] = merged_data['nflId'].fillna(999999)
                    merged_data['jerseyNumber'] = merged_data['jerseyNumber'].astype(object).fillna("")
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
            
            # Load new data with selected columns
            columns_to_select = [
                'old_game_id', 'play_id', 'game_half', 'play_type', 'pass_location', 
                'pass_length', 'run_location', 'run_gap', 'first_down_rush', 
                'first_down_pass', 'first_down_penalty', 'third_down_converted', 
                'third_down_failed', 'fourth_down_converted', 'fourth_down_failed', 
                'rush_attempt', 'pass_attempt', 'lateral_reception', 'lateral_rush', 
                'passer_player_name', 'receiver_player_name', 'rusher_player_name', 
                'receiving_yards', 'rushing_yards', 'interception_player_name', 
                'tackle_for_loss_1_player_name', 'tackle_for_loss_2_player_name',
                'qb_hit_1_player_name', 'qb_hit_2_player_name', 
                'solo_tackle_1_player_name', 'assist_tackle_1_player_name', 
                'assist_tackle_2_player_name', 'tackle_with_assist', 
                'tackle_with_assist_1_player_name', 'tackle_with_assist_2_player_name', 
                'play_type_nfl', 'passer', 'rusher', 'receiver','yardline_100'
            ]

            df = pd.read_csv("assets/nflverse/pbp_2022.csv", usecols=columns_to_select, low_memory=False)
            logger.info("Loaded new data from pbp_2022.csv")

            # Merge with full_merged_data
            temp_final_merged_df = pd.merge(
                full_merged_data,
                df,
                left_on=['gameId', 'playId'],  
                right_on=['old_game_id', 'play_id'], 
                how='left'
            )
            logger.info("Merged possession team data with new data.")

            logger.info(f"Performing Defense Formation Analysis for {possession_team}.")
            # Add defenseFormationString and defenseFormation columns
            all_defense_formations = []
            for (gameId, playId, defensive_team), play_data in temp_final_merged_df.groupby(['gameId', 'playId', 'defensiveTeam']):
                unique_frame_ids = play_data['frameId'].unique()
                defense_formation = ""
                
                for frameId in unique_frame_ids:
                    frame_data = play_data[play_data['frameId'] == frameId]
                    defensive_players_data = frame_data[frame_data['Team'] == defensive_team]
                    
                    position_counts = defensive_players_data['position'].value_counts()
                    frame_formation_str = ', '.join([f"{count}- {position}" for position, count in position_counts.items()])
                    defense_formation += f"Frame {frameId}: " + frame_formation_str + "; "
                
                defense_formation_str = defense_formation.strip('; ')
                
                all_defense_formations.append({
                    'gameId': gameId,
                    'playId': playId,
                    'defenseFormationString': defense_formation_str,
                })
                

            # Convert to DataFrame and merge
            defense_formations_df = pd.DataFrame(all_defense_formations)
            defense_formations_df['defenseFormationString'] = defense_formations_df['defenseFormationString'].str.extract(r'(Frame 1: (.*?));')[1]
            defense_formations_df["defenseFormation"] = defense_formations_df["defenseFormationString"].apply(self.assign_category)
        
            final_merged_df = pd.merge(temp_final_merged_df, defense_formations_df, on=['gameId', 'playId'], how='left')
            logger.info(f"Defense formation added for '{possession_team}'.")

             # Perform Weekly Pass Analysis
            logger.info(f"Performing Weekly Pass Analysis for {possession_team}.")
            weekly_route_analysis_df, weekly_pass_receiver_analysis_df = self.weekly_offense_pass_analysis(final_merged_df)
            logger.info(f"Weekly Pass Analysis Done for {possession_team}.")

            # Perform Pass Analysis
            logger.info(f"Performing Pass Analysis for {possession_team}.")
            route_analysis_df, pass_receiver_analysis_df = self.offense_pass_analysis(final_merged_df)
            logger.info(f"Pass Analysis Done for {possession_team}.")

            if save:
                try:
                    team_directory = os.path.join(self.save_offense_path, possession_team)
                    os.makedirs(team_directory, exist_ok=True)

                    full_file_path = os.path.join(team_directory, f"{possession_team}_full_data.csv")
                    final_merged_df.to_csv(full_file_path, index=False)
                    logger.info(f"Full data for possession team '{possession_team}' saved to '{full_file_path}'.")

                    weekly_route_analysis_path = os.path.join(team_directory, f"{possession_team}_weekly_route_analysis.csv")
                    weekly_route_analysis_df.to_csv(weekly_route_analysis_path, index=False)
                    logger.info(f"Weekly Route analysis data for possession team '{possession_team}' saved to '{weekly_route_analysis_path}'.")

                    weekly_pass_receiver_analysis_path = os.path.join(team_directory, f"{possession_team}_weekly_pass_receiver_analysis.csv")
                    weekly_pass_receiver_analysis_df.to_csv(weekly_pass_receiver_analysis_path, index=False)
                    logger.info(f"Weekly Pass-receiver analysis data for possession team '{possession_team}' saved to '{weekly_pass_receiver_analysis_path}'.")

                    route_analysis_path = os.path.join(team_directory, f"{possession_team}_route_analysis.csv")
                    route_analysis_df.to_csv(route_analysis_path, index=False)
                    logger.info(f"Route analysis data for possession team '{possession_team}' saved to '{route_analysis_path}'.")

                    pass_receiver_analysis_path = os.path.join(team_directory, f"{possession_team}_pass_receiver_analysis.csv")
                    pass_receiver_analysis_df.to_csv(pass_receiver_analysis_path, index=False)
                    logger.info(f"Pass-receiver analysis data for possession team '{possession_team}' saved to '{pass_receiver_analysis_path}'.")

                except Exception as e:
                    logger.error(f"Failed to save data for possession team '{possession_team}': {e}")

            return final_merged_df

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
                    
                    merged_data['nflId'] = merged_data['nflId'].fillna(999999)
                    merged_data['jerseyNumber'] = merged_data['jerseyNumber'].astype(object).fillna("")
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
            
            # Load new data with selected columns
            columns_to_select = [
                'old_game_id', 'play_id', 'game_half', 'play_type', 'pass_location', 
                'pass_length', 'run_location', 'run_gap', 'first_down_rush', 
                'first_down_pass', 'first_down_penalty', 'third_down_converted', 
                'third_down_failed', 'fourth_down_converted', 'fourth_down_failed', 
                'rush_attempt', 'pass_attempt', 'lateral_reception', 'lateral_rush', 
                'passer_player_name', 'receiver_player_name', 'rusher_player_name', 
                'receiving_yards', 'rushing_yards', 'interception_player_name', 
                'tackle_for_loss_1_player_name', 'tackle_for_loss_2_player_name',
                'qb_hit_1_player_name', 'qb_hit_2_player_name', 
                'solo_tackle_1_player_name', 'assist_tackle_1_player_name', 
                'assist_tackle_2_player_name', 'tackle_with_assist', 
                'tackle_with_assist_1_player_name', 'tackle_with_assist_2_player_name', 
                'play_type_nfl', 'passer', 'rusher', 'receiver', 'yardline_100'
            ]

            df = pd.read_csv("assets/nflverse/pbp_2022.csv", usecols=columns_to_select, low_memory=False)
            logger.info("Loaded new data from pbp_2022.csv")

            # Merge with full_merged_data
            temp_final_merged_df = pd.merge(
                full_merged_data,
                df,
                left_on=['gameId', 'playId'],  
                right_on=['old_game_id', 'play_id'], 
                how='left'
            )
            logger.info("Merged possession team data with new data.")

            logger.info(f"Performing Defense Formation Analysis for {defense_team}.")
            # Add defenseFormationString and defenseFormation columns
            all_defense_formations = []
            for (gameId, playId, defensive_team), play_data in temp_final_merged_df.groupby(['gameId', 'playId', 'defensiveTeam']):
                unique_frame_ids = play_data['frameId'].unique()
                defense_formation = ""
                
                for frameId in unique_frame_ids:
                    frame_data = play_data[play_data['frameId'] == frameId]
                    defensive_players_data = frame_data[frame_data['Team'] == defensive_team]
                    
                    position_counts = defensive_players_data['position'].value_counts()
                    frame_formation_str = ', '.join([f"{count}- {position}" for position, count in position_counts.items()])
                    defense_formation += f"Frame {frameId}: " + frame_formation_str + "; "
                
                defense_formation_str = defense_formation.strip('; ')
                
                all_defense_formations.append({
                    'gameId': gameId,
                    'playId': playId,
                    'defenseFormationString': defense_formation_str,
                })
                

            # Convert to DataFrame and merge
            defense_formations_df = pd.DataFrame(all_defense_formations)
            defense_formations_df['defenseFormationString'] = defense_formations_df['defenseFormationString'].str.extract(r'(Frame 1: (.*?));')[1]
            defense_formations_df["defenseFormation"] = defense_formations_df["defenseFormationString"].apply(self.assign_category)
        
            final_merged_df = pd.merge(temp_final_merged_df, defense_formations_df, on=['gameId', 'playId'], how='left')
            logger.info(f"Defense formation added for '{defense_team}'.")

            if save:
                try:
                    team_directory = os.path.join(self.save_defense_path, defense_team)
                    os.makedirs(team_directory, exist_ok=True)

                    full_file_path = os.path.join(team_directory, f"{defense_team}_full_data.csv")
                    final_merged_df.to_csv(full_file_path, index=False)
                    logger.info(f"Full data for defense team '{defense_team}' saved to '{full_file_path}'.")

                except Exception as e:
                    logger.error(f"Error saving data for defense team '{defense_team}': {e}")
                    raise

            return final_merged_df

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



class RouteComboAnalyzer:
    def __init__(self, file_path):
        self.cols = [
            "gameId", "playId", "quarter", "down", "play_type", "offenseFormation",
            "receiverAlignment", "pass_location", "pass_length", "passer_player_name",
            "receiver_player_name", "displayName", "position", "routeRan", "passResult",
            "yardsToGo", "yardlineNumber", "absoluteYardlineNumber","pff_manZone","defenseFormation",
            "yardline_100", "pff_passCoverage"
        ]
        
        self.data = pd.read_csv(file_path, usecols=self.cols, low_memory=False)
        self.temp_pass = self.data.drop_duplicates().reset_index(drop=True)
        self.route_combos_by_quarter_down = {}


    def get_route_combos_by_quarter_down(self, quarter, down, play_type='pass'):
        filtered_df = self.temp_pass[
            (self.temp_pass['quarter'] == quarter) & 
            (self.temp_pass['down'] == down) & 
            (self.temp_pass['play_type'] == play_type)
        ].reset_index(drop=True)

        filtered_df['routeRan'] = filtered_df['routeRan'].fillna('').replace('', np.nan)

        route_combos = (
            filtered_df
            .groupby(['gameId', 'playId', 'down','pass_length', 'offenseFormation', 
                    'receiverAlignment', 'pass_location', 'yardsToGo', 'absoluteYardlineNumber',"pff_manZone","defenseFormation","yardline_100", "pff_passCoverage"])
            .agg(routeCombo=('routeRan', lambda x: ', '.join(x.dropna())),
                passResult=('passResult', 'first')) 
            .reset_index()
        )

        route_combos['strategy'] = route_combos.apply(self.categorize_route_combo, axis=1)
        return route_combos


    def categorize_route_combo(self, row):
        if pd.isna(row['routeCombo']) or pd.isna(row['pass_length']) or pd.isna(row['pass_location']):
            return 'Uncategorized'
        
        routes = row['routeCombo'].split(', ')
        pass_length = row['pass_length']
        pass_location = row['pass_location']
        alignment = row['receiverAlignment']
        formation = row['offenseFormation']
        down = row['down']
        yards_to_go = row['yardsToGo']
        field_position = row['absoluteYardlineNumber']
        quarter = row.get('quarter', None) 
   
        if down == 3 and yards_to_go >= 7 and pass_length == 'deep':
            return 'Third-and-Long Deep Attack'

        if down in [3, 4] and yards_to_go <= 3:
            if formation in ['SINGLEBACK', 'I_FORM']:
                return 'Short Yardage Power Run' if 'RUN' in routes else 'Short Yardage Pass'

        if down == 1:
            if pass_length == 'deep' and formation in ['SINGLEBACK', 'I_FORM']:
                return 'First Down Play Action Deep Pass'

        if down == 2 and yards_to_go <= 3 and pass_length == 'deep':
            return 'Second-and-Short Deep Shot'

        if field_position <= 20:
            if formation == 'SINGLEBACK' and pass_length == 'short':
                return 'Red Zone Short Pass'
            elif formation == 'SHOTGUN' and pass_length == 'deep':
                return 'Red Zone Deep Shotgun Attack'
            elif formation == 'EMPTY' and pass_length == 'short':
                return 'Red Zone Quick Pass'
        
        if 20 < field_position <= 30:
            if formation in ['SINGLEBACK', 'SHOTGUN'] and pass_length == 'deep':
                return 'Scoring Zone Deep Pass'
          
        if 30 < field_position <= 50:
            if pass_length == 'deep' and formation == 'SHOTGUN':
                return 'Midfield Deep Shotgun Attack'
            elif pass_length == 'short' and formation == 'SINGLEBACK':
                return 'Midfield Short Pass'
   
        if formation == 'SHOTGUN' and alignment == '3x1' and pass_length == 'deep' and 'GO' in routes:
            return 'Shotgun Isolation Deep Attack'

        if formation == 'SHOTGUN' and alignment == '3x0':
            return 'Shotgun Overload to Left'

        if formation == 'SINGLEBACK' and alignment == '2x2' and pass_length == 'short':
            return 'Singleback Balanced Short Pass'

        if formation == 'EMPTY' and alignment == '3x2' and pass_length == 'deep':
            return 'Spread Deep Attack from Empty 3x2 Alignment'

        if formation == 'I_FORM' and alignment in ['1x1', '2x1']:
            if pass_length == 'short' and 'SCREEN' in routes:
                return 'I-Formation Screen Play'
            elif pass_length == 'deep':
                return 'Play Action Deep Pass from I-Formation'
                
        if formation == 'PISTOL':
            if pass_length == 'deep' and any(route in ['POST', 'GO'] for route in routes):
                return 'Pistol Deep Attack'
            elif pass_length == 'short':
                return 'Pistol Quick Pass'

        if formation == 'WILDCAT':
            if pass_length == 'short' and 'RUN' in routes:
                return 'Wildcat Run Play'
            elif pass_length == 'deep':
                return 'Wildcat Deep Shot'

        if formation == 'JUMBO':
            if pass_length == 'short' and 'RUN' in routes:
                return 'Power Run from Jumbo Formation'
            elif pass_length == 'deep':
                return 'Play Action Deep Pass from Jumbo Formation'

        if quarter == 4 and pass_length == 'deep':
            return 'Fourth Quarter Deep Shot'

        if pass_location == 'left':
            if alignment in ['3x0', '2x1'] and pass_length == 'deep':
                return 'Deep Left Isolation Pass'
            elif pass_length == 'short':
                return 'Short Left Pass'
            
        if pass_location == 'right':
            if alignment in ['3x0', '2x1'] and pass_length == 'deep':
                return 'Deep Right Isolation Pass'
            elif pass_length == 'short':
                return 'Short Right Pass'

        if pass_location == 'middle':
            if pass_length == 'short':
                return 'Short Middle Pass'
            elif pass_length == 'deep':
                return 'Deep Middle Pass'

        if 'CROSS' in routes and 'OUT' in routes:
            return 'Hi-Lo Route Concept'

        if 'DRAG' in routes and alignment == '2x2':
            return 'Mesh Concept'

        if any(route in ['CROSS', 'SLANT'] for route in routes):
            return 'Spread Misdirection Play'

        if 'SCREEN' in routes:
            return 'Screen Play' if formation in ['I_FORM', 'SINGLEBACK'] else 'General Screen Play'

        if formation in ['SINGLEBACK', 'PISTOL', 'I_FORM'] and pass_length == 'deep' and any(route in ['POST', 'GO'] for route in routes):
            return 'Play Action Deep Pass'

        if formation in ['SHOTGUN', 'EMPTY'] and pass_length == 'deep' and alignment in ['3x1', '3x2'] and all(route in ['GO', 'POST'] for route in routes):
            return 'Four Verticals (Spread Deep Attack with Isolation)'

        if formation in ['SHOTGUN', 'SINGLEBACK'] and any(route in ['OUT', 'CORNER', 'FLAT'] for route in routes) and pass_location in ['left', 'right']:
            return 'Flood Concept (Intermediate Stretch)'

        if formation in ['I_FORM', 'SINGLEBACK', 'PISTOL'] and pass_length == 'short' and any(route in ['HITCH', 'SLANT', 'CROSS'] for route in routes):
            return 'Inside Zone/Run Supportive Pass'

        return 'General Strategy'


    def analyze_route_combos(self):
        for quarter in range(1, 6):
            for down in range(1, 5):
                route_combos_df = self.get_route_combos_by_quarter_down(quarter, down, play_type='pass')
                route_combos_df['quarter'] = quarter
                route_combos_df['down'] = down
                self.route_combos_by_quarter_down[(quarter, down)] = route_combos_df

        combined_df = pd.concat(self.route_combos_by_quarter_down.values(), ignore_index=True)
        return combined_df


    def get_specific_quarter_down(self, quarter, down):
        return self.route_combos_by_quarter_down.get((quarter, down), pd.DataFrame())


    def calculate_strategy_stats(self, df):
        strategy_counts = df['strategy'].value_counts().reset_index()
        strategy_counts.columns = ['strategy', 'count']
        strategy_counts['percentage'] = (strategy_counts['count'] / df.shape[0]) * 100
        return strategy_counts


    def get_overall_strategy_stats(self):
        combined_df = self.analyze_route_combos()
        return self.calculate_strategy_stats(combined_df)


    def get_quarter_down_strategy_stats(self, quarter, down):
        specific_df = self.get_specific_quarter_down(quarter, down)
        if not specific_df.empty:
            return self.calculate_strategy_stats(specific_df)
        else:
            return pd.DataFrame(columns=['strategy', 'count', 'percentage'])
    

    def calculate_success_failure_stats(self):
        combined_df = self.analyze_route_combos()
        combined_df['is_success'] = combined_df['passResult'].apply(lambda x: 1 if x == 'C' else 0)

        strategy_stats = (
            combined_df.groupby('strategy')
            .agg(total_plays=('strategy', 'size'),
                 success_count=('is_success', 'sum'),
                 failure_count=('is_success', lambda x: (1 - x).sum()))
            .reset_index()
        )
        strategy_stats['success_percentage'] = (strategy_stats['success_count'] / strategy_stats['total_plays']) * 100
        strategy_stats['failure_percentage'] = (strategy_stats['failure_count'] / strategy_stats['total_plays']) * 100
        return strategy_stats
    

    def calculate_success_failure_stats_for_quarter_down(self, quarter, down):
        specific_df = self.get_specific_quarter_down(quarter, down)
        
        if specific_df.empty:
            return pd.DataFrame(columns=['strategy', 'total_plays', 'success_count', 'failure_count', 'success_percentage', 'failure_percentage'])

        specific_df['is_success'] = specific_df['passResult'].apply(lambda x: 1 if x == 'C' else 0)

        strategy_stats = (
            specific_df.groupby('strategy')
            .agg(total_plays=('strategy', 'size'),
                 success_count=('is_success', 'sum'),
                 failure_count=('is_success', lambda x: (1 - x).sum()))
            .reset_index()
        )
        strategy_stats['success_percentage'] = (strategy_stats['success_count'] / strategy_stats['total_plays']) * 100
        strategy_stats['failure_percentage'] = (strategy_stats['failure_count'] / strategy_stats['total_plays']) * 100
        return strategy_stats
    
    def calculate_success_failure_stats_all_quarters_downs(self):
        all_stats = []
        
        for quarter in range(1, 6): 
            for down in range(1, 5):  
                stats_df = self.calculate_success_failure_stats_for_quarter_down(quarter, down)
                stats_df['quarter'] = quarter
                stats_df['down'] = down
                all_stats.append(stats_df)
        
        non_empty_stats = [df for df in all_stats if not df.empty]
        combined_stats_df = pd.concat(non_empty_stats, ignore_index=True) if non_empty_stats else pd.DataFrame()
        return combined_stats_df
    


class QBRadarProcessor:
    def __init__(self, file_path, game_id, play_id):
        self.file_path = file_path
        self.game_id = game_id
        self.play_id = play_id
        self.utils = NFLPlotVisualizeUtils()
        self.visualizer = NFLPlotVisualizer(None)
        self.filtered_data = None
        self.receiver_scores = defaultdict(lambda: {'widening': [], 'beam': []})
        self.best_receivers_widening = []
        self.best_receivers_beam = []
        self.initial_ball_position = None
        self.perpendicular_distance_widening = None
        self.perpendicular_distance_beam = None
        self.ball_snapped = False
        self.best_receiver_selected = False
        self.summary_data = {}

    def load_data(self):
        cols = [
            "gameId", "playId","play_type", "frameId", "possessionTeam", "defensiveTeam", "Team", "x", "y", "event", "frameType",
            "yardlineNumber", "absoluteYardlineNumber", "passer_player_name", "receiver_player_name", "displayName",
            "position", "passResult","passLength", "yardsGained", "yardageGainedAfterTheCatch"
        ]
        df = pd.read_csv(self.file_path, usecols=cols, low_memory=False)
        self.filtered_data = df[(df['gameId'] == self.game_id) & (df['playId'] == self.play_id) & (df['play_type'] == 'pass')].copy()
        non_zero_values = self.filtered_data['yardageGainedAfterTheCatch'][self.filtered_data['yardageGainedAfterTheCatch'] != 0]
        #TODO: passLength is given lets see that
        if len(non_zero_values) > 0:
            non_zero_value = non_zero_values.iloc[0]
            self.filtered_data['yardageGainedAfterTheCatch'].replace(0, non_zero_value, inplace=True)
        else:
            self.filtered_data['yardageGainedAfterTheCatch'] = 0 

        self.filtered_data['yardsGainedAtTheCatchEvent'] = self.filtered_data['yardsGained'] - self.filtered_data['yardageGainedAfterTheCatch']


    def convert_display_name(self, display_name):
        if pd.isna(display_name):
            return None
        parts = display_name.split()
        if len(parts) < 2:
            return display_name
        return f"{parts[0][0].upper()}.{parts[1].capitalize()}"

    def process_frames(self):
        for frame_id in self.filtered_data['frameId'].unique():
            frame_data = self.filtered_data[self.filtered_data['frameId'] == frame_id]
            offensive_players = frame_data[frame_data['Team'] == self.filtered_data.possessionTeam.values[0]]
            defensive_players = frame_data[frame_data['Team'] == self.filtered_data.defensiveTeam.values[0]]
            ball_data = frame_data[frame_data['Team'] == 'football']
            events = frame_data['event'].values

            if self.initial_ball_position is None and not ball_data.empty:
                self.initial_ball_position = (ball_data['x'].values[0], ball_data['y'].values[0])
                initial_ball_x = self.initial_ball_position[0]

            qb_data = offensive_players[offensive_players['position'] == 'QB']
            if qb_data.empty:
                continue
            
            defender_positions = [(row['x'], row['y']) for _, row in defensive_players.iterrows()]

            if 'BEFORE_SNAP' in frame_data['frameType'].unique():
                self.ball_snapped = True

            if self.ball_snapped and not self.best_receiver_selected:
                best_receiver_widening, score_widening, best_receiver_beam, score_beam = self.visualizer.find_best_receiver(
                    offensive_players, qb_data.iloc[0], defender_positions
                )
                if best_receiver_widening:
                    self.best_receivers_widening.append(best_receiver_widening)
                if best_receiver_beam:
                    self.best_receivers_beam.append(best_receiver_beam)

                if any(event in frame_data['event'].values for event in ['ball_snap']):
                    self.best_receiver_selected = True

            for receiver in offensive_players[offensive_players['position'].isin(['TE', 'WR', 'RB'])].itertuples():
                receiver_name = receiver.displayName
                self.receiver_scores[receiver_name]['widening'].append(score_widening if receiver_name == best_receiver_widening else 0)
                self.receiver_scores[receiver_name]['beam'].append(score_beam if receiver_name == best_receiver_beam else 0)

            if any(event in events for event in ['pass_outcome_caught', 'pass_outcome_incomplete', 'pass_outcome_touchdown']):
                if best_receiver_widening:
                    receiver_data_widening = offensive_players[offensive_players['displayName'] == best_receiver_widening]
                    if not receiver_data_widening.empty:
                        final_position_widening_x = receiver_data_widening['x'].iloc[0]
                        self.perpendicular_distance_widening = abs(final_position_widening_x - initial_ball_x)

                if best_receiver_beam:
                    receiver_data_beam = offensive_players[offensive_players['displayName'] == best_receiver_beam]
                    if not receiver_data_beam.empty:
                        final_position_beam_x = receiver_data_beam['x'].iloc[0]
                        self.perpendicular_distance_beam = abs(final_position_beam_x - initial_ball_x)

    def calculate_summary(self):
        avg_scores = {
            receiver: {
                'avg_widening': sum(scores['widening']) / max(len(scores['widening']), 1),
                'avg_beam': sum(scores['beam']) / max(len(scores['beam']), 1)
            }
            for receiver, scores in self.receiver_scores.items()
        }

        most_frequent_receiver_widening = None
        most_frequent_receiver_beam = None

        if self.best_receivers_widening:
            most_frequent_receiver_widening = Counter(self.best_receivers_widening).most_common(1)[0][0]
        
        if self.best_receivers_beam:
            most_frequent_receiver_beam = Counter(self.best_receivers_beam).most_common(1)[0][0]

        self.summary_data = {
            "gameId": [self.game_id],
            "playId": [self.play_id],
            "passResult": [self.filtered_data['passResult'].iloc[0]],
            "yardsGained": [self.filtered_data['yardsGained'].iloc[0]],
            "possessionTeam": [self.filtered_data.possessionTeam.values[0] if not self.filtered_data.empty else None],
            "defensiveTeam": [self.filtered_data.defensiveTeam.values[0] if not self.filtered_data.empty else None],
            "passer_name": [self.filtered_data['passer_player_name'].iloc[0] if 'passer_player_name' in self.filtered_data.columns and not self.filtered_data.empty else None],
            "receiver_name": [self.filtered_data['receiver_player_name'].iloc[0] if 'receiver_player_name' in self.filtered_data.columns and not self.filtered_data.empty else None],
            "progressive_widening_search_receiver_name": [self.convert_display_name(most_frequent_receiver_widening) if most_frequent_receiver_widening else None],
            "beam_target_search_receiver_name": [self.convert_display_name(most_frequent_receiver_beam) if most_frequent_receiver_beam else None],
            "progressive_widening_search_widening_receiver_score": [avg_scores.get(most_frequent_receiver_widening, {}).get('avg_widening') if most_frequent_receiver_widening else None],
            "beam_target_search_receiver_score": [avg_scores.get(most_frequent_receiver_beam, {}).get('avg_beam') if most_frequent_receiver_beam else None],
            "beam_target_search_receiver_distance": [self.perpendicular_distance_beam],
            "progressive_widening_search_receiver_distance": [self.perpendicular_distance_widening],
            "yardsGainedAtTheCatchEvent": [self.filtered_data['yardsGainedAtTheCatchEvent'].iloc[0] if 'yardsGainedAtTheCatchEvent' in self.filtered_data.columns and not self.filtered_data.empty else None]
        }

        final_df = pd.DataFrame(self.summary_data)
        return final_df
    
    @staticmethod
    def process_qb_radar_for_teams(team_names, base_file_path, output_base_path="assets/qb_radar"):
        """
        Process QB radar data for each team and save individual and combined files.
        
        Args:
            team_names (list): List of team abbreviations to process.
            base_file_path (str): Path pattern for each team's data file.
            output_base_path (str): Base path to save QB radar output files.
        """
        all_team_data = []
        os.makedirs(output_base_path, exist_ok=True)

        for team_name in team_names:
            team_file_path = base_file_path.format(team_name=team_name)
            output_file_path = os.path.join(output_base_path, f"{team_name}_qb_radar.csv")
            
            try:
                logger.info(f"Reading QB radar GameId PlayId for {team_name}")
                team_df = pd.read_csv(team_file_path, usecols=["gameId", "playId", "play_type"])
                team_df = team_df[team_df['play_type'] == "pass"]

                if team_df.empty:
                    logger.warning(f"No passing plays found for {team_name}. Skipping.")
                    continue

                unique_game_play_ids = team_df[['gameId', 'playId']].drop_duplicates()
                logger.info(f"Succesfully fetched all QB radar GameId PlayId for {team_name}")
                
                play_data_list = []
                logger.info(f"Processing QB radar data for {team_name}")

                for _, row in unique_game_play_ids.iterrows():
                    try:
                        game_id = row['gameId']
                        play_id = row['playId']
                        
                        # Initialize QBRadarProcessor for each unique gameId and playId
                        processor = QBRadarProcessor(team_file_path, game_id, play_id)
                        processor.load_data()
                        processor.process_frames()
                        summary_df = processor.calculate_summary()
                        
                        play_data_list.append(summary_df)
                    except Exception as e:
                        logger.error(f"Error processing play {game_id}-{play_id}: {str(e)}")
                        logger.error(traceback.format_exc()) 
                        continue
   
                
                # Concatenate and save team-specific data
                if play_data_list:
                    team_summary_df = pd.concat(play_data_list, ignore_index=True)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    team_summary_df.to_csv(output_file_path, index=False)
                    logger.info(f"Processed and saved QB radar data for {team_name}")
                    
                    # Append non-empty team summary data to all-team data
                    all_team_data.append(team_summary_df)
                else:
                    logger.warning(f"No valid play data found for team {team_name}.")

            except Exception as e:
                logger.error(f"Error processing QB radar for team '{team_name}': {e}")

        # Save combined data for all teams
        if all_team_data:
            full_summary_df = pd.concat(all_team_data, ignore_index=True)
            full_output_path = os.path.join(output_base_path, "full_qb_radar.csv")
            full_summary_df.to_csv(full_output_path, index=False)
            logger.info(f"All team QB radar data combined and saved to {full_output_path}")
        else:
            logger.warning("No valid team data found for any team.")




class BasePlayerMetrics:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.final_grouped_data = None

    def _scale_ratings(self, grouped, position):
        grouped['position'] = position
        overall_ratings = grouped.groupby(['nflId', 'displayName','Team', 'position']).agg(
            rating=('rating', 'mean'),
            games_played=('gameId', 'nunique'),
        ).reset_index()

        if overall_ratings.empty:
            return pd.DataFrame(columns=['nflId', 'displayName', 'position', 'final_rating'])

        # Scale ratings to 0-10
        overall_ratings['overall_rating_scaled'] = (
            10 * (overall_ratings['rating'] - overall_ratings['rating'].min()) /
            (overall_ratings['rating'].max() - overall_ratings['rating'].min() + 1e-5)
        )

        # Adjust for games played using a smoothing factor
        overall_ratings['adjusted_rating'] = overall_ratings['overall_rating_scaled'] * np.sqrt(
            overall_ratings['games_played'].clip(lower=1)
        )

        # Normalize the adjusted ratings to 0-10
        overall_ratings['final_rating'] = (
            10 * (overall_ratings['adjusted_rating'] - overall_ratings['adjusted_rating'].min()) /
            (overall_ratings['adjusted_rating'].max() - overall_ratings['adjusted_rating'].min() + 1e-5)
        )

        # Handle NaN values and apply rounding rules
        def adjust_rating(rating):
            if pd.isna(rating):  
                return 1
            if rating == 0:
                return 1
            elif 0 <= rating <= 3:
                return int(np.ceil(rating))  
            elif 4 <= rating <= 10:
                return int(rating)  
            else:
                return rating 

        overall_ratings['final_rating'] = overall_ratings['final_rating'].apply(adjust_rating)
        overall_ratings['final_rating'] = overall_ratings['final_rating'].astype(int)
        return overall_ratings[['nflId', 'displayName', 'position', 'Team','final_rating']]


class OffensivePlayerMetrics(BasePlayerMetrics):
    def _calculate_weighted_score(self, grouped, weights):
        """
        Calculate a weighted score based on the given weights for metrics.
        """
        weighted_sum = sum(grouped[metric] * weight for metric, weight in weights.items())
        grouped['weighted_score'] = weighted_sum
        grouped['rating'] = (
            (grouped['weighted_score'] - grouped['weighted_score'].min()) /
            (grouped['weighted_score'].max() - grouped['weighted_score'].min() + 1e-5)
        )
        return grouped

    def calculate_qb_rating(self):
        qb_df = self.dataframe[self.dataframe['position'] == 'QB']
        grouped = qb_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_completions=('passResult', lambda x: (x == 'C').sum()),
            total_pass_attempts=('pass_attempt', 'sum'),
            total_passing_yards=('passingYards', 'sum'),
            total_touchdown_passes=('passResult', lambda x: (x == 'TD').sum()),
            total_interceptions=('passResult', lambda x: (x == 'IN').sum()),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['completion_percentage'] = grouped['total_completions'] / grouped['total_pass_attempts']
        grouped['yards_per_attempt'] = grouped['total_passing_yards'] / grouped['total_pass_attempts']
        grouped['touchdown_percentage'] = grouped['total_touchdown_passes'] / grouped['total_pass_attempts']
        grouped['interception_percentage'] = grouped['total_interceptions'] / grouped['total_pass_attempts']

        weights = {
            'completion_percentage': 0.3,
            'yards_per_attempt': 0.2,
            'touchdown_percentage': 0.2,
            'interception_percentage': -0.1,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'QB')

    def calculate_rb_rating(self):
        rb_df = self.dataframe[self.dataframe['position'] == 'RB']
        grouped = rb_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_rushing_yards=('rushingYards', 'sum'),
            total_receiving_yards=('receivingYards', 'sum'),
            total_yards_after_catch=('yardageGainedAfterTheCatch', 'sum'),
            total_fumbles=('fumbles', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_rushing_yards'] + grouped['total_receiving_yards'] + grouped['total_yards_after_catch']
        grouped['failure'] = grouped['total_fumbles']

        weights = {
            'success': 0.7,
            'failure': -0.2,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'RB')

    def calculate_fb_rating(self):
        fb_df = self.dataframe[self.dataframe['position'] == 'FB']
        grouped = fb_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_rushing_yards=('rushingYards', 'sum'),
            total_receiving_yards=('receivingYards', 'sum'),
            total_yards_after_catch=('yardageGainedAfterTheCatch', 'sum'),
            total_fumbles=('fumbles', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_rushing_yards'] + grouped['total_receiving_yards'] + grouped['total_yards_after_catch']
        grouped['failure'] = grouped['total_fumbles']

        weights = {
            'success': 0.7,
            'failure': -0.2,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'FB')

    def calculate_wr_rating(self):
        wr_df = self.dataframe[self.dataframe['position'] == 'WR']
        grouped = wr_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_receiving_yards=('receivingYards', 'sum'),
            total_yards_after_catch=('yardageGainedAfterTheCatch', 'sum'),
            total_receptions=('hadPassReception', 'sum'),
            total_fumbles=('fumbles', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_receiving_yards'] + grouped['total_yards_after_catch'] + grouped['total_receptions']
        grouped['failure'] = grouped['total_fumbles']

        weights = {
            'success': 0.7,
            'failure': -0.2,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'WR')

    def calculate_te_rating(self):
        te_df = self.dataframe[self.dataframe['position'] == 'TE']
        grouped = te_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_receiving_yards=('receivingYards', 'sum'),
            total_yards_after_catch=('yardageGainedAfterTheCatch', 'sum'),
            total_receptions=('hadPassReception', 'sum'),
            total_fumbles=('fumbles', 'sum'),
            total_pressures_allowed=('pressureAllowedAsBlocker', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_receiving_yards'] + grouped['total_yards_after_catch'] + grouped['total_receptions']
        grouped['failure'] = grouped['total_fumbles'] + grouped['total_pressures_allowed']

        weights = {
            'success': 0.6,
            'failure': -0.3,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'TE')

    def calculate_ol_rating(self, position):
        ol_df = self.dataframe[self.dataframe['position'] == position]
        grouped = ol_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_blocked_players=('blockedPlayerNFLId1', 'count'),
            total_pressures_allowed=('pressureAllowedAsBlocker', 'sum'),
            avg_time_to_pressure=('timeToPressureAllowedAsBlocker', 'mean'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_blocked_players']
        grouped['failure'] = grouped['total_pressures_allowed'] + grouped['avg_time_to_pressure']

        weights = {
            'success': 0.7,
            'failure': -0.2,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, position)

    def calculate_all_offense_ratings(self):
        ratings = [
            self.calculate_qb_rating(),
            self.calculate_rb_rating(),
            self.calculate_fb_rating(),
            self.calculate_wr_rating(),
            self.calculate_te_rating(),
            self.calculate_ol_rating('T'),
            self.calculate_ol_rating('G'),
            self.calculate_ol_rating('C')
        ]
        
        # Filter out empty DataFrames
        non_empty_ratings = [df for df in ratings if not df.empty]
        
        if non_empty_ratings:
            # Ensure all DataFrames have the same columns
            columns = non_empty_ratings[0].columns
            aligned_ratings = [df[columns] for df in non_empty_ratings]
            
            self.final_grouped_data = pd.concat(aligned_ratings, ignore_index=True)
        else:
            # If all DataFrames are empty, create an empty DataFrame with the expected columns
            self.final_grouped_data = pd.DataFrame(columns=['nflId', 'displayName', 'position', 'Team', 'final_rating'])
        
        return self.final_grouped_data



class DefensivePlayerMetrics(BasePlayerMetrics):
    def _calculate_weighted_score(self, grouped, weights):  
        weighted_sum = sum(grouped[metric] * weight for metric, weight in weights.items())
        grouped['weighted_score'] = weighted_sum
        grouped['rating'] = (
            (grouped['weighted_score'] - grouped['weighted_score'].min()) /
            (grouped['weighted_score'].max() - grouped['weighted_score'].min() + 1e-5)
        )
        return grouped
    
    def calculate_de_rating(self):
        de_df = self.dataframe[self.dataframe['position'] == 'DE']
        grouped = de_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_sacks=('sackYardsAsDefense', 'sum'),
            total_tackles_for_loss=('tackleForALoss', 'sum'),
            total_qb_hits=('quarterbackHit', 'sum'),
            total_forced_fumbles=('forcedFumbleAsDefense', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_sacks'] + grouped['total_tackles_for_loss'] + grouped['total_qb_hits']
        grouped['failure'] = grouped['total_forced_fumbles']
        weights = {
            'success': 0.6,
            'failure': -0.2,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'DE')

    def calculate_dt_rating(self):
        dt_df = self.dataframe[self.dataframe['position'] == 'DT']
        grouped = dt_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_sacks=('sackYardsAsDefense', 'sum'),
            total_tackles_for_loss=('tackleForALoss', 'sum'),
            total_qb_hits=('quarterbackHit', 'sum'),
            total_forced_fumbles=('forcedFumbleAsDefense', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_sacks'] + grouped['total_tackles_for_loss'] + grouped['total_qb_hits']
        grouped['failure'] = grouped['total_forced_fumbles']
        weights = {
            'success': 0.6,
            'failure': -0.2,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'DT')
    
    def calculate_nt_rating(self):
        nt_df = self.dataframe[self.dataframe['position'] == 'NT']
        grouped = nt_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_sacks=('sackYardsAsDefense', 'sum'),
            total_tackles_for_loss=('tackleForALoss', 'sum'),
            total_qb_hits=('quarterbackHit', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_sacks'] + grouped['total_tackles_for_loss'] + grouped['total_qb_hits']
        grouped['failure'] = 0  
        weights = {
            'success': 0.6,
            'failure': -0.2,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'NT')

    def calculate_lb_rating(self):
        lb_df = self.dataframe[self.dataframe['position'].isin(['OLB', 'ILB', 'MLB', 'LB'])]
        grouped = lb_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_solo_tackles=('soloTackle', 'sum'),
            total_assisted_tackles=('assistedTackle', 'sum'),
            total_tackles_for_loss=('tackleForALoss', 'sum'),
            total_pass_defensed=('passDefensed', 'sum'),
            total_interceptions=('hadInterception', 'sum'),
            total_interception_yards=('interceptionYards', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_solo_tackles'] + grouped['total_assisted_tackles'] + grouped['total_pass_defensed']
        grouped['failure'] = grouped['total_tackles_for_loss']
        weights = {
            'success': 0.7,
            'failure': -0.3,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'LB')

    def calculate_cb_rating(self):
        cb_df = self.dataframe[self.dataframe['position'] == 'CB']
        grouped = cb_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_pass_defensed=('passDefensed', 'sum'),
            total_interceptions=('hadInterception', 'sum'),
            total_interception_yards=('interceptionYards', 'sum'),
            total_solo_tackles=('soloTackle', 'sum'),
            total_receiving_yards_allowed=('receivingYards', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_pass_defensed'] + grouped['total_interceptions'] + grouped['total_solo_tackles']
        grouped['failure'] = grouped['total_receiving_yards_allowed']
        weights = {
            'success': 0.7,
            'failure': -0.3,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'CB')
    
    def calculate_ss_rating(self):
        ss_df = self.dataframe[self.dataframe['position'] == 'SS']
        grouped = ss_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_tackles=('soloTackle', 'sum'),
            total_pass_defensed=('passDefensed', 'sum'),
            total_interceptions=('hadInterception', 'sum'),
            total_interception_yards=('interceptionYards', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_tackles'] + grouped['total_pass_defensed'] + grouped['total_interceptions']
        grouped['failure'] = 0  
        weights = {
            'success': 0.7,
            'failure': -0.3,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'SS')

    def calculate_fs_rating(self):
        fs_df = self.dataframe[self.dataframe['position'] == 'FS']
        grouped = fs_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_pass_defensed=('passDefensed', 'sum'),
            total_interceptions=('hadInterception', 'sum'),
            total_tackles=('soloTackle', 'sum'),
            total_interception_yards=('interceptionYards', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_pass_defensed'] + grouped['total_interceptions'] + grouped['total_tackles']
        grouped['failure'] = 0  
        weights = {
            'success': 0.7,
            'failure': -0.3,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'FS')

    def calculate_db_rating(self):
        db_df = self.dataframe[self.dataframe['position'] == 'DB']
        grouped = db_df.groupby(['nflId', 'displayName', 'Team', 'gameId', 'playId']).agg(
            total_pass_defensed=('passDefensed', 'sum'),
            total_interceptions=('hadInterception', 'sum'),
            total_tackles=('soloTackle', 'sum'),
            total_interception_yards=('interceptionYards', 'sum'),
            epa=('expectedPointsAdded', 'sum'),
            wpa=('homeTeamWinProbabilityAdded', 'sum')
        ).reset_index()

        grouped['success'] = grouped['total_pass_defensed'] + grouped['total_interceptions'] + grouped['total_tackles']
        grouped['failure'] = 0  # Adjust if there are specific DB failure metrics
        weights = {
            'success': 0.7,
            'failure': -0.3,
            'epa': 0.1,
            'wpa': 0.1
        }

        grouped = self._calculate_weighted_score(grouped, weights)
        return self._scale_ratings(grouped, 'DB')

    def calculate_all_defense_ratings(self):
        ratings = [
            self.calculate_de_rating(),
            self.calculate_dt_rating(),
            self.calculate_nt_rating(), 
            self.calculate_lb_rating(),
            self.calculate_cb_rating(),
            self.calculate_ss_rating(),
            self.calculate_fs_rating(),
            self.calculate_db_rating()
        ]
        

        # Filter out empty DataFrames
        non_empty_ratings = [df for df in ratings if not df.empty]
        
        if non_empty_ratings:
            # Ensure all DataFrames have the same columns
            columns = non_empty_ratings[0].columns
            aligned_ratings = [df[columns] for df in non_empty_ratings]
            
            self.final_grouped_data = pd.concat(aligned_ratings, ignore_index=True)
        else:
            # If all DataFrames are empty, create an empty DataFrame with the expected columns
            self.final_grouped_data = pd.DataFrame(columns=['nflId', 'displayName', 'position', 'Team', 'final_rating'])
        
        return self.final_grouped_data
    
    

class PlayerRatingProcessor:
    def __init__(self, offense_player_rating_base_path, defense_player_rating_base_path, player_ratings_base_path):
        """
        Initialize the processor with base paths for offense, defense, and combined ratings.
        """
        self.offense_player_rating_base_path = offense_player_rating_base_path
        self.defense_player_rating_base_path = defense_player_rating_base_path
        self.player_ratings_base_path = player_ratings_base_path
        os.makedirs(player_ratings_base_path, exist_ok=True)

    
    def process_offense_player_rating(self, team_names):
        """
        Process offensive player ratings for all teams and save combined results.
        """
        all_team_ratings = []
        for team_name in team_names:
            try:
                file_path = os.path.join(self.offense_player_rating_base_path, team_name, f"{team_name}_full_data.csv")
                df = pd.read_csv(file_path, low_memory=False)

                metrics = OffensivePlayerMetrics(df)
                team_ratings = metrics.calculate_all_offense_ratings()
                team_ratings['Team'] = team_name

                # Save individual team ratings
                team_output_path = os.path.join(self.player_ratings_base_path, f"{team_name}_offense_ratings.csv")
                team_ratings.to_csv(team_output_path, index=False)

                all_team_ratings.append(team_ratings)
                logger.info(f"Processed offensive player ratings for {team_name}")
            except Exception as e:
                logger.error(f"Error processing offensive player ratings for {team_name}: {e}")

        # Combine all team ratings
        combined_ratings = pd.concat(all_team_ratings, ignore_index=True)
        combined_output_path = os.path.join(self.player_ratings_base_path, "combined_offense_ratings.csv")
        combined_ratings.to_csv(combined_output_path, index=False)
        logger.info("Combined offensive player ratings saved")

    def process_defense_player_rating(self, team_names):
        """
        Process defensive player ratings for all teams and save combined results.
        """
        all_team_ratings = []
        for team_name in team_names:
            try:
                file_path = os.path.join(self.defense_player_rating_base_path, team_name, f"{team_name}_full_data.csv")
                df = pd.read_csv(file_path, low_memory=False)

                metrics = DefensivePlayerMetrics(df)
                team_ratings = metrics.calculate_all_defense_ratings()
                team_ratings['Team'] = team_name

                # Save individual team ratings
                team_output_path = os.path.join(self.player_ratings_base_path, f"{team_name}_defense_ratings.csv")
                team_ratings.to_csv(team_output_path, index=False)

                all_team_ratings.append(team_ratings)
                logger.info(f"Processed defensive player ratings for {team_name}")
            except Exception as e:
                logger.error(f"Error processing defensive player ratings for {team_name}: {e}")

        # Combine all team ratings
        combined_ratings = pd.concat(all_team_ratings, ignore_index=True)
        combined_output_path = os.path.join(self.player_ratings_base_path, "combined_defense_ratings.csv")
        combined_ratings.to_csv(combined_output_path, index=False)
        logger.info("Combined defensive player ratings saved")

    def process_player_ratings_by_quarter_and_down(self, team_names):
        """
        Process player ratings for each quarter and down and save results.
        """
        for team_name in team_names:
            try:
                # Load team data
                offense_file_path = os.path.join(self.offense_player_rating_base_path, team_name, f"{team_name}_full_data.csv")
                defense_file_path = os.path.join(self.defense_player_rating_base_path, team_name, f"{team_name}_full_data.csv")
                offense_df = pd.read_csv(offense_file_path, low_memory=False)
                defense_df = pd.read_csv(defense_file_path, low_memory=False)

                # Initialize metrics classes
                offense_metrics = OffensivePlayerMetrics(offense_df)
                defense_metrics = DefensivePlayerMetrics(defense_df)

                # Iterate through quarters and downs
                for quarter in range(1, 6):  # Quarters 1-5
                    for down in range(1, 5):  # Downs 1-4
                        # Filter data for the specific quarter and down
                        offense_filtered = offense_df[(offense_df['quarter'] == quarter) & (offense_df['down'] == down)]
                        defense_filtered = defense_df[(defense_df['quarter'] == quarter) & (defense_df['down'] == down)]

                        if not offense_filtered.empty:
                            # Calculate offense ratings
                            offense_metrics.dataframe = offense_filtered
                            offense_ratings = offense_metrics.calculate_all_offense_ratings()
                        else:
                            offense_ratings = pd.DataFrame()
                            logger.warning("No Ratings Found!")

                        if not defense_filtered.empty:
                            # Calculate defense ratings
                            defense_metrics.dataframe = defense_filtered
                            defense_ratings = defense_metrics.calculate_all_defense_ratings()
                        else:
                            defense_ratings = pd.DataFrame()
                            logger.warning("No Ratings Found!")

                        # Combine offense and defense ratings
                        combined_ratings = pd.concat([offense_ratings, defense_ratings], ignore_index=True)

                        # Save the file with the naming convention {quarter}_{down}_all_player_ratings.csv
                        if not combined_ratings.empty:
                            file_name = f"{quarter}_{down}_all_player_ratings.csv"
                            output_path = os.path.join(self.player_ratings_base_path, team_name, file_name)
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            combined_ratings.to_csv(output_path, index=False)
                            logger.info(f"Saved player ratings for quarter {quarter}, down {down} for team {team_name}")

            except Exception as e:
                logger.error(f"Error processing player ratings by quarter and down for {team_name}: {e}")
                

    def concat_player_ratings_by_quarter_and_down(self, team_names):
        """
        Concatenate player ratings for each quarter and down across all teams and save as a single file.
        """
        for quarter in range(1, 6):  # Quarters 1-5
            for down in range(1, 5):  # Downs 1-4
                all_team_ratings = []  # Store ratings for all teams for the current quarter and down
                for team_name in team_names:
                    try:
                        # File path for the specific team's quarter and down
                        team_file_path = os.path.join(
                            self.player_ratings_base_path, 
                            team_name, 
                            f"{quarter}_{down}_all_player_ratings.csv"
                        )
                        if os.path.exists(team_file_path):
                            team_ratings = pd.read_csv(team_file_path)
                            all_team_ratings.append(team_ratings)
                        else:
                            logger.warning(f"File not found: {team_file_path}")
                    except Exception as e:
                        logger.error(f"Error reading file for {team_name}, quarter {quarter}, down {down}: {e}")
                
                if all_team_ratings:
                    # Combine all team ratings for the current quarter and down
                    combined_ratings = pd.concat(all_team_ratings, ignore_index=True)

                    # Save the combined file
                    combined_file_name = f"all_{quarter}_{down}_player_rating.csv"
                    combined_file_path = os.path.join(self.player_ratings_base_path, combined_file_name)
                    combined_ratings.to_csv(combined_file_path, index=False)
                    logger.info(f"Saved combined player ratings: {combined_file_name}")
                else:
                    logger.warning(f"No data to combine for quarter {quarter}, down {down}")


    def process_all_player_ratings(self):
        """
        Process both offense and defense ratings and combine them into a single file.
        """
        # Combine offense and defense ratings
        offense_ratings_path = os.path.join(self.player_ratings_base_path, "combined_offense_ratings.csv")
        defense_ratings_path = os.path.join(self.player_ratings_base_path, "combined_defense_ratings.csv")

        if os.path.exists(offense_ratings_path) and os.path.exists(defense_ratings_path):
            offense_ratings = pd.read_csv(offense_ratings_path)
            defense_ratings = pd.read_csv(defense_ratings_path)

            all_ratings = pd.concat([offense_ratings, defense_ratings], ignore_index=True)
            all_ratings_output_path = os.path.join(self.player_ratings_base_path, "all_player_ratings.csv")
            all_ratings.to_csv(all_ratings_output_path, index=False)
            logger.info("All player ratings combined and saved")
        else:
            logger.error("Cannot combine ratings: One or both combined ratings files are missing")



class PlayGroundSimulatorOffenseDataProcessor:
    def __init__(self, offense_data_path, route_combos_path, player_ratings_path):
        self.offense_data_path = offense_data_path
        self.route_combos_path = route_combos_path
        self.player_ratings_path = player_ratings_path
        self.use_cols_required = [
            "season", "week", "gameId", "playId", "nflId", "quarter", "down", "yardsToGo",
            "absoluteYardlineNumber", "offenseFormation", "play_type", "receiverAlignment",
            "routeRan", "defenseFormation", "pff_manZone", "pff_defensiveCoverageAssignment",
            "yardsGained", "possessionTeam", "defensiveTeam", "Team", "displayName",
            "position", "passer_player_name", "receiver_player_name"
        ]
        self.use_cols_strategy = ["gameId", "playId", "strategy"]
        self.use_cols_player_rating = ["nflId", "final_rating"]
    
    def load_data(self):
        self.offense_df = pd.read_csv(self.offense_data_path, usecols=self.use_cols_required, low_memory=False)
        self.route_combos_df = pd.read_csv(self.route_combos_path, usecols=self.use_cols_strategy, low_memory=False)
        self.player_ratings_df = pd.read_csv(self.player_ratings_path, usecols=self.use_cols_player_rating, low_memory=False)
    
    def process_offense_plays(self):
        results = []
        unique_plays = self.offense_df[['gameId', 'playId']].drop_duplicates()
        
        for _, play in unique_plays.iterrows():
            gameId = play['gameId']
            playId = play['playId']
            
            filtered_data = self.offense_df[(self.offense_df['gameId'] == gameId) & (self.offense_df['playId'] == playId)]
            if filtered_data.empty:
                continue
            
            try:
                season = filtered_data.season.values[0]
                week = filtered_data.week.values[0]
                nflId = filtered_data.nflId.values[0]
                quarter = filtered_data.quarter.values[0]
                down = filtered_data.down.values[0]
                yardsToGo = filtered_data.yardsToGo.values[0]
                absoluteYardlineNumber = filtered_data.absoluteYardlineNumber.values[0]
                passer_player_name = filtered_data.passer_player_name.values[0]
                receiver_player_name = filtered_data.receiver_player_name.values[0]
                offenseFormation = filtered_data.offenseFormation.values[0]
                play_type = filtered_data.play_type.values[0]
                receiverAlignment = filtered_data.receiverAlignment.values[0]
                defenseFormation = filtered_data.defenseFormation.values[0]
                pff_manZone = filtered_data.pff_manZone.values[0]
                yardsGained = filtered_data.yardsGained.values[0]
                
                possession_team = filtered_data['possessionTeam'].iloc[0]
                defensive_team = filtered_data['defensiveTeam'].iloc[0]
                
                offensive_players = (
                    filtered_data[filtered_data['Team'] == possession_team]
                    .drop_duplicates(subset=['displayName'])
                    .head(11)
                )
                defensive_players = (
                    filtered_data[filtered_data['Team'] == defensive_team]
                    .drop_duplicates(subset=['displayName'])
                    .head(11)
                )
                
                offense = {}
                for i, row in enumerate(offensive_players.itertuples(), 1):
                    offense[f'offense_player_{i}_name'] = getattr(row, "displayName", np.nan)
                    offense[f'offense_player_{i}_position'] = getattr(row, "position", np.nan)
                    offense[f'offense_player_{i}_routeRan'] = getattr(row, "routeRan", np.nan)
                    offense[f'offense_player_{i}_nflId'] = getattr(row, "nflId", np.nan)
                    offense[f'offense_player_{i}_rating'] = 0 

                defense = {}
                for i, row in enumerate(defensive_players.itertuples(), 1):
                    defense[f'defense_player_{i}_name'] = getattr(row, "displayName", np.nan)
                    defense[f'defense_player_{i}_position'] = getattr(row, "position", np.nan)
                    defense[f'defense_player_{i}_cover_assignment'] = getattr(row, "pff_defensiveCoverageAssignment", np.nan)
                    defense[f'defense_player_{i}_nflId'] = getattr(row, "nflId", np.nan)
                    defense[f'defense_player_{i}_rating'] = 0

                result = {
                    'season': season,
                    'week': week,
                    'gameId': gameId, 
                    'playId': playId,
                    'nflid': nflId,
                    'quarter': quarter, 
                    'down': down,
                    'yardsToGo': yardsToGo,
                    'absoluteYardlineNumber': absoluteYardlineNumber,
                    'passer_player_name': passer_player_name,
                    'receiver_player_name': receiver_player_name,
                    'offenseFormation': offenseFormation,
                    'play_type': play_type,
                    'receiverAlignment': receiverAlignment,
                    'defenseFormation': defenseFormation,
                    'pff_manZone': pff_manZone,
                    'yardsGained': yardsGained
                }
                result.update(offense)
                result.update(defense)
                results.append(result)
            except Exception as e:
                print(f"Error processing gameId {gameId}, playId {playId}: {e}")

        self.final_df = pd.DataFrame(results)
    
    def merge_route_combos(self):
        self.final_df = pd.merge(self.final_df, self.route_combos_df, on=["gameId", "playId"], how="left")
    
    def merge_player_ratings(self):
        player_ratings_map = self.player_ratings_df.set_index('nflId')['final_rating'].to_dict()
        nflId_columns_to_drop = []

        for col in self.final_df.columns:
            if col.startswith('offense_player_') and col.endswith('_rating'):
                nflId_col = col.replace('_rating', '_nflId')
                if nflId_col in self.final_df.columns:
                    self.final_df[col] = self.final_df[nflId_col].map(player_ratings_map).fillna(0)
                    nflId_columns_to_drop.append(nflId_col) 
            elif col.startswith('defense_player_') and col.endswith('_rating'):
                nflId_col = col.replace('_rating', '_nflId')
                if nflId_col in self.final_df.columns:
                    self.final_df[col] = self.final_df[nflId_col].map(player_ratings_map).fillna(0)
                    nflId_columns_to_drop.append(nflId_col) 

        self.final_df.drop(columns=nflId_columns_to_drop, inplace=True)

        offense_rating_cols = [col for col in self.final_df.columns if col.startswith('offense_player_') and col.endswith('_rating')]
        defense_rating_cols = [col for col in self.final_df.columns if col.startswith('defense_player_') and col.endswith('_rating')]

        self.final_df['average_offense_rating'] = self.final_df[offense_rating_cols].mean(axis=1)
        self.final_df['average_defense_rating'] = self.final_df[defense_rating_cols].mean(axis=1)

        self.final_df['average_offense_rating'] = self.final_df.groupby(['gameId', 'playId'])['average_offense_rating'].transform('first')
        self.final_df['average_defense_rating'] = self.final_df.groupby(['gameId', 'playId'])['average_defense_rating'].transform('first')

                
    def get_final_data(self):
        return self.final_df




class PlayGroundSimulatorDefenseDataProcessor:
    def __init__(self, defense_data_path, route_combos_path, player_ratings_path):
        self.defense_data_path = defense_data_path
        self.route_combos_path = route_combos_path
        self.player_ratings_path = player_ratings_path
        self.use_cols_required = [
            "season", "week", "gameId", "playId", "nflId", "quarter", "down", "yardsToGo",
            "absoluteYardlineNumber", "offenseFormation", "play_type", "receiverAlignment",
            "routeRan", "defenseFormation", "pff_manZone", "pff_defensiveCoverageAssignment",
            "yardsGained", "possessionTeam", "defensiveTeam", "Team", "displayName",
            "position", "passer_player_name", "receiver_player_name"
        ]
        self.use_cols_strategy = ["gameId", "playId", "strategy"]
        self.use_cols_player_rating = ["nflId", "final_rating"]
    
    def load_data(self):
        self.defense_df = pd.read_csv(self.defense_data_path, usecols=self.use_cols_required, low_memory=False)
        self.route_combos_df = pd.read_csv(self.route_combos_path, usecols=self.use_cols_strategy, low_memory=False)
        self.player_ratings_df = pd.read_csv(self.player_ratings_path, usecols=self.use_cols_player_rating, low_memory=False)
    
    def process_defense_plays(self):
        results = []
        unique_plays = self.defense_df[['gameId', 'playId']].drop_duplicates()
        
        for _, play in unique_plays.iterrows():
            gameId = play['gameId']
            playId = play['playId']
            
            filtered_data = self.defense_df[(self.defense_df['gameId'] == gameId) & (self.defense_df['playId'] == playId)]
            if filtered_data.empty:
                continue
            
            try:
                season = filtered_data.season.values[0]
                week = filtered_data.week.values[0]
                nflId = filtered_data.nflId.values[0]
                quarter = filtered_data.quarter.values[0]
                down = filtered_data.down.values[0]
                yardsToGo = filtered_data.yardsToGo.values[0]
                absoluteYardlineNumber = filtered_data.absoluteYardlineNumber.values[0]
                passer_player_name = filtered_data.passer_player_name.values[0]
                receiver_player_name = filtered_data.receiver_player_name.values[0]
                offenseFormation = filtered_data.offenseFormation.values[0]
                play_type = filtered_data.play_type.values[0]
                receiverAlignment = filtered_data.receiverAlignment.values[0]
                defenseFormation = filtered_data.defenseFormation.values[0]
                pff_manZone = filtered_data.pff_manZone.values[0]
                yardsGained = filtered_data.yardsGained.values[0]
                
                possession_team = filtered_data['possessionTeam'].iloc[0]
                defensive_team = filtered_data['defensiveTeam'].iloc[0]
                
                offensive_players = (
                    filtered_data[filtered_data['Team'] == possession_team]
                    .drop_duplicates(subset=['displayName'])
                    .head(11)
                )
                defensive_players = (
                    filtered_data[filtered_data['Team'] == defensive_team]
                    .drop_duplicates(subset=['displayName'])
                    .head(11)
                )
                
                offense = {}
                for i, row in enumerate(offensive_players.itertuples(), 1):
                    offense[f'offense_player_{i}_name'] = getattr(row, "displayName", np.nan)
                    offense[f'offense_player_{i}_position'] = getattr(row, "position", np.nan)
                    offense[f'offense_player_{i}_routeRan'] = getattr(row, "routeRan", np.nan)
                    offense[f'offense_player_{i}_nflId'] = getattr(row, "nflId", np.nan)
                    offense[f'offense_player_{i}_rating'] = 0 

                defense = {}
                for i, row in enumerate(defensive_players.itertuples(), 1):
                    defense[f'defense_player_{i}_name'] = getattr(row, "displayName", np.nan)
                    defense[f'defense_player_{i}_position'] = getattr(row, "position", np.nan)
                    defense[f'defense_player_{i}_cover_assignment'] = getattr(row, "pff_defensiveCoverageAssignment", np.nan)
                    defense[f'defense_player_{i}_nflId'] = getattr(row, "nflId", np.nan)
                    defense[f'defense_player_{i}_rating'] = 0

                result = {
                    'season': season,
                    'week': week,
                    'gameId': gameId, 
                    'playId': playId,
                    'nflid': nflId,
                    'quarter': quarter, 
                    'down': down,
                    'yardsToGo': yardsToGo,
                    'absoluteYardlineNumber': absoluteYardlineNumber,
                    'passer_player_name': passer_player_name,
                    'receiver_player_name': receiver_player_name,
                    'offenseFormation': offenseFormation,
                    'play_type': play_type,
                    'receiverAlignment': receiverAlignment,
                    'defenseFormation': defenseFormation,
                    'pff_manZone': pff_manZone,
                    'yardsGained': yardsGained
                }
                result.update(offense)
                result.update(defense)
                results.append(result)
            except Exception as e:
                print(f"Error processing gameId {gameId}, playId {playId}: {e}")

        self.final_df = pd.DataFrame(results)
    
    def merge_route_combos(self):
        self.final_df = pd.merge(self.final_df, self.route_combos_df, on=["gameId", "playId"], how="left")
    
    def merge_player_ratings(self):
        player_ratings_map = self.player_ratings_df.set_index('nflId')['final_rating'].to_dict()
        nflId_columns_to_drop = []

        for col in self.final_df.columns:
            if col.startswith('offense_player_') and col.endswith('_rating'):
                nflId_col = col.replace('_rating', '_nflId')
                if nflId_col in self.final_df.columns:
                    self.final_df[col] = self.final_df[nflId_col].map(player_ratings_map).fillna(0)
                    nflId_columns_to_drop.append(nflId_col) 
            elif col.startswith('defense_player_') and col.endswith('_rating'):
                nflId_col = col.replace('_rating', '_nflId')
                if nflId_col in self.final_df.columns:
                    self.final_df[col] = self.final_df[nflId_col].map(player_ratings_map).fillna(0)
                    nflId_columns_to_drop.append(nflId_col) 

        self.final_df.drop(columns=nflId_columns_to_drop, inplace=True)

        offense_rating_cols = [col for col in self.final_df.columns if col.startswith('offense_player_') and col.endswith('_rating')]
        defense_rating_cols = [col for col in self.final_df.columns if col.startswith('defense_player_') and col.endswith('_rating')]

        self.final_df['average_offense_rating'] = self.final_df[offense_rating_cols].mean(axis=1)
        self.final_df['average_defense_rating'] = self.final_df[defense_rating_cols].mean(axis=1)

        self.final_df['average_offense_rating'] = self.final_df.groupby(['gameId', 'playId'])['average_offense_rating'].transform('first')
        self.final_df['average_defense_rating'] = self.final_df.groupby(['gameId', 'playId'])['average_defense_rating'].transform('first')

                
    def get_final_data(self):
        return self.final_df
    



class PlayGroundSimulatorDataProcessor:
    def __init__(self, team_names, output_folder="assets/playground/"):
        self.team_names = team_names
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def process_team_files(self):
        offense_dfs = []
        defense_dfs = []

        logger.info("Starting file processing for PlayGroundSimulatorDataProcessor")
        for team_name in self.team_names:
            # File paths
            offense_data_path = f"assets/offesnse-data/{team_name}/{team_name}_full_data.csv"
            defense_data_path = f"assets/defense-data/{team_name}/{team_name}_full_data.csv"
            route_combos_path = f"assets/offesnse-data/{team_name}/combined/full_route_combos.csv"
            player_ratings_path = "assets/player_ratings/all_player_ratings.csv"

            logger.info(f"Processing PlayGroundSimulatorOffenseDataProcessor for team:{team_name}")
            # Process offense data
            offense_processor = PlayGroundSimulatorOffenseDataProcessor(
                offense_data_path=offense_data_path,
                route_combos_path=route_combos_path,
                player_ratings_path=player_ratings_path
            )
            offense_processor.load_data()
            offense_processor.process_offense_plays()
            offense_processor.merge_route_combos()
            offense_processor.merge_player_ratings()
            offense_data = offense_processor.get_final_data()

            # Store individual offense file
            offense_file_path = os.path.join(self.output_folder, f"{team_name}_offense_data.csv")
            logger.info(f"Storing PlayGround Simulator Offense Data for team {team_name}")
            offense_data.drop_duplicates(subset=["gameId", "playId"], inplace=True)
            offense_data.to_csv(offense_file_path, index=False)
            offense_dfs.append(offense_data)
            logger.info(f"Stored PlayGround Simulator Offense Data for team {team_name} at: {offense_file_path}")

            logger.info(f"Processing PlayGroundSimulatorDefenseDataProcessor for team:{team_name}")
            # Process defense data
            defense_processor = PlayGroundSimulatorDefenseDataProcessor(
                defense_data_path=defense_data_path,
                route_combos_path=route_combos_path,
                player_ratings_path=player_ratings_path
            )
            defense_processor.load_data()
            defense_processor.process_defense_plays()
            defense_processor.merge_route_combos()
            defense_processor.merge_player_ratings()
            defense_data = defense_processor.get_final_data()

            # Store individual defense file
            defense_file_path = os.path.join(self.output_folder, f"{team_name}_defense_data.csv")
            logger.info(f"Storing PlayGround Simulator Defense Data for team {team_name}")
            defense_data.drop_duplicates(subset=["gameId", "playId"], inplace=True)
            defense_data.to_csv(defense_file_path, index=False)
            defense_dfs.append(defense_data)
            logger.info(f"Stored PlayGround Simulator Defense Data for team {team_name} at: {defense_file_path}")

        logger.info("Combining all PlayGround Simulator Offense Data")
        # Combine all offense data into one file
        combined_offense_data = pd.concat(offense_dfs, ignore_index=True).drop_duplicates(subset=["gameId", "playId"])
        combined_offense_file = os.path.join(self.output_folder, "all_offense_data.csv")
        combined_offense_data.to_csv(combined_offense_file, index=False)
        logger.info("Combined all PlayGround Simulator Offense Data")


        logger.info("Combining all PlayGround Simulator Defense Data")
        # Combine all defense data into one file
        combined_defense_data = pd.concat(defense_dfs, ignore_index=True).drop_duplicates(subset=["gameId", "playId"])
        combined_defense_file = os.path.join(self.output_folder, "all_defense_data.csv")
        combined_defense_data.to_csv(combined_defense_file, index=False)
        logger.info("Combined all PlayGround Simulator Defense Data")


        logger.info("Combining all PlayGround Simulator Offense & Defense Data in to a Single File")
        # Combine all offense and defense data into a single file
        combined_all_data = pd.concat([combined_offense_data, combined_defense_data], ignore_index=True).drop_duplicates(subset=["gameId", "playId"])
        combined_all_file = os.path.join(self.output_folder, "all_playground_data.csv")
        combined_all_data.to_csv(combined_all_file, index=False)
        logger.info(f"All files processed and stored in {self.output_folder}")

