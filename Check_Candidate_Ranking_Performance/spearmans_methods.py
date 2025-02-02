#########################################################
#                                                       #
# Created on: 24/01/2025                                #
# Created by: Lukas                                     #
#                                                       #
# Updated on: 24/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

#Import
from Candidate_Ranking.ranking_methods import CandidateRanking
from VRP_Solver.distance_calculator import RoadDistanceCalculator
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kruskal


class CalcSpearmans():
    def __init__(self):
        self.correlation_df = pd.DataFrame()
        pass

    def compute_spearman_greedy(self, true_ranking_df, input_df, method, vehicle_cap):
        """
        Handle the 'greedy' method for ranking with spearman.
        """
        candidate_ranker = CandidateRanking()
        correlations = []
        distance_calc = RoadDistanceCalculator()

        for i in range(len(true_ranking_df)):

            #Get predicted ranking
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[true_ranking_column['Rankings'] != len(true_ranking_df)]
            partial_distance_matrix = distance_calc.calculate_distance_matrix(input_df, chosen_company=chosen_company,
                candidate_name=None, method=method, computed_distances_df=None
            )
            predicted_ranking = candidate_ranker.greedy(partial_distance_matrix, chosen_company)

            #Calculate spearmans coeff
            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']

            #Reorder ranking
            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)
            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap] = correlations


    def compute_spearman_bounding_circle(self, true_ranking_df, input_df, input_df_modified, method, vehicle_cap):
        """
        Handle the 'bounding circle' method for ranking.
        """
        candidate_ranker = CandidateRanking()
        correlations = []
        distance_calc = RoadDistanceCalculator()

        for i in range(len(true_ranking_df)):
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[true_ranking_column['Rankings'] != len(true_ranking_df)]

            partial_distance_matrix = distance_calc.calculate_distance_matrix(
                input_df_modified, chosen_company=chosen_company,
                candidate_name=None, method=method, computed_distances_df=None
            )

            #Get ranking using bounding box algorithm
            predicted_ranking = candidate_ranker.bounding_circle(input_df, partial_distance_matrix, chosen_company)

            #Calculate spearmans coeff
            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']
            # Reorder ranking
            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)

            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap] = correlations

    def compute_spearman_accuracy_k_means(self, true_ranking_df, input_df, input_df_modified, vehicle_cap):
        """
        Handle the 'k-means' method for ranking.
        """
        candidate_ranker = CandidateRanking()
        correlations = []
        distance_calc = RoadDistanceCalculator()

        for i in range(len(true_ranking_df)):
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[true_ranking_column['Rankings'] != len(true_ranking_df)]

            # Get full matrix for k-means
            full_matrix = distance_calc.calculate_square_matrix(input_df_modified)
            predicted_ranking = candidate_ranker.k_means(input_df, input_df_modified, full_matrix, chosen_company)

            # Calculate spearmans coeff
            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']
            # Reorder ranking
            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)

            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap]=correlations


    def compute_spearman_accuracy_dbscan(self, true_ranking_df, input_df, input_df_modified, vehicle_cap, ms=2):
        """
        Computes the Spearman correlation between true and predicted rankings using the DBSCAN clustering method.

        Parameters:
        - true_ranking_df (DataFrame): The true rankings of candidates for each company.
        - input_df (DataFrame): The dataset used for clustering.
        - input_df_modified (DataFrame): A modified version of the dataset, used for distance calculations.
        - vehicle_cap (int): The vehicle capacity being evaluated.
        - ms (int, optional): The minimum samples parameter for DBSCAN. Default is 2.

        Process:
        - Extracts true rankings for each company.
        - Computes a distance matrix using `RoadDistanceCalculator`.
        - Determines an appropriate `eps` value for DBSCAN.
        - Predicts rankings using DBSCAN clustering.
        - Calculates the Spearman correlation between true and predicted rankings.
        - Stores correlation results for the given vehicle capacity.

        Returns:
        - None (results are stored in `self.correlation_df`).
        """
        candidate_ranker = CandidateRanking()
        correlations = []
        distance_calc = RoadDistanceCalculator()

        for i in range(len(true_ranking_df)):
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[
                true_ranking_column['Rankings'] != len(true_ranking_df)]

            # Get full matrix for k-means
            full_matrix = distance_calc.calculate_square_matrix(input_df_modified)
            eps = int(round(19 * (len(full_matrix) ** -0.1) + 5, 0))
            predicted_ranking = candidate_ranker.dbscan(input_df, full_matrix, chosen_company, eps, ms)
            # Calculate spearmans coeff
            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']
            # Reorder ranking
            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)

            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap] = correlations

    def _add_ranking(self, ranking):
        """
            Assigns a ranking to each entry in the given DataFrame.

            Parameters:
            - ranking (DataFrame): The DataFrame containing ranking data.

            Returns:
            - DataFrame: The input DataFrame with an added 'Ranking' column,
              where values range from 1 to the length of the DataFrame.
            """
        ranking['Ranking'] = range(1, len(ranking) + 1)
        return ranking

    def perform_test(self, correlations_df, file_size, method):
        """
            Performs the Kruskal-Wallis test to analyze differences in Spearman correlation distributions across truck capacities.

            Parameters:
            - correlations_df (DataFrame): Spearman correlation values for different truck capacities.
            - file_size (int): Size of the input data file.
            - method (str): The ranking method used.

            Outputs:
            - Prints the Kruskal-Wallis test statistic and p-value.
            """
        #Restructure data for kruskal wallis test
        long_df = correlations_df.reset_index().melt(
            id_vars='index',
            var_name='Truck Capacity',
            value_name='Spearman Coefficient'
        ).rename(columns={'index': 'Company'})

        #Group data by Truck Capacity
        groups = [group['Spearman Coefficient'].values for _, group in long_df.groupby('Truck Capacity')]

        #perform Kruskal-Wallis Test
        kruskal_result = kruskal(*groups)
        print(f"Kruskal-Wallis Test Result for {method} , and file size {file_size}:")
        print('statistic:', kruskal_result[0])
        print('p-value:', kruskal_result[1])

    def compute_spearman_accuracy_ml_greedy(self, true_ranking_df, vehicle_cap,RANKING,METHOD,MODEL_TYPE,CHOSEN_CANDIDATES,company_data_to_predict):
        """
            Computes the Spearman correlation between true and predicted rankings using a Greedy-based ML approach.

            Parameters:
            - true_ranking_df (DataFrame): Actual candidate rankings per company.
            - vehicle_cap (int): Vehicle capacity.
            - RANKING, METHOD, MODEL_TYPE (str): Settings for ranking and ML model.
            - CHOSEN_CANDIDATES (list): List of candidates.
            - company_data_to_predict (dict): Mapping of companies to their prediction data.

            Updates:
            - `self.correlation_df` with computed correlations.
            """
        candidate_ranker = CandidateRanking()
        correlations = []

        for i in range(len(true_ranking_df)):
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[
                true_ranking_column['Rankings'] != len(true_ranking_df)]

            #prepare data for predictions
            path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
            model_file = f"Machine_Learning_Train/haversine/TrainedModels/XGBoost/xgboost_booster_haversine_greedy.model"
            scaler_file = f"Machine_Learning_Train/haversine/TrainedModels/XGBoost/scaler_haversine_greedy.pkl"


            chosen_candidates = list(CHOSEN_CANDIDATES)
            chosen_candidates.remove(chosen_company)

            data_to_predict_on = company_data_to_predict[chosen_company]


            #Obtain predicted ranking
            predicted_ranking = candidate_ranker.greedy_ml(data_to_predict_on,model_file,scaler_file,chosen_candidates)

            #Calculate spearmans coeff

            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']
            #Reorder ranking

            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)

            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap] = correlations


    def compute_spearman_accuracy_ml_bounding_circle(self, true_ranking_df, vehicle_cap,RANKING,METHOD,MODEL_TYPE,CHOSEN_CANDIDATES,company_data_to_predict):
        """
            Computes the Spearman correlation between true and predicted rankings using a Bounding Circle-based ML approach.

            Parameters:
            - true_ranking_df (DataFrame): Actual candidate rankings per company.
            - vehicle_cap (int): Vehicle capacity.
            - RANKING, METHOD, MODEL_TYPE (str): Settings for ranking and ML model.
            - CHOSEN_CANDIDATES (list): List of candidates.
            - company_data_to_predict (dict): Mapping of companies to their prediction data.

            Updates:
            - `self.correlation_df` with computed correlations.
            """
        candidate_ranker = CandidateRanking()
        correlations = []

        for i in range(len(true_ranking_df)):
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[
                true_ranking_column['Rankings'] != len(true_ranking_df)]
            #prepare data for ml
            path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
            model_file = f"{path}{MODEL_TYPE}_booster_{METHOD}_{RANKING}.model"
            scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"


            chosen_candidates = list(CHOSEN_CANDIDATES)
            chosen_candidates.remove(chosen_company)

            data_to_predict_on = company_data_to_predict[chosen_company]

            #Obtain predicted ranking
            predicted_ranking = candidate_ranker.bounding_circle_ml(data_to_predict_on,model_file,scaler_file,chosen_candidates)


            # Calculate spearmans coeff

            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']

            #Reorder ranking

            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)

            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap] = correlations



    def compute_spearman_accuracy_ml_k_means(self,true_ranking_df, vehicle_cap,RANKING,METHOD,MODEL_TYPE,CHOSEN_CANDIDATES,company_data_to_predict):
        """
            Computes the Spearman correlation between true and predicted rankings using a K-Means-based ML approach.

            Parameters:
            - true_ranking_df (DataFrame): Actual candidate rankings per company.
            - vehicle_cap (int): Vehicle capacity.
            - RANKING, METHOD, MODEL_TYPE (str): Settings for ranking and ML model.
            - CHOSEN_CANDIDATES (list): List of candidates.
            - company_data_to_predict (dict): Mapping of companies to their prediction data.

            Updates:
            - `self.correlation_df` with computed correlations.
            """
        candidate_ranker = CandidateRanking()
        correlations = []

        for i in range(len(true_ranking_df)):
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[
                true_ranking_column['Rankings'] != len(true_ranking_df)]

            # prepare data for ml
            path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
            model_file = f"{path}{MODEL_TYPE}_booster_{METHOD}_{RANKING}.model"
            scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"

            chosen_candidates = list(CHOSEN_CANDIDATES)
            chosen_candidates.remove(chosen_company)

            data_to_predict_on = company_data_to_predict[chosen_company]



            #Obtain predicted ranking
            predicted_ranking = candidate_ranker.k_means_ml(data_to_predict_on, model_file, scaler_file,
                                                                    CHOSEN_CANDIDATES)


            #Calculate spearmans coeff

            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']
            #Reorder ranking

            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)

            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap] = correlations


    def compute_spearman_accuracy_ml_dbscan(self, true_ranking_df, vehicle_cap,
                                             RANKING, METHOD, MODEL_TYPE,
                                             CHOSEN_CANDIDATES, company_data_to_predict):
        """
            Computes the Spearman correlation between true and predicted rankings using a DBSCAN-based ML approach.

            Parameters:
            - true_ranking_df (DataFrame): Actual candidate rankings per company.
            - vehicle_cap (int): Vehicle capacity.
            - RANKING, METHOD, MODEL_TYPE (str): Settings for ranking and ML model.
            - CHOSEN_CANDIDATES (list): List of candidates.
            - company_data_to_predict (dict): Mapping of companies to their prediction data.

            Updates:
            - `self.correlation_df` with computed correlations.
            """
        candidate_ranker = CandidateRanking()
        correlations = []

        for i in range(len(true_ranking_df)):
            true_ranking_column = true_ranking_df.iloc[:, [i]]
            chosen_company = true_ranking_column.columns[0]
            true_ranking_column.columns = ['Rankings']
            true_ranking_column_without_chosen_comp = true_ranking_column[
                true_ranking_column['Rankings'] != len(true_ranking_df)]

            #prepare data for predictions
            path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
            model_file = f"{path}{MODEL_TYPE}_booster_{METHOD}_{RANKING}.model"
            scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"


            chosen_candidates = list(CHOSEN_CANDIDATES)
            chosen_candidates.remove(chosen_company)

            data_to_predict_on = company_data_to_predict[chosen_company]


            #Obtain predicted ranking
            predicted_ranking = candidate_ranker.dbscan_ml(data_to_predict_on, model_file, scaler_file,
                                                            CHOSEN_CANDIDATES)

            #Calculate spearmans coeff

            predicted_ranking_added = self._add_ranking(predicted_ranking)
            predicted_ranking = predicted_ranking_added['Ranking']
            true_ranking = true_ranking_column_without_chosen_comp['Rankings']
            #Reorder ranking

            predicted_ranking_reordered = predicted_ranking.reindex(true_ranking.index)

            correlation, p_value = spearmanr(predicted_ranking_reordered, true_ranking)
            correlations.append(correlation)

        self.correlation_df[vehicle_cap] = correlations
