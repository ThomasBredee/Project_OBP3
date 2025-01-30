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

    def compute_spearman_accuracy_k_means(self, true_ranking_df, input_df, input_df_modified, method, vehicle_cap):
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

    # def accuracy_dbscan(self, df, input_df, input_df_original, j):
    #     """
    #     Handle the 'dbscan' method for ranking.
    #     """
    #     algorithm = CandidateRanking()
    #     correlations = []
    #     calculator = RoadDistanceCalculator()
    #
    #     for i in range(len(df)):
    #         new_df = df.iloc[:, [i]]
    #         new_df.columns = ['Rankings']
    #         new_df = new_df[new_df['Rankings'] != len(df)]
    #         chosen_company = df.columns[i]
    #         distance_matrix = calculator.calculate_distance_matrix(
    #             input_df, chosen_company=chosen_company,
    #             candidate_name=None, method="haversine", computed_distances_df=None
    #         )
    #
    #         # Get full matrix for DBSCAN
    #         full_matrix = calculator.calculate_square_matrix(input_df)
    #         ranking = algorithm.dbscan(input_df, full_matrix)
    #
    #         # Add ranking and compute Spearman's rank correlation
    #         ranking_added = self._add_ranking(ranking)
    #         ranking_df1 = ranking_added['Ranking']
    #         ranking_df2 = new_df['Rankings']
    #         correlation, p_value = spearmanr(ranking_df1, ranking_df2)
    #
    #         # Print the result
    #         print(f"Spearman's Rank Correlation: {correlation}")
    #         print(f"P-value: {p_value}")
    #         correlations.append(correlation)
    #
    #     self.correlation_df[j] = correlations
    #     return

    def _add_ranking(self, ranking):
        ranking['Ranking'] = range(1, len(ranking) + 1)
        return ranking