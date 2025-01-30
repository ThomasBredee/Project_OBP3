#########################################################
#                                                       #
# Created on: 15/01/2025                                #
# Created by: Lukas and Wester                          #
#                                                       #
# Updated on: 26/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

#Imports
from Candidate_Ranking.ranking_methods import CandidateRanking
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from Input_Transformation.transforming_input import TransformInput
import pandas as pd
import time

TRUCK_CAPACITY = 5
CHOSEN_COMPANY = "Dynamic Industries" #Test client 1
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "haversine" #can change to osrm
EPS = 8
MIN_SAMPLES = 2

if __name__ == "__main__":

    ###Get the data and transformations
    input_df = pd.read_csv("Data/manyLarge.csv")

    check_road_proximity = True
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_clustering = transformer.drop_duplicates(input_df)
    df_modified = transformer.execute_validations(input_df)


    ###Get distance matrix and squared matrix to test the algorithms
    calculator = RoadDistanceCalculator()
    partial_distance_matrix = calculator.calculate_distance_matrix(input_df, chosen_company=CHOSEN_COMPANY,
        candidate_name=None, method=METHOD, computed_distances_df=None)
    squared_matrix = calculator.calculate_square_matrix(df_modified)

    ###Test ranking algorithms
    ranking = CandidateRanking()

    #Test greedy algorithm
    predicted_ranking_greedy = ranking.greedy(partial_distance_matrix, CHOSEN_COMPANY)

    #Test bounding circle algorithm
    predicted_ranking_bounding_circle = ranking.bounding_circle(input_df, partial_distance_matrix, CHOSEN_COMPANY)

    #Test k means algorithm
    predicted_ranking_k_means = ranking.k_means(input_df_clustering, df_modified, squared_matrix, CHOSEN_COMPANY)

    predicted_ranking_dbscan = ranking.dbscan(input_df_clustering, squared_matrix, CHOSEN_COMPANY, EPS, MIN_SAMPLES)

