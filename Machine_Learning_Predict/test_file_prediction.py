#########################################################
#                                                       #
# Created on: 28/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 28/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

from VRP_Solver.distance_calculator import RoadDistanceCalculator
from Candidate_Ranking.ranking_methods import CandidateRanking
from Input_Transformation.transforming_input import TransformInput
from Machine_Learning_Predict.predictor import ModelPredictor
import pandas as pd
from Machine_Learning_Predict.prepare_input import PrepareInput

#######INPUTS FOR MODEL FOR MAIN TEST
TRUCK_CAPACITY =  4
CHOSEN_COMPANY = "Visionary Consulting"
CHOSEN_CANDIDATES = ["Visionary Ventures", "Modern Solutions", "Green Group"]
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "osrm"
path = "Data/many.csv"
MODEL_TYPE = "xgboost"
RANKING = "all"

input_df = pd.read_csv(path)
check_road_proximity = True  # Set true if OSRM container running
transformer = TransformInput(check_road_proximity=check_road_proximity)
input_df_modified = transformer.execute_validations(input_df)
input_df_clustering = transformer.drop_duplicates(input_df)

###test instance single prediction
path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
model_file = f"{path}{MODEL_TYPE}_booster_{METHOD}_{RANKING}.model"
scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"
distance_calc = RoadDistanceCalculator()
input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY,
                                                                  method=METHOD)
full_matrix = distance_calc.calculate_square_matrix(input_df_modified)

ranking = CandidateRanking()
prepare_input = PrepareInput()

if RANKING == "greedy":
    greedy_ranking = ranking.greedy(distance_matrix_ranking, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_greedy(input_df_modified_with_depot, CHOSEN_COMPANY,
                                                                        CHOSEN_CANDIDATES, METHOD, distance_matrix_ranking,
                                                                        TRUCK_CAPACITY, greedy_ranking)

elif RANKING == "bounding_circle":
    bounding_ranking = ranking.bounding_circle(input_df, distance_matrix_ranking, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_bounding_circle(input_df, input_df_modified, input_df_modified_with_depot,
                                                                                 distance_matrix_ranking, TRUCK_CAPACITY, CHOSEN_COMPANY,
                                                                                 CHOSEN_CANDIDATES, METHOD, bounding_ranking)

elif RANKING == "k_means":
    k_means_ranking = ranking.k_means(input_df_clustering, input_df_modified, full_matrix, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_k_means(input_df_clustering, input_df_modified, input_df_modified_with_depot,
                                                                         distance_matrix_ranking, full_matrix, TRUCK_CAPACITY,
                                                                         CHOSEN_COMPANY, CHOSEN_CANDIDATES, METHOD, k_means_ranking)

elif RANKING == "dbscan":
    dbscan_ranking = ranking.dbscan(input_df_clustering, full_matrix, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_dbscan(input_df_clustering, input_df_modified_with_depot,
                                                                        distance_matrix_ranking, full_matrix, TRUCK_CAPACITY,
                                                                        CHOSEN_COMPANY, CHOSEN_CANDIDATES, METHOD, dbscan_ranking)

elif RANKING == "all":
    greedy_ranking = ranking.greedy(distance_matrix_ranking, CHOSEN_COMPANY)
    bounding_ranking = ranking.bounding_circle(input_df, distance_matrix_ranking, CHOSEN_COMPANY)
    k_means_ranking = ranking.k_means(input_df_clustering, input_df_modified, full_matrix, CHOSEN_COMPANY)
    dbscan_ranking = ranking.dbscan(input_df_clustering, full_matrix, CHOSEN_COMPANY, 17, 2)

    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_all(input_df_clustering, input_df_modified, input_df_modified_with_depot,
                                                                     distance_matrix_ranking, full_matrix, TRUCK_CAPACITY, CHOSEN_COMPANY,
                                                                     CHOSEN_CANDIDATES, METHOD, greedy_ranking, bounding_ranking,
                                                                     k_means_ranking, dbscan_ranking)


#data_to_predict_on = data_to_predict_on.drop(columns=["chosen_candidate"])
predictor = ModelPredictor(model_file, scaler_file)
kms = predictor.predict(data_to_predict_on)
rankings = ranking.ranker_ml(data_to_predict_on, model_file, scaler_file, CHOSEN_CANDIDATES)