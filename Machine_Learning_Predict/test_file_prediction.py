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
from VRP_Solver.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
from Machine_Learning_Predict.make_candidate_prediction import ModelPredictor
from Machine_Learning_Predict.prepare_input import PrepareInput
from Input_Transformation.transforming_input import TransformInput
from Machine_Learning_Predict.predictor import ModelPredictor
import pandas as pd
from Machine_Learning_Predict.prepare_input import PrepareInput
import joblib
from scipy.spatial.distance import pdist

#######INPUTS FOR MODEL FOR MAIN TEST
TRUCK_CAPACITY =  4
CHOSEN_COMPANY = "Visionary Consulting"
CHOSEN_CANDIDATES = ["Visionary Ventures"]
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "osrm"
path = "Data/many.csv"
path_model = f"Machine_Learning_Train/{METHOD}/TrainedModels/RF/"
MODEL_TYPE = "random_forest"
RANKING = "k_means"
input_df = pd.read_csv(path)
check_road_proximity = True  # Set true if OSRM container running
transformer = TransformInput(check_road_proximity=check_road_proximity)
input_df_modified = transformer.execute_validations(input_df)
input_df_kmeans = transformer.drop_duplicates(input_df)

###test instance single prediction

#Greedy
RANKING = "greedy"
distance_calc = RoadDistanceCalculator()
input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY,
                                                                  method=METHOD)
ranking = CandidateRanking()
greedy_ranking = ranking.greedy(distance_matrix_ranking, CHOSEN_COMPANY)
prepare_input = PrepareInput()
data_to_predict_on, distance_matrix_vrp = prepare_input.prep_greedy(input_df_modified_with_depot, CHOSEN_COMPANY, CHOSEN_CANDIDATES, METHOD, distance_matrix_ranking,
                TRUCK_CAPACITY, greedy_ranking)

predictor = ModelPredictor(f"{path_model}{MODEL_TYPE}_model_{METHOD}_{RANKING}.pkl", f"{path_model}scaler_{METHOD}_{RANKING}.pkl")
kms = predictor.predict(data_to_predict_on)

#Bounding Circle
RANKING = "bounding_circle"
distance_calc = RoadDistanceCalculator()
input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY,
                                                                 method=METHOD)
ranking = CandidateRanking()
bounding_ranking = ranking.bounding_circle(input_df, distance_matrix_ranking, CHOSEN_COMPANY)

prepare_input = PrepareInput()
data_to_predict_on, distance_matrix_vrp = prepare_input.prep_bounding_circle(input_df, input_df_modified_with_depot, distance_matrix_ranking, TRUCK_CAPACITY, CHOSEN_COMPANY, CHOSEN_CANDIDATES, METHOD, bounding_ranking)
predictor = ModelPredictor(f"{path_model}{MODEL_TYPE}_model_{METHOD}_{RANKING}.pkl", f"{path_model}scaler_{METHOD}_{RANKING}.pkl")
kms = predictor.predict(data_to_predict_on)

#K_means
RANKING = "k_means"
distance_calc = RoadDistanceCalculator()
input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY,
                                                                 method=METHOD)
ranking = CandidateRanking()

full_matrix = distance_calc.calculate_square_matrix(input_df_modified_with_depot)
k_means_ranking = ranking.k_means(input_df_kmeans, input_df_modified, full_matrix, CHOSEN_COMPANY)

prepare_input = PrepareInput()
data_to_predict_on, distance_matrix_vrp = prepare_input.prep_k_means(input_df, input_df_modified_with_depot, distance_matrix_ranking, full_matrix, TRUCK_CAPACITY, CHOSEN_COMPANY, CHOSEN_CANDIDATES, METHOD, k_means_ranking)
predictor = ModelPredictor(f"{path_model}{MODEL_TYPE}_model_{METHOD}_{RANKING}.pkl", f"{path_model}scaler_{METHOD}_{RANKING}.pkl")
kms = predictor.predict(data_to_predict_on)


######INPUTS FOR FULL PREDICTION
