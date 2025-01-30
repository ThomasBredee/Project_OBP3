#########################################################
#                                                       #
# Created on: 11/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 23/01/2025                                #
# Updated by: Dennis Botman                             #
#                                                       #
#########################################################

# Imports
import logging
import time
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from VRP_Solver.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
from Machine_Learning_Predict.make_prediction import ModelPredictor
from Machine_Learning_Predict.prepare_input import PrepareInput
import pandas as pd
import joblib

# Configure logging to clear the log file each time
logging.basicConfig(
    filename='timer_log_main_test_Algorithms_pkg_osrm_manyLarge.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Open the file in write mode
)

def log_step(step_name, start_time):
    execution_time = time.time() - start_time
    logging.info(f'{step_name} completed in {execution_time:.4f} seconds')


def log_substep(substep_name, start_time):
    execution_time = time.time() - start_time
    logging.info(f'  {substep_name} completed in {execution_time:.4f} seconds')


####### INPUTS FOR MODEL FOR MAIN TEST
TRUCK_CAPACITY = 10
CHOSEN_COMPANY = "Dynamic Group"
CHOSEN_CANDIDATE = "Elite Group"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":
    ########## Get data for variable path (many)
    start_time = time.time()
    path = "Data/manyLarge.csv"
    input_df = pd.read_csv(path)
    input_df_modified = input_df.copy()
    input_df_modified['name'] = input_df.groupby('name').cumcount().add(1).astype(str).radd(input_df['name'] + "_")
    log_step("Get data for variable path", start_time)

    ########## Get partial distance matrix
    start_time = time.time()
    distance_calc = RoadDistanceCalculator()
    distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY,
                                                                      method="osrm")
    log_step("Get partial distance matrix", start_time)

    ########## Get ranking
    start_time = time.time()
    substep_time = time.time()

    # SUBSTEP Get full matrix for k-means
    full_matrix = distance_calc.calculate_square_matrix(input_df_modified)
    log_substep("Calculate full matrix for k-means", substep_time)

    substep_time = time.time()
    # SUBSTEP Initialize ranking
    ranking = CandidateRanking()
    log_substep("Initialize ranking", substep_time)

    substep_time = time.time()
    # SUBSTEP Get greedy ranking
    greedy_ranking = ranking.greedy(distance_matrix_ranking, CHOSEN_COMPANY)
    log_substep("Calculate greedy ranking", substep_time)

    substep_time = time.time()
    # SUBSTEP Get circle box ranking
    circle_box_ranking = ranking.bounding_circle(input_df, distance_matrix_ranking, CHOSEN_COMPANY)
    log_substep("Calculate circle box ranking", substep_time)

    substep_time = time.time()
    # SUBSTEP Get k-means ranking
    k_means_ranking = ranking.k_means(input_df, input_df_modified, full_matrix, CHOSEN_COMPANY)
    log_substep("Calculate k-means ranking", substep_time)

    log_step("Get ranking", start_time)

    ########## Get candidate ranking methods (machine learning)
    start_time = time.time()
    substep_time = time.time()

    input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
    # SUBSTEP Initialize prepper
    prepper = PrepareInput()
    log_substep("Initialize data preparation module", substep_time)

    substep_time = time.time()
    # SUBSTEP Create features to predict from input
    prediction_df = prepper.prep_greedy(input_df_modified_with_depot, CHOSEN_COMPANY, greedy_ranking.index, "osrm",
                                        distance_matrix_ranking, TRUCK_CAPACITY, greedy_ranking)
    log_substep("Create features for prediction", substep_time)

    substep_time = time.time()
    # SUBSTEP Get prediction model rf
    path = "Machine_Learning_Train/osrm/TrainedModels/RF/"
    scaler = joblib.load(f"{path}scaler_greedy_osrm.pkl")
    model = joblib.load(f"{path}random_forest_model_greedy_osrm.pkl")
    predictor = ModelPredictor(model, scaler)
    log_substep("Load prediction model and scaler", substep_time)

    substep_time = time.time()
    # SUBSTEP Make prediction
    predicted_df = predictor.predict_for_candidates(prediction_df)
    log_substep("Make predictions", substep_time)

    log_step("Get candidate ranking (machine learning)", start_time)

    ########## Get full distance matrix
    start_time = time.time()
    distance_matrix_vrp = distance_calc.calculate_distance_matrix(input_df_modified_with_depot,
                                                                  chosen_company=CHOSEN_COMPANY,
                                                                  candidate_name=CHOSEN_CANDIDATE, method="osrm",
                                                                  computed_distances_df=distance_matrix_ranking)
    log_step("Get full distance matrix", start_time)

    ########## Solve VRP for single company
    start_time = time.time()
    vrp_solver = VRPSolver()
    print("Solving VRP for Single Company...")
    model_single, current_names_single = vrp_solver.build_model(input_df=input_df_modified_with_depot,
                                                                chosen_company=CHOSEN_COMPANY,
                                                                distance_matrix=distance_matrix_vrp,
                                                                truck_capacity=TRUCK_CAPACITY)
    solution_single, routes_single = vrp_solver.solve(model=model_single, max_runtime=2, display=False,
                                                      current_names=current_names_single, print_routes=True)
    total_distance_single, avg_distance_per_order_single = vrp_solver.calculate_distance_per_order(routes=routes_single,
                                                                                                   distance_matrix=distance_matrix_vrp)
    log_step("Solve VRP for single company", start_time)

    ########## Solve VRP collaboration
    start_time = time.time()
    print("Solving VRP for Collaboration...")
    model_collab, current_names_collab = vrp_solver.build_model(input_df=input_df_modified_with_depot,
                                                                chosen_company=CHOSEN_COMPANY,
                                                                chosen_candidate=CHOSEN_CANDIDATE,
                                                                distance_matrix=distance_matrix_vrp,
                                                                truck_capacity=TRUCK_CAPACITY)
    solution_collab, routes_collab = vrp_solver.solve(model=model_collab, max_runtime=2, display=False,
                                                      current_names=current_names_collab, print_routes=True)
    total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(routes=routes_collab,
                                                                                                   distance_matrix=distance_matrix_vrp)
    log_step("Solve VRP collaboration", start_time)

    ########## Get expected gain
    start_time = time.time()
    predicted_df['Expected Gain'] = avg_distance_per_order_single - predicted_df['Predicted km per order']
    log_step("Get expected gain", start_time)
