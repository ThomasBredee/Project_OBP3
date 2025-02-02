#########################################################
#                                                       #
# Created on: 11/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 02/02/2025                                #
# Updated by: Dennis Botman                             #
#                                                       #
#########################################################

# Imports
import logging
import time
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from VRP_Solver.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
from Machine_Learning_Predict.prepare_input import PrepareInput
from Input_Transformation.transforming_input import TransformInput
from Machine_Learning_Predict.predictor import ModelPredictor
import pandas as pd

# Configure logging to clear the log file each time
logging.basicConfig(
    filename='timer_log_main_operations_haversine_manyLarge.log',
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


####### INPUTS FOR MODEL FOR MAIN TEST #
TRUCK_CAPACITY = 10
CHOSEN_COMPANY = "Dynamic Group"
CHOSEN_CANDIDATE = "Elite Group"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "haversine"

if __name__ == "__main__":
    ########## Get data for variable path (many)
    start_time = time.time()
    path = "Data/manyLarge.csv"
    input_df = pd.read_csv(path)
    substep_time = time.time()
    check_road_proximity = True  # Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_modified = transformer.execute_validations(input_df)
    log_substep("Do validation and transformation on input data", substep_time)
    input_df_clustering = transformer.drop_duplicates(input_df)
    log_step("Get data for variable path", start_time)

    CHOSEN_CANDIDATES = input_df[input_df["name"]!=CHOSEN_COMPANY]["name"].unique()
    ########## Get partial distance matrix
    start_time = time.time()
    distance_calc = RoadDistanceCalculator()
    distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY,
                                                                      method=METHOD)
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
    k_means_ranking = ranking.k_means(input_df_clustering, input_df_modified, full_matrix, CHOSEN_COMPANY)
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
    # SUBSTEP Get prediction model rf
    RANKING = "greedy"
    path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
    model_file = f"{path}XGBoost_booster_{METHOD}_{RANKING}.model"
    scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"

    greedy_ranking = ranking.greedy(distance_matrix_ranking, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepper.prep_greedy(input_df_modified_with_depot, CHOSEN_COMPANY,
                                                                        CHOSEN_CANDIDATES, METHOD,
                                                                        distance_matrix_ranking,
                                                                        TRUCK_CAPACITY, greedy_ranking)

    log_substep("Get data to predict on Greedy", substep_time)

    substep_time = time.time()
    # SUBSTEP Make prediction
    data_to_predict_on = data_to_predict_on.drop(columns=["chosen_candidate"])
    predictor = ModelPredictor(model_file, scaler_file)
    rankings = ranking.ranker_ml(data_to_predict_on, model_file, scaler_file, CHOSEN_CANDIDATES)

    log_substep("Make predictions Greedy ML", substep_time)

    substep_time = time.time()
    # SUBSTEP Get prediction model XGBoost
    RANKING = "bounding_circle"
    path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
    model_file = f"{path}XGBoost_booster_{METHOD}_{RANKING}.model"
    scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"

    bounding_ranking = ranking.bounding_circle(input_df, distance_matrix_ranking, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepper.prep_bounding_circle(input_df, input_df_modified,
                                                                                 distance_matrix_ranking,
                                                                                 TRUCK_CAPACITY, CHOSEN_COMPANY,
                                                                                 CHOSEN_CANDIDATES, METHOD,
                                                                                 bounding_ranking)

    log_substep("Get data to predict on Bounding Circle", substep_time)

    substep_time = time.time()
    # SUBSTEP Make prediction
    data_to_predict_on = data_to_predict_on.drop(columns=["chosen_candidate"])
    predictor = ModelPredictor(model_file, scaler_file)
    rankings = ranking.ranker_ml(data_to_predict_on, model_file, scaler_file, CHOSEN_CANDIDATES)

    log_substep("Make predictions Bounding Circle", substep_time)

    substep_time = time.time()
    # SUBSTEP Get prediction model XGBoost
    RANKING = "k_means"
    path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
    model_file = f"{path}XGBoost_booster_{METHOD}_{RANKING}.model"
    scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"

    k_means_ranking = ranking.k_means(input_df_clustering, input_df_modified, full_matrix, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepper.prep_k_means(input_df_clustering, input_df_modified,
                                                                         distance_matrix_ranking, full_matrix,
                                                                         TRUCK_CAPACITY,
                                                                         CHOSEN_COMPANY, CHOSEN_CANDIDATES, METHOD,
                                                                         k_means_ranking)


    log_substep("Get data to predict on K-Means", substep_time)

    substep_time = time.time()
    # SUBSTEP Make prediction
    data_to_predict_on = data_to_predict_on.drop(columns=["chosen_candidate"])
    predictor = ModelPredictor(model_file, scaler_file)
    rankings = ranking.ranker_ml(data_to_predict_on, model_file, scaler_file, CHOSEN_CANDIDATES)

    log_substep("Make predictions K_Means", substep_time)

    substep_time = time.time()
    # SUBSTEP Get prediction model XGBoost
    RANKING = "k_means"
    path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
    model_file = f"{path}XGBoost_booster_{METHOD}_{RANKING}.model"
    scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"

    dbscan_ranking = ranking.dbscan(input_df_clustering, full_matrix, CHOSEN_COMPANY)
    data_to_predict_on, distance_matrix_vrp = prepper.prep_dbscan(input_df_clustering,
                                                                        input_df_modified_with_depot,
                                                                        distance_matrix_ranking, full_matrix,
                                                                        TRUCK_CAPACITY,
                                                                        CHOSEN_COMPANY, CHOSEN_CANDIDATES, METHOD,
                                                                        dbscan_ranking)

    log_substep("Get data to predict on DBScan", substep_time)

    substep_time = time.time()
    # SUBSTEP Make prediction
    data_to_predict_on = data_to_predict_on.drop(columns=["chosen_candidate"])
    #predictor = ModelPredictor(model_file, scaler_file)
    #rankings = ranking.ranker_ml(data_to_predict_on, model_file, scaler_file, CHOSEN_CANDIDATES)

    log_substep("Make predictions DBScan", substep_time)

    substep_time = time.time()
    # SUBSTEP Get prediction model XGBoost
    RANKING = "all"
    path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
    model_file = f"{path}XGBoost_booster_{METHOD}_{RANKING}.model"
    scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"

    greedy_ranking = ranking.greedy(distance_matrix_ranking, CHOSEN_COMPANY)
    bounding_ranking = ranking.bounding_circle(input_df, distance_matrix_ranking, CHOSEN_COMPANY)
    k_means_ranking = ranking.k_means(input_df_clustering, input_df_modified, full_matrix, CHOSEN_COMPANY)
    dbscan_ranking = ranking.dbscan(input_df_clustering, full_matrix, CHOSEN_COMPANY, 17, 2)

    data_to_predict_on, distance_matrix_vrp = prepper.prep_all(input_df_clustering, input_df_modified,
                                                                     input_df_modified_with_depot,
                                                                     distance_matrix_ranking, full_matrix,
                                                                     TRUCK_CAPACITY, CHOSEN_COMPANY,
                                                                     CHOSEN_CANDIDATES, METHOD, greedy_ranking,
                                                                     bounding_ranking,
                                                                     k_means_ranking, dbscan_ranking)

    log_substep("Get data to predict on Main ML Algorithm", substep_time)

    substep_time = time.time()
    # SUBSTEP Make prediction
    #data_to_predict_on = data_to_predict_on.drop(columns=["chosen_candidate"])
    predictor = ModelPredictor(model_file, scaler_file)
    rankings = ranking.ranker_ml(data_to_predict_on, model_file, scaler_file, CHOSEN_CANDIDATES)

    log_substep("Make predictions Main ML Algorithm", substep_time)

    log_step("Get candidate ranking (machine learning)", start_time)

    ########## Get full distance matrix
    start_time = time.time()
    distance_matrix_vrp = distance_calc.calculate_distance_matrix(input_df_modified_with_depot,
                                                                  chosen_company=CHOSEN_COMPANY,
                                                                  candidate_name=CHOSEN_CANDIDATE, method=METHOD,
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
