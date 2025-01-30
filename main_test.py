#########################################################
#                                                       #
# Created on: 11/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 23/01/2025                                #
# Updated by: Dennis Botman                             #
#                                                       #
#########################################################

#Imports
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from VRP_Solver.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
from Machine_Learning_Predict.make_prediction import ModelPredictor
from Machine_Learning_Predict.prepare_input import PrepareInput
from Input_Transformation.transforming_input import TransformInput
import pandas as pd
import joblib


#######INPUTS FOR MODEL FOR MAIN TEST
TRUCK_CAPACITY =  4
CHOSEN_COMPANY = "Visionary Consulting"
CHOSEN_CANDIDATE = "Visionary Ventures"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":

    ##########Get data for variable path (many)
    ###set your path to the data directory here!
    path = "Data/many.csv"

    ###get the data
    input_df = pd.read_csv(path)
    check_road_proximity = True #Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_kmeans = transformer.drop_duplicates(input_df)
    input_df_modified = transformer.execute_validations(input_df)

    ##########Get partial distance matrix
    ###get distance matrix from chosen company to itself and all other locations in the input df
    distance_calc = RoadDistanceCalculator()
    distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY, method="osrm")

    ##########Get ranking
    ###get distance matrix squared (for kmeans)
    full_matrix =   distance_calc.calculate_square_matrix(input_df_modified)

    ###get candidate ranking methods (heuristics)
    ranking = CandidateRanking()
    greedy_ranking = ranking.greedy(distance_matrix_ranking, CHOSEN_COMPANY)
    circle_box_ranking= ranking.bounding_circle(input_df, distance_matrix_ranking, CHOSEN_COMPANY)
    k_means_ranking = ranking.k_means(input_df_kmeans, input_df_modified, full_matrix, CHOSEN_COMPANY)


    ###get candidate ranking methods (machine learning) / get predicted kilometers per order (For Greedy)
    #preprare data
    input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
    prepper = PrepareInput()
    prediction_df = prepper.prep_greedy(input_df_modified_with_depot, CHOSEN_COMPANY, greedy_ranking.index, "haversine", distance_matrix_ranking,
                                           TRUCK_CAPACITY, greedy_ranking)
    #get pre-trained model
    path = "Machine_Learning_Train/osrm/TrainedModels/RF/"
    scaler = joblib.load(f"{path}scaler_greedy_osrm.pkl")
    model = joblib.load(f"{path}random_forest_model_greedy_osrm.pkl")
    predictor = ModelPredictor(model,scaler)

    #make prediction and get ranking
    predicted_df = predictor.predict_for_candidates(prediction_df)

    ##########Get full distance matrix
    ###get the full distance matrix of best company
    distance_matrix_vrp= distance_calc.calculate_distance_matrix(input_df_modified_with_depot, chosen_company=CHOSEN_COMPANY,
        candidate_name=CHOSEN_CANDIDATE, method="osrm", computed_distances_df=distance_matrix_ranking)

    ###get best route
    vrp_solver = VRPSolver()

    ##########Solve VRP for single company
    # --- Single Company ---
    print("Solving VRP for Single Company...")
    model_single, current_names_single = vrp_solver.build_model(
        input_df=input_df_modified_with_depot,
        chosen_company=CHOSEN_COMPANY,
        distance_matrix=distance_matrix_vrp,
        truck_capacity=TRUCK_CAPACITY
    )

    solution_single, routes_single = vrp_solver.solve(
        model=model_single,
        max_runtime=2,
        display=False,
        current_names=current_names_single,
        print_routes=True
    )
    total_distance_single, avg_distance_per_order_single = vrp_solver.calculate_distance_per_order(
        routes=routes_single,
        distance_matrix=distance_matrix_vrp
    )

    #validate with a plot
    vrp_solver.plotRoute(routes_single, input_df_modified_with_depot)

    ###get expected gain
    predicted_df['Expected Gain'] = avg_distance_per_order_single - predicted_df['Predicted km per order']

    ##########Solve VRP collaboration
    # --- Collaboration ---
    print("Solving VRP for Collaboration...")
    model_collab, current_names_collab = vrp_solver.build_model(
        input_df=input_df_modified_with_depot,
        chosen_company=CHOSEN_COMPANY,
        chosen_candidate=CHOSEN_CANDIDATE,
        distance_matrix=distance_matrix_vrp,
        truck_capacity=TRUCK_CAPACITY
    )
    solution_collab, routes_collab = vrp_solver.solve(
        model=model_collab,
        max_runtime=2,
        display=False,
        current_names=current_names_collab,
        print_routes=True
    )
    total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(
        routes=routes_collab,
        distance_matrix=distance_matrix_vrp
    )

    #validate route
    #vrp_solver.plotRoute(routes_collab, input_df_modified_with_depot)





