#########################################################
#                                                       #
# Created on: 26/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

#Imports
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from VRP_Solver.solver_pyvrp import VRPSolver
from Input_Transformation.transforming_input import TransformInput
import pandas as pd

#######INPUTS FOR MODEL FOR MAIN TEST
TRUCK_CAPACITY =  4
CHOSEN_COMPANY = "Dynamic Industries" #Test client 1
CHOSEN_CANDIDATE = "Swift Technologies" #Test client 2
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "osrm" #haversine other option

if __name__ == "__main__":
    ##########Get data for variable path (many)
    ###set your path to the data directory here!
    path = "Data/medium.csv"

    ###get the data
    input_df = pd.read_csv(path)
    check_road_proximity = True #Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_kmeans = transformer.drop_duplicates(input_df)
    input_df_modified = transformer.execute_validations(input_df)

    ##########Get partial distance matrix
    ###get distance matrix from chosen company to itself and all other locations in the input df
    distance_calc = RoadDistanceCalculator()
    distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df_modified, chosen_company=CHOSEN_COMPANY, method=METHOD)

    ##########Get squared matrix
    ###get distance matrix squared (for kmeans, which assumes symmetry and is therefore always haversine)
    full_squared_matrix = distance_calc.calculate_square_matrix(input_df_modified)

    ##########Get full matrix
    ###get distance matrix squared (for kmeans, which assumes symmetry and is therefore always haversine)
    input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
    full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)

    ##########Get full distance matrix
    ###get the full distance matrix of best company
    distance_matrix_vrp= distance_calc.calculate_distance_matrix(input_df_modified_with_depot, chosen_company=CHOSEN_COMPANY,
        candidate_name=CHOSEN_CANDIDATE, method=METHOD, computed_distances_df=distance_matrix_ranking)

    ##########Get best route distance matrix
    vrp_solver = VRPSolver()

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
        display=True,
        current_names=current_names_single,
        print_routes=True
    )
    total_distance_single, avg_distance_per_order_single = vrp_solver.calculate_distance_per_order(
        routes=routes_single,
        distance_matrix=distance_matrix_vrp
    )
    vrp_solver.plotRoute(routes_single, input_df_modified_with_depot)

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
        display=True,
        current_names=current_names_collab,
        print_routes=True
    )
    total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(
        routes=routes_collab,
        distance_matrix=distance_matrix_vrp
    )

    vrp_solver.plotRoute(routes_collab, input_df_modified_with_depot)

