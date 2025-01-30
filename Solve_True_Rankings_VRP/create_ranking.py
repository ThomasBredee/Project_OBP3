#########################################################
#                                                       #
# Created on: 17/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 24/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

#Imports
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from Input_Transformation.transforming_input import TransformInput
from openpyxl import Workbook
from VRP_Solver.solver_pyvrp import VRPSolver
import pandas as pd


#######INPUTS FROM THE MODEL VARIABLES
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "osrm"
FILE_SIZE = "medium"
RUN_TIME = 1
output_filename = f"VRP_results_{METHOD}_{FILE_SIZE}.xlsx"

if __name__ == "__main__":
    input_df = pd.read_csv(f"Data/{FILE_SIZE}.csv")
    transformer = TransformInput(check_road_proximity=False)
    df_modified = transformer.execute_validations(input_df)

    distance_calc = RoadDistanceCalculator()
    df_with_depot = distance_calc.add_depot(df_modified, LAT_DEPOT, LONG_DEPOT)

    full_distance_matrix = distance_calc.calculate_full_distance_matrix(df_with_depot, method=METHOD)

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for truck_cap in range(2, 21):
            print(f"Processing for truck capacity: {truck_cap}")
            solutions_df = pd.DataFrame(index=input_df['name'].unique(), columns=input_df['name'].unique(),
                                        dtype=float)

            for chosen_company in solutions_df.columns:
                for chosen_candidate in solutions_df.index:
                    if chosen_candidate == chosen_company:
                        solutions_df.at[chosen_candidate, chosen_company] = float('inf')
                        continue

                    vrp_solver = VRPSolver()

                    # Filter the full distance matrix for the relevant rows and columns
                    row_filter = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
                    column_filter = full_distance_matrix.columns.to_series().apply(lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
                    distance_matrix_vrp = full_distance_matrix.loc[row_filter, column_filter]

                    model_collab, current_names_collab = vrp_solver.build_model(
                        input_df=df_with_depot,
                        chosen_company=chosen_company,
                        chosen_candidate=chosen_candidate,
                        distance_matrix=distance_matrix_vrp,
                        truck_capacity=truck_cap
                    )

                    solution_collab, routes_collab = vrp_solver.solve(
                        model=model_collab,
                        max_runtime=RUN_TIME,
                        display=False,
                        current_names=current_names_collab
                    )

                    total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(
                        routes=routes_collab,
                        distance_matrix=distance_matrix_vrp
                    )

                    if solution_collab:
                        solutions_df.at[chosen_candidate, chosen_company] = avg_distance_per_order_collab
                    else:
                        solutions_df.at[chosen_candidate, chosen_company] = float('inf')

            # Saving the results for this truck capacity to the same Excel file but different sheets
            solutions_sheet_name = f"Solutions_Truck_Cap_{truck_cap}"
            solutions_df.to_excel(writer, sheet_name=solutions_sheet_name)
            print(f"Results saved for truck capacity {truck_cap} in sheet {solutions_sheet_name}")

            # Calculate ranks based on the solutions DataFrame
            ranks_df = solutions_df.rank(method='min', axis=0, ascending=True, na_option='keep')
            ranks_df.replace(float('inf'), 'NoSolution', inplace=True)

            # Saving the ranks to another sheet
            ranks_sheet_name = f"Ranks_Truck_Cap_{truck_cap}"
            ranks_df.to_excel(writer, sheet_name=ranks_sheet_name)
            print(f"Ranks saved for truck capacity {truck_cap} in sheet {ranks_sheet_name}")

    print(f"All results saved to {output_filename}")
