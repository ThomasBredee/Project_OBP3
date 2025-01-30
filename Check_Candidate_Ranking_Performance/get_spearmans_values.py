#########################################################
#                                                       #
# Updated on: 24/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

from Input_Transformation.transforming_input import TransformInput
from Check_Candidate_Ranking_Performance.spearmans_methods import CalcSpearmans
import pandas as pd

FILE_SIZE = "medium"
METHOD = "haversine"
file_location = f"VRP_results_{METHOD}_{FILE_SIZE}"
if __name__ == "__main__":

    ###Prepare input
    input_df = pd.read_csv(f"Data/{FILE_SIZE}.csv")
    check_road_proximity = True
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_modified = transformer.execute_validations(input_df)

    ###Create spearmans matrix
    accuracy = CalcSpearmans()
    company_names = input_df['name'].unique()
    accuracy.correlation_df = pd.DataFrame(index=company_names)
    correlation_df = pd.DataFrame(index=company_names)

    for truck_cap in range(2,21,1):
        sheet_name = f"Ranks_Truck_Cap_{truck_cap}"
        true_ranking = pd.read_excel(f"Solve_True_Rankings_VRP/Data_VRP_solved/{METHOD}/{file_location}.xlsx", sheet_name=sheet_name, index_col = 0)
        #accuracy.compute_spearman_greedy(true_ranking, input_df, METHOD, truck_cap)
        #accuracy.compute_spearman_bounding_circle(true_ranking, input_df, input_df_modified,METHOD ,truck_cap)
        accuracy.compute_spearman_accuracy_k_means(true_ranking, input_df, input_df_modified,METHOD ,truck_cap)

    correlation_df = round(accuracy.correlation_df,3)