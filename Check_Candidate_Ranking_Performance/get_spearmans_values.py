#########################################################
#                                                       #
# Updated on: 24/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

from Input_Transformation.transforming_input import TransformInput
from Check_Candidate_Ranking_Performance.spearmans_methods import CalcSpearmans
import pandas as pd
from Candidate_Ranking.ranking_methods import CandidateRanking
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from Check_Candidate_Ranking_Performance.ml_prepper import MLPrepper


MACHINE_LEARNING = True
FILE_SIZE = "mini"
METHOD = "osrm"
RANKING = "greedy"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
MODEL_TYPE = "xgboost"
file_location = f"Solve_True_Rankings_VRP/Data_VRP_solved/{METHOD}/VRP_results_{METHOD}_{FILE_SIZE}"

if __name__ == "__main__":

    ###Prepare input
    input_df = pd.read_csv(f"Data/{FILE_SIZE}.csv")
    check_road_proximity = False
    distance_calc = RoadDistanceCalculator()
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df = transformer.drop_duplicates(input_df)
    input_df_modified = transformer.execute_validations(input_df)
    input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)

    ###Create spearmans matrix
    accuracy = CalcSpearmans()
    company_names = input_df['name'].unique()
    accuracy.correlation_df = pd.DataFrame(index=company_names)
    correlation_df = pd.DataFrame(index=company_names)

    full_matrix = distance_calc.calculate_square_matrix(input_df_modified)
    if METHOD == "haversine":
        full_matrix = distance_calc.calculate_square_matrix(input_df_modified)
    elif METHOD == "osrm":
        full_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified)

    ranking = CandidateRanking()
    distance_matrix_for_candidates = ranking.clean_column_and_index_labels(full_matrix)
    CHOSEN_CANDIDATES = distance_matrix_for_candidates.columns.unique()

    company_data_to_predict = {}
    company_distance_matrix_vrp = {}
    prepare_input = MLPrepper()
    #Loop over each chosen company

    #use SMALL range for machine learning algorithms
    #for truck_cap in range(2,21,1):
    for truck_cap in [3, 8, 12, 18]:
        print('truck_cap',truck_cap)
        sheet_name = f"Ranks_Truck_Cap_{truck_cap}"
        true_ranking = pd.read_excel(f"{file_location}.xlsx", sheet_name=sheet_name, index_col = 0)
        if MACHINE_LEARNING is False:
            if RANKING == "greedy":
                accuracy.compute_spearman_greedy(true_ranking, input_df_modified, METHOD, truck_cap)
            elif RANKING =="bounding_circle":
                accuracy.compute_spearman_bounding_circle(true_ranking, input_df, input_df_modified, METHOD, truck_cap)
            elif RANKING == "k_means":
                accuracy.compute_spearman_accuracy_k_means(true_ranking, input_df, input_df_modified, truck_cap)
            elif RANKING =="dbscan":
                accuracy.compute_spearman_accuracy_dbscan(true_ranking, input_df, input_df_modified, truck_cap)

        if MACHINE_LEARNING is True:
            for chosen_company in CHOSEN_CANDIDATES:
                # Filter chosen candidates for the current company
                company_candidates = list(CHOSEN_CANDIDATES)
                company_candidates.remove(chosen_company)

                distance_matrix_ranking = full_matrix[full_matrix.index.str.startswith(chosen_company)]

                #change ranking for desired algorithm
                if RANKING == "greedy":
                    greedy_ranking = ranking.greedy(distance_matrix_ranking, chosen_company)
                    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_greedy(input_df_modified_with_depot,
                                                                                        chosen_company, company_candidates,
                                                                                        METHOD, distance_matrix_ranking,
                                                                                        truck_cap, greedy_ranking)
                    company_data_to_predict[chosen_company] = data_to_predict_on
                    company_distance_matrix_vrp[chosen_company] = distance_matrix_vrp

                elif RANKING == "bounding_circle":
                    bounding_ranking = ranking.bounding_circle(input_df, distance_matrix_ranking, chosen_company)
                    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_bounding_circle(
                        input_df, input_df_modified, input_df_modified_with_depot, distance_matrix_ranking,
                        truck_cap, chosen_company, company_candidates, METHOD, bounding_ranking)

                    company_data_to_predict[chosen_company] = data_to_predict_on
                    company_distance_matrix_vrp[chosen_company] = distance_matrix_vrp

                elif RANKING == "k_means":
                    k_means_ranking = ranking.k_means(input_df, input_df_modified, full_matrix, chosen_company)
                    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_k_means(input_df, input_df_modified,
                                                                                         input_df_modified_with_depot,
                                                                                         distance_matrix_ranking,
                                                                                         full_matrix,
                                                                                         truck_cap, chosen_company,
                                                                                         CHOSEN_CANDIDATES, METHOD,
                                                                                         k_means_ranking)
                    company_data_to_predict[chosen_company] = data_to_predict_on
                    company_distance_matrix_vrp[chosen_company] = distance_matrix_vrp

                elif RANKING == "dbscan":
                    EPS = 19*len(input_df)**(-0.1)+5
                    MS = 2
                    dbscan_ranking = ranking.dbscan(input_df, full_matrix, chosen_company,EPS, MS)
                    data_to_predict_on, distance_matrix_vrp = prepare_input.prep_dbscan(input_df, input_df_modified,
                                                                                        input_df_modified_with_depot,
                                                                                        distance_matrix_ranking,
                                                                                        full_matrix, truck_cap,
                                                                                        chosen_company, CHOSEN_CANDIDATES,
                                                                                        METHOD, dbscan_ranking, ms=2)
                    company_data_to_predict[chosen_company] = data_to_predict_on
                    company_distance_matrix_vrp[chosen_company] = distance_matrix_vrp

            if RANKING == "greedy":
                accuracy.compute_spearman_accuracy_ml_greedy(true_ranking, truck_cap,
                                                             RANKING, METHOD, MODEL_TYPE,
                                                             CHOSEN_CANDIDATES, company_data_to_predict)
            elif RANKING == "bounding_circle":
                accuracy.compute_spearman_accuracy_ml_bounding_circle(true_ranking, truck_cap,
                                                                      RANKING, METHOD, MODEL_TYPE, CHOSEN_CANDIDATES,
                                                                      company_data_to_predict)
            elif RANKING == "k_means":
                accuracy.compute_spearman_accuracy_ml_k_means(true_ranking, truck_cap,
                                                              RANKING, METHOD, MODEL_TYPE, CHOSEN_CANDIDATES,
                                                              company_data_to_predict)


            elif RANKING == "dbscan":
                accuracy.compute_spearman_accuracy_ml_dbscan(true_ranking, truck_cap,
                                                             RANKING, METHOD, MODEL_TYPE, CHOSEN_CANDIDATES,
                                                             company_data_to_predict)

    correlation_df = round(accuracy.correlation_df,3)
    print(correlation_df)
    print('mean: ',correlation_df.values.mean())
    accuracy.perform_test(correlation_df,FILE_SIZE,METHOD)