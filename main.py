from Dashboard.dashboard import Dashboard
import streamlit as st
import time
import pandas as pd
import numpy as np
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from VRP_Solver.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
from Machine_Learning_Predict.predictor import ModelPredictor
from Machine_Learning_Predict.prepare_input import PrepareInput
from Input_Transformation.transforming_input import TransformInput
import joblib

LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":

    start_time_overall = time.time()

    dashboard = Dashboard()
    distance_calc = RoadDistanceCalculator()
    ranking = CandidateRanking()
    prepare_input = PrepareInput()

    if st.session_state.update1 and st.session_state.firsttime1 == False:
        st.sidebar.write(
            '<div style="text-align: center; color: red; font-weight: bold; font-style: italic;">'
            'Not up-to-date! <br>'
            'Recalculate Ranking and VRP.'
            '</div>',
            unsafe_allow_html=True
        )

# TODO: _ranking_using_machine_learning() should return the same format as ranking.greedy().....................
    def _ranking_using_machine_learning(ranking):
        temp_ranker = CandidateRanking()
        vrp_solver = VRPSolver()
        # distance_calc = RoadDistanceCalculator()
        if st.session_state.ml_choice:
            CHOSEN_CANDIDATES = \
                st.session_state.input_df[st.session_state.input_df["name"] != st.session_state.company_1][
                    "name"].unique()
        else:
            CHOSEN_CANDIDATES = [st.session_state.selected_candidate]

        heuristic = st.session_state.heuristic
        if heuristic == "greedy":
            data_to_predict_on, distance_matrix_vrp = prepare_input.prep_greedy(
                st.session_state.input_df_modified_with_depot,
                st.session_state.company_1,
                CHOSEN_CANDIDATES,
                st.session_state.distance,
                st.session_state.distance_matrix_ranking,
                st.session_state.vehicle_capacity,
                ranking
            )
        elif heuristic == "bounding_circle":
            data_to_predict_on, distance_matrix_vrp = prepare_input.prep_bounding_circle(
                st.session_state.input_df,
                st.session_state.input_df_modified_with_depot,
                st.session_state.distance_matrix_ranking,
                st.session_state.vehicle_capacity,
                st.session_state.company_1,
                CHOSEN_CANDIDATES,
                st.session_state.distance,
                ranking
            )
        elif heuristic == "k_means":
            distance_calc = RoadDistanceCalculator()
            data_to_predict_on, distance_matrix_vrp = prepare_input.prep_k_means(
                st.session_state.input_df_clustering,
                st.session_state.input_df_modified,
                st.session_state.distance_matrix_ranking,
                st.session_state.full_matrix,
                st.session_state.vehicle_capacity,
                st.session_state.company_1,
                CHOSEN_CANDIDATES,
                st.session_state.distance,
                ranking
            )

            distance_matrix_vrp = distance_calc.calculate_distance_matrix(st.session_state.input_df_modified_with_depot,
                                                                          chosen_company=st.session_state.company_1,
                                                                          candidate_name=CHOSEN_CANDIDATES[0],
                                                                          method=st.session_state.distance,
                                                                          computed_distances_df=st.session_state.distance_matrix_ranking)

        elif heuristic == "dbscan":
            data_to_predict_on, distance_matrix_vrp = prepare_input.prep_dbscan(
                st.session_state.input_df_clustering,
                st.session_state.input_df_modified_with_depot,
                st.session_state.distance_matrix_ranking,
                st.session_state.full_matrix,
                st.session_state.vehicle_capacity,
                st.session_state.company_1,
                CHOSEN_CANDIDATES,
                st.session_state.distance,
                ranking
            )

        # Drop rows where the index contains any string in CHOSEN_CANDIDATES
        distance_matrix_vrp = distance_matrix_vrp[
            ~distance_matrix_vrp.index.str.contains('|'.join(CHOSEN_CANDIDATES), case=False, na=False)]

        # Drop columns where the column name contains any string in CHOSEN_CANDIDATES
        distance_matrix_vrp = distance_matrix_vrp.loc[:,
                              ~distance_matrix_vrp.columns.str.contains('|'.join(CHOSEN_CANDIDATES), case=False,
                                                                        na=False)]

        model_single, current_names_single = vrp_solver.build_model(
            input_df=st.session_state.input_df_modified_with_depot,
            chosen_company=st.session_state.company_1,
            distance_matrix=distance_matrix_vrp,
            truck_capacity=st.session_state.vehicle_capacity,
        )

        solution_single, routes_single = vrp_solver.solve(
            model=model_single,
            max_runtime=1,
            display=False,
            current_names=current_names_single,
            print_routes=True
        )
        total_distance_single, avg_distance_per_order_single = vrp_solver.calculate_distance_per_order(
            routes=routes_single,
            distance_matrix=distance_matrix_vrp
        )

        prediction_feats = data_to_predict_on.drop(columns=["chosen_candidate"]).columns
        path = f"Machine_Learning_Train/{st.session_state.distance}/TrainedModels/XGBoost/"
        model_file = f"{path}xgboost_booster_{st.session_state.distance}_{st.session_state.heuristic}.model"
        scaler_file = f"{path}scaler_{st.session_state.distance}_{st.session_state.heuristic}.pkl"
        predictor = ModelPredictor(model_file, scaler_file)

        if st.session_state.ml_choice:
            return temp_ranker.ranker_ml(data_to_predict_on[prediction_feats], model_file, scaler_file,
                                         CHOSEN_CANDIDATES)
        else:
            if heuristic == "k_means" or heuristic == "dbscan":  # Because of different scaling clusterin algos
                return max(1.0, np.ceil(
                    predictor.predict(data_to_predict_on[prediction_feats])[0] - avg_distance_per_order_single))
            else:
                return max(1.0, np.ceil(
                    avg_distance_per_order_single - predictor.predict(data_to_predict_on[prediction_feats])[0]))

    # Check if ranking needs to be executed
    if st.session_state.execute_Ranking and st.session_state.input_df is not None:
        st.session_state.firsttime1 = False

        # Create different matrices, used for ranking:
        st.session_state.distance_matrix_ranking = distance_calc.calculate_distance_matrix(st.session_state.input_df_modified,
                                                                          chosen_company=st.session_state.company_1,
                                                                          method=st.session_state.distance)
        st.session_state.full_matrix = distance_calc.calculate_square_matrix(st.session_state.input_df_modified)
        st.session_state.input_df_modified_with_depot = distance_calc.add_depot(st.session_state.input_df_modified,
                                                                                LAT_DEPOT, LONG_DEPOT)
        if st.session_state.distance == "osrm":
            check_road_proximity = True
        else:
            check_road_proximity = False
        transformer = TransformInput(check_road_proximity=check_road_proximity)
        st.session_state.input_df_clustering = transformer.drop_duplicates(st.session_state.input_df)


        # Calculate ranking......
        if st.session_state.heuristic == "greedy":
            greedy_ranking = ranking.greedy(st.session_state.distance_matrix_ranking,
                                                      st.session_state.company_1)
            if st.session_state.ml_choice:
                st.session_state.ranking = _ranking_using_machine_learning(greedy_ranking)
            else:
                st.session_state.ranking = greedy_ranking

        elif st.session_state.heuristic == "bounding_circle":
            bounding_circle_ranking = ranking.bounding_circle(st.session_state.input_df,
                                                               st.session_state.distance_matrix_ranking,
                                                               st.session_state.company_1)
            if st.session_state.ml_choice:
                st.session_state.ranking = _ranking_using_machine_learning(bounding_circle_ranking)
            else:
                st.session_state.ranking = bounding_circle_ranking

        elif st.session_state.heuristic == "k_means":
            k_means_ranking = ranking.k_means(st.session_state.input_df_clustering,
                                                         st.session_state.input_df_modified,
                                                         st.session_state.full_matrix,
                                                         st.session_state.company_1)
            if st.session_state.ml_choice:
                st.session_state.ranking = _ranking_using_machine_learning(k_means_ranking)
            else:
                st.session_state.ranking = k_means_ranking

        elif st.session_state.heuristic == "dbscan":
            db_scan_ranking = ranking.dbscan(st.session_state.input_df_clustering,
                                                      st.session_state.full_matrix,
                                                      st.session_state.company_1)
            if st.session_state.ml_choice:
                st.session_state.ranking = _ranking_using_machine_learning(db_scan_ranking)
            else:
                st.session_state.ranking = db_scan_ranking

        elif st.session_state.heuristic == "machine_learning":
            st.session_state.ml_choice = True
            greedy_ranking = ranking.greedy(st.session_state.distance_matrix_ranking, st.session_state.company_1)
            bounding_ranking = ranking.bounding_circle(st.session_state.input_df, st.session_state.distance_matrix_ranking, st.session_state.company_1)
            k_means_ranking = ranking.k_means(st.session_state.input_df_clustering, st.session_state.input_df_modified, st.session_state.full_matrix, st.session_state.company_1)
            dbscan_ranking = ranking.dbscan(st.session_state.input_df_clustering, st.session_state.full_matrix, st.session_state.company_1)

            CHOSEN_CANDIDATES = \
                st.session_state.input_df[st.session_state.input_df["name"] != st.session_state.company_1][
                    "name"].unique()

            data_to_predict_on, distance_matrix_vrp = prepare_input.prep_all(st.session_state.input_df_clustering, st.session_state.input_df_modified,
                                                                             st.session_state.input_df_modified_with_depot,
                                                                             st.session_state.distance_matrix_ranking, st.session_state.full_matrix,
                                                                             st.session_state.vehicle_capacity, st.session_state.company_1,
                                                                             CHOSEN_CANDIDATES, st.session_state.distance, greedy_ranking,
                                                                             bounding_ranking,
                                                                             k_means_ranking,dbscan_ranking)
            prediction_feats = data_to_predict_on.columns
            path = f"Machine_Learning_Train/{st.session_state.distance}/TrainedModels/XGBoost/"
            model_file = f"{path}xgboost_booster_{st.session_state.distance}_all.model"
            scaler_file = f"{path}scaler_{st.session_state.distance}_all.pkl"
            predictor = ModelPredictor(model_file, scaler_file)


            st.session_state.ranking = ranking.ranker_ml(data_to_predict_on[prediction_feats], model_file, scaler_file,st.session_state.input_df[st.session_state.input_df["name"] != st.session_state.company_1]["name"].unique())

        st.session_state.execute_Ranking = False
        st.session_state.show_Ranking = True

    if st.session_state.show_Ranking and st.session_state.input_df is not None:
        st.session_state.execute_Ranking = False
        st.session_state.show_Ranking = True
        dashboard.display_ranking()

        csv_file, file_name = dashboard.download(type="ranking")
        st.sidebar.download_button(
            label='Download Ranking',
            data=csv_file,
            file_name=file_name,
            mime="text/csv"
        )

    # Check if VRP execution is triggered
    if st.session_state.execute_VRP and st.session_state.selected_candidate is not None and st.session_state.input_df is not None:
        st.session_state.execute_VRP = False
        st.session_state.show_VRP = True
        st.session_state.distance_matrix_vrp = distance_calc.calculate_distance_matrix(st.session_state.input_df_modified_with_depot,
                                                                      chosen_company=st.session_state.company_1,
                                                                      candidate_name=st.session_state.selected_candidate,
                                                                      method=st.session_state.distance,
                                                                      computed_distances_df=st.session_state.distance_matrix_ranking)
        st.session_state.vrp_solver = VRPSolver()

        print("Solving VRP for Collaboration...")
        model_collab, current_names_collab = st.session_state.vrp_solver.build_model(
            input_df=st.session_state.input_df_modified_with_depot,
            chosen_company=st.session_state.company_1,
            chosen_candidate=st.session_state.selected_candidate,
            distance_matrix=st.session_state.distance_matrix_vrp,
            truck_capacity=st.session_state.vehicle_capacity
        )
        st.session_state.solution_collab, st.session_state.routes_collab = st.session_state.vrp_solver.solve(
            model=model_collab,
            max_runtime=1,
            display=False,
            current_names=current_names_collab,
            print_routes=True
        )
        total_distance_collab, avg_distance_per_order_collab = st.session_state.vrp_solver.calculate_distance_per_order(
            routes=st.session_state.routes_collab,
            distance_matrix=st.session_state.distance_matrix_vrp
        )

        st.session_state.execute_VRP = False
        st.session_state.show_VRP = True


    if st.session_state.show_VRP and st.session_state.input_df is not None:
        if st.session_state.update2 and st.session_state.firsttime2 == False:
            st.write('<div style="text-align: left; color: red; font-weight: bold; font-style: italic;">'
                     'Not up-to-date! <br>'
                     'Recalculate VRP. <br>'
                     '<br>'
                     '</div>',
                     unsafe_allow_html=True
                     )
        st.write("### VRP Solution")

        st.session_state.solution_print = pd.DataFrame(st.session_state.routes_collab, index=[f"Route_{i}" for i in range(len(st.session_state.routes_collab))])

        col1, col2, col3 = st.columns([15, 1, 2.5])
        with col1:
            st.write(st.session_state.solution_print)
        with col3:
            csv_file, file_name = dashboard.download(type="vrp")
            st.download_button(
                label='Download VRP',
                data=csv_file,
                file_name=file_name,
                mime="text/csv"
            )

        #start_time_test = time.time()
        dashboard.showmap(st.session_state.routes_collab, st.session_state.input_df_modified_with_depot)
        #print("\nTESTING UNITS", time.time() - start_time_test, "seconds\n")

    if st.session_state.input_df is None:
        dashboard.clear_all()

    print("Overall re-initialization time took:", round(time.time() - start_time_overall,4), "seconds")