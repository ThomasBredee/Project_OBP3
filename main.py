###TEST PUSH###

from Dashboard.dashboard import Dashboard
import streamlit as st
import time
import pandas as pd

from VRP_Solver.distance_calculator import RoadDistanceCalculator
from VRP_Solver.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
from Machine_Learning_Predict.make_prediction import ModelPredictor
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
    prepper = PrepareInput()

    if st.session_state.update1 and st.session_state.firsttime1 == False:
        st.sidebar.write(
            '<div style="text-align: center; color: red; font-weight: bold; font-style: italic;">'
            'Not up-to-date! <br>'
            'Recalculate Ranking and VRP.'
            '</div>',
            unsafe_allow_html=True
        )

    # Check if ranking needs to be executed
    if st.session_state.execute_Ranking and st.session_state.input_df is not None:
        st.session_state.firsttime1 = False

        st.session_state.distance_matrix_ranking = distance_calc.calculate_distance_matrix(st.session_state.input_df_modified,
                                                                          chosen_company=st.session_state.company_1,
                                                                          method=st.session_state.distance)
        st.session_state.full_matrix = distance_calc.calculate_square_matrix(st.session_state.input_df_modified)

        # Generate ranking based on heuristic
        # TODO: later implementeren: greedy_ranking, bounding_circle_ranking, en k_means_ranking, zodra de prepper werkt.
        #  hier al vast een poging gedaan om greedy_ranking te implementeren op basis van st.session_state.ml_choice
        if st.session_state.heuristic == "greedy":
            st.session_state.ranking = ranking.greedy(st.session_state.distance_matrix_ranking,
                                                      st.session_state.company_1)
        elif st.session_state.heuristic == "bounding_circle":
            st.session_state.ranking = ranking.bounding_circle(st.session_state.input_df,
                                                               st.session_state.distance_matrix_ranking,
                                                               st.session_state.company_1)
        elif st.session_state.heuristic == "k_means":
            st.session_state.ranking = ranking.k_means(st.session_state.input_df_kmeans,
                                                         st.session_state.input_df_modified,
                                                         st.session_state.full_matrix,
                                                         st.session_state.company_1)
        #TODO: deze input van ranking.dbscan() nog even checken
        elif st.session_state.heuristic == "dbscan":
            st.session_state.ranking = ranking.dbscan(st.session_state.input_df,
                                                      st.session_state.reduced_distance_df,
                                                      st.session_state.company_1)

        ###get candidate ranking methods (machine learning) / get predicted kilometers per order (For Greedy)
        # preprare data
        st.session_state.input_df_modified_with_depot = distance_calc.add_depot(st.session_state.input_df_modified,
                                                                                LAT_DEPOT, LONG_DEPOT)


        #TODO: DEZE PREPPER VERWERKEN HIERBOVEN IN DE IF STATEMENTS, ZODAT ER EEN PREDICT VARIABLE KLAARSTAAT
        # IN DE SESSION_STATE DIE OPGEROEPEN KAN WORDEN IN DISPLAY RANKING.
        # prediction_df = prepper.prep_greedy(st.session_state.input_df_modified_with_depot,
        #                                     st.session_state.company_1,
        #                                     st.session_state.greedy_ranking.index,  <--- st.session.latest_ranking.index
        #                                     "haversine",
        #                                     st.session_state.distance_matrix_ranking,
        #                                     st.session_state.vehicle_capacity,
        #                                     st.session_state.greedy_ranking)       <--- st.session.lastest_ranking
        # # get pre-trained model
        # path = "Machine_Learning_Train/osrm/TrainedModels/RF/"
        # scaler = joblib.load(f"{path}scaler_greedy_osrm.pkl")
        # model = joblib.load(f"{path}random_forest_model_greedy_osrm.pkl")
        # predictor = ModelPredictor(model, scaler)
        # #make prediction and get ranking
        # predicted_df = predictor.predict_for_candidates(prediction_df)

        #print(st.session_state.ranking)
        st.session_state.execute_Ranking = False
        st.session_state.show_Ranking = True
        # print("Ranking took:",  round(time.time() - start_time,4), "seconds")

    if st.session_state.show_Ranking and st.session_state.input_df is not None:
        # Display the ranking
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
            max_runtime=2,
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