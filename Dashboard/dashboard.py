#########################################################
#                                                       #
# Created on: 14/01/2025                                #
# Created by: Thomas                                    #
#                                                       #
# Created on: 31/01/2025                                #
# Created by: Thomas                                    #
#                                                       #
#########################################################

import streamlit as st
import pandas as pd
import warnings
import folium
from streamlit_folium import st_folium
from st_aggrid import AgGrid, GridOptionsBuilder
from Dashboard.gridbuilder import PINLEFT, PRECISION_THREE, draw_grid
from Input_Transformation.transforming_input import TransformInput
import pandas as pd
from Candidate_Ranking.ranking_methods import CandidateRanking
from Machine_Learning_Predict.predictor import ModelPredictor
from Machine_Learning_Predict.prepare_input import PrepareInput
import joblib
import random
import time
import numpy as np

from VRP_Solver.solver_pyvrp import VRPSolver
from VRP_Solver.distance_calculator import RoadDistanceCalculator

warnings.filterwarnings('ignore')


class Dashboard:
    def __init__(self):
        #print("Initializing Dashboard")
        st.set_page_config(page_title='Cross-Collaboration Dashboard', page_icon=":bar_chart", layout='wide')
        st.title(" :bar_chart: Collaboration Dashboard")
        st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

        defaults = {
        #True / False's
            "execute_Ranking": False,
            "show_Ranking": False,
            "execute_VRP": False,
            "show_VRP": False,
            "update1": False,
            "firsttime1": True,
            "update2": False,
            "firsttime2": True,

        #Dashboard input saves
            "selected_candidate": None,
            "vehicle_capacity": None,
            "company_1": None,
            "heuristic": None,
            "ml_choice": True,
            "distance": None,

        #Output saves:
            "greedy_ranking": None,
            "circle_box_ranking": None,
            "k_means_ranking": None,

            "df_display": None,
            "vrp_solver": None,
            "model": None,
            "current_names": None,
            "solution_collab": None,
            "routes_collab": None,
            "expected_gain": None,
            "solution_print": None,

        #Different versions of matrices
            "input_df": None,
            "input_df_kmeans": None,
            "input_df_modified": None,
            "distance_matrix_ranking": None,
            "full_matrix": None,
            "input_df_modified_with_depot": None,
            "distance_matrix_vrp": None,
        # Used for calculations:
            "row_number": 0,
            "data1": None,
            "data2": None
        }

        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

        # File uploader
        fl = st.file_uploader(":file_folder: Upload a file", type=(["csv"]))

        #start_time = time.time()
        if fl is not None:
            self._process_file(fl)
        else:
            st.session_state.input_df = None

        self.prepare_input = PrepareInput()
        #print("Processing file took {} seconds".format(time.time() - start_time))

    def _process_file(self, fl):
        if fl.name.endswith('.csv'):
            st.session_state.input_df = pd.read_csv(fl, encoding="ISO-8859-1")
            #print(st.session_state.input_df)
        elif fl.name.endswith(('.xlsx', '.xls')):
            st.session_state.input_df = pd.read_excel(fl, engine='openpyxl')
        else:
            st.error("Unsupported file type!")
            st.stop()

        check_road_proximity = True  # Set true if OSRM container running
        transformer = TransformInput(check_road_proximity=check_road_proximity)
        st.session_state.input_df_kmeans = transformer.drop_duplicates(st.session_state.input_df)
        st.session_state.input_df_modified = transformer.execute_validations(st.session_state.input_df)

        st.write(f"Uploaded file: {fl.name}")

        st.sidebar.header("Choose your filter: ")
        filters = list(st.session_state.input_df["name"].unique())
        company = st.sidebar.selectbox("Pick your Company:", filters, key="company_select")
        if company != st.session_state.company_1:
            st.session_state.company_1 = company
            self.reset_right_panel()  # Reset right panel when company changes

        # Use a number input for vehicle capacity with no maximum value
        VC = st.sidebar.number_input(
            "Pick your Capacity:",
            min_value=2,
                value=2,
            max_value=20,
            step=1,
            #key='vehicle_capacity'
        )
        if VC != st.session_state.vehicle_capacity:
            st.session_state.vehicle_capacity = VC
            st.session_state.update1 = True
            st.session_state.update2 = True

        if VC != st.session_state.vehicle_capacity:
            st.session_state.vehicle_capacity = VC
            self.reset_right_panel()  # Reset right panel when capacity changes

        heuristics = list(["greedy", "bounding_circle", "k_means", "dbscan", "machine_learning"])
        heuristic = st.sidebar.selectbox("Pick your Heuristic:", heuristics)#, key='heuristics_choice')
        if heuristic != st.session_state.heuristic:
            st.session_state.heuristic = heuristic
            st.session_state.update1 = True
            st.session_state.update2 = True
        if heuristic != st.session_state.heuristic:
            st.session_state.heuristic = heuristic
            self.reset_right_panel()  # Reset right panel when heuristic changes

        if st.session_state.heuristic == "dbscan":
            st.sidebar.write('<div style="text-align: center; color: red; font-style: italic;">'
                             'Make sure to use a big input file, (>= many.csv) <br>'
                             '<br>'
                             '</div>',
                             unsafe_allow_html=True
                             )

        distance_choices = list(["haversine", "osrm"])
        distance = st.sidebar.selectbox("Pick your Distance:", distance_choices)#, key='distance_choice')
        if distance != st.session_state.distance:
            st.session_state.distance = distance
            st.session_state.update1 = True
            st.session_state.update2 = True
            self.reset_right_panel()

        if st.session_state.heuristic != "machine_learning":
            ml = st.sidebar.selectbox(
                "Do you want to employ Machine Learning?",
                ("False", "True")
            )
            ml_save = ml == "True"
            if ml_save != st.session_state.ml_choice:
                st.session_state.ml_choice = ml == "True"
                st.session_state.update1 = True
                st.session_state.update2 = True

            if st.session_state.ml_choice:
                st.sidebar.write('<div style="text-align: center; color: red; font-style: italic;">'
                                'Takes extra time <br>'
                                '<br>'
                                '</div>',
                                unsafe_allow_html=True
                             )
        else:
            st.session_state.ml_choice = True

        if st.sidebar.button("Get Ranking"):
            st.session_state.update1 = False
            st.session_state.firsttime1 = True
            st.session_state.execute_Ranking = True
            st.session_state.show_Ranking = True
            st.session_state.display_VRP = False

    def _ranking_using_machine_learning(self, ranking):
        temp_ranker = CandidateRanking()
        vrp_solver = VRPSolver()
        #distance_calc = RoadDistanceCalculator()
        if st.session_state.ml_choice:
            CHOSEN_CANDIDATES = \
            st.session_state.input_df[st.session_state.input_df["name"] != st.session_state.company_1]["name"].unique()
        else:
            CHOSEN_CANDIDATES = [st.session_state.selected_candidate]

        heuristic = st.session_state.heuristic
        if heuristic == "greedy":
            data_to_predict_on, distance_matrix_vrp = self.prepare_input.prep_greedy(
                st.session_state.input_df_modified_with_depot,
                st.session_state.company_1,
                CHOSEN_CANDIDATES,
                st.session_state.distance,
                st.session_state.distance_matrix_ranking,
                st.session_state.vehicle_capacity,
                ranking
            )
        elif heuristic == "bounding_circle":

            ranking = ranking.rename(columns={'Total Distance': 'Percentage Overlap'})
            data_to_predict_on, distance_matrix_vrp = self.prepare_input.prep_bounding_circle(
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
            ranking = ranking.rename(columns={'Total Distance': 'Percentage Overlap'})
            data_to_predict_on, distance_matrix_vrp = self.prepare_input.prep_k_means(
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
                                                                          method=st.session_state.distance ,
                                                                          computed_distances_df=st.session_state.distance_matrix_ranking)

        elif heuristic == "dbscan":
            ranking = ranking.rename(columns={'Total Distance': 'Percentage'})
            data_to_predict_on, distance_matrix_vrp = self.prepare_input.prep_dbscan(
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


        if heuristic != "machine_learning":

            distance_matrix_vrp = distance_matrix_vrp[~distance_matrix_vrp.index.str.contains('|'.join(CHOSEN_CANDIDATES), case=False, na=False)]

            # Drop columns where the column name contains any string in CHOSEN_CANDIDATES
            distance_matrix_vrp = distance_matrix_vrp.loc[:, ~distance_matrix_vrp.columns.str.contains('|'.join(CHOSEN_CANDIDATES), case=False, na=False)]

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
            return "Machine Leanring"
        else:
            if heuristic == "k_means" or heuristic == "dbscan": #Because of different scaling clusterin algos
                return max(1.0, np.ceil(avg_distance_per_order_single -  predictor.predict(data_to_predict_on[prediction_feats])[0]))
            else:
                return max(1.0, np.ceil(avg_distance_per_order_single - predictor.predict(data_to_predict_on[prediction_feats])[0]))

    def display_ranking(self):
        st.write("### Potential Candidates")

        col1, col2, col3, col4 = st.columns([1.8, 0.2, 1.5, 1])
        with col1:
            if st.session_state.ml_choice:
                formatter = {
                    'Company': ('Company', {'pinned': 'left', 'width': 150}),
                    'Total Distance': ('Total Distance (Meters)',
                                       {'type': ['numericColumn', 'customNumericFormat'], 'precision': 3,
                                        'width': 200}),
                }
            else:
                formatter = {'Company': ('Company', {'pinned': 'left', 'width': 150})}

            st.session_state.df_display = st.session_state.ranking.reset_index()
            st.session_state.df_display.rename(columns={'index': 'Company'}, inplace=True)

            st.session_state.row_number = st.number_input('Number of rows', min_value=0, value=3)
            st.session_state.data1 = draw_grid(
                st.session_state.df_display.head(st.session_state.row_number),
                formatter=formatter,
                fit_columns=True,
                selection='single',
                use_checkbox=True,
                max_height=300
            )

            # Debugging - Print Selected Rows
            selected_rows = st.session_state.data1.get("selected_rows", pd.DataFrame())

            # Ensure selected_rows is a valid DataFrame and not empty
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                new_candidate = selected_rows.iloc[0]['Company']

                # Only update if the selected candidate has changed
                if new_candidate != st.session_state.selected_candidate:
                    st.session_state.selected_candidate = new_candidate
                    st.session_state.update2 = True

        previous_candidate = st.session_state.get("previous_candidate", None)

        if (
                "selected_candidate" in st.session_state
                and st.session_state.selected_candidate
                and st.session_state.selected_candidate != previous_candidate
        ):
            st.session_state.expected_gain = self._ranking_using_machine_learning(st.session_state.ranking)
            st.session_state.previous_candidate = st.session_state.selected_candidate

        # Render expected gain if available
        with col3:
            if not st.session_state.ml_choice:  # Ensure ML is False before showing expected gain
                if "expected_gain" in st.session_state and st.session_state.selected_candidate:
                    st.write(f"The Expected gain for {st.session_state.selected_candidate} is:",
                             st.session_state.expected_gain, "kilometers per order")

        # Solve VRP button
        with col4:
            if st.session_state.selected_candidate:
                if col4.button("Solve VRP"):
                    st.session_state.execute_VRP = True
                    st.session_state.update2 = False
                    st.session_state.firsttime2 = False

    def download(self, type="ranking"):
        if type == "ranking":
            csv_file = st.session_state.ranking.to_csv(index=True)
            file_name = f"Ranking_Export_{st.session_state.company_1}.csv"
        elif type == "vrp":

            #print(st.session_state.solution)
            csv_file = st.session_state.solution_print.to_csv(index=True)
            file_name = f"VRP_Export_{st.session_state.company_1}.csv"
        return csv_file, file_name

    def showmap(self, route_input, df_input):
        # Extract base company names (before the last underscore)
        def get_base_name(name):
            if "_" in name:
                return "_".join(name.split("_")[:-1])
            return name

        df_input['base_name'] = df_input['name'].apply(get_base_name)

        # Map each base company name to a unique color
        unique_companies = df_input['base_name'].unique()
        company_colors = ["blue", "green", "red"]
        company_color_map = {company: company_colors[i % len(company_colors)] for i, company in
                             enumerate(unique_companies)}

        # Assign unique colors for each truck's route
        truck_colors = [
            "blue", "orange", "red", "purple", "cyan",
            "magenta", "lime", "brown", "pink",
            "gold", "silver", "darkcyan", "indigo", "green",
            "black", "white", "lightblue", "darkgreen", "gray"
        ]
        route_colors = [truck_colors[i % len(truck_colors)] for i in range(len(route_input))]

        # Create a folium map centered on the depot of the first route
        depot_coords = None
        for truck_route in route_input:
            first_location = truck_route[0]
            depot_row = df_input[df_input['name'] == first_location]
            if not depot_row.empty:
                depot_coords = (depot_row.iloc[0]['lat'], depot_row.iloc[0]['lon'])
                break

        if not depot_coords:
            st.error("Depot not found in the input DataFrame.")
            return

        route_map = folium.Map(location=depot_coords, zoom_start=8)

        # Process each truck's route
        for idx, truck_route in enumerate(route_input):
            route_coordinates = []
            for location in truck_route:
                location_row = df_input[df_input['name'] == location]
                if location_row.empty:
                    st.warning(f"Location '{location}' not found in the input DataFrame.")
                    continue

                lat, lon = location_row.iloc[0]['lat'], location_row.iloc[0]['lon']
                route_coordinates.append((lat, lon))

                # Add markers for each location in the truck's route
                base_name = get_base_name(location)
                color = company_color_map.get(base_name, "gray")
                folium.Marker(
                    location=(lat, lon),
                    popup=f"{location} ({base_name})",
                    icon=folium.Icon(color=color if location != "Depot" else "black"),
                ).add_to(route_map)

            # Draw a polyline for the truck's route if it has coordinates
            if route_coordinates:
                folium.PolyLine(
                    locations=route_coordinates,
                    color=route_colors[idx],
                    weight=2.5,
                    opacity=0.8,
                    tooltip=f"Truck {idx + 1}"
                ).add_to(route_map)
            #else:
             #   st.warning(f"Route #{idx + 1} has no valid locations and will not be displayed.")

        # Display the map in Streamlit
        st_folium(route_map, width=800, height=600)

    def clear_all(self):
        defaults = {
            # True / False's
            "execute_Ranking": False,
            "show_Ranking": False,
            "execute_VRP": False,
            "show_VRP": False,
            "update1": False,
            "firsttime1": True,
            "update2": False,
            "firsttime2": True,

            # Dashboard input saves
            "selected_candidate": None,
            "vehicle_capacity": None,
            "company_1": None,
            "heuristic": None,
            "distance": None,

            # Output saves:
            "greedy_ranking": None,
            "circle_box_ranking": None,
            "k_means_ranking": None,

            "df_display": None,
            "vrp_solver": None,
            "model": None,
            "current_names": None,
            "solution_collab": None,
            "routes_collab": None,
            "expected_gain": None,
            "solution_print": None,

            # Different versions of matrices
            "input_df": None,
            "input_df_kmeans": None,
            "input_df_modified": None,
            "distance_matrix_ranking": None,
            "full_matrix": None,
            "input_df_modified_with_depot": None,
            "distance_matrix_vrp": None,
            # Used for calculations:
            "row_number": 0,
            "data1": None,
            "data2": None
        }
        for key, default in defaults.items():
            st.session_state[key] = default

    def reset_right_panel(self):
        """Reset the right panel when the left column changes."""
        st.session_state.show_Ranking = False
        st.session_state.show_VRP = False
        st.session_state.selected_candidate = None
        st.session_state.expected_gain = None
        st.session_state.ranking = None
        st.session_state.solution_collab = None
        st.session_state.routes_collab = None


