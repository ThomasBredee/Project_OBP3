#########################################################
#                                                       #
# Created on: 23/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# updates on: 30/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

#Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from itertools import permutations
import joblib
from Machine_Learning_Train.get_features_training_data import DataFramePreparer
from Input_Transformation.transforming_input import TransformInput
from VRP_Solver.distance_calculator import RoadDistanceCalculator

#Global Variables for this file
METHOD = "haversine"
RANKING = "all"
MODEL_TYPE = "xgboost"
MAX_TRUCK_CAP = 20
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
MIN_SAMPLES = 2

class ModelTester:
    def __init__(self, model_file, scaler_file):
        #Load the XGBoost model and the scaler trained on the train fraction of the generated data
        self.model = xgb.Booster()
        self.model.load_model(model_file)
        self.scaler = joblib.load(scaler_file)

    def evaluate(self, df, input_features):
        """
        Evaluates the performance of the model on a provided dataset using specified input features

        :param df: DataFrame containing the dataset to be evaluated, including both features and the target variable
        :param input_features: List of column names to be used as input features for the model

        :return: Prints the performance metrics including Mean Squared Error, R-squared, and Average Divergence
        """

        #Prep data
        X_test = df[input_features]
        y_test = df['real_km_order']
        X_test_scaled = self.scaler.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)

        #Make predictions
        predictions = self.model.predict(dtest)

        #Get info on metrics about how model is performing (OUT-SAMPLE)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        divergence = np.abs(y_test - predictions).mean()

        print("Evaluation Results:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²): {r2:.2%}")
        print(f"Average Misprediction (Divergence): {divergence:.2f} km/order")

    def generate_test_data(self, input_df, input_df_modified, max_truck_cap, method, company_pairs):
        """
        Generates the test data based to make predictions on for different pairs of chosen candidates and chosen companies

        :param input_df: Df containing the original input data
        :param input_df_modified: Modified version of the input DataFrame that includes underscores for each company location
        :param max_truck_cap: Maximum truck capacity used for certain calculations and constraints
        :param method: The method used for distance calculation (haversine or osrm)
        :param company_pairs: A list of tuples, where each tuple is a pair of companies

        :return: Df containing rows of generated features for each company pair, designed to test distance and ranking methods
        """

        rows = []

        #Initialize distance calculator and data preparer objects
        distance_calc = RoadDistanceCalculator()
        preparer = DataFramePreparer()

        #Prep data
        input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)

        #Calculate the full distance matrix and adapt it based on the specified ranking method
        if RANKING == "greedy" or  RANKING == "bounding_circle":
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,
                                                                                method=method)
            full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)

        elif RANKING == "k_means":
            df_input_clustering = transformer.drop_duplicates(input_df)
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,
                                                                                method=method)
            squared_distance_df_kmeans = distance_calc.calculate_square_matrix(input_df_modified)
        elif RANKING == "dbscan":
            df_input_clustering = transformer.drop_duplicates(input_df)
            EPS = int(round(19 * len(df_input_clustering) ** -0.1 + 5, 0))
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,
                                                                                method=method)
            MIN_SAMPLES = 2
            squared_distance_df_dbscan = distance_calc.calculate_square_matrix(input_df_modified)

        elif RANKING == "all":
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,
                                                                                method=METHOD)
            full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)
            df_input_clustering = transformer.drop_duplicates(input_df)
            squared_distance_df_kmeans = distance_calc.calculate_square_matrix(input_df_modified_with_depot)
            squared_distance_df_dbscan = distance_calc.calculate_square_matrix(input_df_modified)
            MIN_SAMPLES = 2
            EPS = int(round(19 * (len(df_input_clustering) ** -0.1) + 5, 0))


        #Process each company pair to create test data
        for pair in company_pairs:
            print(f"Processing company pair: {pair}")
            chosen_company, chosen_candidate = pair

            #Slice distance matrix based on company pair
            row_filter_vrp = full_distance_matrix.index.to_series().apply(
                lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
            column_filter = full_distance_matrix.columns.to_series().apply(
                lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
            distance_matrix_vrp = full_distance_matrix.loc[row_filter_vrp, column_filter]

            #Create feature row based on the ranking method
            if RANKING == "greedy":
                row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
                distance_matrix_ranking = full_distance_matrix.loc[
                    row_filter_ranking, full_distance_matrix_ranking.columns]
                row = preparer.get_features_greedy(input_df_modified, distance_matrix_ranking, distance_matrix_vrp,
                                                   max_truck_cap, chosen_company, chosen_candidate)
            elif RANKING == "bounding_circle":
                row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
                distance_matrix_ranking = full_distance_matrix.loc[
                    row_filter_ranking, full_distance_matrix_ranking.columns]
                row = preparer.get_features_bounding_circle(input_df, input_df_modified, distance_matrix_ranking,
                                                            distance_matrix_vrp, max_truck_cap, chosen_company,
                                                            chosen_candidate)
            elif RANKING == "k_means":
                row = preparer.get_features_k_means(df_input_clustering, input_df_modified, squared_distance_df_kmeans,
                                                    distance_matrix_vrp, max_truck_cap, chosen_company,
                                                    chosen_candidate)
            elif RANKING == "dbscan":
                row = preparer.get_features_dbscan(df_input_clustering, input_df_modified, squared_distance_df_dbscan,
                                                   distance_matrix_vrp, max_truck_cap, chosen_company, chosen_candidate,
                                                   EPS, MIN_SAMPLES)
            elif RANKING == "all":
                row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
                distance_matrix_ranking = full_distance_matrix.loc[row_filter_ranking, full_distance_matrix_ranking.columns]
                row = preparer.get_all_features(input_df, df_input_clustering, input_df_modified, distance_matrix_ranking, squared_distance_df_kmeans,
                         squared_distance_df_dbscan, distance_matrix_vrp, max_truck_cap, chosen_company, chosen_candidate, EPS, MIN_SAMPLES)

            rows.append(row)

        results_df = pd.DataFrame(rows)

        return results_df


if __name__ == "__main__":

    #Get model
    path = f"Machine_Learning_Train/{METHOD}/TrainedModels/XGBoost/"
    model_file = f"{path}{MODEL_TYPE}_booster_{METHOD}_{RANKING}.model"
    scaler_file = f"{path}scaler_{METHOD}_{RANKING}.pkl"
    test_model = ModelTester(model_file, scaler_file)

    #Get test df without features and validate input and make transformations and pairs
    test_df = pd.read_csv(f"Machine_Learning_Train/{METHOD}/TestData/test_split_{RANKING}_{METHOD}.csv", index_col=0)
    transformer = TransformInput(check_road_proximity=True)
    input_df_modified = transformer.execute_validations(test_df)
    company_pairs = [list(pair) for pair in permutations(test_df['name'].unique(), 2)]

    #Create Test data
    test_data = test_model.generate_test_data(test_df, input_df_modified, MAX_TRUCK_CAP, METHOD, company_pairs)
    feature_cols = [col for col in test_data.columns if col not in ["real_km_order", "chosen_company", "chosen_candidate"]]

    #Evaluate trained model
    test_model.evaluate(test_data, feature_cols)
