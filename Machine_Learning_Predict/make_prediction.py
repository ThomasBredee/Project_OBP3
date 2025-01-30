#########################################################
#                                                       #
# Created on: 22/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd


class ModelPredictor:
    def __init__(self, model, scaler):
        """
        Initialize the predictor with a trained model and scaler.

        Args:
            model: Trained machine learning model.
            scaler: Pre-fitted scaler for data normalization.
        """
        self.model = model
        self.scaler = scaler

    def predict_for_candidates(self, prediction_data):
        """
        Predict values for each chosen candidate in the input DataFrame.

        Args:
            input_df (pd.DataFrame): DataFrame with input data for prediction.
                                    Must contain 'chosen_candidate' and feature columns.
            output_file (str): Filename to save the predictions.

        Returns:
            pd.DataFrame: DataFrame containing predictions indexed by chosen candidate.
        """
        # Ensure required columns are present
        feature_columns = ['totalsum', 'number_stops', 'Max_to_depot', 'Min_to_depot', 'vehicle_cap',
                           'greedy_total_sum']

        # Extract features and scale them
        features = prediction_data[feature_columns]

        scaled_features = self.scaler.transform(features)

        # Make predictions
        predictions = self.model.predict(scaled_features)

        # Create a DataFrame with chosen_candidate as index and predictions as the first column
        prediction_df = pd.DataFrame(predictions, index=prediction_data['chosen_candidate'], columns=['Predicted km per order'])

        df_sorted = prediction_df.sort_values(by='Predicted km per order')

        return df_sorted
