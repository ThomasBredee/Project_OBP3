#########################################################
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 28/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################
import numpy as np
import joblib
import xgboost as xgb

class ModelPredictor:
    def __init__(self, model_file, scaler_file):
        # Load the XGBoost model directly if saved as a .model file
        self.model = xgb.Booster()
        self.model.load_model(model_file)  # Assuming the model was saved with XGBoost's save_model
        self.scaler = joblib.load(scaler_file)

    def predict(self, input_data):
        """
        Make predictions using the loaded model and scaler.
        :param input_data: A numpy array or a list of input features.
        :return: Predicted values.
        """
        # Check if input_data is a list and convert it to numpy array if necessary
        if isinstance(input_data, list):
            input_data = np.array(input_data)

        # Ensure input_data is 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        # Scale the features using the loaded scaler
        scaled_data = self.scaler.transform(input_data)

        # Convert scaled data to DMatrix before prediction
        dmatrix_data = xgb.DMatrix(scaled_data)

        # Make predictions using the scaled data
        predictions = self.model.predict(dmatrix_data)

        return predictions