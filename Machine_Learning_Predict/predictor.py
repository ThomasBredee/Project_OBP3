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

class ModelPredictor:
    def __init__(self, model_path, scaler_path):
        """
        Initialize the predictor class by loading the model and scaler from specified paths.
        :param model_path: Path to the saved model file (.pkl)
        :param scaler_path: Path to the saved scaler file (.pkl)
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

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

        # Make predictions using the scaled data
        predictions = self.model.predict(scaled_data)

        return predictions