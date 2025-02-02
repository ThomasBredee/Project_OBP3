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
import joblib
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

#Global Variables for this file
METHOD = "haversine"
RANKING = "all"
MODEL_TYPE = "xgboost"
TOTAL_TRAILS = 150

class ModelTrainer:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.scaler = StandardScaler()

    def train(self, df, train_columns):
        """
        Train the model using the sampled data and a list of trained features and save the model and scalar

        :param df: DataFrame containing the training data
        :param train_columns: List of columns to use as training features

        :return: Trained model booster (to predict on) and the scaler object
        """

        #Prep data
        X = df[train_columns]
        y = df['real_km_order']
        X_scaled = self.scaler.fit_transform(X)  # Scale features for model input

        #Optimize model using optuna
        if self.model_type == "xgboost":
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective(trial, X_scaled, y), n_trials=TOTAL_TRAILS)
            best_params = study.best_params
            best_params['n_estimators'] = best_params.pop('num_boost_round') #Boosting_rounds <-> amount of trees (estimators)
            self.model = xgb.XGBRegressor(**best_params)
            self.model.fit(X_scaled, y)

        #Get info on metrics about how model is performing (IN-SAMPLE), and save model and scalar
        predictions = self.model.predict(X_scaled)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        divergence_per_order = np.sum(np.abs(predictions - y)) / len(y)

        print(f"Model trained with MSE: {mse:.4f}")
        print(f"Divergence per Order: {divergence_per_order:.4f}")
        print(f"R^2 Score: {r2:.4f}")

        booster = self.model.get_booster()
        booster.save_model(f"{MODEL_TYPE}_booster_{METHOD}_{RANKING}.model")
        joblib.dump(self.scaler, f"scaler_{METHOD}_{RANKING}.pkl", compress=3)

        return booster, self.scaler

    def objective(self, trial, X, y):
        """
        Optuna optimization method using the TPESampler to train on 4-folds and predict on 5th fold each training trial to combat overfitting

        :param trial: Optuna trial object which suggests hyperparameters for each trial
        :param X: Feature data
        :param y: Target variable

        :return: Minimum Root Mean Squared Error from cross-validation results
        """

        #Hyperparameters guess for the trial
        num_boost_round = trial.suggest_int('num_boost_round', 50, 500) #<- amount of trees (n_est)
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True)
        }

        #Do cross-validation using guessed parameters and return the minimum RMSE
        cv_results = xgb.cv(
            params=params,
            dtrain=xgb.DMatrix(X, label=y),
            num_boost_round=num_boost_round,
            nfold=5,
            metrics='rmse',
            early_stopping_rounds=10,
        )
        return cv_results['test-rmse-mean'].min()


if __name__ == "__main__":
    #Get generated training data
    data = pd.read_csv(f"Machine_Learning_Train/{METHOD}/TrainData/generated_training_data_{RANKING}_{METHOD}_10000_rows.csv")
    train_df = data.drop_duplicates(keep='first')

    #Get feature where model will predict on
    features = [col for col in train_df.columns if col not in ["real_km_order", "chosen_company", "chosen_candidate"]]

    #Train model
    trainer = ModelTrainer(MODEL_TYPE)
    booster, scaler = trainer.train(train_df, features)
