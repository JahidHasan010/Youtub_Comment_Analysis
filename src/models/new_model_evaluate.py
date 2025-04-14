import logging
import warnings
from dotenv import load_dotenv
from src.data.data_transformation import DataTransformation
from src.utils import read_congif
import mlflow
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from mlflow import MlflowClient

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
load_dotenv()

# dagshub_token = os.getenv('DAGSHUB_TOKEN')



# Get the DagsHub token from the environment
dagshub_token = os.getenv('DAGSHUB_TOKEN')
dagshub_user = os.getenv('DAGSHUB_USER')

# new 
if dagshub_token and dagshub_user:
    # Set MLflow authentication environment variables
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    
    # Set the MLflow tracking URI
   
    print("DagsHub login successful!")
else:
    print("DagsHub token not found.")

class ModelTraining:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = read_congif()
        self.best_accuracy = 0
        self.best_model_name = None
        self.best_model = None

    def evaluate_models(self, models):
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            logging.info(f"{model_name} Accuracy: {accuracy}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model_name = model_name
                self.best_model = model

        return self.best_model_name, self.best_model

    def train_and_log_best_model(self):
        try:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(**self.config['model_params']['random_forest']),
                "XGBoost": xgb.XGBClassifier(**self.config['model_params']['xgboost']),
                "SVM": SVC(**self.config['model_params']['svm']),
            }

            best_model_name, best_model = self.evaluate_models(models)
            logging.info(f"Best model is {best_model_name} with accuracy {self.best_accuracy}")

            with mlflow.start_run(run_name=f"Best Model: {best_model_name}"):
                y_pred = best_model.predict(self.x_test)

                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                confusion_mat = confusion_matrix(self.y_test, y_pred)
                classification_rep = classification_report(self.y_test, y_pred, output_dict=True)

                mlflow.log_metric("Accuracy", self.best_accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)

                for label, metrics in classification_rep.items():
                    if isinstance(metrics, dict):
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(f"{label} {metric_name}", metric_value)

                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
                plt.savefig("reports/confusion_matrix.png")
                plt.close()
                mlflow.log_artifact("reports/confusion_matrix.png")
                mlflow.sklearn.log_model(best_model, "best_model")
                run_id = mlflow.active_run().info.run_id
                model_version = mlflow.register_model(f"runs:/{run_id}/best_model", "Best_Model")
                self.set_model_alias("Best_Model", model_version.version)
        except Exception as e:
            logging.error(f"Error in training or logging the model: {e}")
            raise

    def set_model_alias(self, model_name, version):
        try:
            client = MlflowClient()
            client.set_registered_model_alias(model_name, "dev", version=version)
        except Exception as e:
            logging.error(f"Error in registering the model alias: {e}")

if __name__ == "__main__":
    clean_data_path = 'data/processed/clean.csv'
    transformer_path = 'models/transformer.pkl'
    transformation = DataTransformation(clean_data_path, transformer_path)
    x_train, x_test, y_train, y_test = transformation.process()
    trainer = ModelTraining(x_train, x_test, y_train, y_test)
    trainer.train_and_log_best_model()

# python src/models/new_model_evaluate.py
