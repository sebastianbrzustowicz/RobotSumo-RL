import mlflow
import yaml

class MLflowManager:
    def __init__(self, experiment_name="Mino_Sumo"):
        mlflow.set_experiment(experiment_name)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def start_run(self, run_name, config):
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(config)

    def log_metrics(self, step, metrics):
        mlflow.log_metrics(metrics, step=step)

    def end_run(self):
        mlflow.end_run()