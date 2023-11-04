start cmd /k "luigid --background --logdir ./logs"
start cmd /k "mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns"
start http://localhost:8082  # Luigi web interface
start http://localhost:5000  # MLflow web interface