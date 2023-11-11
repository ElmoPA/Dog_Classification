import json
import luigi
import mlflow
import itertools
from train import Train

def generate_params_combination(grid_params_dict):
    key, values = zip(*grid_params_dict.items())
    for value_combination in itertools.product(*values):
        yield dict(zip(key, value_combination))

class GridSearch(luigi.Task):

    grid_params = luigi.Parameter()
    experiment_name = luigi.Parameter(default='default experiment')

    def outputs(self):
        return None

    def requires(self):
        return None

    def run(self):
        grid_params_dict = json.loads(self.grid_params)
        grid_params_dict['called'] = [self.experiment_name]

        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        with mlflow.start_run(run_name=self.experiment_name) as exp:
            luigi.build(
                [Train(**task_param) for task_param in generate_params_combination(grid_params_dict)],
                local_scheduler=True
            )

if __name__ == '__main__':
    grid_params = {
        'model': ['ResNet50', 'ResNet18'],
        'epochs': [17]
    }
    grid_params_str = json.dumps(grid_params)
    luigi.build([GridSearch(experiment_name='test1', grid_params=grid_params_str)], local_scheduler=True)