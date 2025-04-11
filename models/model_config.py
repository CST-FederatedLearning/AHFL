import JSON
from streamlit import json


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.training_params = config["training_params"]
        self.model_params = config["model_params"]
