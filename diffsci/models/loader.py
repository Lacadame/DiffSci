import json
import pathlib

import porenet.models


def load_model(config_path, model_identifier,
               aseval=True, device='cuda'):
    """
    Loads a model and its configuration based on a JSON configuration file
        and a model identifier.

    Parameters:
    - config_path: str or pathlib, path to production model folder.
    - model_identifier: str, unique identifier for the model
        configuration within the JSON file.
    - aseval: bool, whether to switch to evaluation mode (default: True).
    - device: str or None, device to load the model on (default: 'cuda').
              If None, then it is loaded as it is (to CPU)

    Returns:
    - model: The loaded model instance ready for use.
    - config: The configuration instance used for the model.
    """

    # Transform to pathlib.Path (if it is not already)
    config_path = pathlib.Path(config_path)

    # Load JSON configuration
    with open(config_path/"models.json", 'r') as file:
        config = json.load(file)

    # Extract model configuration
    model_config = config[model_identifier]['load']

    # Dynamically evaluate the model and config constructors
    ModelClass = eval(model_config['model'])
    ConfigClass = eval(model_config['config'])

    # Load the model with its configuration
    module = porenet.models.KarrasModule.load_from_checkpoint(
        config_path/model_identifier,  # Adjust the path as needed
        model=ModelClass,
        config=ConfigClass,
        conditional=model_config['conditional'],
        mask=model_config['mask']
    )

    # Optionally, switch to evaluation mode
    if aseval:
        module.model.eval()
    if device is not None:
        module.to(device)
    return module, ConfigClass


def list_models(config_path):
    """
    Lists the available models and their configurations from the JSON
    configuration file.

    Parameters:
    - config_path: str, path to the JSON configuration file.

    Returns:
    - models_list: dict, a dictionary where keys are model identifiers
                   and values are their configurations.
    """

    # Transform to pathlib.Path (if it is not already)
    config_path = pathlib.Path(config_path)

    # Load JSON configuration
    with open(config_path/"models.json", 'r') as file:
        config = json.load(file)

    models_list = {}

    for model_identifier, settings in config.items():
        models_list[model_identifier] = {
            'training': settings.get('training',
                                     'No training configuration provided.'),
            'load': settings.get('load',
                                 'No load configuration provided.')
        }

    return models_list
