import json
import os
import pickle
import importlib
import scanpy as sc
import re
from platform import architecture

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def load_data(path):
    if path.endswith('.h5ad'):
        return sc.read_h5ad(path)
    elif path.endswith('.h5'):
        return sc.read_10x_h5(path)
    elif os.path.isdir(path):
        return sc.read_10x_mtx(path)
    elif path.endswith('.csv'):
        return sc.read_csv(path)
    elif path.endswith('.txt') or path.endswith('.tsv'):
        return sc.read_text(path, delimiter='\t')
    else:
        raise ValueError(f"Unrecognized format: {path}")

# region Set device to Apple Metal, CUDA, or CPU in that order
def specify_device():
    """
    Function to specify which device the model should be located on.  Attempt to use Apple Metal as first resort, followed by
    CUDA, before settling on CPU
    :return: device specification
    """
    # Use Apple Metal as first resort
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # Fall back to CUDA if available (for non-Mac systems)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    # Use CPU as last resort
    else:
        device = torch.device("cpu")

    return device
# endregion

# region Create Optimizer from training_config file
def create_optimizer(model, training_config):
    """
    Create optimizer based on parameter configuration
    :param model:
    :param training_config:
    :return:
    """
    optimizer_type = training_config['optimizer']
    lr = training_config['learning_rate']
    weight_decay = training_config['weight_decay']

    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Add other optimizers as needed
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
# endregion

# region Create Learning Rate Scheduler
def create_scheduler(optimizer, training_config):
    """
    Creates a ReduceLROnPlateau learning rate scheduler from the provided training configuration dictionary.
    :param optimizer: The PyTorch optimizer whose learning rate will be managed by the scheduler
    :param training_config: A dictionary containing scheduler hyperparameters: 'scheduler_mode',
    'scheduler_factor', 'scheduler_patience', 'scheduler_verbose', and 'min_lr'
    :return: A configured ReduceLROnPlateau scheduler instance
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=training_config['scheduler_mode'],  # Reduce LR when validation loss stops decreasing
        factor=training_config['scheduler_factor'],  # Multiply LR by this factor when reducing
        patience=training_config['scheduler_patience'],  # Wait this many epochs for improvement before reducing LR
        verbose=training_config['scheduler_verbose'],  # Print message when LR is reduced
        min_lr=training_config['min_lr']  # Don't reduce LR below this value
    )
    return scheduler
# endregion

# region Load Tensor Datasets
def load_tensors(save_location, search_pattern='tensors.pkl'):
    """
    Load the scaled training, validation, and testing tensors used for model training and evaluation
    :param save_location: directory location with the pickled dictionary
    :param search_pattern: the pattern to identify the pickled dictionary of the tensors.  Currently, the tensor dictionary is created
    in src/training/training.py
    :return:
    """

    # Filter files that contain the search pattern
    matching_files = [file for file in os.listdir(save_location) if search_pattern in file]

    if not matching_files:
        print(f"No files containing '{search_pattern}' found in {save_location}")
        return None
    elif len(matching_files) > 1:
        print(f"Multiple files containing '{search_pattern}' found in {save_location}")
    else:
        with open(os.path.join(save_location,matching_files[0]), 'rb') as f:
            scaled_tensors_dictionary = pickle.load(f)
    return scaled_tensors_dictionary
# endregion

# region Cross-Validation Utility Functions

# endregion

# region Load Previously Saved Model
def get_model_class_name(save_location):
    """
    Get the class name of the model from the architecture.txt file that is saved with every model training.  The function
    expects the file name to be architecture.txt and the model class name to be in the first line of the .txt file
    :param save_location: Directory where the architecture.txt file is located (should be a run within the experiments
    directory)
    :return: the class_name as a string
    """
    architecture_file_path = os.path.join(save_location,'architecture.txt')

    with open(architecture_file_path, 'r') as f:
        first_line = f.readline().strip()

    # Extract class name (there should be an open parenthesis after the class name)
    if '(' in first_line:
        class_name = first_line.split('(')[0].strip()
        return class_name
    else:
        # If no parenthesis, return the whole line
        return first_line

def load_previous_model(save_location, metadata_extension='metadata.json'):
    """
    Load a previously trained model from a saved state dictionary and a metadata.json file that contains the model class name,
    the name of the .pt file, the input dimensions, and the batch size used in training
    :param save_location: The directory location where the necessary information is saved
    :param metadata_extension: the name for the metadata file.  By default, the metadata files are saved as metadata.json, but
    in case they are saved as anything else this parameter can be changed
    :return:
    """
    metadata_file = os.path.join(save_location, metadata_extension)
    with open(metadata_file, 'r') as f:
        metadata_dictionary = json.load(f)

    model_file = metadata_dictionary['model_file']
    model_class_name = metadata_dictionary['model_class']
    input_dim = metadata_dictionary['input_dim']

    module_name = "src.models.models"

    try:
        # Import the module
        module = importlib.import_module(module_name)

        # Get the class from the module
        model_class = getattr(module, model_class_name)

        # Instantiate the model
        model = model_class(input_dim=input_dim)

        # Load the weights
        model_path = os.path.join(save_location, model_file)
        model.load_state_dict(torch.load(model_path))

        model.eval()  # Set to evaluation mode

        print(f"Successfully loaded {model_class_name} from {save_location}")
        return model

    except ImportError:
        print(f"Could not import module: {module_name}")
        print("Make sure your model classes are in the correct module")
        return None
    except AttributeError:
        print(f"Could not find class '{model_class_name}' in module {module_name}")
        print("Make sure the class name in metadata matches your model class definition")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
# endregion

# region Get the Input Dimension of Linear Input Dimension
def get_linear_input_dimension(model):
    """
    Get the linear input dimension of a model contingent on the model class having 'input_dim' as a parameter
    :param model: the neural network model, currently this function is expecting nn.Sequential
    :return: input dimension
    """
    if hasattr(model, 'input_dim'):
        return model.input_dim
    elif hasattr(model, 'model') and isinstance(model.model[0], nn.Linear):
        return model.model[0].in_features
# endregion

# region Load Metadata file
def load_metadata_file(save_location, metadata_extension='metadata.json'):
    """
    Loads a JSON metadata file from the specified directory and returns its contents as a dictionary.
    :param save_location: The directory containing the metadata file
    :param metadata_extension: The filename of the metadata file. Default is 'metadata.json'
    :return: A dictionary containing the metadata key-value pairs
    """
    metadata_file = os.path.join(save_location, metadata_extension)
    with open(metadata_file, 'r') as f:
        metadata_dictionary = json.load(f)

    return metadata_dictionary
# endregion

# region Analyze Feature Importance
def read_feature_impotances(save_location, feature_importance_file_name='feature_importance_checkpoint_final.pkl'):
    """
    Loads a pickled feature importances list from the specified directory.
    :param save_location: The directory containing the feature importances file
    :param feature_importance_file_name: The filename of the pickled feature importances. Default is
    'feature_importance_checkpoint_final.pkl'
    :return: The deserialized feature importances list (list of tuples of gene name and importance value)
    """
    with open(os.path.join(save_location,feature_importance_file_name), 'rb') as file:
        feature_importances_list = pickle.load(file)

    return feature_importances_list

def remove_irrelevant_features(feature_importances_list, minimum_importance=0.0, maximum_importance=None):
    """
    Function that takes a list of feature_importances tuples (gene name and importance as a npfloat64 value)
    and removes any features that don't meet the specified minimum and maximum thresholds
    :param feature_importances_list: A list of tuples where the first tuple entry is the gene name and the second tuple
    entry is the calculated feature importance as numpy64
    :param minimum_importance: the minimum importance of a feature for it to be kept as part of the return feature list.
    Default value is 0.0
    :param maximum_importance: the maximum importance of a feature for it to be kept as part of the return feature list.
    Default value is None
    :return: a condensed list of features that have the specified minimum and maximum importance, a list of features that were
    removed due to this, and revised list of features including their calculated importance
    """

    feature_return_list = []
    removed_feature_list = []
    revised_feature_importance_list = []
    for feature_tuple in feature_importances_list:
        if maximum_importance is not None:
            if maximum_importance > feature_tuple[1] > minimum_importance:
                feature_return_list.append(feature_tuple[0])
                revised_feature_importance_list.append(feature_tuple)
            else:
                removed_feature_list.append(feature_tuple[0])
        else:
            if feature_tuple[1] > minimum_importance:
                feature_return_list.append(feature_tuple[0])
                revised_feature_importance_list.append(feature_tuple)
            else:
                removed_feature_list.append(feature_tuple[0])

    return feature_return_list, removed_feature_list, revised_feature_importance_list

def save_relevant_feature_list(save_location, save_relevant_feature_list=True, save_removed_feature_list=False, save_relevant_feature_and_importance_list=False,
                               minimum_importance=0.0, maximum_importance=None):
    """
    Saves the relevant features based on a

    :param save_location: Location to save the feature lists
    :param save_relevant_feature_list: Save the list containing the relevant features as determined in remove_irrelevant_features.py. Default is True
    :param save_removed_feature_list: Save the list containing the irrelevant features as determined in remove_irrelevant_features.py. Default is False
    :param save_relevant_feature_and_importance_list: Save the list containing the relevant features and their importance as determined in
    remove_irrelevant_features.py. Default is False
    :param minimum_importance: The minimum importance threshold of a feature to keep it (parameter for remove_irrelevant_features)
    :param maximum_importance: The maximum importance threshold of a feature to keep it (parameter for remove_irrelevant_features)
    :return: None
    """

    feature_importances = read_feature_impotances(save_location=save_location,
                                                  feature_importance_file_name='feature_importance_checkpoint_final.pkl')
    print(feature_importances)
    feature_return_list, removed_features, new_feature_list = remove_irrelevant_features(feature_importances,
                                                                                         minimum_importance=minimum_importance,
                                                                                         maximum_importance=maximum_importance)
    if save_relevant_feature_list is True:
        with open(os.path.join(save_location,'relevant_features.pkl'), 'wb') as file:
            pickle.dump(feature_return_list, file)

    if save_removed_feature_list is True:
        with open(os.path.join(save_location,'irrelevant_features.pkl'), 'wb') as file:
            pickle.dump(removed_features, file)

    if save_relevant_feature_and_importance_list is True:
        with open(os.path.join(save_location,'relevant_feature_importances.pkl'), 'wb') as file:
            pickle.dump(new_feature_list, file)
# endregion

# region Extract Experiment Info From Metadata
def extract_experiment_info(metadata_dictionary, information_location = 'model_file'):
    """
    Extract experiment and filtration information from a metadata.json file that follows the pattern:
    "best_model_{experiment}_{filtration}.pt"
    :param metadata_dictionary: The dictionary containing the metadata information.  Should be in json format
    :param information_location: The key value for the metadata dictionary that corresponds to the experiment value
    :return:
    """
    filename = metadata_dictionary[information_location]

    # Remove prefix "best_model_" and suffix ".pt"
    if not filename.startswith("best_model_") or not filename.endswith(".pt"):
        return None, None

    content = filename[11:-3]  # Remove "best_model_" (11 chars) and ".pt" (3 chars)

    # Match the experimental run pattern using regex
    import re
    match = re.match(r"(rna-seq_run_[12])_(.*)", content)

    if match:
        experiment = match.group(1)
        filtration = match.group(2)
        return experiment, filtration

    # Fallback method if the pattern doesn't match
    parts = content.split("_", 3)
    if len(parts) >= 4:
        experiment = "_".join(parts[:3])
        filtration = "_".join(parts[3:])
        return experiment, filtration

    return content, None
# endregion

# region Ask Yes/No Question
def ask_yes_no_question(question, default=None):
    """
    Ask a yes/no question and return a True or False result
    :param question: The question to ask
    :param default: Default answer, if none the user must provide an answer
    :return: True for yes and False for no
    """

    valid = {"yes": True, "ye": True, "y": True, "no": False, "n": False}

    while True:
        print(question + "[yes/no]:")
        choice = input().lower()

        # Handle empty response with default
        if choice == "" and default is not None:
            return valid[""]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no'")
# endregion