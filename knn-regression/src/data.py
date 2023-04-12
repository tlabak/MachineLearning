import numpy as np
import pandas as pd


def load_data(data_path):
    """
    This function loads the data in data_path csv into two numpy arrays:
    examples (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of examples.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        examples (np.array): numpy array of size (N, K) containing the N examples, each which
            has K different attributes
        targets (np.array): numpy array of size (N, ) containing the N targets.
        attribute_names (list): list of strings containing names of each attribute
            (headers of csv)
    """
    if data_path.endswith('gz'):
        df = pd.read_csv(data_path, compression='gzip')
    else:
        df = pd.read_csv(data_path)

    feature_columns = [col for col in df.columns if col != "class"]
    examples = df[feature_columns].to_numpy()
    target = df[["class"]].to_numpy()

    return examples, target, feature_columns
