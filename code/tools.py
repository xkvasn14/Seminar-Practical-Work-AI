"""
    Function used for various purposes. Not included in the main code
"""
import os

import pandas as pd


def get_cluster_numbers_from_files(data_path: str = "data_in_use"):
    """
        Should print all non-tested normalized, annotated files. Prints unique sequence of cluster annotations.
    """
    files = os.listdir(data_path)
    filtered_files = [f for f in files if '_dsvm' not in f and '_lstm' not in f and '_stgcn' not in f]
    transition_results = {}

    for file_name in filtered_files:
        file_path = os.path.join(data_path, file_name)
        try:
            data = pd.read_csv(file_path)
            if 'Cluster' not in data.columns:
                raise ValueError(f"Cluster column not found in {file_name}")
            cluster_transitions = data['Cluster'][data['Cluster'] != data['Cluster'].shift()].tolist()
            transition_results[file_name] = cluster_transitions
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    for file_name, transitions in transition_results.items():
        print(f"{file_name}: {transitions}")


if __name__ == "__main__":
    get_cluster_numbers_from_files("data_in_use")
