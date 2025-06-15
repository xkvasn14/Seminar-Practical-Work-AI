"""
This script builds up the posture analysis script and specifically annotates data only for the purposes of the stgcn_new scripts.
!!! RUN WHOLE POSTURE_ANALYSIS SCRIPT FIRST !!! (normalization, annotation, data separation)
"""
import os

import pandas as pd


def re_annotate_for_stgcn_new(path_to_file, label):
    df = pd.read_csv(path_to_file)
    print(path_to_file, label)
    df["Cluster"] = label
    parts = path_to_file.split("/")
    file = parts[-1]
    file_name, ext = file.split(".")
    new_file = file_name + "_single_label." + ext
    parts[-1] = new_file
    out_path = "/".join(parts)
    print(f"Saving annotated file to {out_path}")
    df.to_csv(out_path, index=False)


def data_annotation_stgcn(path_to_datas: str = "data_in_use", do_annotation: bool = False, labels: list = []):
    if path_to_datas.endswith("/"):
        path_to_datas = path_to_datas[:-1]

    paths_to_files = os.listdir(path_to_datas)

    index = 0
    for path in paths_to_files:
        if path.endswith(".csv"):
            if "tmp_file" in path or "_dsvm" in path or "_lstm" in path or "_stgcn" in path:
                pass
            else:
                if do_annotation:
                    re_annotate_for_stgcn_new(path_to_datas + "/" + path, labels[index])
                    index += 1
                else:
                    print(path)


if __name__ == "__main__":
    """ 
    !!! number of labels must equal the number of annotated files - there is no check, but it won't work otherwise... 
    """
    labels = [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 3, 4]
    data_annotation_stgcn(path_to_datas="data_in_use", do_annotation=True, labels=labels)
