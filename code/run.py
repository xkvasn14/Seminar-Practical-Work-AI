from CNN import train_cnn
from graphs_from_models import make_graphs
from data_annotation_stgcn import data_annotation_stgcn
from training_STGCN import train_stgcn
from test_STGCN import test_stgcn
from training_LSTM import train_lstm
from test_LSTM import test_lstm
from test_DSVM import test_dsvm
from training_DSVM import train_dsvm
from posture_analysis import posture_analysis_pipeline

if __name__ == "__main__":
    posture_analysis_pipeline("../data", do_normalization=True, do_annotation=True)

    train_dsvm(dataset_path='data_in_use/tmp_file.csv', model_path="best_dsvm.pkl")
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_012.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_022.csv", save_csv=True)

    train_lstm(dataset_path='data_in_use/tmp_file.csv', model_path='best_lstm.pth')
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_012.csv", model_lstm_path="best_lstm.pth",
              save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_022.csv", model_lstm_path="best_lstm.pth",
              save_csv=True)

    train_stgcn(dataset_path="data_in_use/tmp_file.csv", model_path="best_stgcn.pth")
    test_stgcn(normalized_data_path=f"data_in_use/data_12343658_1_012.csv", model_stgcn_path="best_stgcn.pth",
               save_csv=True)
    test_stgcn(normalized_data_path=f"data_in_use/data_12343658_1_022.csv", model_stgcn_path="best_stgcn.pth",
               save_csv=True)

    # labels = [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 3, 4]
    # data_annotation_stgcn(path_to_datas="data_in_use", do_annotation=True, labels=labels)
    # train_cnn(dataset_path="data_in_use", model_path="best_cnn.pth", test=True, train=True)

    make_graphs("data_in_use")
