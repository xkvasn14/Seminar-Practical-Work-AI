from CNN import train_cnn
from graphs_from_models import make_graphs, make_confusion_matrices
from data_annotation_stgcn import data_annotation_stgcn
from training_STGCN import train_stgcn
from test_STGCN import test_stgcn
from training_LSTM import train_lstm
from test_LSTM import test_lstm
from test_DSVM import test_dsvm
from training_DSVM import train_dsvm
from posture_analysis import posture_analysis_pipeline


def test_all():
    print("TESTS FROM DSVM:")
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_012.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_013.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_014.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_015.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_016.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_017.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_018.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_019.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_020.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_021.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_022.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_023.csv", save_csv=True)
    test_dsvm(normalized_data_path="data_in_use/data_12343658_1_024.csv", save_csv=True)

    print("TESTS FROM LSTM:")
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_012.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_013.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_014.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_015.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_016.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_017.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_018.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_019.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_020.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_021.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_022.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_023.csv", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_024.csv", save_csv=True)

    print("TESTS FROM STGCN:")
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_012.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_013.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_014.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_015.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_016.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_017.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_018.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_019.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_020.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_021.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_022.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_023.csv", save_csv=True)
    test_stgcn(normalized_data_path="data_in_use/data_12343658_1_024.csv", save_csv=True)


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

    # # # if you also wish to train and test CNN architecture
    # # labels = [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 3, 4]
    # # data_annotation_stgcn(path_to_datas="data_in_use", do_annotation=True, labels=labels)
    # # train_cnn(dataset_path="data_in_use", model_path="best_cnn.pth", test=True, train=True)

    # test all data on DSVM, LSTM and STGCN
    test_all()

    make_graphs("data_in_use")
    make_confusion_matrices("data_in_use")
