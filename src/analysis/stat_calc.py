import numpy as np

def calculateF1ScoreValue(train_list, val_list, test_list):
    """ Calculates the mean and standard deviation of F1 macro score based on results from three different seeds"""
    print("F1 Macro")

    f1_mean_train = np.mean(train_list)
    f1_sd_train = np.std(train_list, ddof=1)
    print(f"Train: {f1_mean_train:.3f} ± {f1_sd_train:.3f}")


    f1_mean_val = np.mean(val_list)
    f1_sd_val = np.std(val_list, ddof=1)
    print(f"Validation: {f1_mean_val:.3f} ± {f1_sd_val:.3f}")


    f1_mean_test = np.mean(test_list)
    f1_sd_test = np.std(test_list, ddof=1)
    print(f"Test: {f1_mean_test:.3f} ± {f1_sd_test:.3f}")



def calculateAccuracyScoreValue(train_list, val_list, test_list):
    """ Calculates the mean and standard deviation of accuracy score based on results from three different seeds"""
    print("Accuracy")

    acc_mean_train = np.mean(train_list)
    acc_sd_train = np.std(train_list, ddof=1)
    print(f"Train:{acc_mean_train:.3f} ± {acc_sd_train:.3f}")


    acc_mean_val = np.mean(val_list)
    acc_sd_val = np.std(val_list, ddof=1)
    print(f"Validation:{acc_mean_val:.3f} ± {acc_sd_val:.3f}")


    acc_mean_test = np.mean(test_list)
    acc_sd_test = np.std(test_list, ddof=1)
    print(f"Test:{acc_mean_test:.3f} ± {acc_sd_test:.3f}")


def calculateTopKAccuracyScoreValue(train_list, val_list, test_list):
    """ Calculates the mean and standard deviation of top-k accuracy score based on results from three different seeds"""
    print("Top-2 Accuracy")

    topk_mean_train = np.mean(train_list)
    topk_sd_train = np.std(train_list, ddof=1)
    print(f"Train: {topk_mean_train:.3f} ± {topk_sd_train:.3f}")


    topk_mean_val = np.mean(val_list)
    topk_sd_val = np.std(val_list, ddof=1)
    print(f"Validation: {topk_mean_val:.3f} ± {topk_sd_val:.3f}")


    topk_mean_test = np.mean(test_list)
    topk_sd_test = np.std(test_list, ddof=1)
    print(f"Test: {topk_mean_test:.3f} ± {topk_sd_test:.3f}")


if __name__ == "__main__":

    # For SmolVM2 Experiment

    train_f1 = [0.314960,0.17878,0.16833]
    val_f1 = [0.20561,0.15994,0.14853]
    test_f1 = [0.22689,0.14920,0.17469]
    calculateF1ScoreValue(train_f1, val_f1, test_f1)

    train_topk = [0.62109,0.51750,0.51750]
    val_topk = [0.54932,0.50194,0.51219]
    test_topk = [0.54716,0.50333,0.51461]

    calculateTopKAccuracyScoreValue(train_topk, val_topk, test_topk)

    train_acc = [0.38033,0.31565,0.30728]
    val_acc = [0.30901,0.29698,0.30064]
    test_acc = [0.32697,0.29847,0.31308]
    calculateAccuracyScoreValue(train_acc, val_acc, test_acc)



