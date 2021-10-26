import numpy as np
import time
import os
import scipy.io
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

# from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor, MLPClassifier
import scipy
import seaborn as sns

from data_tools import *
from dataset import *
from utils import CDF_plot


# ----------- SVM for comparision (use same data) -----------
# to compare time as well, use feature_flag=False and the same dataset function
def svm_regressor(data_train, data_test):

    # assign data for svm baseline
    cir_train, err_train, label_train = data_train
    cir_test, err_test, label_test = data_test

    # extract feature (to compare time)
    train_time = time.time()
    features_train = feature_extraction(cir_train)
    
    # error regression
    # train_time = time.time()
    clf_reg = make_pipeline(StandardScaler(), SVR(gamma='auto'))
    clf_reg.fit(features_train, err_train)
    svr_train_time = time.time() - train_time

    test_time = time.time()    
    features_test = feature_extraction(cir_test)
    err_est = clf_reg.predict(features_test)
    svr_test_time = (time.time() - test_time) / features_test.shape[0]

    # reshape
    err_test = err_test.reshape(err_test.shape[0])
    rmse_error = (np.sum((err_est - err_test) ** 2) / err_test.shape[0]) ** 0.5
    abs_error = (np.sum(np.abs(err_est - err_test)) / err_test.shape[0])
    print("SVM Regression Results: rmse %f, abs %f, time %f/%f" % (rmse_error, abs_error, svr_train_time, svr_test_time))

    # original
    err_test = err_test.reshape(err_test.shape[0])
    rmse_org = (np.sum((err_test) ** 2) / err_test.shape[0]) ** 0.5
    abs_org = (np.sum(np.abs(err_test)) / err_test.shape[0])
    print("Dataset Errors: rmse %f, abs %f" % (rmse_org, abs_org))

    return np.abs(err_est - err_test), np.abs(err_test), svr_train_time, svr_test_time


# only use cir, extract feature itself and count time
def svm_classifier(data_train, data_test):

    # assign data for svm baseline
    cir_train, err_train, label_train = data_train
    cir_test, err_test, label_test = data_test

    # extract feature (to compare time)
    train_time = time.time()
    features_train = feature_extraction(cir_train)
    
    # label classification
    clf_cls = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_cls.fit(features_train, label_train.squeeze())
    svc_train_time = time.time() - train_time

    test_time = time.time()
    features_test = feature_extraction(cir_test)
    label_est = clf_cls.predict(features_test)
    svc_test_time = (time.time() - test_time) / features_test.shape[0]

    # reshape
    label_test = label_test.reshape(label_test.shape[0])
    accuracy = np.sum(label_est == label_test) / label_test.shape[0]
    print("SVM Classification Result: accuracy %f, time %f/%f" % (accuracy, svc_train_time, svc_test_time))

    return accuracy, svc_train_time, svc_test_time


def mlp_regressor(data_train, data_test):

    # assign data for mlp baseline
    cir_train, err_train, label_train = data_train
    cir_test, err_test, label_test = data_test

    # extract feature (to compare time)
    train_time = time.time()
    features_train = feature_extraction(cir_train)

    # error regression
    mlp_reg = MLPRegressor(
        solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50, 20),
        random_state=1, max_iter=600
    )
    mlp_reg.fit(features_train, err_train)
    mlp_train_time = time.time() - train_time

    test_time = time.time()
    features_test = feature_extraction(cir_test)
    err_est = mlp_reg.predict(features_test)
    # test_score = mlp_reg.score(features_test, err_test)
    mlp_test_time = (time.time() - test_time) / features_test.shape[0]

    # reshape
    err_test = err_test.reshape(err_test.shape[0])
    rmse_error = (np.sum((err_est - err_test) ** 2) / err_test.shape[0]) ** 0.5
    abs_error = (np.sum(np.abs(err_est - err_test)) / err_test.shape[0])
    print("MLP Regression Results: rmse %f, abs %f, time %f/%f" % (rmse_error, abs_error, mlp_train_time, mlp_test_time))

    return np.abs(err_est - err_test), np.abs(err_test), mlp_train_time, mlp_test_time


def mlp_classifier(data_train, data_test):

    # assign data for mlp baseline
    cir_train, err_train, label_train = data_train
    cir_test, err_test, label_test = data_test

    # extract feature (to compare time)
    train_time = time.time()
    features_train = feature_extraction(cir_train)

    # label classification
    mlp_cls = MLPClassifier(random_state=1, max_iter=600)
    mlp_cls.fit(features_train, label_train.squeeze())
    mlp_train_time = time.time() - train_time

    test_time = time.time()
    features_test = feature_extraction(cir_test)
    label_est = mlp_cls.predict(features_test)
    # test_score = mlp_cls.score(feature_test, label_test)
    # label_est_soft = mlp_cls.predict_proba(features_test)
    mlp_test_time = (time.time() - test_time) / features_test.shape[0]

    # reshape
    label_test = label_test.reshape(label_test.shape[0])
    accuracy = np.sum(label_est == label_test) / label_test.shape[0]
    print("MLP Classification Result: accuracy %f, time %f/%f" % (accuracy, mlp_train_time, mlp_test_time))

    return accuracy, mlp_train_time, mlp_test_time


def CDF_plot_baseline(data, save_path, num=200, title=None):
    error_est, error_gt = data

    # est results
    data_est = np.abs(error_est - error_gt)
    blocks_num_est = num
    pred_error_max_est = np.max(data_est)
    step_est = pred_error_max_est / blocks_num_est
    
    pred_error_cnt_est = np.zeros((blocks_num_est + 1,))
    for i in range(data_est.shape[0]):
        index_est = int(data_est[i] / step_est)  # normalize values to (0, 1)
        pred_error_cnt_est[index_est] = pred_error_cnt_est[index_est] + 1
    pred_error_cnt_est = pred_error_cnt_est / np.sum(pred_error_cnt_est)

    CDF_est = np.zeros((blocks_num_est + 1,))
    for i in range(blocks_num_est + 1):  # accumulate error to CDF
        if i == 0:
            CDF_est[i] = pred_error_cnt_est[i]
        else:
            CDF_est[i] = CDF_est[i - 1] + pred_error_cnt_est[i]

    plt.plot(np.linspace(0, pred_error_max_est, num=blocks_num_est + 1), CDF_est, color='red')

    # original error
    data_org = np.abs(error_gt)
    blocks_num_org = num
    pred_error_max_org = np.max(data_org)
    pred_error_cnt_org = np.zeros((blocks_num_org + 1,))
    step_org = pred_error_max_org / blocks_num_org

    # normalize to (0, 1) by dividing max
    for i in range(data_org.shape[0]):
        index_org = int(data_org[i] / step_org)
        pred_error_cnt_org[index_org] = pred_error_cnt_org[index_org] + 1
    pred_error_cnt_org = pred_error_cnt_org / np.sum(pred_error_cnt_org)

    # accumulate error at each point to CDF
    CDF_org = np.zeros((blocks_num_org + 1,))
    for i in range(blocks_num_org + 1):
        if i == 0:
            CDF_org[i] = pred_error_cnt_org[i]
        else:
            CDF_org[i] = CDF_org[i - 1] + pred_error_cnt_org[i]
        
    plt.plot(np.linspace(0, pred_error_max_org, num=blocks_num_org + 1), CDF_org, color='black')

    plt.legend(["Mitigated Error", "Original Range Error"])
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join(save_path, "CDF_est.png"))
    plt.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ewine", help="dataset for usage, ewine or zenodo")
    parser.add_argument("--dataset_env", type=str, default="nlos", help="dataset (zenodo) of different environments")
    parser.add_argument("--mode", type=str, default="full", help="mode to assign train and test data")
    parser.add_argument("--split_factor", type=float, default=0.8, help="split factor for train and test data")
    opt = parser.parse_args()

    # assign different roots of each dataset
    if opt.dataset_name == 'zenodo':
        root = './data/data_zenodo/dataset.pkl'
    elif opt.dataset_name == 'ewine':
        filepaths = ['./data/data_ewine/dataset1/tag_room0.csv',
                     './data/data_ewine/dataset1/tag_room1.csv',
                     './data/data_ewine/dataset2/tag_room0.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part0.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part1.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part2.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part3.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part4.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part5.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part6.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part7.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part8.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part9.csv']
        root = filepaths

    # assign data for training and testing
    data_train, data_test, feature_train, feature_test = err_mitigation_dataset(
        root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env, split_factor=opt.split_factor,
        scaling=True, mode=opt.mode, feature_flag=False
    )

    # ------------- train and test svm ------------
    error_svm, error_gt, svr_train_time, svr_test_time = svm_regressor(data_train, data_test)
    print('error_svm_abs: ', error_svm[0:10])
    print('error_gt: ', error_gt[0:10])
    save_path = "./saved_results/%s_%s/SVR_reg" % (opt.dataset_env, opt.mode)
    os.makedirs(save_path, exist_ok=True)
    data_svm = error_svm, error_gt
    norm_cdf = scipy.stats.norm.cdf(np.abs(error_svm - error_gt))
    sns.lineplot(x=np.abs(error_svm - error_gt), y=norm_cdf)
    plt.show()
    CDF_plot_baseline(data_svm, save_path)
    # ewine
    # svr: rmse 0.318274, abs 0.183702, time 70.490080/21.803473
    # svc: accuracy 0.830523, time 137.347341/25.464936
    # zenodo

    # env label classification using svm
    svm_accuracy, svc_train_time, svc_test_time = svm_classifier(data_train, data_test)
    # print('label_est: ', label_est[0:10])
    # print('label_test: ', label_test[0:10])

    # --------------- train and test mlp -------------
    error_mlp, error_gt, mlpr_train_time, mlpr_test_time = mlp_regressor(data_train, data_test)
    print('error_mlp_abs: ', error_mlp[0:10])
    print('error_gt: ', error_gt[0:10])
    save_path = "./saved_results/%s_%s/MLP_reg" % (opt.dataset_env, opt.mode)
    os.makedirs(save_path, exist_ok=True)
    data_mlp = error_mlp, error_gt
    norm_cdf = scipy.stats.norm.cdf(np.abs(error_mlp-error_gt))
    sns.lineplot(x=np.abs(error_mlp-error_gt), y=norm_cdf)
    plt.show()
    CDF_plot_baseline(data_mlp, save_path)

    res_svm = np.abs(error_svm - error_gt)
    res_mlp = np.abs(error_mlp - error_gt)
    CDF_plot(err_arr=error_gt, color='purple')
    CDF_plot(err_arr=res_svm, color='c')
    CDF_plot(err_arr=res_mlp, color='brown')
    plt.legend(["Original error", "SVM", "MLP"], loc='lower right')
    plt.savefig(os.path.join(save_path, "CDF_%s_%s.png" % (opt.dataset_name, opt.dataset_env)))
    plt.close()

    mlp_accuracy, mlpc_train_time, mlpc_test_time = mlp_classifier(data_train, data_test)

