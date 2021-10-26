import time
import sys
import scipy.io as io

from utils import *
from dataset import *
from model_sep import *
from baseline import *

import logging


# separated testing for evaluation during training
def test_identifier(opt, device, result_path, model_path, dataloader, network, epoch, data_raw):
    # different for val and test: result_path, epoch

    # Load models from path
    if epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "INet_%d.pth" % epoch)))
        network.eval()
    else:
        print("No saved models in dirs.")

    # Evaluation initialization
    accuracy = 0.0
    start_time = time.time()

    for i, batch in enumerate(dataloader):

        # Set model input
        cir_gt = batch["CIR"]
        label_gt = batch["Label"]
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            label_gt = label_gt.to(device=device, dtype=torch.int64)

        with torch.no_grad():

            # Generate estimations (one-hot encoding \epsilon_los, \epsilon_nlos)
            label_est = network(cir_gt)

            # Evaluation metrics
            time_test = (time.time() - start_time) / 500  # batch_size
            time_avg = time_test / (i + 1)
            label_gt = label_gt.squeeze()
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gt).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

        # Print log
        sys.stdout.write(
            "\r[Val Data: %s/%s] [Model Type: Identifier%s] [Test Epoch: %d] [Batch: %d/%d] [accuracy %f] [Test Time: "
            "%f] "
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, epoch, i, len(dataloader), accuracy_avg,
               time_avg)
        )
        logging.info(
            "\r[Val Data: %s/%s] [Model Type: Identifier%s] [Test Epoch: %d] [Batch: %d/%d] [accuracy %f] [Test Time: "
            "%f] "
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, epoch, i, len(dataloader), accuracy_avg,
               time_avg)
        )

    # compare accuracies
    data_train, data_test = data_raw
    svm_accuracy, svc_train_time, svc_test_time = svm_classifier(data_train, data_test)
    mlp_accuracy, mlpc_train_time, mlpc_test_time = mlp_classifier(data_train, data_test)
    sys.stdout.write(
        "\r[Comparison results] [Acc: ours %f, svm %f, mlp %f] [Inference time: ours %f, svm %f, mlp %f] "
        % (accuracy_avg, svm_accuracy, mlp_accuracy, time_avg, svc_test_time, mlpc_test_time)
    )
    logging.info(
        "\r[Comparison results] [Acc: ours %f, svm %f, mlp %f] [Inference time: ours %f, svm %f, mlp %f] "
        % (accuracy_avg, svm_accuracy, mlp_accuracy, time_avg, svc_test_time, mlpc_test_time)
    )


def test_estimator(opt, device, result_path, model_path, dataloader, network, epoch, data_raw): 
    # different for val and test: result_path, epoch

    # Load models from path
    if epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % epoch)))
        network.eval()
    else:
        print("No saved models in dirs.")

    # Evaluation initialization
    rmse_error = 0.0
    abs_error = 0.0
    start_time = time.time()

    for i, batch in enumerate(dataloader):

        # Set model input
        cir_gt = batch["CIR"]
        err_gt = batch["Err"]
        label_gt = batch["Label"]
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()

        with torch.no_grad():

            # Generate estimations
            label_input = label_gt.numpy().astype('int64')
            label_cat = to_categorical(
                label_input, num_columns=opt.num_classes
            )
            label_cat = torch.tensor(label_cat)
            label_cat = label_cat.to(device=device, dtype=torch.int64)
            err_mu, err_sigma, err_sri = network(label_cat, cir_gt)  # err_est

            # Evaluation metrics
            rmse_error += (torch.mean((err_mu - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_mu - err_gt))
            time_test = (time.time() - start_time) / 500  # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)

            # range residual error arrays
            err_real = err_gt.cpu().numpy()
            err_fake = err_mu.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))
        
        # Print log
        sys.stdout.write(
            "\r[Val: %s/%s] [Model Type: Estimator%s] [Test Epoch: %d] [Batch: %d/%d] [Error: rmse %f, abs %f] [Test "
            "Time: %f] "
            % (opt.dataset_name, opt.dataset_env, opt.estimator_type, epoch, i, len(dataloader), rmse_avg, abs_avg,
               time_avg)
        )
        logging.info(
            "\r[Val: %s/%s] [Model Type: Estimator%s] [Test Epoch: %d] [Batch: %d/%d] [Error: rmse %f, abs %f] [Test "
            "Time: %f] "
            % (opt.dataset_name, opt.dataset_env, opt.estimator_type, epoch, i, len(dataloader), rmse_avg, abs_avg,
               time_avg)
        )

    # CDF plotting of residual error
    res_sri = np.abs(err_real_arr - err_fake_arr)
    data_train, data_test = data_raw
    err_svm, err_gt_s, _, _ = svm_regressor(data_train, data_test)
    # err_mlp, err_gt_m, _, _ = mlp_regressor(data_train, data_test)
    res_svm = np.abs(err_svm)
    # res_mlp = np.abs(err_mlp)
    resi_ori = np.asarray([item[0] for item in err_real_arr])
    resi_svm = np.asarray(res_svm)
    # resi_mlp = np.asarray(res_mlp)
    resi_sri = np.asarray([item[0] for item in res_sri])
    # print("residual error array", resi_ori, resi_sri, resi_svm, resi_mlp)
    CDF_plot(err_arr=resi_ori, color='y')
    CDF_plot(err_arr=resi_sri, color='purple')
    CDF_plot(err_arr=resi_svm, color='c')
    # CDF_plot(err_arr=resi_mlp, color='brown')
    plt.legend(["Original error", "Our method", "SVM"], loc='lower right')  # , "MLP"
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, epoch)))
    plt.close()


def test_sri_network(opt, device, result_path, model_path, dataloader, net_i, net_e, epoch_i, epoch_e, data_raw):
    
    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'test_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load models from path
    if epoch_i != 0:
        net_i.load_state_dict(torch.load(os.path.join(model_path, "INet_%d.pth" % epoch_i)))
        net_i.eval()
    else:
        print("No saved Identifier models in dirs.")
    if epoch_e != 0:
        net_e.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % epoch_e)))
        net_e.eval()
    else:
        print("No saved Estimator models in dirs.")

    # Evaluation initialization
    rmse_error = 0.0
    abs_error = 0.0
    accuracy = 0.0
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        
        # Set model input
        cir_gt = batch["CIR"]
        err_gt = batch["Err"]
        label_gt = batch["Label"]
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()
            label_gt = label_gt.to(device=device, dtype=torch.int64)

        with torch.no_grad():

            # Generate estimations
            label_est = net_i(cir_gt)
            label_input0 = np.zeros((cir_gt.shape[0], 1)).astype('int64')
            label_sim_0 = to_categorical(
                label_input0, num_columns=opt.num_classes
            )
            label_sim_0 = torch.tensor(label_sim_0)
            label_sim_0 = label_sim_0.to(device=device, dtype=torch.int64)
            err_mu_0, err_sig_0, err_sri_0 = net_e(label_sim_0, cir_gt)
            label_input1 = np.ones((cir_gt.shape[0], 1)).astype('int64')
            label_sim_1 = to_categorical(
                label_input1, num_columns=opt.num_classes
            )
            label_sim_1 = torch.tensor(label_sim_1)
            label_sim_1 = label_sim_1.to(device=device, dtype=torch.int64)
            err_mu_1, err_sig_1, err_sri_1 = net_e(label_sim_1, cir_gt)
            
            # Evaluation metrics
            err_est = label_est[:, 0] * err_mu_0.squeeze() + label_est[:, 1] * err_mu_1.squeeze()
            err_est = err_est.unsqueeze(1)
            # print("checking 1: ", label_est[0], err_mu_0[0], err_est[0])
            # print(err_est.size(), err_mu_0.size())
            # likelihood = label_est[:, 0] * err_sri_0 + label_est[:, 1] * err_sri_1
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / err_est.shape[0]  # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gt = label_gt.squeeze()
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gt).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)
            
            # range residual error arrays
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        sys.stdout.write(
            "\r[Test: %s/%s] [Model Type: Identifier%s_Estimator%s] [Test Epoch: i_%d/e_%d] [Batch: %d/%d] [Error: "
            "accuracy %f, rmse %f, abs %f] [Test Time: %f] "
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, opt.estimator_type, epoch_i, epoch_e, i,
            len(dataloader), accuracy_avg, rmse_avg, abs_avg, time_avg)
        )
        logging.info(
            "\r[Test: %s/%s] [Model Type: Identifier%s_Estimator%s] [Test Epoch: i_%d/e_%d] [Batch: %d/%d] [Error: "
            "accuracy %f, rmse %f, abs %f] [Test Time: %f] "
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, opt.estimator_type, epoch_i, epoch_e, i,
            len(dataloader), accuracy_avg, rmse_avg, abs_avg, time_avg)
        )

    # CDF plotting of residual error
    res_sri = np.abs(err_real_arr - err_fake_arr)
    data_train, data_test = data_raw
    err_svm, err_gt_s, _, _ = svm_regressor(data_train, data_test)
    err_mlp, err_gt_m, _, _ = mlp_regressor(data_train, data_test)
    res_svm = np.abs(err_svm)
    # res_mlp = np.abs(err_mlp)
    resi_ori = np.asarray([item[0] for item in err_real_arr])
    resi_svm = np.asarray(res_svm)
    # resi_mlp = np.asarray(res_mlp)
    resi_sri = np.asarray([item[0] for item in res_sri])
    # print("checking 2: ")
    # print(resi_ori, resi_svm, resi_mlp, resi_sri)
    CDF_plot(err_arr=resi_ori, color='c')  # marker='o'
    CDF_plot(err_arr=resi_sri, color='y')  # marker='*'
    CDF_plot(err_arr=resi_svm, color='purple')  # marker='x'
    # CDF_plot(err_arr=resi_mlp, color='b')
    plt.legend(["Original error", "Our method", "SVM"], loc='lower right')  # , "MLP"
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, epoch_e)))
    print("Saving CDF figures in %s." % result_path)
    plt.close()
    io.savemat(os.path.join(result_path, "residual_sri_%s_%s_%d" % (opt.dataset_name, opt.dataset_env, epoch_e)),
               {'residual_sri': resi_sri})
    io.savemat(os.path.join(result_path, "residual_svm_%s_%s_%d" % (opt.dataset_name, opt.dataset_env, epoch_e)),
               {'residual_svm': resi_svm})
    # io.savemat(os.path.join(result_path, "residual_mlp_%s_%s_%d" % (opt.dataset_name, opt.dataset_env, epoch_e)),
    #            {'residual_mlp': resi_mlp})
    io.savemat(os.path.join(result_path, "residual_gt_%s_%s_%d" % (opt.dataset_name, opt.dataset_env, epoch_e)),
               {'residual_gt': resi_ori})

    # compare accuracies
    data_train, data_test = data_raw
    svm_acc, _, svc_time = svm_classifier(data_train, data_test)
    mlp_acc, _, mlpc_time = mlp_classifier(data_train, data_test)
    logging.info(
        "\r[Comparison results] [RMSE: ours %f, MAE: ours %f] [Acc: ours %f, svm %f, "
        "mlp %f] [Inference time: ours %f, svm %f, mlp %f] "
        % (rmse_avg, abs_avg, accuracy_avg, svm_acc, mlp_acc, time_avg, svc_time, mlpc_time)
    )
    logging.info(
        "\r[Comparison results] [RMSE: ours %f, MAE: ours %f] [Acc: ours %f, svm %f, "
        "mlp %f] [Inference time: ours %f, svm %f, mlp %f] "
        % (rmse_avg, abs_avg, accuracy_avg, svm_acc, mlp_acc, time_avg, svc_time, mlpc_time)
    )
