import torch
import torch.nn as nn
import numpy as np
import array
import torch.optim as optim
import argparse
import random
from singleRBF_KRR import RBFRegressor
from cla_LAB_ori import LAB_ori
from data_reg import SinDataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.optim import lr_scheduler
import random
FONTSIZE = 13

seed = 30
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def testFun(adp_model, data_x, data_y):
    #
    batch = 128
    adp_model.eval()
    pred_y_list = []
    cnt = 0
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            pred_y_list.append(adp_model(data_x[:, cnt: cnt + batch]).detach().reshape(-1))
        else:
            pred_y_list.append(adp_model(data_x[:, cnt:]).detach().reshape(-1))
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list, 0)
    pred_y = pred_y.view(max(pred_y.shape),1)
    print('\t\t the rsse loss: ', format(adp_model.rsse_loss(pred=pred_y, target=data_y)))
    # print('\t\t the mae loss:', format(adp_model.mad_loss(pred=pred_y, target=data_y))
    print('\t\t the rmse loss:', format(adp_model.rmse_loss(pred=pred_y, target=data_y)))

    return pred_y


def IniTrainKernel(train_x, val_x, train_y, val_y, test_x, test_y):
    weight_ini = torch.sqrt(torch.tensor(300).float() / train_x.shape[0]) * torch.ones(train_x.shape)
    # adp_model = LABRBF(x_sv=train_x, y_sv=train_y, weight_ini=weight_ini)
    adp_model = LAB_ori(x_sv=train_x, y_sv=train_y, weight_ini=weight_ini)
    # adp_model.double()
    adp_model.train()

    # plt.figure(2)
    # plt.subplot(3, 2, 1)
    pred_y_list = []
    print('the train error rate loss: ')
    pred_y_list.append(testFun(adp_model, train_x, train_y))
    print('the validation error rate loss: ')
    pred_y_list.append(testFun(adp_model, val_x, val_y))
    print('the test error rate loss: ')
    pred_y_list.append(testFun(adp_model, test_x, test_y))
    pred_y = torch.cat(pred_y_list, 0).reshape(-1)
    data_x = torch.cat((train_x[0,:], val_x[0,:], test_x[0,:]), 0)
    sorted, index = torch.sort(data_x)
    # plt.plot(data_x[index], pred_y[index], 'k-', label='decision function')

    # plt.plot(train_x[0,:], train_y, 'r+', label='train data', markersize=13)
    # plt.plot(val_x[0, :], val_y, 'b2', label='val data', markersize=13)
    # plt.plot(test_x[0, :], test_y, 'g|', label='test data', markersize=13)
    # plt.legend(loc="lower left")

    # plt.figure(5)
    # plt.subplot(2, 2, 2)
    # plt.plot(data_x[index], pred_y[0, index], 'k-', label='decision function')
    # plt.plot(train_x[0, :], train_y, 'r+', label='train data', markersize=13)
    # plt.plot(val_x[0, :], val_y, 'b2', label='val data', markersize=13)
    # plt.plot(test_x[0, :], test_y, 'g|', label='test data', markersize=13)
    # plt.legend(loc="lower left")

    # plt.show()
    # build the optimizer
    optimizer = optim.Adam(adp_model.parameters(), lr=1e-1)
    optimizer.zero_grad()
    # calculate total training index
    max_iters = 3000
    batch_size = 16
    train_num = len(val_y)
    train_indexes = []
    while len(train_indexes) < max_iters * batch_size:
        train_index = np.arange(0, train_num)
        train_index = np.random.permutation(train_index)
        train_indexes.extend(train_index.tolist())

    train_indexes = train_indexes[: max_iters * batch_size]
    # train the Kernel
    loss_list = []
    adp_model.train()
    for iter_id in range(max_iters):
        # scheduler.step()
        ind_tmp = train_indexes[iter_id*batch_size:(iter_id+1)*batch_size]
        val_pred = adp_model(x_train=val_x[:, ind_tmp])
        val_loss = adp_model.rsse_loss(pred=val_pred, target=val_y[ind_tmp])  # or rsse_loss
        optimizer.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(adp_model.parameters(), max_norm=10, norm_type='inf')
        optimizer.step()
        if iter_id == 0 or iter_id % 100 == 99:
            print('[{}] loss={}'.format(iter_id, val_loss))
            for name, params in adp_model.named_parameters():
                print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())

        tmp = val_loss.detach().numpy()
        loss_list.append(tmp)
    # test
    adp_model.eval()
    #
    # plt.figure(2)
    # plt.subplot(3, 2, 2)
    pred_y_list = []
    print('the train error rate loss: ')
    pred_y_list.append(testFun(adp_model, train_x, train_y))
    print('the validation error rate loss: ')
    pred_y_list.append(testFun(adp_model, val_x, val_y))
    print('the test error rate loss: ')
    pred_y_list.append(testFun(adp_model, test_x, test_y))
    pred_y = torch.cat(pred_y_list, 0).reshape(-1)
    data_x = torch.cat((train_x[0, :], val_x[0, :], test_x[0, :]), 0)
    sorted, index = torch.sort(data_x)
    # plt.plot(data_x[index], pred_y[index], 'k-', label='decision function')
    # plt.plot(train_x[0,:], train_y, 'r+', label='train data', markersize=13)
    # plt.plot(val_x[0, :], val_y, 'b2', label='val data', markersize=13)
    # plt.plot(test_x[0, :], test_y, 'g|', label='test data', markersize=13)
    # plt.legend(loc="lower left")
    #
    # plt.subplot(3, 2, 3)
    # plt.plot(loss_list[1:], '-', label='loss norm')
    # plt.legend(loc="lower left")

    weight_mat = adp_model.weight.data
    return weight_mat, pred_y


def preprocessX(data_sv_x, data_train_x, data_test_x):
    for fea_id in range(data_sv_x.shape[0]):
        max_x = max([data_sv_x[fea_id, :].max(), data_train_x[fea_id, :].max()])
        min_x = min([data_sv_x[fea_id, :].min(), data_train_x[fea_id, :].min()])
        delta_x = max_x - min_x
        # print(delta_x)
        if delta_x == 0:
            delta_x = 1
        mid_x = (max_x + min_x) / 2
        data_sv_x[fea_id, :] = (data_sv_x[fea_id, :] - mid_x) / delta_x
        data_train_x[fea_id, :] = (data_train_x[fea_id, :] - mid_x) / delta_x
        data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mid_x) / delta_x
    data_sv_x = data_sv_x * 2
    data_train_x = data_train_x * 2
    data_test_x = data_test_x * 2

    return  data_sv_x, data_train_x, data_test_x


if __name__ == '__main__':
    # build the dataset
    dataset_con = SinDataset(100)  #StepDataset(200)#

    global delta_y
    delta_y = 2

    data_train_x, data_train_y = dataset_con.get_sv_data()
    data_val_x, data_val_y = dataset_con.get_val_data()
    data_test_x, data_test_y = dataset_con.get_test_data()

    train_x = torch.cat([data_train_x, data_val_x], dim=1)
    train_y = torch.cat([data_train_y, data_val_y], dim=0)

    weight_mat, pred_y = IniTrainKernel(data_train_x, data_val_x, data_train_y, data_val_y, data_test_x, data_test_y)
    print(weight_mat)

    plt.figure(5)
    ax = plt.subplot(2, 2, 4)
    data_x = torch.cat((train_x[0, :], data_test_x[0, :]), 0).cpu()
    sorted, index = torch.sort(data_x)
    plt.plot(data_x[index], pred_y[index], 'k-', label='Decision Function')
    plt.plot(data_train_x[0, :], data_train_y, 'ko', markerfacecolor='white',
             label='Support Data', markersize=13)
    plt.plot(data_val_x[0, :], data_val_y, 'k+', markersize=10)
    plt.plot(data_train_x[0, :], data_train_y, 'k+', markersize=10)
    plt.plot(data_test_x[0, :], data_test_y, 'k+', label='Sample Data', markersize=10)
    plt.legend(loc="lower left", fontsize=FONTSIZE)
    ax.set_title("(d) LAB RBF Kernel", fontsize=FONTSIZE)

    regressor = RBFRegressor(x_train=data_train_x , y_train=data_train_y , sigma=20, lamda=0.01)
    regressor.eval()

    plt.figure(5)
    pred_y_list = []
    print('the train error rate loss: ')
    tmp = testFun(regressor, train_x , train_y.double()).reshape(-1)
    pred_y_list.append(tmp.cpu())
    print('the test error rate loss: ')
    tmp = testFun(regressor, data_test_x , data_test_y.double()).reshape(-1)
    pred_y_list.append(tmp.cpu())

    pred_y = torch.cat(pred_y_list, 0).reshape(-1)
    ax = plt.subplot(2, 2, 2)
    plt.plot(data_x[index], pred_y[index], 'k-', label='Decision Function')
    plt.plot(data_train_x.reshape(-1).cpu(), data_train_y.cpu(), 'ko', markerfacecolor='white',
             label='Support Data', markersize=13)
    plt.plot(data_train_x.reshape(-1).cpu(), data_train_y.cpu(), 'k+', markersize=10)
    plt.plot(data_test_x.reshape(-1).cpu(), data_test_y.cpu(), 'k+', markersize=10)
    plt.plot(data_val_x.reshape(-1).cpu(), data_val_y.cpu(), 'k+', label='Sample Data',
             markersize=10)
    plt.legend(loc="lower left", fontsize=FONTSIZE)
    ax.set_title('(b) RBF Kernel  ($\Theta=20$)', fontsize=FONTSIZE)
    # Define Rectangle Parameters
    # rect = Rectangle((1.3, -1.0), 1.2, 2.1, edgecolor='red', facecolor='none', linewidth=3)
    # # Add Rectangle to Plot
    # ax.add_patch(rect)

    regressor = RBFRegressor(x_train=data_train_x , y_train=data_train_y , sigma=100, lamda=0.001)
    regressor.eval()

    plt.figure(5)
    ax = plt.subplot(2, 2, 3)
    pred_y_list = []
    print('the train error rate loss: ')
    tmp = testFun(regressor, train_x , train_y.double()).reshape(-1)
    pred_y_list.append(tmp.cpu())
    print('the test error rate loss: ')
    tmp = testFun(regressor, data_test_x , data_test_y.double()).reshape(-1)
    pred_y_list.append(tmp.cpu())

    pred_y = torch.cat(pred_y_list, 0).reshape(-1)
    data_x = torch.cat((train_x[0, :], data_test_x[0, :]), 0).cpu()
    sorted, index = torch.sort(data_x)
    plt.plot(data_x[index], pred_y[index], 'k-', label='Decision Function')
    plt.plot(data_train_x.reshape(-1).cpu(), data_train_y.cpu(), 'ko', markerfacecolor='white',
             label='Support Data', markersize=13)
    plt.plot(data_train_x.reshape(-1).cpu(), data_train_y.cpu(), 'k+', markersize=10)
    plt.plot(data_test_x.reshape(-1).cpu(), data_test_y.cpu(), 'k+', markersize=10)
    plt.plot(data_val_x.reshape(-1).cpu(), data_val_y.cpu(), 'k+', label='Sample Data',
             markersize=10)
    plt.legend(loc="lower left", fontsize=FONTSIZE)
    ax.set_title('(c) RBF Kernel ($\Theta=100$)', fontsize=FONTSIZE)
    # Define Rectangle Parameters
    # rect = Rectangle((0.05, -0.15), 1.1, 1.2, edgecolor='red', facecolor='none', linewidth=3)
    # # Add Rectangle to Plot
    # ax.add_patch(rect)
    plt.show()
