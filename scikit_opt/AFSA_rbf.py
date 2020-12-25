#-*-coding:utf-8-*-
import numpy as np
import json
import sys
import os
import traceback
import copy
pathDir = os.path.dirname(__file__)
curPath = os.path.abspath(pathDir)
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)
from RBF.rbf_train import get_data_test
# from sko.AFSA import AFSA
from scikit_opt.func import FuncSet
from scikit_opt.AFSA import AFSA
def func(x):
    x1, x2 = x
    return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2




# def rbf_train_all(low, high, data_x_number_day, data_y_number_day, hidden_num, ratio, max_step):
#     n1, n2, n3 = np.shape(data_x_number_day)
#     for i in range(n1):
#         w, C, delta = evolve(low, high, data_x_number_day[i], data_y_number_day[i], hidden_num, ratio, max_step, i)
#         save_model_result(C, delta, w, i)


if __name__ == '__main__':
    # path_file = "E:\\git\\RBF_AFSA_GSA\\data\\300730.SZ\\300730.SZ.csv"
    path_file = "E:\\git\\RBF_AFSA_GSA\\data\\close\\000017.csv"
    data_x, data_y, test_x, test_y = get_data_test(path_file)


    hidden_num = 40
    m = 15
    d = hidden_num * (m + 2)
    bound = np.tile([[-10], [10]], d)
    func = FuncSet(data_x[0], data_y[0], para=[40, 15])
    afsa = AFSA(func.func_rbf, n_dim=d, size_pop=50, max_iter=300,
                max_try_num=100, step=0.2, visual=0.3, bound1=-10, bound2=10,
                q=0.98, delta=0.5)
    best_x, best_y = afsa.run()
    print(best_x, best_y)
