import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


def sigmoid(x):
    """
    隐含层和输出层对应的函数法则
    """
    return 1/np.sum(1+np.exp(-x))


def BP(data_tr, data_te, hidden_number, maxiter=600):
    # --pandas是基于numpy设计的，效率略低
    # 为提高处理效率，转换为数组
    data_tr, data_te = np.array(data_tr), np.array(data_te)
    n, m = np.shape(data_tr)
    # --隐层输入
    # -1： 代表的是隐层的阈值
    net_in = np.array([0.0, 0, -1])
    w_mid = np.random.rand(3, 4)  # 隐层权值阈值（-1x其中一个值：阈值）

    # 输出层输入
    # -1：代表输出层阈值
    out_in = np.array([0.0, 0, 0, 0, -1])
    w_out = np.random.rand(hidden_number+1)  # 输出层权值阈值（-1x其中一个值：阈值）
    delta_w_out = np.zeros([5])  # 存放输出层权值阈值的逆向计算误差
    delta_w_mid = np.zeros([3, hidden_number])  # 存放因此能权值阈值的逆向计算误差
    yita = 1.75  # η： 学习速率
    Err = np.zeros([maxiter])  # 记录总体样本每迭代一次的错误率

    # 1.样本总体训练的次数
    for it in range(maxiter):

        # 衡量每一个样本的误差
        err = np.zeros([len(data_tr)])

        # 2.训练集训练一遍
        for j in range(n):
            net_in[:m] = data_tr[j]  # 存储当前对象前两个属性值
            # real = data_tr[j, 2]
            real = data_te[j]

            # 3.当前对象进行训练
            for i in range(4):
                out_in[i] = sigmoid(np.sum(net_in * w_mid[:, i]))  # 计算输出层输入
            res = sigmoid(np.sum(out_in * w_out))  # 获得训练结果

            err[j] = abs(real - res)

            # --先调节输出层的权值与阈值
            delta_w_out = yita * res * (1 - res) * (real - res) * out_in  # 权值调整
            delta_w_out[4] = -yita * res * (1 - res) * (real - res)  # 阈值调整
            w_out = w_out + delta_w_out

            # --隐层权值和阈值的调节
            for i in range(hidden_number):
                # 权值调整
                delta_w_mid[:, i] = yita * out_in[i] * (1 - out_in[i]) * w_out[i] * res * (1 - res) * (
                            real - res) * net_in
                # 阈值调整
                delta_w_mid[2, i] = -yita * out_in[i] * (1 - out_in[i]) * w_out[i] * res * (1 - res) * (real - res)
            w_mid = w_mid + delta_w_mid
        Err[it] = err.mean()
    plt.plot(Err)
    plt.show()

    # 存储预测误差
    err_te = np.zeros([100])

    # 预测样本100个
    # for j in range(100):
    #     net_in[:2] = data_te[j, :2]  # 存储数据
    #     real = data_te[j, 2]  # 真实结果
    #
    #     # net_in和w_mid的相乘过程
    #     for i in range(4):
    #         # 输入层到隐层的传输过程
    #         out_in[i] = sigmoid(sum(net_in * w_mid[:, i]))
    #     res = sigmoid(sum(out_in * w_out))  # 网络预测结果输出
    #     err_te[j] = abs(real - res)  # 预测误差
    #     print('res:', res, ' real:', real)
    #
    # plt.plot(err_te)
    # plt.show()


def get_data():
    path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\000017_.csv"

    with open(path, 'r') as file:
        temp = csv.DictReader(file)
        data_dir = [row['close'] for row in temp]
    input_number = 15
    data_x = []
    data_y = []
    data_dir.reverse()
    for i in range(input_number, len(data_dir)):
        temp_1 = []
        for each in range(i - input_number, i):
            temp_1.append(float(data_dir[each]))
        data_x.append(temp_1)
        data_y.append(float(data_dir[i]))
    return data_x, data_y


def cder():
    w_mid = np.random.rand(3, 4)
    print(w_mid)
    print(w_mid[:1])
    a1 = [6.26, 6.31, 6.29, 6.45]
    return np.array(a1) * w_mid[:1]


if "__main__" == __name__:
    # print(1 / (1 + np.e**(-15.797)))
    print()

    # datr = cder()
    # print(datr)
    # print("============")
    # print(np.sum(datr))
    # 1.读取样本
    # data_x, data_y = get_data()
    # # data_tr = pd.read_csv("5.2 data_tr.txt")
    # # data_te = pd.read_csv("5.2 data_tr.txt")
    # BP(data_x, data_y, 20, maxiter=600)
