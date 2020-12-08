import numpy as np
import copy
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    :param x:
    :return:
    """
    # return 1/(1+np.exp(-x))
    return x


def BP_train(A, Y, hidden_num, ratio, step_max):
    n, m = np.shape(A)
    A = np.array(A)
    step = 0
    # w_mid = np.random.rand(hidden_num, m)
    w_mid = [[0.25, 0.45, 0.62, 0.17], [0.44, 0.36, 0.75, 0.83], [0.12, 0.47, 0.53, 0.63]]
    # w_out = np.random.rand(hidden_num)
    w_out = [0.44, 0.35, 0.72]
    # delta_w_mid = np.zeros([hidden_num, m])
    # delta_b_mid = np.zeros([hidden_num])
    # b_mid = np.random.rand(hidden_num)
    b_mid = [0.5, 0.7, 0.4]
    while step < step_max:
        res = []
        step += 1
        out_in = np.zeros([n, hidden_num])
        # err = np.zeros([n])
        # 进行训练
        cost_ = 0
        # delta_w_out = 0
        # delta_b_out = 0
        for j in range(n):
            # 对第j个对象进行训练
            net_in = copy.deepcopy(A[j])
            # real = copy.deepcopy(Y[j])
            # 3.当前对象进行训练
            for i in range(hidden_num):

                out_in[j][i] = sigmoid(np.sum(np.array(net_in) * np.array(w_mid[i])) + b_mid[i])
                # w_mid[:, i]:指第i行
            res.append(np.sum(out_in[j] * w_out))
        errors = np.array(res) - np.array(Y)
        # 反向传播
        for i in range(hidden_num):
            delta_w_out = 0
            delta_w_mid = 0
            delta_b_mid = 0
            for j in range(n):
                # net_in = copy.deepcopy(A[j])
                # delta_w_out += out_in[j][i] #* errors[j]
                delta_w_out += out_in[j][i] * errors[j]
                delta_w_mid += A[j] * w_out[i] * errors[j]
                # * errors[j]
                delta_b_mid += w_out[i] * errors[j]
                # errors[j] *
            print(delta_w_mid)
            w_out[i] = w_out[i] - ratio * delta_w_out / n
            w_mid[i] = w_mid[i] - ratio * delta_w_mid / n
            b_mid[i] = b_mid[i] - ratio * delta_b_mid / n
            # delta_w_out += ratio * res[j] * (1 - res[j]) * (real - res[j]) * out_in
            # print("==============3=======")
            # print(np.sum(out_in))
            # print(delta_w_out)

                # 权值调整
                # delta_w_mid[i] += ratio * out_in[i] * (1 - out_in[i]) * w_out[i] * res[j] * (1 - res[j]) * (
                #              real - res[j]) * net_in
                # 阈值调整
        # print(type(w_mid[1]))
        # print(w_mid[1])
        # print()
        # print()
        for t in range(n):
            cost_ += errors[t] ** 2
        if step % 10 == 0:
            print("\t-------- iter: ", step, " ,cost: ", cost_)
            if cost_ < 0.01:
                # 如果损失函数值小于3则停止迭代
                break
        plt.clf()
        plt.plot(Y)
        plt.plot(res)
        plt.pause(0.2)
    plt.show()


if __name__ == "__main__":
    a1 = [6.26, 6.31, 6.29, 6.45]
    a2 = [6.31, 6.29, 6.45, 6.32]
    a3 = [6.29, 6.45, 6.32, 6.29]
    a4 = [6.45, 6.32, 6.29, 6.23]
    a5 = [6.32, 6.29, 6.23, 6.25]
    Y = [6.32, 6.29, 6.23, 6.25, 6.24]
    A = [a1, a2, a3, a4, a5]
    # w_h_1 = np.random.uniform(0, 1, (3, len(a1)))

    # w_h_2 =np.random.rand(3)

    BP_train(A, Y, 3, 0.1, 5000)
# print(y_pre)
# print(h_1)
# print(h_2)