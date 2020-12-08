import numpy as np
import csv

import matplotlib.pyplot as plt


def get_data():
    path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\000017_.csv"

    with open(path, 'r') as file:
        temp = csv.DictReader(file)
        data_dir = [row['close'] for row in temp]

    # print(data_dir)
    # print(len(data_dir))
    input_number = 5
    data_x = []
    data_y = []
    data_dir.reverse()
    for i in range(input_number, len(data_dir)):
        temp_1 = []
        for each in range(i - 4, i):
            temp_1.append(float(data_dir[each]))
        data_x.append(temp_1)
        data_y.append(data_dir[i])
    # for l in range(len(data_x)):
    #     print(data_x[l])
    #     print(data_y[l])
    return data_x, data_y


def fun_activity(x, c, delta):
    """
    隐藏层神经元的激活函数
    :param x: 输入向量(一条数据)
    :param c: 向量（一个中心）
    :param delta: 一个数值
    :return: 一个隐藏层神经元的输出
    """
    t = 0
    for i in range(len(x)):
        t += round((float(x[i]) - round(float(c[i])))**2)
    hidden_out = np.e**(-1.0 * t / (2*delta * delta))

    return hidden_out


def get_error(w, data_y, data_x, b, c, delta):
    results_pre = get_results(data_x, w, b, c, delta)
    dif_pre_lab = []
    for i in range(len(data_y)):
        dif_pre_lab.append(float(results_pre[i]) - float(data_y[i]))
    return dif_pre_lab


def get_cost(w, data_y, data_x, b, c, delta):
    cost_ = 0
    er = get_results(data_x, w, b, c, delta)
    for i in range(len(data_y)):
        cost_ += (float(er[i]) - float(data_y[i]))**2
    return cost_


def get_results(data_x, w, b, c, delta):
    results = []
    n, m = np.shape(data_x)
    n1, m1 = np.shape(c)
    for j in range(n):
        result = 0
        for i in range(n1):
            result += w[i] * fun_activity(data_x[j], c[i], delta[i]) + b[i]
        results.append(result)
    return results


def rbf_train(hidden_number, input_data_x, input_data_y, max_steps, alpha):
    """

    :param hidden_number: 隐藏层数量
    :param input_data_x: 输入数据
    :param input_data_y: 数据标签
    :param max_steps: 最大迭代次数
    :return:
    """
    # 隐藏层数量 hidden_number; 输入数据input_data(数据input_data_x和标签input_data_y);
    n, m = np.shape(input_data_x)
    w = np.random.rand(hidden_number)
    b = np.random.rand(hidden_number)
    c = np.random.uniform(5, 7, (hidden_number, m))
    delta = np.random.rand(hidden_number)
    pso_w = PSO(100, 10, w, input_data_y, input_data_x, b, c, delta)
    w = pso_w.evolve()
    steps = 0
    while steps < max_steps:
        error = get_error(w, input_data_y, input_data_x, b, c, delta)
        # 反向传播
        # print(error)
        for hidden_ in range(hidden_number):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for i in range(n):
                sum1 += w[hidden_] * error[i] * np.e**(-1 * (np.linalg.norm(input_data_x[i] - c[hidden_]))**2 / 2
                                                       * (delta[hidden_])**2) * (input_data_x[i] - c[hidden_])
                sum2 += w[hidden_] * error[i] * np.e**(-1 * (np.linalg.norm(input_data_x[i] - c[hidden_]))**2 / 2
                                                       * (delta[hidden_])**2) * \
                        (np.linalg.norm((input_data_x[i] - c[hidden_]))) * delta[hidden_]**(-3)
                sum3 += error[i] * np.e**(-1 * (np.linalg.norm(input_data_x[i] - c[hidden_]))**2 / 2
                                          * (delta[hidden_])**2)
            # delta_center = (w[hidden_] / (delta[hidden_] * delta[hidden_])) * (sum1/n)
            # delta_delta = (w[hidden_] / (delta[hidden_] * delta[hidden_]
            #                                    * delta[hidden_])) * (sum2/n)
            # delta_w = (sum3/n)
            # 更新参数
            # w[hidden_] = w[hidden_] - alpha * delta_w
            # c[hidden_] = c[hidden_] - alpha * delta_center
            # delta[hidden_] = delta[hidden_] - alpha * delta_delta
            w[hidden_] = w[hidden_] - alpha * sum3 / n
            c[hidden_] = c[hidden_] - alpha * sum1 / n
            delta[hidden_] = delta[hidden_] - alpha * sum2 / n
        if steps % 10 == 0:
            cost = (1.0 / 2) * get_cost(w, input_data_y, input_data_x, b, c, delta)
            print("\t-------- iter: ", steps, " ,cost: ", cost)
            if cost < 3:
                # 如果损失函数值小于3则停止迭
                break

        results_pre = get_results(input_data_x, w, b, c, delta)
        plt.clf()
        plt.plot(input_data_y)
        # plt.plot(results_pre,)

        # plt.scatter(self.x[:, 0], self.x[:, 1], s=60, color='k')
        # plt.scatter(self.x[:, 0], self.x[:, 1], s=60, color='k')
        # plt.xlim(0, 50)
        # plt.ylim(4, 8)
        plt.pause(0.01)
        steps += 1
    plt.show()
    return delta, w, b, c


if __name__ == '__main__':
    from deep_python.Optimization_algorithm.NET_work.RBF_PSO import PSO
    # from deep_python.Optimization_algorithm.NET_work.RBF_PSO_w_c import PSO
    data_x, data_y = get_data()
    delta, w, b, c = rbf_train(8, data_x, data_y, 40000, 0.05)
