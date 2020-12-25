import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)
# from deep_python.Optimization_algorithm.NET_work.RBF.rbf_train import get_predict
from RBF.rbf_train import get_predict

import copy


def load_model(file_center, file_delta, file_w):
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            print(lines)
            for x in lines:
                print(x)
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return model

    center = get_model(file_center)
    # 2、导入rbf函数扩展常数
    delta = get_model(file_delta)
    # 3、导入隐含层到输出层之间的权重
    w = get_model(file_w)
    return center, delta, w


def save_predict(file_name, pre):
    """
    保存最终的预测结果
    :param file_name:
    :param pre: 最终的预测结果
    :return:
    """
    f = open(file_name, "w")
    m = np.shape(pre)[0]
    result = []
    for i in range(m):
        result.append(str(pre[i, 0]))
        f.write("\n".join(result))
    f.close()


def get_data_test():
    # path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\000017_.csv"
    path_10 = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\close\\000017.csv"
    # path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\close\\000017.csv"
    path = "E:\\git\\RBF_AFSA_GSA\\data\\000017.SZ\\000017.SZ.csv"
    with open(path, 'r') as file:
        temp = csv.DictReader(file)
        data_dir1 = [row['close'] for row in temp]
    with open(path, 'r') as file1:
        temp1 = csv.DictReader(file1)
        # data_dir2 = [row['volume'] for row in temp1]
        data_dir2 = [row['vol'] for row in temp1]

    # 定义使用前几天的数据来进行预测
    # number_day = 10
    number_day = 20
    input_number = 15
    data_dir1.reverse()
    data_dir2.reverse()
    data_x_number_day = []
    data_y_number_day = []
    test_x_number_day = []
    test_y_number_day = []

    for j in range(number_day):
        data_x = []
        data_y = []
        for i in range(input_number, len(data_dir1[j: 1 + j + len(data_dir1) - number_day])):
            data_dir1_1 = copy.deepcopy(data_dir1[j: 1 + j + len(data_dir1) - number_day])
            temp_1 = []
            for each in range(i - input_number, i):
                temp_1.append(float(data_dir1_1[each]))

            # 将成交量加入数据集
            # for each2 in range(i - input_number, i):
            #     temp_1.append(float(data_dir2[each2])/10000)
            data_x.append(temp_1)
            data_y.append(float(data_dir1_1[i]))
        test_x = data_x.pop()
        test_y = data_y.pop()
        data_x_number_day.append(data_x)
        data_y_number_day.append(data_y)
        test_x_number_day.append(test_x)
        test_y_number_day.append(test_y)
    return data_x_number_day, data_y_number_day, test_x_number_day, test_y_number_day


def predict_num_day(number):
    # result = []
    data_x, data_y, test_x, test_y = get_data_test()

    print('获取数据成功')
    # for i in range(number):
    #
    #     center, delta, weight = load_model("center_%d.csv.txt" % i, "delta_%d.csv.txt" % i, "weight_%d.csv.txt" % i)
    #
    #     result.append(get_predict(weight, center, delta, test_x[i]))
    # print(result)
    print(test_y)
    # [         5.99,   5.7,    5.56,   5.63,   5.52,   5.59,   5.59,   5.55,   5.52,   5.61, ==  5.54, 5.58, 5.61, 5.51, 5.45, 5.38, 5.25, 5.28, 5.27, 5.25]
    result1 = [5.8763, 5.6454, 5.6456, 5.6367, 5.5234, 5.5964, 5.5941, 5.5557, 5.5249, 5.6321,
               5.5302, 5.3201, 5.6196, 5.5183, 5.4319, 5.4028, 5.2637, 5.2772, 5.2761, 5.2484]


    result2 = [5.8561, 5.5953, 5.5456, 5.6926, 5.5494, 5.5743, 5.5574, 5.5456, 5.5549, 5.6671,
               5.5429, 5.8992, 5.6204, 5.5228, 5.4566, 5.3818, 5.2526, 5.2833, 5.2757, 5.2558]

              # 5.54,    5.58,   5.61,   5.51,  5.45,   5.38,   5.25,   5.28,   5.27,   5.25
    result3 = [5.7874, 5.6751, 5.6634, 5.6354, 5.5038, 5.5064, 5.5762, 5.5645, 5.5234, 5.7063,
               5.5269, 5.7093, 5.7025, 5.6447, 5.4523, 5.3893, 5.2579, 5.2811, 5.2894, 5.2681]
              # 5.54,    5.58,   5.61,   5.51,  5.45,   5.38,   5.25,   5.28,   5.27,   5.25
    result4 = [5.7480, 5.6059, 5.5354, 5.6065, 5.5242, 5.5864, 5.6497, 5.5585, 5.4983, 5.6228,
               5.5481, 5.8994, 5.6911, 5.5136, 5.4627, 5.3761, 5.2602, 5.2774, 5.2541, 5.2609]
              # 5.54,    5.58,   5.61,   5.51,  5.45,   5.38,   5.25,   5.28,   5.27,   5.25
    result5 = [5.9870, 5.6013, 5.6156, 5.5160, 5.4445, 5.4866, 5.5793, 5.5273, 5.5677, 5.5636,
               5.5639, 5.8467, 5.7641, 5.5114, 5.4508, 5.3891, 5.2571, 5.3071, 5.3058, 5.3161]
              # 5.54,    5.58,   5.61,   5.51,  5.45,   5.38,   5.25,   5.28,   5.27,   5.25
    result6 = [5.9847, 5.6296, 5.5584, 5.5839, 5.5370, 5.6073, 5.6701, 5.4369, 5.6748, 5.5870,
               5.5596, 5.8812, 5.7030, 5.5171, 5.4508, 5.3828, 5.2561, 5.2905, 5.3084, 5.2594]
    """
    result1 = [5.8763, 5.6454, 5.6456, 5.6567, 5.5134, 5.5764, 5.6341, 5.5457, 5.5049, 5.6321]
    result2 = [5.7874, 5.6751, 5.6634, 5.6354, 5.5038, 5.5064, 5.5762, 5.5645, 5.5234, 5.7063]
    result3 = [5.8561, 5.5953, 5.5456, 5.6926, 5.5494, 5.5743, 5.5574, 5.5456, 5.5549, 5.6671]
    result4 = [5.7480, 5.6059, 5.5354, 5.6065, 5.5242, 5.5864, 5.6497, 5.5585, 5.4983, 5.6228]
    result5 = [5.9870, 5.6013, 5.6156, 5.5160, 5.4445, 5.4866, 5.5793, 5.5273, 5.5677, 5.5636]
    result6 = [5.9847, 5.6296, 5.5584, 5.5839, 5.5370, 5.6073, 5.6701, 5.4369, 5.6748, 5.5870]
    """
    # p = np.array(result) - np.array(test_y)
    p1 = np.array(result1) - np.array(test_y)
    p2 = np.array(result2) - np.array(test_y)
    p3 = np.array(result3) - np.array(test_y)
    p4 = np.array(result4) - np.array(test_y)
    print(result5)
    print(test_y)
    p5 = np.array(result5) - np.array(test_y)
    p6 = np.array(result6) - np.array(test_y)
    cost1 = 0
    cost2 = 0
    cost2_1 = 0
    cost2_2 = 0

    cost3_1 = 0
    cost3_2 = 0

    cost4_1 = 0
    cost4_2 = 0

    cost5_1 = 0
    cost5_2 = 0

    cost6_1 = 0
    cost6_2 = 0

    cost7_1 = 0
    cost7_2 = 0

    print('预测结果')
    # for i in range(len(result)):
    #     cost1 += (p[i] ** 2) ** (1 / 2)
    #     cost2 += np.abs(p[i]/test_y[i])
    #     print(cost2*100)
    # 平均误差
    len_data = len(result1)
    for i in range(len_data):
        cost2_1 += (p1[i] ** 2)
        cost2_2 += np.abs(p1[i]/test_y[i])
        print(np.abs(p1[i]/test_y[i])*100)
    print('--------------------------')
    for i in range(len_data):
        cost3_1 += (p2[i] ** 2)
        cost3_2 += np.abs(p2[i]/test_y[i])
        print(np.abs(p2[i]/test_y[i]) * 100)
    print('--------------------------')
    for i in range(len_data):
        cost4_1 += (p3[i] ** 2)
        cost4_2 += np.abs(p3[i]/test_y[i])
        print(np.abs(p3[i]/test_y[i]) * 100)
    print('--------------------------')
    for i in range(len_data):
        cost5_1 += (p4[i] ** 2)
        cost5_2 += np.abs(p4[i]/test_y[i])
        print(np.abs(p4[i]/test_y[i]) * 100)
    print('--------------------------')
    for i in range(len_data):
        cost6_1 += (p5[i] ** 2)
        cost6_2 += np.abs(p5[i]/test_y[i])
        if i in [9, 10, 11]:
            print("------%s" % p5[i])
        print(np.abs(p5[i]/test_y[i]) * 100)
    print('--------------------------')
    for i in range(len(result1)):
        cost7_1 += (p6[i] ** 2)
        cost7_2 += np.abs(p6[i]/test_y[i])
        print(np.abs(p6[i]/test_y[i]) * 100)
    print("--------------RMSE------------")
    print("GSA+AFSA+RBF:RMSE\t=" + '\t%f' % (cost2_1/len_data)**(1/2))
    print("AFSA+RBF:RMSE\t=" + '\t%f' % (cost3_1 / len_data) ** (1 / 2))
    print("GSA+RBF:RMSE\t=" + '\t%f' % (cost4_1 / len_data) ** (1 / 2))
    print("RBF:RMSE\t=" + '\t%f' % (cost5_1 / len_data) ** (1 / 2))
    print("ARIMS:RMSE\t=" + '\t%f' % (cost6_1 / len_data)**(1 / 2))
    print("LSTM:RMSE\t=" + '\t%f' % (cost7_1 / len_data)**(1 / 2))
    print("--------------MRE------------")
    print("GSA+AFSA+RBF:MRE\t=" + '\t%f' % (cost2_2 / len_data))
    print("AFSA+RBF:MRE\t=" + '\t%f' % (cost3_2 / len_data))
    print("GSA+RBF:MRE\t=" + '\t%f' % (cost4_2 / len_data))
    print("RBF:MRE\t=" + '\t%f' % (cost5_2 / len_data))
    print("ARIMS:MRE\t=" + '\t%f' % (cost6_2 / len_data))
    print("LSTM:MRE\t=" + '\t%f' % (cost7_2 / len_data))


    dates = ('3/22',
             '3/23',
             '3/26',
             '3/27',
             '3/28',
             '3/29',
             '3/30',
             '4/02',
             '4/03',
             '4/04',
             '4/09',
             '4/10',
             '4/11',
             '4/12',
             '4/13',
             '4/16',
             '4/17',
             '4/18',
             '4/19',
             '4/20')
    size = 6
    txt = ['AFSA_GA_RBF', 'AFSA_RBF', 'GSA_RBF', 'RBF', 'ARIMA', 'LSTM']
    plt.clf()
    plt.bar(np.arange(6), [(cost2_1 / len_data)**(1 / 2),
                           (cost3_1 / len_data)**(1 / 2),
                           (cost4_1 / len_data)**(1 / 2),
                           (cost5_1 / len_data)**(1 / 2),
                           (cost6_1 / len_data)**(1 / 2),
                           (cost7_1 / len_data)**(1 / 2)
                           ], label=txt)

    plt.show()

    plt.clf()
    plt.bar(np.arange(6), [cost2_2 / len_data,
                           cost3_2 / len_data,
                           cost4_2 / len_data,
                           cost5_2 / len_data,
                           cost6_2 / len_data,
                           cost7_2 / len_data
                           ], label=txt)

    plt.show()

    plt.clf()
    plt.xticks(np.arange(20), dates, rotation=0, fontsize=size)
    plt.yticks(fontsize=size)
    plt.plot(test_y, 'k-')
    # plt.plot(result1, 'ro--')
    # plt.plot(result2, 'ro--')
    # plt.plot(result3, 'ro--')
    plt.plot(result4, 'k:')
    # plt.legend(['True value', 'GSA+AFSA+RBF', 'GSA+RBF', 'AFSA+RBF', 'RBF'], fontsize=size)
    # plt.legend(['True value', 'GSA+AFSA+RBF', 'GSA+RBF'], fontsize=size+5)
    # plt.legend(['True value', 'GSA+AFSA+RBF', 'AFSA+RBF'], fontsize=size+5)
    # plt.xlabel()
    # plt.legend(['True value', 'GSA+AFSA+RBF', 'RBF'], fontsize=size+5)

    # plt.legend(['True value', 'GSA+AFSA+RBF'], fontsize=size)
    # plt.legend(['True value', 'AFSA+RBF'], fontsize=size + 5)
    plt.legend(['True value', 'GSA+RBF'], fontsize=size+5)

    # plt.xlabel()
    plt.legend(['True value', 'RBF'], fontsize=size+5)

    plt.show()

    # plt.clf()
    # plt.plot(test_y)
    # plt.plot(result)
    # plt.show()


if __name__ == "__main__":
    predict_num_day(10)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}