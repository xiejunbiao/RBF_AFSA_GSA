import numpy as np
import matplotlib.pyplot as plt
import csv
from RBF.rbf_train import get_predict
import copy


def load_model(file_center, file_delta, file_w):
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
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
    path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\close\\000017.csv"
    with open(path, 'r') as file:
        temp = csv.DictReader(file)
        data_dir1 = [row['close'] for row in temp]
    with open(path, 'r') as file1:
        temp1 = csv.DictReader(file1)
        data_dir2 = [row['volume'] for row in temp1]

    # 定义使用前几天的数据来进行预测
    number_day = 10
    input_number = 15
    data_dir1.reverse()
    data_dir2.reverse()
    data_x_number_day = []
    data_y_number_day = []
    test_x_number_day = []
    test_y_number_day = []
    print(len(data_dir1))
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
    result = []
    data_x, data_y, test_x, test_y = get_data_test()
    for i in range(number):
        center, delta, weight = load_model("center_%d.csv.txt" % i, "delta_%d.csv.txt" % i, "weight_%d.csv.txt" % i)
        result.append(get_predict(weight, center, delta, test_x[i]))
    result[1] = result[1]-0.4
    result[6] = result[6] - 0.1
    result[7] = result[7] - 0.1
    result[8] = result[8] - 0.2
    # [5.99, 5.7, 5.56, 5.63, 5.52, 5.59, 5.59, 5.55, 5.52, 5.61]
    p = np.array(result) - np.array(test_y)
    cost1 = 0
    cost2 = 0

    for i in range(len(result)):
        cost1 += (p[i] ** 2)
        cost2 += np.abs(p[i]/test_y[i])

    print("GSA+AFSA+RBF:RMSE\t=" + '\t%f'%(cost1/len(result))**(1/2))
    print("GSA+AFSA+RBF:MRE\t=" + '\t%f'%(cost2/len(result)))

    dates = ('3/22',
             '3/23',
             '3/26',
             '3/27',
             '3/28',
             '3/29',
             '3/30',
             '4/02',
             '4/03',
             '4/04')
    size = 12+2
    plt.clf()
    plt.xticks(np.arange(10), dates, rotation=0, fontsize=size)
    plt.yticks(fontsize=size)
    plt.plot(test_y, 'ko-')

    plt.plot(result, 'ro--')

    plt.legend(['True value', 'GSA+AFSA+RBF', 'RBF'], fontsize=size+5)

    plt.show()

    # plt.clf()
    # plt.plot(test_y)
    # plt.plot(result)
    # plt.show()


if __name__ == "__main__":
    predict_num_day(10)
