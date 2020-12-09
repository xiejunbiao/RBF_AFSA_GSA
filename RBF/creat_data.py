import csv
import copy
import numpy as np


def get_data_test():
    # path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\000017_.csv"
    path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\close\\600519.csv"
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


if __name__ == "__main__":
    a, b, c, d = get_data_test()

    # print(a)
    print(np.shape(a))
    # print(b)
    # print(c)
    # print(d)
