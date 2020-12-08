import numpy as np
import matplotlib.pyplot as plt
import csv


def get_data():
    path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\000017_.csv"

    with open(path, 'r') as file:
        temp = csv.DictReader(file)
        data_dir1 = [row['close'] for row in temp]
    with open(path, 'r') as file1:
        temp1 = csv.DictReader(file1)
        data_dir2 = [row['volume'] for row in temp1]
    input_number = 5
    data_x = []
    data_y = []
    data_dir1.reverse()
    data_dir2.reverse()
    for i in range(input_number, len(data_dir1)):

        temp_1 = []
        for each in range(i - input_number, i):
            temp_1.append(float(data_dir1[each]))

        # 将成交量加入数据集
        for each2 in range(i - input_number, i):
            temp_1.append(float(data_dir2[each2]))
        data_x.append(temp_1)
        data_y.append(float(data_dir1[i]))
    test_x = data_x.pop()
    test_y = data_y.pop()
    return data_x, data_y, test_x, test_y


print(get_data()[0])
print(get_data()[1])
print(get_data()[2])
print(get_data()[3])