import matplotlib.pyplot as plt
import numpy as np


def printResult():
    """
    结果显示：作图
    :return:
    """

    data1 = []
    data2 = []

    path_data2 = "E:\\研究生\\期刊\\谢俊标\\论文_构建中。。\\" \
                 "小论文-股票预测\\pic\\损失迭代曲线\\data3.txt"
    with open(path_data2, encoding='utf-8') as f:
        data = f.readlines()
        for i in data:
            tmp = str(i).split('\t')
            data1.append(float(tmp[0]))
            data2.append(float(tmp[1]))
    f.close()
    print(data2)
    # f.writelines(str(y2))

    plt.plot(data1, 'k-')
    plt.plot(data2, 'k:')
    size = 12
    plt.legend(['optimal value', 'average value'], fontsize=size + 5)
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.xlabel("Iteration", font2)
    plt.ylabel("function value", font2)
    # plt.title("GSA+AFSA")
    plt.show()


if __name__ == '__main__':
    printResult()
