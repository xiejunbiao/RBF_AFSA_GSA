import matplotlib.pylab as plt
import pandas as pd
import numpy as np


pd.set_option('display.float_format', lambda x1: '%.6f' % x1)  # pandas
np.set_printoptions(precision=6, suppress=True)  # numpy
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# layout1 = (1, 2)
# data_train_ax = plt.subplot2grid(layout1, (0, 0))
# data_diff_ax = plt.subplot2grid(layout1, (0, 1))


def plot_2(name_list, data_list):
    # name_list = ['A', 'B', 'C', 'D']
    # name_list = name_list
    # num_list = [10, 15, 16, 28]
    num_list = data_list[0]
    # num_list2 = [10, 12, 18, 26]
    num_list2 = data_list[1]
    x = list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, num_list, width=width, label='MRE', fc='b')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, num_list2, width=width, label='RMSE', tick_label=name_list, fc='g')
    plt.legend()
    plt.show()


# 显示高度
def autolabel(rects):

    for rect in rects:

        # height = rect.get_height()
        height = 0.01
        # plt.text(rect.get_x()+rect.get_width()/2. - 0.25, 1.02*height, "{:.3f}".format(height))
        plt.text(4, 0.02, 'nihao')


def plot_1(name_list, num_list, num, type_):

    if type_ == 'MRE':
        type_n = 'MRE'
        num_list = [n * 100 for n in num_list]
    else:
        type_n = 'RMSE'
        # num_list = [n * 10 for n in num_list]
    type_n = num + '-' + type_n

    # name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # num_list = [33, 44, 53, 16, 11, 17, 17, 10]
    size = 9
    plt.xticks(np.arange(len(name_list)), name_list, rotation=0, fontsize=size)
    plt.yticks(fontsize=size)
    plt.title(type_n)
    # autolabel(plt.bar(range(len(num_list)), num_list, width=0.5, color='rgb'))
    autolabel(plt.plot(range(len(num_list)), num_list, 'ko-'))
    # autolabel(plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list))
    # plt.legend()
    plt.show()


def plot_3(name_list, mre_list1, rmse_list2, num, type_):
    num_list = [n * 100 for n in mre_list1]
    rmse_list2 = [n * 10 for n in rmse_list2]
    # if type_ == 'MRE':
    #     type_n = 'MRE'
    #     num_list = [n * 100 for n in mre_list1]
    # else:
    #     type_n = 'RMSE'
        # num_list = [n * 10 for n in num_list]
    # type_n = num + '-' + type_n

    # name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # num_list = [33, 44, 53, 16, 11, 17, 17, 10]
    size = 10
    plt.xticks(np.arange(len(name_list)), name_list, rotation=0, fontsize=size)
    plt.yticks(fontsize=size)
    # plt.title(type_n)
    # autolabel(plt.bar(range(len(num_list)), num_list, width=0.5, color='rgb'))
    plt.plot(range(len(num_list)), num_list, 'ko-')
    plt.plot(range(len(rmse_list2)), rmse_list2, 'ro--')
    plt.plot(range(len(rmse_list2)), [1.06, 1.06, 1.06, 1.06, 1.06, 1.06], 'w--')
    for i in range(len(name_list)):
        plt.text(i-0.05, num_list[i]+0.02, "{:.3f}".format(num_list[i]))
        plt.text(i-0.05, rmse_list2[i]+0.02, "{:.3f}".format(rmse_list2[i]))
    # autolabel(plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list))
    # plt.legend()

    # plt.xlabel()
    plt.legend(['MRE(%)', 'RMSE(/10)', num], fontsize=size)
    plt.show()


def get_result(path):
    data = pd.read_csv(path, index_col=0, parse_dates=[0])

    def get_rrmse(data_o, data_f):
        return np.sqrt((((data_f - data_o)/data_o)*((data_f - data_o)/data_o)).sum())/(data_o.shape[0])

    def get_mre(data_o, data_f):
        return (abs(data_f - data_o)/data_o).sum()/(data_o.shape[0])

    def get_rmse(data_o, data_f):
        return np.sqrt(((data_f - data_o)*(data_f - data_o)).sum()/(data_o.shape[0]))
        # return (abs(data_f - data_o)/data_o).sum()/data_o.shape[0]

    txt = ['AFSA_GA_RBF', 'AFSA_RBF', 'GSA_RBF', 'RBF', 'ARIMA', 'LSTM']

    def get_mre_rmse(number):
        # txt = ['AFSA_GA_RBF', 'AFSA_RBF', 'GSA_RBF', 'RBF', 'ARIMA', 'LSTM']
        index_o = 'close_' + number
        rmse = []
        rrmse = []
        mre = []
        for i in txt:
            rmse.append(get_rmse(data[index_o], data['close_' + str(number) + '_' + i]))
            mre.append(get_mre(data[index_o], data['close_' + str(number) + '_' + i]))
            rrmse.append(get_rrmse(data[index_o], data['close_' + str(number) + '_' + i]))
        return rmse, mre, rrmse

    # num_list = ['000017', '300730', '002628', '300144']
    num_list = ['000017', '300730', '002628', '300144']
    for each_num in num_list:
        rmse_list, mre_list, rrmse_list = get_mre_rmse(each_num)
        print('rrmse_list', rrmse_list)
        print('rmse_list', rmse_list)
        print(mre_list)
        # plot_2(txt, [mre_list, rmse_list])
        # plot_1(txt, mre_list, each_num, 'MRE')
        # plot_1(txt, rmse_list, each_num, 'RMSE')
        plot_3(txt, mre_list, rmse_list, each_num, 'RMSE')


if __name__ == '__main__':
    path_r = 'E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\预测结果过表.csv'
    print('__________11111111111111____________')
    get_result(path_r)
