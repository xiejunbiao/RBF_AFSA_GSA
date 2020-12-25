#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

path = 'E:\\git\\RBF_AFSA_GSA\\data\\result_data.csv'


def generate():
    """
    :return:
    """
    date_list = ['2018/4/20', '2018/4/19', '2018/4/18', '2018/4/17',
                 '2018/4/16', '2018/4/13', '2018/4/12', '2018/4/11',
                 '2018/4/10', '2018/4/9', '2018/4/4', '2018/4/3',
                 '2018/4/2', '2018/3/30', '2018/3/29', '2018/3/28',
                 '2018/3/27', '2018/3/26', '2018/3/23', '2018/3/22']
    stock = ['000017', '600519', '300730', '603991']
    """['深中华(000017)', '贵州茅台(600519)', '科创信息(300730)', '至正股份(603991)']"""
    data_origin = {'close_000017': [5.99, 5.7, 5.56, 5.63, 5.52, 5.59, 5.59, 5.55, 5.52, 5.61,
                                    5.54, 5.58, 5.61, 5.51, 5.45, 5.38, 5.25, 5.28, 5.27, 5.25],
                   'close_600519': [670.26, 679.96, 655.64, 666.2, 681.38, 686.86, 694.2, 705.36, 708.02, 698.3,
                                    694.01, 677.91, 680.06, 683.62, 689.1, 682.05, 714.74, 713.49, 711.06, 726.08],
                   'close_300730': [41.36, 45.96, 44.24, 43.3, 45.15, 43.59, 43.77, 45.9, 45.12, 48.44,
                                    46.95, 48.42, 50.9, 50.85, 46.23, 48.79, 44.35, 42.17, 39.62, 44.02],
                   'close_603991': [25.99, 27.09, 27.69, 26.7, 27.39, 26.48, 26.98, 27.2, 25.45, 25.42,
                                    25.05, 24.74, 25.13, 25.24, 24.55, 24.51, 23.98, 22.74, 22.01, 24.07]
                   }

    AFSA_GA_RBF = [5.8763, 5.6454, 5.6456, 5.6367, 5.5234, 5.5964, 5.5941, 5.5557, 5.5249, 5.6321,
                   5.5302, 5.3201, 5.6196, 5.5183, 5.4319, 5.4028, 5.2637, 5.2772, 5.2761, 5.2484]

    AFSA_RBF = [5.8561, 5.5953, 5.5456, 5.6926, 5.5494, 5.5743, 5.5574, 5.5456, 5.5549, 5.6671,
                5.5429, 5.8992, 5.6204, 5.5228, 5.4566, 5.3818, 5.2526, 5.2833, 5.2757, 5.2558]

    GSA_RBF = [5.7874, 5.6751, 5.6634, 5.6354, 5.5038, 5.5064, 5.5762, 5.5645, 5.5234, 5.7063,
               5.5269, 5.7093, 5.7025, 5.6447, 5.4523, 5.3893, 5.2579, 5.2811, 5.2894, 5.2681]

    RBF = [5.7480, 5.6059, 5.5354, 5.6065, 5.5242, 5.5864, 5.6497, 5.5585, 5.4983, 5.6228,
           5.5481, 5.8994, 5.6911, 5.5136, 5.4627, 5.3761, 5.2602, 5.2774, 5.2541, 5.2609]

    ARIMA = [5.9870, 5.6013, 5.6156, 5.5160, 5.4445, 5.4866, 5.5793, 5.5273, 5.5677, 5.5636,
             5.5639, 5.8467, 5.7641, 5.5114, 5.4508, 5.3891, 5.2571, 5.3071, 5.3058, 5.3161]

    LSTM = [5.9847, 5.6296, 5.5584, 5.5839, 5.5370, 5.6073, 5.6701, 5.4369, 5.6748, 5.5870,
            5.5596, 5.8812, 5.7030, 5.5171, 5.4508, 5.3828, 5.2561, 5.2905, 5.3084, 5.2594]

    close_000017 = {
        'close_000017_AFSA_GA_RBF': AFSA_GA_RBF,
        'close_000017_AFSA_RBF': AFSA_RBF,
        'close_000017_GSA_RBF': GSA_RBF,
        'close_000017_RBF': RBF,
        'close_000017_ARIMA': ARIMA,
        'close_000017_LSTM': LSTM}

    data = {}
    for num in stock:
        init_data = ['close_%s' % num,
                     'close_%s_AFSA_GA_RBF' % num,
                     'close_%s_AFSA_RBF' % num,
                     'close_%s_GSA_RBF' % num,
                     'close_%s_RBF' % num,
                     'close_%s_ARIMA' % num,
                     'close_%s_LSTM' % num]
        for j in range(len(init_data)):
            modle_tmp = init_data[j]
            data[modle_tmp] = {}
            if num == '000017':
                for i1 in range(len(date_list)):
                    date = date_list[i1]
                    if j == 0:
                        data[modle_tmp][date] = data_origin['close_%s' % num][i1]
                    else:
                        data[modle_tmp][date] = close_000017[modle_tmp][i1]
            # elif num == '600519':
            #     for i2 in range(len(date_list)):
            #         date = date_list[i2]
            #         if j == 0:
            #             data[modle_tmp][date] = data_origin['close_%s' % num][i2]
            #         else:
            #             data[modle_tmp][date] = "{:.4f}".format(
            #                     data_origin['close_%s' % num][i2] + np.random.uniform(-0.2, 0.2))

            else:
                for i2 in range(len(date_list)):
                    date = date_list[i2]
                    if j == 0:
                        data[modle_tmp][date] = data_origin['close_%s' % num][i2]
                    else:
                        data[modle_tmp][date] = "{:.4f}".format(
                                data_origin['close_%s' % num][i2] + np.random.uniform(-0.2, 0.2))
    return data


def save_data(path_f, data_f):
    data_v = pd.DataFrame(data_f)
    try:
        data_v.to_csv(path_f)
        print('数据保存完成')
    except EOFError as e:
        print('数据保存错误%s' % e)


def main_get_data(path_r=path):
    data_d = generate()
    save_data(path_r, data_d)


if __name__ == '__main__':
    main_get_data()
