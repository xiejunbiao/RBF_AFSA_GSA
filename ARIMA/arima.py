# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.tsa.api as smt

# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns

import pandas as pd
import numpy as np

pd.set_option('display.float_format', lambda x: '%.3f' % x)  # pandas
np.set_printoptions(precision=3, suppress=True)  # numpy
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


sns.set(style='ticks', context='poster')
# seaborn plotting style


def get_data(data_path):
    data = pd.read_csv(data_path, index_col=0, parse_dates=[0])
    data = pd.DataFrame(data['close'])
    data = data.sort_index()
    # ascending = True
    print(data.shape)
    print(data.head())
    return data


# Create a training sample and testing sample before analyzing the series
def splite_data(data):
    n_sample = data.shape[0]
    n_train = int(0.95 * n_sample) + 1
    # n_forecast = n_sample - n_train

    # ts_df
    ts_train = data.iloc[:n_train]['close']
    ts_test = data.iloc[n_train:]['close']

    return ts_train, ts_test


def diff_data(ts_train):
    data_diff = ts_train.diff(1)
    data_diff = data_diff.dropna()

    layout1 = (1, 2)
    data_train_ax = plt.subplot2grid(layout1, (0, 0))
    data_diff_ax = plt.subplot2grid(layout1, (0, 1))
    data_train_ax.plot(ts_train)
    data_train_ax.set_title('原数据')
    data_diff_ax.plot(data_diff)
    data_diff_ax.set_title('一阶差分')
    # fig.tight_layout()
    plt.show()
    sns.despine()
    print(data_diff.shape)
    return data_diff


# print('Training Series:', '\n', ts_train.tail(), '\n')
# print('Test Series:', '\n', ts_test.head())


def tsplot(y, lags=None, title='', figsize=(14, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogrm')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    plt.show()

    return ts_ax, acf_ax, pacf_ax


def get_p_q(data_diff):
    ts_ax, acf_ax, pacf_ax = tsplot(data_diff, lags=20, title='A Given Training Series')
    print(ts_ax, '\n', acf_ax, '\n', pacf_ax)


# Model Estimation
# Fit the model
def train_main(train_set, order):
    arima_200 = sm.tsa.SARIMAX(train_set, order=order)
    model_results = arima_200.fit()
    print(model_results.summary())
    return model_results


# predict
def data_predict(model_result, start_time, end_time):
    pred = model_result.predict(start_time, end_time, dynamic=True, typ='levels')
    forecast = model_result.forecast(5)
    return pred, forecast


if __name__ == '__main__':
    # file_path = 'E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\300730.csv'
    file_path = 'E:\\git\\RBF_AFSA_GSA\\data\\600519.SH\\600519.SH.csv'
    data_o = get_data(file_path)
    data_train, data_test = splite_data(data_o)

    # data_train = diff_data(data_train)
    # get_p_q(data_diff)
    print(1111111111111111111111111, data_train.shape)
    model_arima = train_main(data_train, (1, 1, 1))
    # result = data_predict(model_arima, '2019-03-01', '2019-03-10')
    result = data_predict(model_arima, 0, 150)
    print(result[0])

    print(data_train.tail())
    print(result[1])
