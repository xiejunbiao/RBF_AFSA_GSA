# -*- coding:utf-8 -*-
import tushare as ts
import os
from sqlalchemy import create_engine
import pandas as pd
import math
import csv


token = 'a8d9d0fd801d46978fd203025f6db22d7821e73423bfdfabd8b03f53'
ts.set_token(token)
# pro = ts.pro_api()

pro = ts.pro_api()

def get_stock(path, number):
    # data = pro.query('daily', ts_code='number', start_date='20180102', end_date='20180307')
    # data_test = pro.query('daily', ts_code='number', start_date='20180308', end_date='20180321')
    # data_L = pro.query('daily', ts_code='number', start_date='20180322', end_date='20180404')

    # data = pro.daily(ts_code='number', start_date='20180102', end_date='20180307')
    # data_test = pro.daily(ts_code='number', start_date='20180308', end_date='20180321')
    # data_L = pro.daily(ts_code='number', start_date='20180322', end_date='20180404')
    # return ts.get_hist_data(number, start='2018-01-12', end='2018-03-03')
    data = pro.daily(ts_code=number, start_date='20180102', end_date='20180307')
    data_test = pro.daily(ts_code=number, start_date='20190308', end_date='20180321')
    data_L = pro.daily(ts_code=number, start_date='20180322', end_date='20180404')
    # data = ts.get_hist_data(number, start='2018-01-02', end='2018-03-07')
    # data_test = ts.get_hist_data(number, start='2019-03-08', end='2018-03-21')
    # data_L = ts.get_hist_data(number, start='2018-03-22', end='2018-04-04')
    data.to_csv(path + '/%s.csv' % number)
    data_test.to_csv(path + '/%s.csv' % (number+'_test'))
    data_L.to_csv(path + '/%s.csv' % ('L' + number))


def get_concept_classified():
    return ts.get_concept_classified()


def get_all_data(number_list, path):
    for i in number_list:
        path_t = path + "/%s" % i
        print(path + "/%s" % i)
        check_dir_exist(path_t)
        get_stock(path_t, i)


def check_dir_exist(dir):
    # 坚持目录是否存在，不存在则创建
    if not os.path.exists(dir):
        os.mkdir(dir)


if __name__ == '__main__':
    path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data"
    get_all_data(['300730.SZ', '002628.SZ', '300144.SZ', '000017.SZ'], path)

    # '002628'
    # pro = ts.pro_api()
    #
    # # 查询当前所有正常上市交易的股票列表
    #
    # data = pro.stock_basic(exchange='',
    #                        list_status='L',
    #                        fields='ts_code,symbol,name,area,industry,list_date')
    # msg = pro.stock_company(exchange='',
    #                         fields='ts_code,chairman,manager,secretary,reg_capital,setup_date,province')
    # # print(msg)
    # msg.to_csv('msg_stack.csv')
    # data.to_csv('all_stack.csv')
    # print(data)
    # '300144'
    # '300730'
    #
    # print(get_concept_classified())
