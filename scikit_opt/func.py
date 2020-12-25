#!-*-coding:utf-8-*-
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21
# @Author  : github.com/xiejunbiao
import numpy as np


class FuncSet(object):
    def __init__(self, data_x, data_y, modle=None, para=None):
        self.data_x = data_x
        self.data_y = data_y
        self.modle = modle
        self.modle_para = para

    def func_rbf(self, chrom):

        num, m = self.modle_para[0], self.modle_para[1]
        # print(np.shape(chrom))
        # print(np.shape(chrom[:num]))
        # print(np.shape(self.arr_size(chrom[num:-num], m)))
        # print(np.shape(chrom[-num:]))
        # return 1 / float(self.fun_c(chrom[:num], self.arr_size(chrom[num:-num], m), chrom[-num:],
        #                                self.data_x, self.data_y)[1])
        return float(self.fun_c(chrom[:num], self.arr_size(chrom[num:-num], m), chrom[-num:],
                                self.data_x, self.data_y)[1])
    def fun_c(self, w, C, delta, A, Y):
        # print(len(w))
        # print(np.shape(C))
        # print(np.shape(A))
        n, m = np.shape(A)
        hidden_out = []
        # 正向传播
        for j in range(n):
            hidden_out_temp = []

            for i in range(len(C)):
                # print(delta[i])
                # print(np.e ** (-1 * (np.linalg.norm(np.array(A[j]) - np.array(C[i])) ** 2) / (2 * delta[i] ** 2)))
                hidden_out_temp.append(
                        np.e**(-1 * (np.linalg.norm(np.array(A[j]) - np.array(C[i]))**2) / (2 * delta[i]**2)))
            hidden_out.append(hidden_out_temp)
        # print(len(hidden_out))
        # print(len(np.mat(w).T))
        y_pre = np.mat(hidden_out) * np.mat(w).T
        errors = y_pre - np.mat(Y).T
        cost_ = 0
        # 计算损失
        for t in range(n):
            cost_ += errors[t]**2
        return errors, cost_, y_pre

    def arr_size(self, arr, size):
        s = []
        for i in range(0, int(len(arr)), size):
            c = arr[i:i + size]
            s.append(c)
        return s


if __name__ == '__main__':
    # FuncSet()
    pass