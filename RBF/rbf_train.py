import numpy as np
import csv
import copy
import matplotlib.pyplot as plt
import pymysql


# 引力搜索算法
class GSA(object):
    class Particle(object):
        def __init__(self, population_size, dim,  C, delta, A, Y):
            self.C = C
            self.delta = delta
            self.A = A
            self.Y = Y
            self.population_size = population_size
            self.dim = dim
            self.x_bound = [-10, 10]
            # self.x_bound = [400, 850]

            # 初始化每个粒子的位置
            self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                       (self.population_size, self.dim))
            # 初始化每个粒子在每个维度上的速度
            self.v = np.random.rand(self.population_size, self.dim)

        def cal_fitness(self, x):
            # 计算每个粒子的适应度self.x是粒子的所有位置 self.fitness是一个一维数组
            fitness = self.calculate_fitness(x, self.C, self.delta, self.A, self.Y)
            return fitness

        def calculate_fitness(self, x, C, delta, A, Y):
            num, m = np.shape(C)
            fitness1 = []
            for i in range(len(x)):
                fitness1.append(float(
                    get_result(x[i][:num], arr_size(x[i][num:-num], m), x[i][-num:], A, Y)[1]))
            return np.array(fitness1)

    def dist(self, x, y):
        vec1 = np.array(x)
        vec2 = np.array(y)
        distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
        return distance

    # 获取引力系数
    def get_Gravity_Coefficient(self, max_stepts):
        p = 20
        g0 = 5
        g = []
        for i in range(1, max_stepts+1):
            g.append(g0*(np.e**(-p*(i / max_stepts))))
        return g

    def evolve(self, max_steps, number, dim, C, delta, A, Y):
        population = self.Particle(number, dim, C, delta, A, Y)
        # population.x = np.array([[-4.5, -5.5], [-6.2, 3.1], [2.5, 7.6], [3.5, 8.5]])
        # population.v = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
        # 初始化粒子的位置和速度
        population_v = population.v
        population_x = population.x
        size, m = np.shape(population_x)
        # 这里定义每个粒子的适应度
        fitness = population.cal_fitness(population_x)
        fitness_min = np.max(fitness)
        fitness_max = np.min(fitness)
        for step in range(max_steps):
            # print(g[step])
            g = self.get_Gravity_Coefficient(max_steps)
            fitness = population.cal_fitness(population_x)
            if np.max(fitness) > fitness_max:
                 fitness_min = np.max(fitness)
            if np.min(fitness) < fitness_min:
                fitness_max = np.min(fitness)
            quality_ = []
            sum1 = 0
            # 根据粒子的适应度计算每个粒子的质量
            for each in range(size):
                # 此处的粒子质量可能存在零
                # 因为我们这里定义的粒子不能存在无质量的粒子故在进行操作+ 0.0002       (1/np.e)
                quality_.append((fitness[each] - fitness_min) / (fitness_max - fitness_min))
                sum1 += quality_[each]
            for each1 in range(size):
                quality_[each1] = quality_[each1] / sum1
            f = np.zeros((size, m), dtype=float)
            b = np.random.rand(m)
            # 根据万有引力定律计算每个粒子在每个方向上所受到的力
            for i in range(size):
                # 每个粒子
                for j in range(size):
                    if i != j:
                        dist_ij = self.dist(population_x[i], population_x[j])
                        mult_ = g[step] * (quality_[j]) / dist_ij
                        for k in range(m):
                            f[i][k] += b[k] * mult_ * (population_x[j][k] - population_x[i][k])
            a = copy.deepcopy(f)
            for i in range(size):
                for j in range(m):
                    population_v[i][j] = 0.5 * population_v[i][j] + a[i][j]
                    population_x[i][j] = population_x[i][j] + population_v[i][j]
            if dim == 2:
                plt.clf()
                plt.scatter(population_x[:, 0], population_x[:, 1], s=30, color='r')
                pop1 = np.array([[-4.5, -5.5], [-6.2, 3.1], [2.5, 7.6], [3.5, 8.5]])
                plt.scatter(pop1[:, 0], pop1[:, 1], s=30, color='b')
                plt.scatter(0, 0, s=40, color='g')
                # plt.xlim(-100, 100)
                # plt.ylim(-100, 100)
                plt.pause(0.01)
            global_best_fitness = np.min(fitness)
            print('best fitness: %.5f, mean fitness: %.5f' % (global_best_fitness, np.average(fitness)))
        pg = population.x[np.argmin(fitness)]
        num, m = np.shape(C)
        # return self.pg
        return pg[:num], arr_size(pg[num:-num], m), pg[-num:]


# 粒子群算法
class PSO(object):
    def __init__(self, population_size, max_steps, d, C, delta, A, Y):
        self.C = C
        self.delta = delta
        self.A = A
        self.Y = Y
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = d  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [-10, 10]  # 解空间范围
        # 初始化粒子群位置
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim))
        # 初始化粒子群速度
        self.v = np.random.rand(self.population_size, self.dim)
        #  np.random.rand(3, 2)
        # array([[0.14022471, 0.96360618],  # random
        #        [0.37601032, 0.25528411],  # random
        #        [0.49313049, 0.94909878]])  # random
        fitness = self.calculate_fitness1(self.x, C, delta, A, Y)    # 所有粒子的适应度
        self.p = self.x  # 个体的最佳位置
        # 每个粒子个体有个自己的最佳位置

        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        # np.argmin(fitness)返回数组fitness中最小值的索引
        # 返回沿轴的最小值的索引。

        self.individual_best_fitness = fitness  # 个体的最优适应度
        # 每个个体有个自己的适应度（个体最优适应度）

        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度

    def calculate_fitness1(self, x, C, delta, A, Y):
        num, m = np.shape(C)
        fitness1 = []
        for i in range(len(x)):
            fitness1.append(float(
                get_result(x[i][:num], arr_size(x[i][num:-num], m), x[i][-num:], A, Y)[1]))
        return np.array(fitness1)

    def calculate_fitness(self, x):
        """
        此方法可以自己定义
        :param x: 输入一个population_size×dim
        :return: 一个一维数组（代表每个粒子的适应度）
        """
        return np.sum(np.square(x), axis=1)
    #   np.square(x)将x中的元素进行平方
    #   np.sum(）
    # 当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
    # 当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列

    def evolve(self):

        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            # 随机产生一个0-1之间的数字
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            # 速度更新

            self.x = self.v + self.x
            # 位置更新

            # plt.clf()
            # plt.scatter(self.x[:, 0], self.x[:, 1], s=60, color='k')
            # plt.xlim(self.x_bound[0], self.x_bound[1])
            # plt.ylim(self.x_bound[0], self.x_bound[1])
            # plt.pause(0.01)
            fitness = self.calculate_fitness1(self.x, self.C, self.delta, self.A, self.Y)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.average(fitness)))
        num, m = np.shape(self.C)
        # return self.pg
        return self.pg[:num], arr_size(self.pg[num:-num], m), self.pg[-num:]


# 初始化鱼群和适应度计算
class AFSIndividual:
    """
    鱼群初始化类
    """
    def __init__(self, vardim, bound):
        """

        :param vardim: 维度
        :param bound: 界限范围
        """
        self.vardim = vardim
        self.bound = bound
        self.len = self.vardim
        self.chrom = np.zeros(self.len)

    def generate(self):
        """
        随机生成一个 个体 鱼
        :return:
        """
        len = self.len
        rnd = np.random.random(size=len)
        self.velocity = np.random.random(size=len)
        for i in range(0, len):
            # 生成一个位置
            self.chrom[i] = self.bound[0, i] + (self.bound[1, i] - self.bound[0, i]) * rnd[i]
        self.bestPosition = np.zeros(len)
        self.bestFitness = 0.

    def calculateFitness(self, C, delta, A, Y):
        """
        计算每个人工鱼的适应度
        :return:
        """
        # self.fitness = rbf_train.(self.vardim, self.chrom, self.bound)
        num, m = np.shape(C)
        # print(self.chrom[:num])
        # print(arr_size(self.chrom[num:-num], m))
        print(self.chrom[-num:])
        print(np.shape(self.chrom))
        self.fitness = 1/float(get_result(self.chrom[:num], arr_size(self.chrom[num:-num], m), self.chrom[-num:], A, Y)[1])
        # self.fitness = 1/float(get_result(self.chrom, C, delta, A, Y)[1])


# 改进的鱼群算法
class ArtificialFishSwarm:
    """
    定义一个人工鱼群的一个类
    """

    def __init__(self, sizepop, vardim, bound, MAXGEN, params, C, delta, A, Y):
        """
        构造函数
        :param sizepop: 鱼群规模
        :param vardim: 变量的维度
        :param bound: 变量的边界2*vardim
        :param MAXGEN: 结束条件（最大迭代次数）
        :param params: 算法所需参数，它是一个由[visual, step, delta, try_num]组成的列表。
                        visual：视野
                        step:步长
                        delta:拥挤度因子
                        try_num:觅食行为尝试的次数
        """
        self.quality = []
        self.A = A
        # 数据集
        self.Y = Y
        # 数据集标签
        self.C = C
        self.delta = delta
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.step = []
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        # [[0,0][0,0][0,0][0,0][0,0][0,0][0,0]...]
        self.t = 0
        self.lennorm = 6000
        self.close_of_print = 0  # 控制标量输出，开关量0不输出，非零时输出

    def initialize(self):
        """
        初始化一个鱼群
        :return:
        """
        for i in range(0, self.sizepop):
            # 初始化鱼群和鱼群状态
            # 实例化一个类
            ind = AFSIndividual(self.vardim, self.bound)
            # 产生一个鱼
            ind.generate()
            self.population.append(ind)
            self.step.append(np.tile([self.params[1]], self.vardim))  # 2

    def evaluation(self, x, C, delta, A, Y):
        """
        评估个人的适合度
        :param x:
        :return:
        """
        x.calculateFitness(C, delta, A, Y)

    def forage(self, x):
        """
        觅食行为
        :param x:
        :return:
        """
        newInd = copy.deepcopy(x)
        found = False
        for i in range(0, self.params[3]):
            indi = self.randSearch(x, self.params[0])
            if indi.fitness > x.fitness:
                newInd.chrom = x.chrom + np.random.random(self.vardim) * self.params[1] * self.lennorm * (
                        indi.chrom - x.chrom) / np.linalg.norm(indi.chrom - x.chrom)
                newInd = indi
                found = True
                break
        if not(found):
            newInd = self.randSearch(x, self.params[1])
        return newInd

    def randSearch(self, x, searLen):
        """
        人工鱼群随机搜索行为
        :param x:
        :param searLen:视野范围
        :return:
        """
        ind = copy.deepcopy(x)
        ind.chrom += np.random.uniform(-1, 1,
                                       self.vardim) * searLen * self.lennorm
        for j in range(0, self.vardim):
            if ind.chrom[j] < self.bound[0, j]:
                ind.chrom[j] = self.bound[0, j]
            if ind.chrom[j] > self.bound[1, j]:
                ind.chrom[j] = self.bound[1, j]
        self.evaluation(ind, self.C, self.delta, self.A, self.Y)
        return ind

    def huddle(self, i, x, quality, step):
        """
        聚群行为
        :param x:
        :return:
        """
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        # 记录下在x视野内的鱼
        for i in range(1, self.sizepop):
            # 1=======改进
            # if (dist[i] > 0) and (dist[i] < self.params[0] * self.lennorm):
            if (dist[i] > 0) and (dist[i] <
                                  self.params[0] * (1 + quality * self.sizepop * np.random.rand()) *
                                  self.lennorm):
                index.append(i)
        nf = len(index)
        if self.close_of_print != 0:
            print('huddle', nf)
        if nf > 0:
            xc = np.zeros(self.vardim)
            population_chrom = []   # 2
            population_finess = []  # 2
            population_chrom.append(x.chrom)    # 2
            population_finess.append(x.fitness)     # 2
            for i in range(0, nf):
                population_chrom.append(self.population[index[i]].chrom)    # 2
                population_finess.append(self.population[index[i]].fitness)     # 2
                # 将当前鱼感知域内的人工鱼按向量相加求和
                xc += self.population[index[i]].chrom

            # 求在x视野内的n个鱼的平均位置
            xc = xc / nf
            # print(xc)
            # 生成一个新的鱼并将中心位置赋值给该鱼
            cind = AFSIndividual(self.vardim, self.bound)
            cind.chrom = xc
            cind.calculateFitness(self.C, self.delta, self.A, self.Y)

            population_chrom.append(cind.chrom)  # 2
            population_finess.append(cind.fitness)  # 2
            if (cind.fitness / nf) > (self.params[2] * x.fitness):

                # 当人工鱼符合条件时更新该人工鱼的移动步长（按照以下方式）
                acle = self.acceleration_afsa(x, cind, population_chrom, population_finess)    # 2
                step = step + acle   # 2
                xnext = x.chrom + np.random.random(
                    self.vardim) * step * self.lennorm * (xc - x.chrom) / np.linalg.norm(xc - x.chrom)

                # 此处的 self.lennorm 代表什么意思是否可以直接去掉？？
                # xnext = x.chrom + np.random.random(
                #     self.vardim) * self.params[1] * self.lennorm * (xc - x.chrom) / np.linalg.norm(xc - x.chrom)
                # 判断人工鱼的移动是否离开了解空间
                # 如果大于最大界限那么就让其等于最大界限
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd, self.C, self.delta, self.A, self.Y)
                # print "hudding"
                return newInd, step
            else:
                return self.forage(x), step
        else:
            return self.forage(x), step

    def follow(self, x, quality, step):
        """
        人工鱼群追尾行为：
        探索当前领域内（dij < visual)的伙伴中Yj最大的伙伴Xj ，如果Yc/nf  >  dYi
        表明伙伴Xj状态具有较高的食物浓度，且不拥挤，则朝着伙伴Xj的方向前进一步，执行觅食行为
        :param x:
        :return:
        """
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.sizepop):
            # if (dist[i] > 0) and (dist[i] < self.params[0] * self.lennorm):
            if (dist[i] > 0) and (dist[i] <
                                  self.params[0] * (1 + quality * self.sizepop * np.random.rand()) *
                                  self.lennorm):
                index.append(i)
        nf = len(index)
        if self.close_of_print != 0:
            print('follow', nf)
        if nf > 0:
            best = -999999999.
            bestIndex = 0
            population_chrom = []  # 2
            population_finess = []  # 2
            population_chrom.append(x.chrom)    # 2
            population_finess.append(x.fitness)     # 2
            for i in range(0, nf):
                population_chrom.append(self.population[index[i]].chrom)  # 2
                population_finess.append(self.population[index[i]].fitness)  # 2
                if self.population[index[i]].fitness > best:
                    best = self.population[index[i]].fitness
                    bestIndex = index[i]
            population_chrom[bestIndex:bestIndex] = []    # 2
            population_finess[bestIndex:bestIndex] = []   # 2
            population_chrom.append(self.population[bestIndex].chrom)   # 2
            population_finess.append(self.population[bestIndex].fitness)    # 2

            if (self.population[bestIndex].fitness / nf) > (self.params[2] * x.fitness):

                # 当人工鱼符合条件时更新该人工鱼的移动步长（按照以下方式）
                acle = self.acceleration_afsa(x, self.population[bestIndex], population_chrom, population_finess)  # 2
                step = step + acle   # 2
                xnext = x.chrom + np.random.random(
                    self.vardim) * step * self.lennorm * \
                                    (self.population[bestIndex].chrom - x.chrom) / np.linalg.norm(
                    self.population[bestIndex].chrom - x.chrom)         # 2
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd, self.C, self.delta, self.A, self.Y)
                # print "follow"
                return newInd, step
            else:
                return self.forage(x), step
        else:
            return self.forage(x), step

    def solve(self):
        """
        迭代人工鱼群算法
        :return:
        """

        self.initialize()

        # 对当前鱼群进行评估
        for i in range(0, self.sizepop):
            # print(self.delta)

            self.evaluation(self.population[i], self.C, self.delta, self.A, self.Y)

            self.fitness[i] = self.population[i].fitness

        fitness_min = np.max(self.fitness)
        fitness_max = np.min(self.fitness)
        sum0 = 0
        # 根据粒子的适应度计算每个粒子的质量
        for each in range(self.sizepop):
            # 此处的粒子质量可能存在零
            # 因为我们这里定义的粒子不能存在无质量的粒子故在进行操作+ (1/np.e)
            self.quality.append((self.fitness[each] - fitness_min + (1 / np.e)) / (fitness_max - fitness_min+ (1 / np.e)))
            sum0 += self.quality[each]
        for each1 in range(self.sizepop):
            self.quality[each1] = self.quality[each1] / sum0

        # 最优适应度
        # 个体最优
        # best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[int(bestIndex)])
        # 鱼群平均适应度
        # 全局最优
        self.avefitness = np.mean(self.fitness)
        # 个体适应度
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        # 全局适应度
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.MAXGEN - 1:
            self.t += 1
            # newpop = []
            for i in range(0, self.sizepop):
                # 每条鱼进行寻优

                # 聚群行为
                xi1, step_temp1 = self.huddle(i, self.population[i], self.quality[i], self.step[i])
                xi2, step_temp2 = self.follow(self.population[i], self.quality[i], self.step[i])
                # 公告板记录每条鱼的状态
                if xi1.fitness > xi2.fitness:
                    self.population[i] = xi1
                    self.fitness[i] = xi1.fitness
                    self.step[i] = step_temp1
                else:
                    self.population[i] = xi2
                    self.fitness[i] = xi2.fitness
                    self.step[i] = step_temp2

                # 以下为优化部分
                fitness_min = np.max(self.fitness)
                fitness_max = np.min(self.fitness)

            sum0 = 0
            # 根据粒子的适应度计算每个粒子的质量
            for each in range(self.sizepop):
                # 此处的粒子质量可能存在零
                # 因为我们这里定义的粒子不能存在无质量的粒子故在进行操作+ (1/np.e)

                self.quality.append((self.fitness[each] - fitness_min + (1 / np.e)) / (fitness_max - fitness_min + (1 / np.e)))
                sum0 += self.quality[each]
            for each1 in range(self.sizepop):
                self.quality[each1] = self.quality[each1] / sum0
            #

            best = np.max(self.fitness)

            # 获得最大值的索引
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[int(bestIndex)])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:")
        # print(self.best.chrom)
        self.printResult()
        num, m = np.shape(self.C)
        return self.best.chrom[:num], arr_size(self.best.chrom[num:-num], m), self.best.chrom[-num:]
        # return self.best.chrom

    def distance(self, x):
        """
        计算x与其他鱼的距离
        :param x:当前人工鱼和其他鱼的距离列表
        :return: 返回一个数组
        """
        dist = np.zeros(self.sizepop)
        for i in range(0, self.sizepop):
            dist[i] = np.linalg.norm(x.chrom - self.population[i].chrom) / 6000
        return dist

    def acceleration_afsa(self, x, y, population_chrom, population_finess):
        """
        加速度的计算
        :param x:
        :param y:
        :param population_chrom: 列表的第一个存放的是当前鱼状态  最后一个存放的是目标鱼的状态
        :param population_finess: 列表的第一个存放的是当前鱼的适应度最后一个存放的是目标鱼的适应度
        :return:
        """

        fitness_min = max(population_finess)
        fitness_max = min(population_finess)
        sum0 = 0
        quality_part = []
        # 根据粒子的适应度计算每个粒子的质量
        for each in range(len(population_chrom)):
            # 此处的粒子质量可能存在零
            # 因为我们这里定义的粒子不能存在无质量的粒子故在进行操作+ (1/np.e)

            quality_part.append((population_finess[each] - fitness_min + (1 / np.e)) / (fitness_max - fitness_min+ (1 / np.e)))
            sum0 += quality_part[each]
        # for each1 in range(len(population_chrom)):
        #     quality_part[each1] = quality_part[each1] / sum0
        quality_part_x = quality_part[0] / sum0
        quality_part_y = quality_part[-1] / sum0
        p = 20
        g0 = 100
        a = len(population_chrom)*((x.chrom - y.chrom) * (g0 * (np.e ** (-p * (self.t / self.MAXGEN)))) *
                                   (quality_part_x * quality_part_y)/(dist_x_y(x.chrom, y.chrom) + 0.01))/quality_part_x
        return a

    def printResult(self):
        """
        结果显示：作图
        :return:
        """

        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        # plt.plot(x, y1, 'r')
        # plt.plot(x, y2, 'g')
        # plt.plot(x, y1, 'r', label='optimal value')
        # plt.plot(x, y2, 'g', label='average value')
        plt.plot(y1, 'k-')
        plt.plot(y2, 'k:')
        size = 12
        plt.legend(['optimal value', 'average value'], fontsize=size+5)
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
        plt.xlabel("Iteration", font2)
        plt.ylabel("function value", font2)
        plt.title("GSA+AFSA")
        plt.show()


# 鱼群算法
class ArtificialFishSwarm_1:
    """
    定义一个人工鱼群的一个类
    """

    def __init__(self, sizepop, vardim, bound, MAXGEN, params, C, delta, A, Y):
        """
        构造函数
        :param sizepop: 鱼群规模
        :param vardim: 变量的维度
        :param bound: 变量的边界2*vardim
        :param MAXGEN: 结束条件（最大迭代次数）
        :param params: 算法所需参数，它是一个由[visual, step, delta, try_num]组成的列表。
                        visual：视野
                        step:步长
                        delta:拥挤度因子
                        try_num:觅食行为尝试的次数
        """
        self.quality = []
        self.A = A
        # 数据集
        self.Y = Y
        # 数据集标签
        self.C = C
        self.delta = delta
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.step = []
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        # [[0,0][0,0][0,0][0,0][0,0][0,0][0,0]...]
        self.t = 0
        self.lennorm = 6000
        self.path_data = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\data_Optimized_curve\\000017"

    def initialize(self):
        """
        初始化一个鱼群
        :return:
        """
        for i in range(0, self.sizepop):
            # 初始化鱼群和鱼群状态
            # 实例化一个类
            ind = AFSIndividual(self.vardim, self.bound)
            # 产生一个鱼
            ind.generate()
            self.population.append(ind)
            self.step.append(np.tile([self.params[1]], self.vardim))    #  2

    def evaluation(self, x, C, delta, A, Y):
        """
        评估个人的适合度
        :param x:
        :return:
        """
        x.calculateFitness(C, delta, A, Y)

    def forage(self, x):
        """
        觅食行为
        :param x:
        :return:
        """
        newInd = copy.deepcopy(x)
        found = False
        for i in range(0, self.params[3]):
            indi = self.randSearch(x, self.params[0])
            if indi.fitness > x.fitness:
                newInd.chrom = x.chrom + np.random.random(self.vardim) * self.params[1] * self.lennorm * (
                        indi.chrom - x.chrom) / np.linalg.norm(indi.chrom - x.chrom)
                newInd = indi
                found = True
                break
        if not(found):
            newInd = self.randSearch(x, self.params[1])
        return newInd

    def randSearch(self, x, searLen):
        """
        人工鱼群随机搜索行为
        :param x:
        :param searLen:视野范围
        :return:
        """
        ind = copy.deepcopy(x)
        ind.chrom += np.random.uniform(-1, 1,
                                       self.vardim) * searLen * self.lennorm
        for j in range(0, self.vardim):
            if ind.chrom[j] < self.bound[0, j]:
                ind.chrom[j] = self.bound[0, j]
            if ind.chrom[j] > self.bound[1, j]:
                ind.chrom[j] = self.bound[1, j]
        self.evaluation(ind, self.C, self.delta, self.A, self.Y)
        return ind

    def huddle(self, x):
        """
        聚群行为
        :param x:
        :return:
        """
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        # 记录下在x视野内的鱼
        for i in range(1, self.sizepop):
            # 1=======改进
            if (dist[i] > 0) and (dist[i] < self.params[0] * self.lennorm):
            # if (dist[i] > 0) and (dist[i] <
            #                       self.params[0] * (1 + quality * self.sizepop * np.random.rand()) *
            #                       self.lennorm):
                index.append(i)
        nf = len(index)
        if nf > 0:
            xc = np.zeros(self.vardim)
            # population_chrom = []   # 2
            # population_finess = []  # 2
            # population_chrom.append(x.chrom)    # 2
            # population_finess.append(x.fitness)     # 2
            for i in range(0, nf):
                # population_chrom.append(self.population[index[i]].chrom)    # 2
                # population_finess.append(self.population[index[i]].fitness)     # 2
                # 将当前鱼感知域内的人工鱼按向量相加求和
                xc += self.population[index[i]].chrom

            # 求在x视野内的n个鱼的平均位置
            xc = xc / nf
            # print(xc)
            # 生成一个新的鱼并将中心位置赋值给该鱼
            cind = AFSIndividual(self.vardim, self.bound)
            cind.chrom = xc
            cind.calculateFitness(self.C, self.delta, self.A, self.Y)

            # population_chrom.append(cind.chrom)  # 2
            # population_finess.append(cind.fitness)  # 2
            if (cind.fitness / nf) > (self.params[2] * x.fitness):

                # 当人工鱼符合条件时更新该人工鱼的移动步长（按照以下方式）
                # acle = self.acceleration_afsa(x, cind, population_chrom, population_finess)    # 2
                # step = step + acle   # 2
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (xc - x.chrom) / np.linalg.norm(xc - x.chrom)

                # 此处的 self.lennorm 代表什么意思是否可以直接去掉？？
                # xnext = x.chrom + np.random.random(
                #     self.vardim) * self.params[1] * self.lennorm * (xc - x.chrom) / np.linalg.norm(xc - x.chrom)
                # 判断人工鱼的移动是否离开了解空间
                # 如果大于最大界限那么就让其等于最大界限
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd, self.C, self.delta, self.A, self.Y)
                # print "hudding"
                return newInd
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def follow(self, x):
        """
        人工鱼群追尾行为：
        探索当前领域内（dij < visual)的伙伴中Yj最大的伙伴Xj ，如果Yc/nf  >  dYi
        表明伙伴Xj状态具有较高的食物浓度，且不拥挤，则朝着伙伴Xj的方向前进一步，执行觅食行为
        :param x:
        :return:
        """
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.sizepop):
            if (dist[i] > 0) and (dist[i] < self.params[0] * self.lennorm):
            # if (dist[i] > 0) and (dist[i] <
            #                       self.params[0] * (1 + quality * self.sizepop * np.random.rand()) *
            #                       self.lennorm):
                index.append(i)
        nf = len(index)
        if nf > 0:
            best = -999999999.
            bestIndex = 0
            # population_chrom = []  # 2
            # population_finess = []  # 2
            # population_chrom.append(x.chrom)    #2
            # population_finess.append(x.fitness)     #2
            for i in range(0, nf):
                # population_chrom.append(self.population[index[i]].chrom) #2
                # population_finess.append(self.population[index[i]].fitness) #2
                if self.population[index[i]].fitness > best:
                    best = self.population[index[i]].fitness
                    bestIndex = index[i]
            # population_chrom[bestIndex:bestIndex] = []    #2
            # population_finess[bestIndex:bestIndex] = []   #2
            # population_chrom.append(self.population[bestIndex].chrom)   #2
            # population_finess.append(self.population[bestIndex].fitness)    #2

            if (self.population[bestIndex].fitness / nf) > (self.params[2] * x.fitness):

                # 当人工鱼符合条件时更新该人工鱼的移动步长（按照以下方式）
                # acle = self.acceleration_afsa(x, self.population[bestIndex], population_chrom, population_finess)  # 2
                # step = step + acle   # 2
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * \
                                    (self.population[bestIndex].chrom - x.chrom) / np.linalg.norm(
                    self.population[bestIndex].chrom - x.chrom)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd, self.C, self.delta, self.A, self.Y)
                # print "follow"
                return newInd
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def solve(self):
        """
        迭代人工鱼群算法
        :return:
        """

        self.initialize()

        # 对当前鱼群进行评估
        for i in range(0, self.sizepop):
            self.evaluation(self.population[i], self.C, self.delta, self.A, self.Y)

            self.fitness[i] = self.population[i].fitness

        # 最优适应度
        # 个体最优
        # best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[int(bestIndex)])
        # 鱼群平均适应度
        # 全局最优
        self.avefitness = np.mean(self.fitness)
        # 个体适应度
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        # 全局适应度
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.MAXGEN - 1:
            self.t += 1
            for i in range(0, self.sizepop):
                # 每条鱼进行寻优

                # 聚群行为
                xi1 = self.huddle(self.population[i])
                xi2 = self.follow(self.population[i])
                # 公告板记录每条鱼的状态
                if xi1.fitness > xi2.fitness:
                    self.population[i] = xi1
                    self.fitness[i] = xi1.fitness
                else:
                    self.population[i] = xi2
                    self.fitness[i] = xi2.fitness

            best = np.max(self.fitness)

            # 获得最大值的索引
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[int(bestIndex)])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:")
        # print(self.best.chrom)
        self.printResult()
        num, m = np.shape(self.C)
        return self.best.chrom[:num], arr_size(self.best.chrom[num:-num], m), self.best.chrom[-num:]

    def distance(self, x):
        """
        计算x与其他鱼的距离
        :param x:当前人工鱼和其他鱼的距离列表
        :return: 返回一个数组
        """
        dist = np.zeros(self.sizepop)
        for i in range(0, self.sizepop):
            dist[i] = np.linalg.norm(x.chrom - self.population[i].chrom) / 6000
        return dist

    def acceleration_afsa(self, x, y, population_chrom, population_finess):
        """
        加速度的计算
        :param x:
        :param y:
        :param population_chrom: 列表的第一个存放的是当前鱼状态  最后一个存放的是目标鱼的状态
        :param population_finess: 列表的第一个存放的是当前鱼的适应度最后一个存放的是目标鱼的适应度
        :return:
        """

        fitness_min = max(population_finess)
        fitness_max = min(population_finess)
        sum0 = 0
        quality_part = []
        # 根据粒子的适应度计算每个粒子的质量
        for each in range(len(population_chrom)):
            # 此处的粒子质量可能存在零
            # 因为我们这里定义的粒子不能存在无质量的粒子故在进行操作+ (1/np.e)
            quality_part.append((population_finess[each] - fitness_min + (1 / np.e)) / (fitness_max - fitness_min+ (1 / np.e)))
            sum0 += quality_part[each]
        # for each1 in range(len(population_chrom)):
        #     quality_part[each1] = quality_part[each1] / sum0
        quality_part_x = quality_part[0] / sum0
        quality_part_y = quality_part[-1] / sum0
        p = 20
        g0 = 100
        a = len(population_chrom)*((x.chrom - y.chrom) * (g0 * (np.e ** (-p * (self.t / self.MAXGEN)))) *
                                   (quality_part_x * quality_part_y)/(dist_x_y(x.chrom, y.chrom) + 0.01))/quality_part_x
        return a

    def printResult(self):
        """
        结果显示：作图
        :return:
        """
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        # plt.plot(x, y1, 'r', label='optimal value')
        # plt.plot(x, y2, 'g', label='average value')
        f = open(self.path_data + "//000017.txt", 'a+', encoding='utf-8')
        t1 = ",".join(["{:.3f}".format(i) for i in y1])
        t2 = ",".join(["{:.3f}".format(i) for i in y2])
        f.write(t1+'\n')
        f.write(t2+'\n')
        # f.writelines(str(y2))
        f.close()
        plt.plot(y1, 'k-')
        plt.plot(y2, 'k:')
        size = 12
        plt.legend(['optimal value', 'average value'], fontsize=size + 5)
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
        plt.xlabel("Iteration", font2)
        plt.ylabel("function value", font2)
        # plt.title("GSA+AFSA")

        # plt.xlabel("Iteration")
        # plt.ylabel("function value")
        # plt.title("AFSA")
        plt.legend()
        plt.show()


def dist_x_y(x, y):
    vec1 = np.array(x)
    vec2 = np.array(y)
    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return distance


def arr_size(arr, size):
    s = []
    for i in range(0, int(len(arr)), size):
        c = arr[i:i + size]
        s.append(c)
    return s


# 加上成交量和测试数据
def get_data_test(path, number_day=10, input_number=15):
    # path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\000017_.csv"
    # path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\close\\000017.csv"
    print(path)
    with open(path, 'r') as file:
        temp = csv.DictReader(file)
        data_dir1 = [row['close'] for row in temp]
    # print(data_dir1)
    with open(path, 'r') as file1:
        temp1 = csv.DictReader(file1)

        data_dir2 = [row['volume'] for row in temp1]
        # data_dir2 = [row['vol'] for row in temp1]

    # 定义使用前几天的数据来进行预测
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
            # data_dir2_1 = copy.deepcopy(data_dir2[j: 1 + j + len(data_dir2) - number_day])
            temp_1 = []
            for each in range(i - input_number, i):
                temp_1.append(float(data_dir1_1[each]))

            # 将成交量加入数据集
            # for each2 in range(i - input_number, i):
            #     temp_1.append(float(data_dir2_1[each2])/10000)
            data_x.append(temp_1)
            data_y.append(float(data_dir1_1[i]))
        test_x = data_x.pop()
        test_y = data_y.pop()
        data_x_number_day.append(data_x)
        data_y_number_day.append(data_y)
        test_x_number_day.append(test_x)
        test_y_number_day.append(test_y)
    return data_x_number_day, data_y_number_day, test_x_number_day, test_y_number_day


def get_data():
    path = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\000017_.csv"

    with open(path, 'r') as file:
        temp = csv.DictReader(file)
        data_dir = [row['close'] for row in temp]
    input_number = 15
    data_x = []
    data_y = []
    data_dir.reverse()
    for i in range(input_number, len(data_dir)):
        temp_1 = []
        for each in range(i - input_number, i):
            temp_1.append(float(data_dir[each]))
        data_x.append(temp_1)
        data_y.append(float(data_dir[i]))
    return data_x, data_y


def get_result(w, C, delta, A, Y):
    # print(len(w))
    # print(np.shape(C))
    # print(np.shape(A))
    n,   m = np.shape(A)
    hidden_out = []
    # 正向传播
    for j in range(n):
        hidden_out_temp = []

        for i in range(len(C)):
            # print(delta[i])
            # print(np.e ** (-1 * (np.linalg.norm(np.array(A[j]) - np.array(C[i])) ** 2) / (2 * delta[i] ** 2)))
            hidden_out_temp.append(
                np.e ** (-1 * (np.linalg.norm(np.array(A[j]) - np.array(C[i])) ** 2) / (2 * delta[i] ** 2)))
        hidden_out.append(hidden_out_temp)
    # print(len(hidden_out))
    # print(len(np.mat(w).T))
    y_pre = np.mat(hidden_out) * np.mat(w).T
    errors = y_pre - np.mat(Y).T
    cost_ = 0
    # 计算损失
    for t in range(n):
        cost_ += errors[t] ** 2
    return errors, cost_, y_pre


def get_predict(w, C, delta, A):
    result = 0
    for i in range(len(C)):
        result += w[i][0] * np.e ** (-1 * (np.linalg.norm(np.array(A)-np.array(C[i])) ** 2) / (2 * delta[i][0] ** 2))
    return result


def evolve(low, high, A, Y, hidden_num, ratio, max_step,i):
    """
    :param A: 训练集
    :param Y: 训练集标签
    :param hidden_num: 隐藏层数量
    :param ratio: 学习率
    :param max_step: 最大迭代次数
    :return:模型的权重、径向基核函数中心C、径向基核函数宽度
    """
    n, m = np.shape(A)
    C = np.random.uniform(-20, 20, (hidden_num, m))

    delta = np.random.rand(hidden_num)
    w = np.random.rand(hidden_num)
    # 不使用鱼群算法优化直接使用上面的参数初始化(将上面的w权重取消注释)

    # 算法优化部分（使用优化算法进行权值和径向基函数的中心初始化）
    C1 = copy.deepcopy(C)
    delta1 = copy.deepcopy(delta)
    A1 = copy.deepcopy(A)
    Y1 = copy.deepcopy(Y)
    d = hidden_num * (m + 2)

    # 鱼群算法
    bound = np.tile([[-10], [10]], d)
    # 1,1改进的鱼群算法
    afs_1 = ArtificialFishSwarm(5, d, bound, 50, [0.001, 0.01, 0.618, 10], C1, delta1, A1, Y1)
    w, C, delta = afs_1.solve()

    # 1,2鱼群算法
    # afs = ArtificialFishSwarm_1(60, d, bound, 50, [0.001, 0.0001, 0.618, 40], C1, delta1, A1, Y1)
    # w, C, delta = afs.solve()

    # 2粒子群算法
    # pso = PSO(60, 10, d, C1, delta1, A1, Y1)
    # w, C, delta = pso.evolve()

    # 3引力搜索算法
    # gsa = GSA()
    # w, C, delta =gsa.evolve(10, 60, d, C1, delta1, A1, Y1)
    # print(w)
    # print(C)
    # print(delta)

    steps = 0
    # 训练
    while steps < max_step:
        errors, cost_, y_pre = get_result(w, C, delta, A, Y)
        # hidden_out = []
        # # 正向传播
        # for j in range(n):
        #     hidden_out_temp = []
        #     for i in range(hidden_num):
        #         hidden_out_temp.append(
        #             np.e ** (-1 * (np.linalg.norm(np.array(A[j]) - np.array(C[i])) ** 2) / (2 * delta[i] ** 2)))
        #     hidden_out.append(hidden_out_temp)
        #
        # y_pre = np.mat(hidden_out) * np.mat(w).T
        # errors = y_pre - np.mat(Y).T

        # 反向传播
        for k in range(len(C)):
            sum1_w = 0
            sum2_c = 0
            sum3_delta = 0
            for k1 in range(n):
                sum1_w += errors[k1] * \
                          (np.e ** (-1 * (np.linalg.norm(np.array(A[k1]) - np.array(C[k])) ** 2) / (2 * delta[k] ** 2)))
                sum2_c += (w[k] * errors[k1] *
                           (np.e ** (-1 * (np.linalg.norm(np.array(A[k1]) - np.array(C[k])) ** 2) / (
                                       2 * delta[k] ** 2)))
                           / delta[k] ** 2) * (np.array(A[k1]) - np.array(C[k]))
                sum3_delta += (w[k] * errors[k1] *
                               (np.e ** (-1 * (np.linalg.norm(np.array(A[k1]) - np.array(C[k])) ** 2)
                                         / (2 * delta[k] ** 2)))) * (
                                          np.linalg.norm(np.array(A[k1]) - np.array(C[k])) ** 2) / (delta[k] ** 3)
            # 更新参数
            w[k] = w[k] - ratio * sum1_w / n
            C[k] = C[k] - ratio * sum2_c / n
            delta[k] = delta[k] - ratio * sum3_delta / n

        if steps % 10 == 0:
            print("\t-------- iter: ", steps, " ,cost: ", cost_)
            if cost_ < 0.01:
                # 如果损失函数值小于0.01则停止迭代
                break
        plt.clf()
        plt.title("Real value versus predicted value")
        plt.plot(Y, 'r')
        plt.plot(y_pre, 'g')
        plt.legend(['True value', 'Predicted value'])

        # plt.xticks(np.arange(37), dates, rotation=80)

        # plt.xlabel('GSA+RBF')
        # plt.xlabel('AFSA+RBF')
        plt.xlabel('GSA+AFSA+RBF')
        plt.ylabel('Predicted value and True value')
        plt.text(27, 5.2, 'cost=%f' % cost_)
        models = ['RBF', 'GSA_AFSA_RBF', 'GSA_RBF', 'AFSA_RBF']
        plt.xlabel(models[2])
        plt.savefig('E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\image'
                    '\\' + models[2] + '\\%d.jpg' % steps)
        plt.pause(0.02)
        steps += 1
    return w, C, delta


def save_model_result(center, delta, weight, i):
    """
     保存最终的模型
    :param center:
    :param delta:
    :param weight:
    :param i:
    :return:
    """
    def write_file(file_name, source):
        f = open(file_name, "w")
        if len(np.shape(source)) == 2:
            m, n = np.shape(source)
            for i1 in range(m):
                tmp = []
                for j in range(n):
                    tmp.append(str(source[i1, j]))
                f.write("\t".join(tmp) + "\n")
        else:
            n = len(source)
            tmp = []
            for j in range(n):
                tmp.append(str(source[j]))
                f.write("\t".join(tmp) + "\n")
                tmp = []
        f.close()
    write_file("center_%d.csv.txt" % i, center)
    write_file("delta_%d.csv.txt" % i, delta)
    write_file("weight_%d.csv.txt" % i, weight)
    print("==============第%d个数据集的参数保存成功=============" % (i+1))
    # write_file('train_result.txt', result)


def rbf_train_all(low, high, data_x_number_day, data_y_number_day, hidden_num, ratio, max_step):
    n1, n2, n3 = np.shape(data_x_number_day)
    for i in range(n1):
        w, C, delta = evolve(low, high, data_x_number_day[i], data_y_number_day[i], hidden_num, ratio, max_step, i)
        save_model_result(C, delta, w, i)


def database_connect(database_name):
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                      passwd='123456', database=database_name)
    return db


# 将数据存入数据库
def InsertData(database_conn, TableName, dic):
    try:
        cur = database_conn.cursor()
        COLstr = ''  # 列的字段
        ROWstr = ''  # 行字段

        ColumnStyle = ' VARCHAR(20)'
        for key in dic.keys():
            COLstr = COLstr + ' ' + key+ ColumnStyle + ','
            ROWstr = (ROWstr + '"%s"' + ',') % (dic[key])
        # 推断表是否存在，存在运行try。不存在运行except新建表，再insert
        try:
            cur.execute("SELECT * FROM  %s" % (TableName))
            cur.execute("INSERT INTO %s VALUES (%s)" % (TableName, ROWstr[:-1]))

        except pymysql.Error as e:
            cur.execute("CREATE TABLE %s (%s)" % (TableName, COLstr[:-1]))
            cur.execute("INSERT INTO %s VALUES (%s)" % (TableName, ROWstr[:-1]))
        database_conn.commit()
        cur.close()

    except pymysql.Error as e:
        print("Mysql Error %d: %s" % (e.args[0], e.args[1]))


def main(dir, file_name):
    conn = database_connect("dataset_stock")
    InsertData(conn, file_name, dir)
    print("已存入第个数据，第%d条记录")
    conn.close()


if __name__ == '__main__':
    # path_file = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\300730.csv"
    # E:\\git\\RBF_AFSA_GSA\\data\\300730.SZ\\300730.SZ.csv
    # path_file = "E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\data\\close\\000017.csv"
    path_file = "E:\\git\\RBF_AFSA_GSA\\data\\300730.SZ\\300730.SZ.csv"

    data_x, data_y, test_x, test_y = get_data_test(path_file)

    # for i in range(len(data_x)):
    #     for j in range(len(data_x[i])):
    #         print(data_x[i][j], 'lable:', data_y[i][j])

    low = 10
    high = 30
    rbf_train_all(low, high, data_x, data_y, 40, 0.9, 3000)
    plt.show()
