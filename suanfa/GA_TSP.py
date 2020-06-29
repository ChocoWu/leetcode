#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu
from abc import abstractmethod
import numpy as np
import pandas as pd
from scipy import spatial
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import os


class GA_TSP:
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=[], constraint_ueq=[]):
        self.func = func
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = constraint_eq  # a list of unequal constraint functions with c[i] <= 0
        self.constraint_ueq = constraint_ueq  # a list of equal functions with ceq[i] = 0

        self.Chorm = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        self.has_constraint = False
        self.len_chrom = self.n_dim
        self.crtbp()

        # self.FitV_history = []
        self.best_x_history = []
        self.best_y_history = []

        self.all_history_Y = []
        self.all_history_FitV = []

    def crtbp(self):
        # create the population
        tmp = np.random.rand(self.size_pop, self.len_chrom)
        self.Chrom = tmp.argsort(axis = 1)
        return self.Chrom

    def chrom2x(self):
        self.X = self.Chrom
        return self.X

    def x2y(self):
        self.Y_raw = np.array([self.func(x) for x in self.X])
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    def ranking(self):
        # GA select the biggest one, but we want to minimize func, so we put a negative here
        self.FitV = -self.Y

    def selection_tournament_faster(self, tourn_size = 3):
        '''
        Select the best individual among *tournsize* randomly chosen
        Same with `selection_tournament` but much faster using numpy
        individuals,
        :param self:
        :param tourn_size:
        :return:
        '''
        aspirants_idx = np.random.randint(self.size_pop, size = (self.size_pop, tourn_size))
        aspirants_values = self.FitV[aspirants_idx]
        winner = aspirants_values.argmax(axis = 1)  # winner index in every team
        sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
        self.Chrom = self.Chrom[sel_index, :]
        return self.Chrom

    def crossover_pmx(self):
        '''
        Executes a partially matched crossover (PMX) on Chrom.
        For more details see [Goldberg1985]_.
        :param self:
        :return:
        .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
       salesman problem", 1985.
        '''
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        for i in range(0, size_pop, 2):
            Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
            cxpoint1, cxpoint2 = np.random.randint(0, self.len_chrom - 1, 2)
            if cxpoint1 >= cxpoint2:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1 + 1
            # crossover at the point cxpoint1 to cxpoint2
            pos1_recorder = {value: idx for idx, value in enumerate(Chrom1)}
            pos2_recorder = {value: idx for idx, value in enumerate(Chrom2)}
            for j in range(cxpoint1, cxpoint2):
                value1, value2 = Chrom1[j], Chrom2[j]
                pos1, pos2 = pos1_recorder[value1], pos2_recorder[value2]
                Chrom1[j], Chrom1[pos1] = Chrom1[pos1], Chrom1[j]
                Chrom2[j], Chrom2[pos2] = Chrom2[pos2], Chrom2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2

            self.Chrom[i], self.Chrom[i + 1] = Chrom1, Chrom2
        return self.Chrom

    def mutation_TSP_1(self):
        for i in range(self.size_pop):
            for j in range(self.n_dim):
                if np.random.rand() < self.prob_mut:
                    n = np.random.randint(0, self.len_chrom, 1)
                    self.Chrom[i, j], self.Chrom[i, n] = self.Chrom[i, n], self.Chrom[i, j]
        return self.Chrom

    def run(self, max_iter = None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.X = self.chrom2x()
            self.Y = self.x2y()
            self.ranking()
            self.selection_tournament_faster()
            self.crossover_pmx()
            self.mutation_TSP_1()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.best_x_history.append(self.X[generation_best_index, :])
            self.best_y_history.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

        global_best_index = np.array(self.best_y_history).argmin()
        global_best_X = self.best_x_history[global_best_index]
        global_best_Y = self.func(global_best_X)
        return global_best_X, global_best_Y

    def get_name(self):
        return 'GA_TSP'


if __name__ == '__main__':

    cities_file = ["5_city.txt", "10_city.txt", "20_city.txt", "30_city.txt", "50_city.txt"]
    for city_file in cities_file:
        print(city_file)
        best_d = float('inf')
        best_rountine = None
        best_y_history = []
        func_name = None

        city = pd.read_csv(city_file, header = None, names = ['id', "x", "y"])

        points_coordinate = city[['x', 'y']].values[1:-1]  # generate coordinate of points
        num_points = len(points_coordinate)
        # num_iter = num_points * 10

        # 各个城市之间的距离矩阵
        distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric = 'euclidean')
        # print(f'distance_matrix:\n{distance_matrix}')

        def cal_total_distance(routine):
            num_points, = routine.shape
            return sum(
                [distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

        for num_iter in range(num_points * 10, 5000, 200):
            f = GA_TSP(func = cal_total_distance, n_dim = num_points, size_pop = 20, max_iter = num_iter,
                       prob_mut = 0.05)
            best_points, best_distance = f.run()
            func_name = f.get_name()
            if best_d > best_distance:
                best_d = best_distance
                best_rountine = best_points
                best_y_history = f.best_y_history
                print(num_iter)
                print(func_name, best_points, best_distance, cal_total_distance(best_points))

        fig, ax = plt.subplots(1, 2, figsize = (10, 8))
        ax[0].set_title(func_name+city_file.split('_')[0])
        ax[0].plot(best_y_history)
        ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Distance")
        ax[1].set_title(func_name + '_' + city_file.split('_')[0])
        best_points_ = np.concatenate([best_rountine, [best_rountine[0]]])
        best_points_coordinate = points_coordinate[best_points_, :]  # 点的坐标
        ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
                   marker = 'o', markerfacecolor = 'blue', color = 'red', linestyle = '-')
        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[1].set_xlabel("Longitude")
        ax[1].set_ylabel("Latitude")
        plt.subplots_adjust(wspace = 0.4)
        path_name = city_file.split('_')[0] + '.jpg'
        plt.savefig(os.path.join('GA', path_name))
        # plt.show()
        with open('GA/log.txt', 'a') as f:
            f.write(str(best_d) + '\n')
            best_rountine = [str(i) for i in best_rountine]
            f.write(' '.join(best_rountine))
        # plt.show()
        print(best_d)
        print(best_rountine)



