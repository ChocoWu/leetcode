#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import math
import copy
import numpy as np
import pandas as pd
from scipy import spatial
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import os


class Tabu_TSP():
    def __init__(self, city, n_points, size_pop=10, max_iter=20):
        self.city = city
        self.n_points = n_points
        self.size_pop = size_pop
        self.max_iter = max_iter

        self.distance_matrix = None
        self.first_solution = None
        self.distance_of_first_solution = None

        self.best_x_history = []  # 记录各代的最佳情况
        self.best_y_history = []  # 记录各代的最佳情况
        self.best_x, self.best_y = None, None
        self.y_history = []

        self.coordinate_dict = {}
        self.init_citys()
        self.generate_distance_matrix()
        self.generate_first_solution()

    def init_citys(self):
        """
        输入的参数为城市的数目, 返回的size个城市的位置坐标
        :param size:[5, 10, 20, 30, 50]
        :return:
        """
        with open(self.city, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split(',')
                index = int(l[0])
                x = float(l[1])
                y = float(l[2])
                self.coordinate_dict[index] = (x, y)

    def generate_distance_matrix(self):  # 生成距离矩阵
        d = np.zeros((self.n_points + 2, self.n_points + 2))
        for i in range(self.n_points + 1):
            for j in range(self.n_points + 1):
                if i == j:
                    continue
                if d[i][j] != 0:
                    continue
                x1 = self.coordinate_dict[i][0]
                y1 = self.coordinate_dict[i][1]
                x2 = self.coordinate_dict[j][0]
                y2 = self.coordinate_dict[j][1]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if i == 0:
                    d[i][j] = d[self.n_points + 1][j] = d[j][i] = d[j][self.n_points + 1] = distance
                else:
                    d[i][j] = d[j][i] = distance
        self.distance_matrix = d

    def generate_first_solution(self):
        """
        产生初始化的解
        :param size: 旅行商问题中城市的数目
        :return:
        """
        path_list = list(range(self.n_points + 2))
        self.first_solution = self.shuffle(path_list)
        self.distance_of_first_solution = self.path_length(self.distance_matrix, np.array(self.first_solution)[1:-1]-1, self.n_points)

    def shuffle(self, my_list):  # 起点和终点不能打乱
        temp_list = my_list[1:-1]
        np.random.shuffle(temp_list)
        shuffle_list = my_list[:1] + temp_list + my_list[-1:]
        return shuffle_list

    def path_length(self, d_matrix, path_list, size):  # 计算路径长度
        length = 0
        for i in range(size):
            if i == size - 1:
                length += d_matrix[path_list[i]+1][path_list[0]+1]
            else:
                length += d_matrix[path_list[i]+1][path_list[i + 1]+1]
        return length

    def find_neighborhood(self, path_list):
        """
        对提供的path_list进行一步操作，然后产生其邻域
        :param path_list:
        :param d_matrix:
        :return:
        """
        neighborhood_of_solution = []
        for n in path_list[1:-1]:
            idx1 = path_list.index(n)
            for kn in path_list[1:-1]:
                idx2 = path_list.index(kn)
                if n == kn:
                    continue
                _tmp = copy.deepcopy(path_list)
                _tmp[idx1] = kn
                _tmp[idx2] = n

                distance = self.path_length(self.distance_matrix, np.array(_tmp)[1:-1]-1, len(path_list) - 2)
                _tmp.append(distance)

                if _tmp not in neighborhood_of_solution:
                    neighborhood_of_solution.append(_tmp)

        indexOfLastItemInTheList = len(neighborhood_of_solution[0]) - 1

        neighborhood_of_solution.sort(key = lambda x: x[indexOfLastItemInTheList])
        return neighborhood_of_solution

    def run(self):
        """

        :param first_solution: 初始化解路径
        :param distance_of_first_solution: 初始化路径的距离
        :param d_matrix: 各个城市之间的距离
        :param iters: 迭代的次数
        :param size: 禁忌表的大小
        :return: best_solution_ever, best_cost
        """
        count = 1
        solution = self.first_solution
        tabu_list = list()
        best_cost = self.distance_of_first_solution
        self.best_x_history.append(np.array(self.first_solution))
        self.best_y_history.append(self.distance_of_first_solution)
        while count <= self.max_iter:
            neighborhood = self.find_neighborhood(solution)
            index_of_best_solution = 0
            best_solution = neighborhood[index_of_best_solution]
            best_cost_index = len(best_solution) - 1

            found = False
            while found is False:
                i = 0
                while i < len(best_solution):
                    if best_solution[i] != solution[i]:
                        first_exchange_node = best_solution[i]
                        second_exchenge_node = solution[i]
                        break
                    i = i + 1

                if [first_exchange_node, second_exchenge_node] not in tabu_list and \
                        [second_exchenge_node, first_exchange_node] not in tabu_list:
                    tabu_list.append([first_exchange_node, second_exchenge_node])
                    found = True
                    solution = best_solution[:-1]
                    cost = neighborhood[index_of_best_solution][best_cost_index]
                    self.y_history.append(cost)
                    if cost < best_cost:
                        best_cost = cost
                        self.best_x_history.append(np.array(solution))
                        self.best_y_history.append(cost)
                else:
                    index_of_best_solution = index_of_best_solution + 1
                    if index_of_best_solution > len(neighborhood) - 1:
                        break
                    best_solution = neighborhood[index_of_best_solution]
            if len(tabu_list) >= self.size_pop:
                tabu_list.pop(0)
            count += count
        best_generation = np.array(self.best_y_history).argmin()
        self.best_x = self.best_x_history[best_generation]
        self.best_y = self.best_y_history[best_generation]
        return self.best_x, self.best_y

    def get_name(self):
        return 'Tabu_TSP'


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
        # distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric = 'euclidean')
        # print(f'distance_matrix:\n{distance_matrix}')

        for num_iter in range(num_points * 10, 5000, 200):
            f = Tabu_TSP(city = city_file, n_points = num_points, size_pop = 20, max_iter = num_iter)
            func_name = f.get_name()
            best_points, best_distance = f.run()
            best_points = best_points[1:-1] - 1
            if best_d > best_distance:
                best_d = best_distance
                best_rountine = best_points
                y_history = f.best_y_history
                print(num_iter)
                print(func_name, best_points[1:-1], best_distance, f.path_length(f.distance_matrix, best_points, num_points))

        fig, ax = plt.subplots(1, 2, figsize = (10, 8))
        ax[0].set_title(func_name+city_file.split('_')[0])
        ax[0].plot(y_history)
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
        plt.savefig(os.path.join('Tabu', path_name))
        # plt.show()
        with open('Tabu/log.txt', 'w') as f:
            f.write(str(best_d) + '\n')
            best_rountine = [str(i) for i in best_rountine]
            f.write(' '.join(best_rountine))
        # plt.show()
        print(best_d)
        print(best_rountine)
