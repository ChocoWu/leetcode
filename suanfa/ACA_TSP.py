import numpy as np
import pandas as pd
from scipy import spatial
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import os


class ACA_TSP:
    def __init__(self, func, n_points,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 alpha=1, beta=2, rho=0.1,
                 ):
        self.func = func
        self.n_dim = n_points  # 城市数量
        self.size_pop = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_points, n_points))  # 避免除零错误

        self.Tau = np.ones((n_points, n_points))  # 信息素矩阵，每次迭代都会更新
        self.Table = np.zeros((size_pop, n_points)).astype(np.int)  # 某一代每个蚂蚁的爬行路径
        self.y = None  # 某一代每个蚂蚁的爬行总距离
        self.x_best_history, self.best_y_history = [], []  # 记录各代的最佳情况
        self.best_x, self.best_y = None, None

    def run(self):
        for i in range(self.max_iter):  # 对每次迭代
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta  # 转移概率，无须归一化。
            for j in range(self.size_pop):  # 对每个蚂蚁
                self.Table[j, 0] = 0  # start point，其实可以随机，但没什么区别
                for k in range(self.n_dim - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.Table[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # 在这些点中做选择
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # 概率归一化
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # 计算距离
            y = np.array([self.func(i) for i in self.Table])

            # 顺便记录历史最好情况
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :], y[index_best]
            self.x_best_history.append(x_best)
            self.best_y_history.append(y_best)

            # 计算需要新涂抹的信息素
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # 每个蚂蚁
                for k in range(self.n_dim - 1):  # 每个节点
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                    delta_tau[n1, n2] += 1 / y[j]  # 涂抹的信息素
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
                delta_tau[n1, n2] += 1 / y[j]  # 涂抹信息素

            # 信息素飘散+信息素涂抹
            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_generation = np.array(self.best_y_history).argmin()
        self.best_x = self.x_best_history[best_generation]
        self.best_y = self.best_y_history[best_generation]
        return self.best_x, self.best_y

    def get_name(self):
        return 'ACA_TSP'


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

        def cal_total_distance(routine):
            num_points, = routine.shape
            return sum(
                [distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


        for num_iter in range(num_points * 10, 5000, 200):
            f = ACA_TSP(func = cal_total_distance, n_points = num_points,
                        size_pop = 20, max_iter = num_iter,
                        distance_matrix = distance_matrix)
            best_points, best_distance = f.run()
            func_name = f.get_name()
            if best_d > best_distance:
                best_d = best_distance
                best_rountine = best_points
                best_y_history = f.best_y_history
                print(num_iter)
                print(func_name, best_points, best_distance, cal_total_distance(best_points))

        fig, ax = plt.subplots(1, 2, figsize = (10, 8))
        ax[0].set_title(func_name + '_' + city_file.split('_')[0])
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
        plt.savefig(os.path.join('ACA', path_name))
        # plt.show()
        with open('ACA/log.txt', 'a') as f:
            f.write(str(best_d) + '\n')
            best_rountine = [str(i) for i in best_rountine]
            f.write(' '.join(best_rountine))
        print(best_d)
        print(best_rountine)
