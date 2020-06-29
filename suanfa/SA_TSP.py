import numpy as np
import pandas as pd
from scipy import spatial
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import os


class SA_TSP:
    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=50, max_stay_counter=150, **kwargs):
        assert T_max > T_min > 0, 'T_max > T_min > 0'

        self.func = func
        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（also called Long of Chain）
        self.max_stay_counter = max_stay_counter  # stop if best_y stay unchanged over max_stay_counter times

        self.n_dims = len(x0)

        self.best_x = np.array(x0)  # initial solution
        self.best_y = self.func(self.best_x)
        self.T = self.T_max
        self.iter_cycle = 0
        self.best_y_history = [self.best_y]
        self.best_x_history = [self.best_x]

    def get_new_x(self, x):
        x_new = x.copy()
        SWAP, REVERSE, TRANSPOSE = 0, 1, 2

        def swap(x_new):
            n1, n2 = np.random.randint(0, len(x_new) - 1, 2)
            if n1 >= n2:
                n1, n2 = n2, n1 + 1
            x_new[n1], x_new[n2] = x_new[n2], x_new[n1]
            return x_new

        def reverse(x_new):
            n1, n2 = np.random.randint(0, len(x_new) - 1, 2)
            if n1 >= n2:
                n1, n2 = n2, n1 + 1
            x_new[n1:n2] = x_new[n1:n2][::-1]

            return x_new

        def transpose(x_new):
            # randomly generate n1 < n2 < n3. Notice: not equal
            n1, n2, n3 = sorted(np.random.randint(0, len(x_new) - 2, 3))
            n2 += 1
            n3 += 2
            slice1, slice2, slice3, slice4 = x_new[0:n1], x_new[n1:n2], x_new[n2:n3 + 1], x_new[n3 + 1:]
            x_new = np.concatenate([slice1, slice3, slice2, slice4])
            return x_new

        new_x_strategy = np.random.randint(3)
        if new_x_strategy == SWAP:
            x_new = swap(x_new)
        elif new_x_strategy == REVERSE:
            x_new = reverse(x_new)
        elif new_x_strategy == TRANSPOSE:
            x_new = transpose(x_new)

        return x_new

    def cool_down(self):
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0
        while True:
            for i in range(self.L):
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                # Metropolis
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.best_y_history.append(self.best_y)
            self.best_x_history.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                stop_code = 'Cooled to final temperature'
                break
            if stay_counter > self.max_stay_counter:
                stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break

        return self.best_x, self.best_y

    def get_name(self):
        return 'SA_TSP'


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
            f = SA_TSP(func = cal_total_distance, x0 = range(num_points), T_max = 100, T_min = 1, L = num_iter)
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
        plt.savefig(os.path.join('SA', path_name))
        # plt.show()
        with open('SA/log.txt', 'a') as f:
            f.write(str(best_d) + '\n')
            best_rountine = [str(i) for i in best_rountine]
            f.write(' '.join(best_rountine))
        # plt.show()
        print(best_d)
        print(best_rountine)
