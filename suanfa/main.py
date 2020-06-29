import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from ACA_TSP import ACA_TSP
from SA_TSP import SA_TSP
from GA_TSP import GA_TSP
from Tabu_TSP import Tabu_TSP
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import os


def calculate(city_file, algorithm="GA"):
    city = pd.read_csv(city_file, header=None, names=['id', "x", "y"])

    points_coordinate = city[['x', 'y']].values[1:-1]  # generate coordinate of points
    num_points = len(points_coordinate)
    num_iter = 500

    # 各个城市之间的距离矩阵
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    # print(f'distance_matrix:\n{distance_matrix}')

    # 计算routine路线的总距离长度
    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    f = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=num_iter)
    if algorithm is 'ACA':
        f = ACA_TSP(func=cal_total_distance, n_points=num_points,
                      size_pop=20, max_iter=num_iter,
                      distance_matrix=distance_matrix)
    elif algorithm is 'GA':
        f = GA_TSP(func = cal_total_distance, n_dim = num_points, size_pop = 20, max_iter = num_iter,
                   prob_mut=0.05)
    elif algorithm is 'Tabu':
        f = Tabu_TSP(city=city_file, n_points=num_points, size_pop=20, max_iter=num_iter)

    if algorithm is 'Tabu':
        func_name = f.get_name()
        best_points, best_distance = f.run()
        best_points = best_points[1:-1] - 1
        print(func_name, best_points, best_distance, f.path_length(f.distance_matrix, best_points, num_points))
    else:
        func_name = f.get_name()
        best_points, best_distance = f.run()
        print(func_name, best_points, best_distance, cal_total_distance(best_points))

    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].set_title(func_name)
    ax[0].plot(f.best_y_history)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Distance")
    ax[1].set_title(func_name)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]  # 点的坐标
    ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
                marker='o', markerfacecolor='blue', color='red', linestyle='-')
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    plt.subplots_adjust(wspace=0.4)
    path_name = '_'.join([city_file.split('_')[0], algorithm]) + '.jpg'
    plt.savefig(os.path.join('img', path_name))
    # plt.show()

    return num_points, cal_total_distance(best_points)


def plot_distance():

    cities_file = ["5_city.txt", "10_city.txt", "20_city.txt", "30_city.txt", "50_city.txt"]
    sa_distance = []
    aca_distance = []
    ga_distance = []
    tabu_distance = []
    for city in cities_file:
        sa_n_points, sa_d = calculate(city, algorithm='SA')
        sa_distance.append([sa_n_points, sa_d])
        aca_n_points, aca_d = calculate(city, algorithm='ACA')
        aca_distance.append([aca_n_points, aca_d])
        ga_n_points, ga_d = calculate(city, algorithm = 'GA')
        ga_distance.append([ga_n_points, ga_d])
        tabu_n_points, tabu_d = calculate(city, algorithm = 'Tabu')
        tabu_distance.append([tabu_n_points, tabu_d])
    sa_distance = np.array(sa_distance)
    aca_distance = np.array(aca_distance)
    ga_distance = np.array(ga_distance)
    tabu_distance = np.array(tabu_distance)
    print(f'sa_distance: {sa_distance}')
    print(f"aca_distance: {aca_distance}")
    print(f'ga_distance.{ga_distance}')
    print(f'tabu_distance.{tabu_distance}')
    figure, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(sa_distance[:, 0], sa_distance[:, 1],
             marker='o', markerfacecolor='blue', color='blue', linestyle='-', label='SA_TSP')
    ax1.plot(aca_distance[:, 0], aca_distance[:, 1],
             marker='o', markerfacecolor='green', color='green', linestyle='-', label='ACA_TSP')
    ax1.plot(ga_distance[:, 0], ga_distance[:, 1],
             marker='o', markerfacecolor='yellow', color='yellow', linestyle='-', label='GA_TSP')
    ax1.plot(tabu_distance[:, 0], tabu_distance[:, 1],
             marker = 'o', markerfacecolor = 'orange', color = 'orange', linestyle = '-', label = 'Tabu_TSP')
    ax1.legend()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_xlabel("num_cities")
    ax1.set_ylabel("best_distance")
    plt.savefig('img/best_d.jpg')
    # plt.show()


if __name__ == '__main__':
    plot_distance()

