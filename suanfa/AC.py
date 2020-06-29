'''
ALPHA:信息启发因子,信息浓度对于路径选择所占的比重

BETA: 期望启发式因子,BETA越大,蚁群就越容易选择局部较短路径,
      这时算法的收敛速度加快;但是随机性却不高,容易得到局部的最优解.

RHO:  信息挥发因子,RHO过小,在各路径下残留的信息素过多,导致无效的路径
      被持续搜索,影响到算法的收敛速率;RHO过大无效的路径虽然可以被排除搜索,
      但是有效地路径也会被放弃搜索，影响到最后的最优值搜索.

'''
import copy
import random
import threading
import tkinter
from functools import reduce
import sys

(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
# 城市数目,蚁群数目
(city_num, ant_num) = (50, 50)

# 城市对应点的坐标
distance_x = [
    178, 272, 176, 171, 650, 499, 267, 703, 408, 437, 491, 74, 532, 416, 626,
    42, 271, 359, 163, 508, 229, 576, 147, 560, 35, 714, 757, 517, 64, 314,
    675, 690, 391, 628, 87, 240, 705, 699, 258, 428, 614, 36, 360, 482, 666,
    597, 209, 201, 492, 294]
distance_y = [
    170, 395, 198, 151, 242, 556, 57, 401, 305, 421, 267, 105, 525, 381, 244,
    330, 395, 169, 141, 380, 153, 442, 528, 329, 232, 48, 498, 265, 343, 120,
    165, 50, 433, 63, 491, 275, 348, 222, 288, 490, 213, 524, 244, 114, 104,
    552, 70, 425, 227, 331]

# 城市距离和信息素,采用二维数组的形式存储
# 初始化城市距离为0，信息素浓度为1
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]


class Ant(object):
    # 初始化
    def __init__(self, ID):
        self.ID = ID
        self.__clean_data()  # 初始化出生点

    # 初始化数据
    def __clean_data(self):
        self.path = []  # 当前蚂蚁的行走路径
        self.total_distance = 0.0  # 当前蚂蚁行走的总长度
        self.move_count = 0  # 当前蚂蚁的行走次数
        self.current_city = -1  # 当前蚂蚁的所在城市
        self.open_table_city = [True for each in range(city_num)]  # 探索城市的状态,True表示未被探索果,False表示已经被探索过

        city_index = random.randint(0, city_num - 1)  # 随机初始生成点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choose_next_city(self):
        next_city = -1
        select_city_prob = [0.0 for each in range(city_num)]  # 存储选择每个城市的概率
        total_prob = 0.0

        # 获取去每个城市的概率
        for index in range(city_num):
            if self.open_table_city[index]:  # 如果index城市没有被探索过,就计算选择这个城市的概率;已经探索过的话不可以再重探索这个城市
                try:
                    # 计算概率,与信息浓度成正比,与距离成反比
                    select_city_prob[index] = pow(pheromone_graph[self.current_city][index], ALPHA) * pow((1.0 / distance_graph[self.current_city][index]), BETA)
                    total_prob += select_city_prob[index]
                except ZeroDivisionError as e:
                    print('Ant ID:{ID},current city:{current},target city{target}'.format(ID=self.ID,current=self.current_city,target=index))
                    sys.exit(1)

        # 采用轮盘赌方法选择下一个行走的城市
        if total_prob > 0.0:
            # 产生一个随机概率
            temp_prob = random.uniform(0.0, total_prob)
            for index in range(city_num):
                if self.open_table_city[index]:
                    # 轮次相减
                    temp_prob -= select_city_prob[index]
                    if temp_prob < 0.0:
                        next_city = index
                        break

        # 如果next_city=-1,则没有利用轮盘赌求出要去的城市
        # 通过随机生成的方法生成一个城市
        if next_city == -1:
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # 如果next_city已经被访问过，则需要重新生成
                next_city = random.randint(0, city_num - 1)

        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0

        for i in range(1, city_num):
            end = self.path[i]
            start = self.path[i - 1]
            temp_distance += distance_graph[start][end]

        # 回路
        start = city_num-1
        end = 0
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):
        # 初始化数据
        self.__clean_data()

        # 搜索路径,遍历完所有的城市为止
        while self.move_count < city_num:
            next_city = self.__choose_next_city()
            # 移动
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()


class TSP(object):
    # 初始化
    def __init__(self, root, width=800, height=600, city_num=50):

        # 创建画布
        self.root = root
        self.width = width
        self.height = height

        # 城市数目初始化为city_num
        self.city_num = city_num

        self.canvas = tkinter.Canvas(root,
                                     width=self.width,
                                     height=self.height,
                                     bg='#EBEBEB',
                                     xscrollincrement=1,
                                     yscrollincrement=1)

        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("TSP蚁群算法(n:初始化,e:开始搜索,s:停止搜索,q:退出程序)")
        self.__r = 5
        self.__lock = threading.RLock()  # 线程锁
        self.bindEvents()
        self.new()

        # 计算城市之间的距离
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow(distance_x[i] - distance_x[j], 2) + pow(distance_y[i] - distance_y[j], 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] = float(int(temp_distance + 0.5))

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best = Ant(-1)  # 初始最优解
        self.best.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 更改标题
    def title(self, s):
        self.root.title(s)

    # 初始化
    def new(self, evt=None):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()  # 清除信息
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象

        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上画出初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆,半径为self.__r
            node = self.canvas.create_oval(
                x - self.__r,
                y - self.__r,
                x + self.__r,
                y + self.__r,
                fill="#ff0000",  # 填充白色
                outline="#000000",  # 轮框白色
                tags="node")
            self.nodes2.append(node)

            # 显示坐标
            self.canvas.create_text(x,
                                    y - 10,
                                    text='(' + str(x) + ',' + str(y) + ')',
                                    fill='black')

        # 初始化城市之间的信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0
        self.best = Ant(-1)  # 初始最优解
        self.best.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按照order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
            return i2

        # order[-1]初始点
        reduce(line2, order, order[-1])

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quit(self, evt=None):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print("\n程序已经退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt=None):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    # 开始搜索
    def search_path(self, evt=None):
        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索每一条路径
                ant.search_path()
                # 判断是否是最优解
                if ant.total_distance < self.best.total_distance:
                    self.best = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_graph()
            print("迭代次数:", self.iter, "最佳路径总距离", int(self.best.total_distance))
            # 连线
            self.line(self.best.path)
            # 设置标题
            self.title("TSP蚁群算法(n:初始化,e:开始搜索,s:停止搜索,q:退出程序),迭代次数:%d,总距离%d" % (self.iter,int(self.best.total_distance)))
            # 更新画布
            self.canvas.update()
            self.iter += 1

        # 更新信息素

    def __update_pheromone_graph(self):
        temp_pheromone = [[0.0 for i in range(city_num)] for j in range(city_num)]

        for ant in self.ants:
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                # 留下的信息素浓度与蚂蚁所走路径总距离成反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]
            # 更新尾部到头部的信息浓度
            temp_pheromone[ant.path[city_num-1]][ant.path[0]] += Q/ant.total_distance
            temp_pheromone[ant.path[0]][ant.path[city_num-1]] = temp_pheromone[ant.path[city_num-1]][ant.path[0]]

        # 更新信息素,新的信息素等于信息素增量加上没有挥发掉的信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = (1 - RHO) * pheromone_graph[i][j] + temp_pheromone[i][j]

        # 按键响应

    def bindEvents(self):

        self.root.bind("q", self.quit)  # 退出程序
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)  # 停止搜索
        self.root.bind("n", self.new)  # 初始化

        # 主循环

    def mainloop(self):
        self.root.mainloop()


if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()
