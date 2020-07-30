#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu
import math

"""
有一个正整数n，找到一系列的完全平方数（1，4，9，...）,它们的和为n，并且这些完全平方数的数量最少。返回完全平方数的最少数量。
1.贪心算法，从小于n的最大完全平方数开始查找，但是这个不一定是最优的，可能12=9+1+1+1，或者12=4+4+4，则最少应该是3
2. 直接遍历所有可能的完全平方数，当n=23时，即：
    a。先找离23最近的完全平方数为16，23=16+7，然后再找7的最少完全平方数，7=4+1+1+1，最后返回值为5
    b。再找完全平方数9，23=9+14，然后找14的最少完全平方数，14=9+4+1，最后返回值为4
    c。再找完全平方数4，23=4+19，然后找19的最少完全平方数，19=16+1+1+1，最后返回值为5
    d。再找1完全平方数1，23=1+22，然后再找22的最少完全平方数，22=9+9+4，则最后的返回值为4
    综上，则23的最少完全平方数为4
假设：整数n的完全平方数最少数量可以通过小于n的结果得到，满足动态规划的条件
"""


class Solution1:
    def num_squares(self, n: int) -> int:
        """
        每次处理当前步骤中所有可能的情况，一旦可以完成，则返回所需的步骤。
        :param n:
        :return:
        """
        from collections import deque
        if n == 0:
            return 0
        queue = deque([n])
        step = 0
        visited = set()

        while queue:
            step += 1
            l = len(queue)
            for _ in range(l):
                tmp = queue.pop()
                for i in range(1, int(tmp ** 0.5) + 1):
                    diff = tmp - i ** 2
                    if diff == 0:
                        return step
                    if diff not in visited:
                        visited.add(diff)
                        queue.appendleft(diff)
        return step


class Solution2:
    def num_sqares(self, n):
        """
        对solution1中的方法进行了一些裁剪，即考虑如果diff就是一个完全平方数，则无需再计算其他的，直接返回step+1
        :param n:
        :return:
        """
        from collections import deque
        if n == 0 or n == 1:
            return n
        if int(n ** 0.5) ** 2 == n:
            return 1
        queue = deque([n])
        candidates = set([i ** 2 for i in range(1, int(n ** 0.5)+1)])
        step = 0
        while queue:
            step += 1
            l = len(queue)
            for _ in range(l):
                tmp = queue.pop()
                for x in candidates:
                    val = tmp - x
                    if val in candidates:
                        return step + 1
                    elif val > 0:
                        queue.appendleft(val)


class Solution3:
    def num_squares(self, n):
        """
        使用动态规划
        dp[i] = min(dp[i], dp[i - j * j] + 1) j <= i ** 0.5 + 1
        :param n:
        :return:
        """
        dp = [float("inf")] * (n + 1)
        dp[0] = 0
        for i in range(1, n+1):
            for j in range(1, int(i**0.5)+1):
                dp[i] = min(dp[i], dp[i-j*j]+1)
        return dp[n]


class Solution4:
    _dp = [0]

    def num_squares(self, n):
        """
        dp[i] = min(dp[i], dp[- j * j] + 1) j <= i ** 0.5 + 1
        :param n:
        :return:
        """
        dp = self._dp
        while len(dp) <= n:
            dp += list((min(dp[-i * i] for i in range(1, int(len(dp) ** 0.5 + 1))) + 1,))
        return dp[n]


class Solution5:

    def is_square(self, n):
        sq = int(math.sqrt(n))
        return sq * sq == n

    def num_squares(self, n):
        """
        Lagrange 四平方定理： 任何一个正整数都可以表示成不超过四个整数的平方之和。
        :param n:
        :return:
        """
        if self.is_square(n):
            return 1
        while (n & 3) == 0:
            n >>= 2
        if (n & 7) == 7:
            return 4
        sq = int(math.sqrt(n)) + 1
        for i in range(1, sq):
            if self.is_square(n - i * i):
                return 2
        return 3


if __name__ == '__main__':
    s = Solution5()
    print(s.num_squares(12))


