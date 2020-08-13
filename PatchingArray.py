#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

# 给定一个已经排序好的数组，问：最少需要往数据中添加多少个数才能使得1-n的数都能由数组中若干个数的和得到
# 并返回添加数组之后的数组
# 注意数组中的元素不能重复使用
# 解析：https://www.cnblogs.com/grandyang/p/5165821.html


class Solution:
    def minPatches(self, nums, n):
        ans = 0
        i = 0
        a = 0  # 数组元素加和
        while a < n:
            if i < len(nums) and nums[i] <= a + 1:
                a += nums[i]
                i += 1
            else:
                ans += 1
                a += a + 1
        return ans

    def minPatches1(self, nums, n):
        miss = 1
        k = len(nums)
        i = 0
        while miss <= n:
            if i >= len(nums) or nums[i] > miss:
                nums.insert(i, miss)
            miss += nums[i]
            i += 1
        return len(nums) - k, nums


if __name__ == '__main__':
    s = Solution()
    nums = [1, 2, 4, 11, 30]
    # nums = [1, 2]
    ans = s.minPatches(nums, 50)
    print(ans)
    anss, new_nums = s.minPatches1(nums, 50)
    print(anss)
    assert anss == ans

