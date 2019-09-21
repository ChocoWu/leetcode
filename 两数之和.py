#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

"""
给定一个整数数组 nums 和一个目标值 target，
请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
"""


class Solution:
    def twoSum(self, nums, target):
        hash_map = {}
        for idx, num in enumerate(nums):
            another_num = target - num
            if another_num in hash_map:
                return [hash_map[another_num], idx]
            hash_map[num] = idx
        return None


if __name__ == '__main__':
    s = Solution()
    res = s.twoSum([1, 9, 2, 3], 10)
    print(res)
