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
            if target - num in hash_map:
                return [idx, hash_map[target-num]]
            hash_map[target-num] = idx
