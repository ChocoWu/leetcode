#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

"""
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空。
    nums1 = [1, 3]
    nums2 = [2]
    则中位数是 2.0

先对数组进行拼接，然后进行排序
根据长度返回拼接后的中位数的长度
"""


class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """

        :param nums1: List
        :param nums2:  List
        :return: Float
        """
        nums1.extend(nums2)
        nums1 = sorted(nums1)
        if len(nums1) % 2 == 0:
            return (nums1[int(len(nums1) / 2) - 1] + nums1[int(len(nums1) / 2)]) / 2
        else:
            return nums1[int(len(nums1) / 2)]
    #     n = len(nums1) + len(nums2)
    #     if n % 2 == 1:
    #         return self.findKth(nums1, nums2, n / 2 + 1)
    #     else:
    #         smaller = self.findKth(nums1, nums2, n / 2)
    #         bigger = self.findKth(nums1, nums2, n / 2 + 1)
    #         return (smaller + bigger) / 2.0
    #
    # def findKth(self, A, B, k):
    #     if len(A) == 0:
    #         return B[int(k - 1)]
    #     if len(B) == 0:
    #         return A[int(k - 1)]
    #     if k == 1:
    #         return min(A[0], B[0])
    #
    #     a = A[int(k / 2) - 1] if len(A) >= k / 2 else None
    #     b = B[int(k / 2) - 1] if len(B) >= k / 2 else None
    #
    #     if b is None or (a is not None and a < b):
    #         return self.findKth(A[int(k / 2):], B, int(k - k // 2))


if __name__ == '__main__':
    nums1 = [1, 2, 3, 4, 5, 6]
    # nums2 = [2, 3, 4, 5]
    nums2 = [7, 8, 9, 10]
    # nums1 = [1, 2, 3, 4, 5, 6]
    print(sorted(nums1))
    s = Solution()
    print(s.findMedianSortedArrays(nums1, nums2))


