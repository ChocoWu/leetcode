#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

"""
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        flag = 0
        res = ListNode(0)
        r = res
        while l1 or l2:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            temp = x + y + flag
            flag = temp // 10
            r.next = ListNode(temp % 10)
            r = r.next
            if l1 is None:
                l1 = l1.next
            if l2 is None:
                l2 = l2.next
        if flag > 0:
            r.next = ListNode(flag)
        return res.next
