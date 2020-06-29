#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

"""
判断两棵树是否相同， 结构相同，在相同的节点处值相同
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.right = None
        self.left = None


class Solution:
    # def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    def isSameTree(self, p, q):
        """
        :param p: TreeNode
        :param q: TreeNode
        :return: bool
        """
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
