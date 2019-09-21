#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu


class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __repr__(self):
        return str(self.data)

    def isEmpty(self):
        return self.length == 0

    def update(self, dataOrNode):
        item = None
        if isinstance(dataOrNode, Node):
            item = dataOrNode
        else:
            item = Node(dataOrNode)
        if not self.head:
            self.head = item
            self.length += 1
        else:
            node = self.head
            while node.next:
                node = node.next
            node.next = item
            self.length += 1

    def delete(self, index):
        if self.isEmpty():
            print("this chain table is empty")
            return
        if index > self.length:
            print('out of index')
            return
        if index == 0:
            # 删除的为头节点
            self.head = self.head.next
            self.length -= 1
            return


if __name__ == '__main__':
    res = Node(10)
    print(res.data)


