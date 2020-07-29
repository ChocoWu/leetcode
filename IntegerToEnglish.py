#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

"""
将一个非负整数转换成英文表达式形式，给定的数小于2^31-1
example:
input:    1234567
output:   One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven
"""


def get(num):

    # 不超过19的整数表达
    ge = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Sixteen', 'Seventeen', 'Nineteen']
    # 十位不小于2的表达
    shi = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
    gewei = num % 10
    shiwei = (num // 10) % 10
    baiwei = num // 100

    pre = 0
    ans = ''
    if baiwei != 0:
        pre = 1
        ans = ans + ge[baiwei] + ' Hundred'
    if shiwei > 1:
        if pre:
            ans = ans + ' '
        pre = 1
        ans = ans + shi[shiwei]
    if shiwei == 1 or gewei != 0:
        if shiwei == 1:
            gewei = gewei + 10
        if pre:
            ans = ans + ' '
        ans = ans + ge[gewei]
    return ans


def number_to_words(num):
    # num : int
    if num == 0:
        return "Zero"
    rear = ["", "Thousand", "Million", "Billion"]

    nums = []
    for i in range(4):
        nums.append(num % 1000)
        num = num // 1000
    ans = ""
    pre = 0
    for i in range(3, -1, -1):
        if not nums[i]:
            continue
        if pre:
            ans = ans + " "
        pre = 1
        ans = ans + get(nums[i]) + " " + rear[i]

    return ans


if __name__ == '__main__':
    num = 1234567
    print(number_to_words(num))
