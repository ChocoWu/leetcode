#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

"""
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        substring = {}
        i = 0
        max_len = 0
        sub_len = 0
        while i < len(s):
            if s[i] in substring:
                i = substring[s[i]]
                i += 1
                if len(substring) > max_len:
                    max_len = len(substring)
                substring = {}
            else:
                substring[s[i]] = i
                sub_len = len(substring)
                i += 1
        if max_len > sub_len:
            return max_len
        else:
            return sub_len

    def func2(self, s: str) -> int:
        """
        返回最长的结果，则每次找到重复的就需要将索引i定为不重复的位置，j - i + 1 表示当前找到的不重复的子序列的长度，
        ans表示目前为止最长的子序列的长度
        :param s:
        :return:
        """
        st = {}
        i, ans = 0, 0
        for j in range(len(s)):
            if s[j] in st:
                i = max(st[s[j]], i)
            ans = max(ans, j - i + 1)
            st[s[j]] = j + 1
        return ans


if __name__ == '__main__':
    s = Solution()
    print(s.func2("abcabc"))
