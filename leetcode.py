import re
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from copy import deepcopy
from functools import reduce
from itertools import combinations_with_replacement, permutations
from math import comb, factorial, inf
from operator import xor
from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.val) + ", " + str(self.left) + ", " + str(self.right)


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# Definition for a Node.
class Node:
    def __init__(self, x: int, next: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    # LC 1. Two Sum (Easy)
    # https://leetcode.com/problems/two-sum/
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # # Naive solution with double loops - O(n²)T, O(1)S
        # Memoization of seen numbers - O(n)TS
        d = {}
        for i, num in enumerate(nums):
            if num in d:
                return [d[num], i]
            d[target - num] = i

    # LC 2. Add Two Numbers (Medium)
    # https://leetcode.com/problems/add-two-numbers/
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        # Naive solution: get each node val as char, then get n1 and n2, then sum them and create the res linked list -> Won't work on bigint overflow test case
        # Optimized solution
        n1, n2 = l1, l2
        dummy = ListNode()
        node = dummy
        carry = 0
        while n1 or n2 or carry:
            val = carry
            if n1:
                val += n1.val
                n1 = n1.next
            if n2:
                val += n2.val
                n2 = n2.next
            carry, val = divmod(val, 10)
            new_node = ListNode(val)
            node.next = new_node
            node = new_node
        return dummy.next

    # LC 3. Longest Substring Without Repeating Characters
    # https://leetcode.com/problems/longest-substring-without-repeating-characters/
    def lengthOfLongestSubstring(self, s: str) -> int:
        # Hash table and double pointers - O(|s|)TS
        h = {}
        i, j, maxi = 0, 0, 0
        for j in range(len(s)):
            c = s[j]
            if c in h and i <= h[c]:
                i = h[c] + 1
            h[c] = j
            maxi = max(maxi, j - i + 1)
        return maxi

    # LC 4. Median of Two Sorted Arrays (Hard)
    # https://leetcode.com/problems/median-of-two-sorted-arrays/
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # # With package
        # from statistics import median
        # return median(nums1 + nums2)

        # # Without package
        # s = sorted(nums1 + nums2)
        # return (s[n//2-1]/2.0+s[n//2]/2.0, s[n//2])[n % 2] if n else None

        # Binary search - O(log(min(m,n)))T, O(1)S
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        imin, imax, half_len = 0, m, (m + n + 1) // 2
        while imin <= imax:
            i = (imin + imax) // 2
            j = half_len - i
            if i < m and nums2[j - 1] > nums1[i]:
                imin = i + 1
            elif i > 0 and nums1[i - 1] > nums2[j]:
                imax = i - 1
            else:
                if i == 0:
                    max_left = nums2[j - 1]
                elif j == 0:
                    max_left = nums1[i - 1]
                else:
                    max_left = max(nums1[i - 1], nums2[j - 1])
                if (m + n) % 2 == 1:
                    return float(max_left)
                if i == m:
                    min_right = nums2[j]
                elif j == n:
                    min_right = nums1[i]
                else:
                    min_right = min(nums1[i], nums2[j])
                return (max_left + min_right) / 2.0

    # LC 5. Longest Palindromic Substring
    # https://leetcode.com/problems/longest-palindromic-substring/
    def longestPalindrome(self, s: str) -> str:
        # # DP - O(n²)TS
        # dp = [[False]*len(s) for _ in range(len(s)) ]
        # for i in range(len(s)):
        #     dp[i][i]=True
        # ans=s[0]
        # for j in range(len(s)):
        #     for i in range(j):
        #         if s[i]==s[j] and (dp[i+1][j-1] or j==i+1):
        #             dp[i][j]=True
        #             if j-i+1>len(ans):
        #                 ans=s[i:j+1]
        # return ans

        # Recursive (expand from center solution) - O(n²)T, O(1)T
        def expand(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1 : r]

        res = ""
        for i in range(len(s)):
            subs1 = expand(i - 1, i + 1)
            subs2 = expand(i, i + 1)
            if len(subs1) > len(res):
                res = subs1
            if len(subs2) > len(res):
                res = subs2
        return res

    # LC 6. Zigzag Conversion
    # https://leetcode.com/problems/zigzag-conversion/
    def convert(self, s: str, numRows: int) -> str:
        # # My solution (double pass) - O(|s|)TS
        # if len(s) <= numRows or numRows == 1:
        #     return s
        # d = []
        # i = 0
        # while i < len(s):
        #     d.append(s[i : i + numRows])
        #     i += numRows
        #     j = numRows - 2
        #     while j and i < len(s):
        #         d.append(s[i])
        #         i += 1
        #         j -= 1
        # res = []
        # for m in range(numRows):
        #     for k in range(len(d)):
        #         if k % (numRows - 1) == 0:
        #             if m < len(d[k]):
        #                 res.append(d[k][m])
        #         else:
        #             if (k) % (numRows - 1) == numRows - m - 1:
        #                 res.append(d[k])
        # return "".join(res)

        # LC solution (single pass) - same but a bit faster
        if numRows == 1 or numRows >= len(s):
            return s
        rows = [[] for _ in range(numRows)]
        index = 0
        step = -1
        for char in s:
            rows[index].append(char)
            if index == 0:
                step = 1
            elif index == numRows - 1:
                step = -1
            index += step
        for i in range(numRows):
            rows[i] = "".join(rows[i])
        return "".join(rows)
