import bisect
import collections
import functools
import heapq
import itertools
import math
import re
from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict, deque
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

        # # Binary search - O(log(min(m,n)))T, O(1)S
        # if len(nums1) > len(nums2):
        #     nums1, nums2 = nums2, nums1
        # m, n = len(nums1), len(nums2)
        # imin, imax, half_len = 0, m, (m + n + 1) // 2
        # while imin <= imax:
        #     i = (imin + imax) // 2
        #     j = half_len - i
        #     if i < m and nums2[j - 1] > nums1[i]:
        #         imin = i + 1
        #     elif i > 0 and nums1[i - 1] > nums2[j]:
        #         imax = i - 1
        #     else:
        #         if i == 0:
        #             max_left = nums2[j - 1]
        #         elif j == 0:
        #             max_left = nums1[i - 1]
        #         else:
        #             max_left = max(nums1[i - 1], nums2[j - 1])
        #         if (m + n) % 2 == 1:
        #             return float(max_left)
        #         if i == m:
        #             min_right = nums2[j]
        #         elif j == n:
        #             min_right = nums1[i]
        #         else:
        #             min_right = min(nums1[i], nums2[j])
        #         return (max_left + min_right) / 2.0

        # Binary search (LC solution) - O(log(min(m,n)))T, O(1)S
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        low, high = 0, m
        while low <= high:
            partitionX = (low + high) // 2
            partitionY = (m + n + 1) // 2 - partitionX
            maxX = float("-inf") if partitionX == 0 else nums1[partitionX - 1]
            maxY = float("-inf") if partitionY == 0 else nums2[partitionY - 1]
            minX = float("inf") if partitionX == m else nums1[partitionX]
            minY = float("inf") if partitionY == n else nums2[partitionY]
            if maxX <= minY and maxY <= minX:
                if (m + n) % 2 == 0:
                    return (max(maxX, maxY) + min(minX, minY)) / 2
                else:
                    return max(maxX, maxY)
            elif maxX > minY:
                high = partitionX - 1
            else:
                low = partitionX + 1

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

        # Recursive (expand from center solution) - O(n²)T, O(n)S
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

    # LC 8. String to Integer (atoi) (Medium)
    # https://leetcode.com/problems/string-to-integer-atoi/
    def myAtoi(self, s):
        # Lots of edge cases, not worth doing it so here is the final solution
        i = 0
        n = len(s)
        while i < n and s[i] == " ":  # skipping space characters at the beginning
            i += 1
        positive = 0
        negative = 0
        if i < n and s[i] == "+":
            positive += 1  # number of positive signs at the start in string
            i += 1
        if i < n and s[i] == "-":
            negative += 1  # number of negative signs at the start in string
            i += 1
        ans = 0.0
        while i < n and "0" <= s[i] <= "9":
            ans = ans * 10 + (ord(s[i]) - ord("0"))  # converting character to integer
            i += 1
        if negative > 0:  # if negative sign exists
            ans = -ans
        if (
            positive > 0 and negative > 0
        ):  # if both +ve and -ve signs exist, Example: +-12
            return 0
        INT_MAX = 2**31 - 1
        INT_MIN = -(2**31)
        if ans > INT_MAX:  # if ans > 2^31 - 1
            ans = INT_MAX
        if ans < INT_MIN:  # if ans < -2^31
            ans = INT_MIN
        return int(ans)

    # LC 9. Palindrome Number (Easy)
    # https://leetcode.com/problems/palindrome-number/
    def isPalindrome(self, x: int) -> bool:
        # # One-line code
        # return str(x) == str(x)[::-1]

        # # Half str check
        # from math import floor, ceil
        # s = str(x)
        # mid = len(s) / 2
        # left = floor(mid)
        # right = ceil(mid)
        # return s[:left] == s[right:][::-1]

        # Without str convertion
        if x < 0:
            return False
        reversed_x = 0
        n = x
        while x > 0:
            last_digit = x % 10
            x = x // 10
            reversed_x = reversed_x * 10 + last_digit
        return n == reversed_x

    # LC 10. Regular Expression Matching (Hard)
    # https://leetcode.com/problems/regular-expression-matching/
    def isMatch(self, s: str, p: str) -> bool:
        # # Cheating with built-in function
        # p = p.replace("**", "")
        # return re.fullmatch(p, s)

        # DP - O(mn)TS
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[0][j] = dp[0][j - 2]
            else:
                dp[0][j] = j > 1 and p[j - 2] == "*" and dp[0][j - 2]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == ".":
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == "*":
                    dp[i][j] = (
                        dp[i][j - 2]
                        or (p[j - 2] == s[i - 1] or p[j - 2] == ".")
                        and dp[i - 1][j]
                    )
                else:
                    dp[i][j] = False
        return dp[m][n]

    # LC 11. Container With Most Water (Medium)
    # https://leetcode.com/problems/container-with-most-water/
    def maxArea(self, height: List[int]) -> int:
        # O(n)T, O(1)S
        maxi, start, end = 0, 0, len(height) - 1
        while start != end:
            cur_area = min(height[start], height[end]) * (end - start)
            maxi = max(maxi, cur_area)
            if height[start] < height[end]:
                start += 1
            else:
                end -= 1
        return maxi

    # LC 12. Integer to Roman (Medium)
    # https://leetcode.com/problems/integer-to-roman/
    def intToRoman(self, num: int) -> str:
        d = OrderedDict(
            {
                1000: "M",
                900: "CM",
                500: "D",
                400: "CD",
                100: "C",
                90: "XC",
                50: "L",
                40: "XL",
                10: "X",
                9: "IX",
                5: "V",
                4: "IV",
                1: "I",
            }
        )
        res = ""
        for n in d.keys():
            while n <= num:
                res += d[n]
                num -= n
        return res

    # LC 13. Roman to Integer (Easy)
    # https://leetcode.com/problems/roman-to-integer/
    def romanToInt(self, s: str) -> int:
        # Constraint: 1 <= num <= 3999
        d = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
            "IV": 4,
            "IX": 9,
            "XL": 40,
            "XC": 90,
            "CD": 400,
            "CM": 900,
        }
        i, res = 0, 0
        while i < len(s):
            val = d.get(s[i : i + 2], 0)
            if val:
                res += val
                i += 2
                continue
            res += d[s[i : i + 1]]
            i += 1
        return res

    # LC 14. Longest Common Prefix
    # https://leetcode.com/problems/longest-common-prefix/
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # # First solution: sort by s lengths and compare all to s0
        # if len(strs) == 1:
        #     return strs[0]
        # strs = sorted(strs, key=len)
        # res = strs[0]
        # ls0 = len(res)
        # if ls0 == 0:
        #     return res
        # for s in strs[1:]:
        #     tmp = ""
        #     for j in range(len(res)):
        #         if res[j] == s[j]:
        #             tmp += s[j]
        #         else:
        #             break
        #     res = tmp
        # return res

        # LC solution: sort alphabetically and compare first & last
        ans = ""
        v = sorted(strs)
        first = v[0]
        last = v[-1]
        for i in range(min(len(first), len(last))):
            if first[i] != last[i]:
                return ans
            ans += first[i]
        return ans

    # LC 15. 3Sum (Medium)
    # https://leetcode.com/problems/3sum/
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # # O(n^2) with sort
        # nums.sort()
        # answer = []
        # for i in range(len(nums) - 2):
        #     if nums[i] > 0:
        #         break
        #     if i > 0 and nums[i] == nums[i - 1]:
        #         continue
        #     l = i + 1
        #     r = len(nums) - 1
        #     while l < r:
        #         total = nums[i] + nums[l] + nums[r]
        #         if total < 0:
        #             l += 1
        #         elif total > 0:
        #             r -= 1
        #         else:
        #             triplet = [nums[i], nums[l], nums[r]]
        #             answer.append(triplet)
        #             while l < r and nums[l] == triplet[1]:
        #                 l += 1
        #             while l < r and nums[r] == triplet[2]:
        #                 r -= 1
        # return answer

        # Solution with triple lists (negatives, zeros, positives)
        res = set()
        # 1. Split nums into three lists: negative numbers, positive numbers, and zeros
        n, p, z = [], [], []
        for num in nums:
            if num > 0:
                p.append(num)
            elif num < 0:
                n.append(num)
            else:
                z.append(num)
        # 2. Create a separate set for negatives and positives for O(1) look-up times
        N, P = set(n), set(p)
        # 3. If there is at least 1 zero in the list, add all cases where -num exists in N and num exists in P
        #   i.e. (-3, 0, 3) = 0
        if z:
            for num in P:
                if -1 * num in N:
                    res.add((-1 * num, 0, num))
        # 3. If there are at least 3 zeros in the list then also include (0, 0, 0) = 0
        if len(z) >= 3:
            res.add((0, 0, 0))
        # 4. For all pairs of negative numbers (-3, -1), check to see if their complement (4)
        #   exists in the positive number set
        for i in range(len(n)):
            for j in range(i + 1, len(n)):
                target = -1 * (n[i] + n[j])
                if target in P:
                    res.add(tuple(sorted([n[i], n[j], target])))
        # 5. For all pairs of positive numbers (1, 1), check to see if their complement (-2)
        #   exists in the negative number set
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                target = -1 * (p[i] + p[j])
                if target in N:
                    res.add(tuple(sorted([p[i], p[j], target])))
        return res

    # LC 16. 3Sum Closest
    # https://leetcode.com/problems/3sum-closest/
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # Double pointers solution
        closest = float("inf")
        nums.sort()
        for i in range(len(nums) - 2):
            l, r = i + 1, len(nums) - 1
            while l < r:
                sum3 = nums[i] + nums[l] + nums[r]
                if sum3 == target:
                    return target
                elif sum3 < target:
                    l += 1
                else:
                    r -= 1
                if abs(sum3 - target) < abs(closest - target):
                    closest = sum3
        return closest

    # LC 17. Letter Combinations of a Phone Number (Medium)
    # https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        h = {
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"],
        }
        res = []
        for l in itertools.product(*[h[d] for d in digits]):
            res.append("".join(l))
        return res

    # LC 18. 4Sum (Medium)
    # https://leetcode.com/problems/4sum/
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if len(nums) < 4:
            return []
        nums.sort()
        res = []
        for i in range(len(nums) - 3):
            if i and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums) - 2):
                if j != i + 1 and nums[j] == nums[j - 1]:
                    continue
                l, r = j + 1, len(nums) - 1
                while l < r:
                    sum4 = nums[i] + nums[j] + nums[l] + nums[r]
                    if sum4 == target:
                        res.append([nums[i], nums[j], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        while l < r and nums[r] == nums[r - 1]:
                            r -= 1
                        l += 1
                        r -= 1
                    elif sum4 < target:
                        l += 1
                    else:
                        r -= 1
        return res

    # LC 19. Remove Nth Node From End of List (Medium)
    # https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # # One-pass with array to save nodes - O(|list|)TS
        # dummy = ListNode()
        # dummy.next = head
        # arr = []
        # curr = dummy
        # while curr:
        #     arr.append(curr)
        #     curr = curr.next
        # arr[-n-1].next = arr[-n].next
        # return dummy.next

        # Two pointers fast & slow - O(|list|)T, O(1)S
        # Send a fast pointer n steps ahead and then start iterating with slow pointer till the fast pointer reaches the end
        fast = head
        slow = head
        for i in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return head

    # LC 20. Valid Parentheses (Easy)
    # https://leetcode.com/problems/valid-parentheses/
    def isValidParentheses(self, s: str) -> bool:
        maps = {")": "(", "}": "{", "]": "["}
        pipe = []
        for char in s:
            if char not in maps:
                pipe.append(char)
                continue
            if not pipe or pipe.pop() != maps[char]:
                return False
        return not pipe

    # LC 21. Merge Two Sorted Lists (Easy)
    # https://leetcode.com/problems/merge-two-sorted-lists/
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        # O(n)T, O(1)S
        dummy = ListNode()
        curr = dummy
        while list1 and list2:
            if list1.val < list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next
        curr.next = list1 or list2
        return dummy.next

    # LC 22. Generate Parentheses (Medium)
    # https://leetcode.com/problems/generate-parentheses/
    def generateParenthesis(self, n: int) -> List[str]:
        # DP
        dp = []
        dp.append([""])
        for i in range(1, n + 1):
            cur = []
            for j in range(i):
                left = dp[j]
                right = dp[i - j - 1]
                for l in left:
                    for r in right:
                        cur.append("(" + l + ")" + r)
            dp.append(cur)
        return dp[n]

    # LC 23. Merge k Sorted Lists (Hard)
    # https://leetcode.com/problems/merge-k-sorted-lists/
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # In-place sorting using heappush and heappop - O(nlogk)T, O(k)S
        if not lists or all(not nd for nd in lists):
            return None
        h = []
        for nd in lists:
            if nd:
                heapq.heappush(h, (nd.val, id(nd), nd))
        _, _, head = heapq.heappop(h)
        curr = head
        if curr.next:
            heapq.heappush(h, (curr.next.val, id(curr.next), curr.next))
        while h:
            _, _, curr.next = heapq.heappop(h)
            curr = curr.next
            if curr.next:
                heapq.heappush(h, (curr.next.val, id(curr.next), curr.next))
        return head

    # LC 24. Swap Nodes in Pairs (Medium)
    # https://leetcode.com/problems/swap-nodes-in-pairs/
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy.next = head
        prev = dummy
        while cur and cur.next:
            prev.next = cur.next
            cur.next = cur.next.next
            prev.next.next = cur
            prev, cur = cur, cur.next
        return dummy.next

        # # More readable
        # dummy = ListNode()
        # dummy.next = head
        # curr = dummy
        # while curr.next and curr.next.next:
        #     first = curr.next
        #     second = curr.next.next
        #     first.next = second.next
        #     second.next = first
        #     curr.next = second
        #     curr = first
        # return dummy.next

    # LC 25. Reverse Nodes in k-Group (Hard)
    # https://leetcode.com/problems/reverse-nodes-in-k-group/
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # # Recursive solution - O(n)T, O(n/k)S
        # curr = head
        # count = 0
        # while curr and count != k:
        #     curr = curr.next
        #     count += 1
        # if count == k:
        #     curr = self.reverseKGroup(curr, k)
        #     while count:
        #         tmp = head.next
        #         head.next = curr
        #         curr = head
        #         head = tmp
        #         count -= 1
        #     head = curr
        # return head

        # Iterative solution - O(n)T, O(1)S
        dummy = jump = ListNode()
        dummy.next = l = r = head
        while True:
            count = 0
            while r and count < k:
                r = r.next
                count += 1
            if count == k:
                pre, cur = r, l
                for _ in range(k):
                    cur.next, cur, pre = pre, cur.next, cur
                jump.next, jump, l = pre, l, r
            else:
                return dummy.next

    # LC 26. Remove Duplicates from Sorted Array (Easy)
    # https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    def removeDuplicates(self, nums: List[int]) -> int:
        # # In-place sorting - O(n log n)T, O(n)S
        # nums[:] = sorted(set(nums))
        # return len(nums)

        # Two pointers - O(n)T, O(1)S
        j = 0
        for i in range(1, len(nums)):
            if nums[j] != nums[i]:
                j += 1
                nums[j] = nums[i]
        return j + 1

        # # Using .pop() - O(n)T, O(1)S
        # i = 1
        # while i < len(nums):
        #     if nums[i] == nums[i - 1]:
        #         nums.pop(i)
        #     else:
        #         i += 1
        # return len(nums)

        # # Using OrderedDict.fromkeys() - O(n)T, O(n)S
        # nums[:] =  collections.OrderedDict.fromkeys(nums)
        # return len(nums)

    # LC 27. Remove Element (Easy)
    # https://leetcode.com/problems/remove-element/
    def removeElement(self, nums: List[int], val: int) -> int:
        # # Using .pop() - O(n)T, O(1)S
        # i = 0
        # while i < len(nums):
        #     if nums[i] == val:
        #         nums.pop(i)
        #     else:
        #         i += 1
        # return len(nums)

        # Two pointers - O(n)T, O(1)S
        j = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[j] = nums[i]
                j += 1
        return j

    # LC 28. Find the Index of the First Occurrence in a String (Easy)
    # https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/
    def strStr(self, haystack: str, needle: str) -> int:
        # # One-liner - O(|haystack|*|needle|)T, O(1)S
        # return haystack.find(needle)

        # # Sliding window - O(|haystack|*|needle|)T, O(1)S
        # i, j = 0, len(needle)
        # while j <= len(haystack):
        #     if haystack[i:j] == needle:
        #         return i
        #     i += 1
        #     j += 1
        # return -1

        # KMP algorithm - O(|haystack|+|needle|)T, O(|needle|)S
        lps = [0] * len(needle)  # "longest proper prefix that is also a suffix"
        # Preprocessing
        pre = 0
        for i in range(1, len(needle)):
            while pre > 0 and needle[i] != needle[pre]:
                pre = lps[pre - 1]
            if needle[pre] == needle[i]:
                pre += 1
                lps[i] = pre
        # Main algorithm
        n = 0  # needle index
        for h in range(len(haystack)):
            while n > 0 and needle[n] != haystack[h]:
                n = lps[n - 1]
            if needle[n] == haystack[h]:
                n += 1
            if n == len(needle):
                return h - n + 1
        return -1

    # LC 29. Divide Two Integers (Medium)
    # https://leetcode.com/problems/divide-two-integers/
    def divide(self, dividend: int, divisor: int) -> int:
        # # With bit manipulation: TODO
        # With len(range())
        sign = 1
        if (dividend >= 0 and divisor < 0) or (dividend < 0 and divisor >= 0):
            sign = -1
        dividend, divisor = abs(dividend), abs(divisor)
        res = len(range(0, dividend - divisor + 1, divisor))
        if sign == -1:
            res = -res
        return min(max(res, -(2**31)), 2**31 - 1)

    # LC 30. Substring with Concatenation of All Words (Hard)
    # https://leetcode.com/problems/substring-with-concatenation-of-all-words/
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        # LC solution
        # Words counters - O(|s|*|words|)T, O(|words|)S
        wordLength = len(words[0])
        substrLength = wordLength * len(words)
        expectedWordCounts = collections.Counter(words)
        result = []
        # Trying each way to split `s`
        # into consecutive words of length `substrLength`
        for offset in range(wordLength):
            wordCounts = {word: 0 for word in expectedWordCounts.keys()}
            # Start with counting words in the first substring
            for i in range(offset, substrLength + offset, wordLength):
                word = s[i : i + wordLength]
                if word in wordCounts:
                    wordCounts[word] += 1
            if wordCounts == expectedWordCounts:
                result.append(offset)
            # Then iterate the other substrings
            # by adding a word at the end and removing the first word
            for start in range(
                offset + wordLength,
                len(s) - substrLength + 1,
                wordLength,
            ):
                removedWord = s[start - wordLength : start]
                addedWord = s[start + substrLength - wordLength : start + substrLength]
                if removedWord in wordCounts:
                    wordCounts[removedWord] -= 1
                if addedWord in wordCounts:
                    wordCounts[addedWord] += 1
                if wordCounts == expectedWordCounts:
                    result.append(start)
        return result

    # LC 31. Next Permutation (Medium)
    # https://leetcode.com/problems/next-permutation/
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # # LC solutions

        # # Brute force - O(n!)T, O(n)S
        # def swap(i, j):
        #     nums[i], nums[j] = nums[j], nums[i]
        # def reverse(i):
        #     nums[i:] = reversed(nums[i:])
        # n = len(nums)
        # i = n - 2
        # while i >= 0 and nums[i] >= nums[i + 1]:
        #     i -= 1
        # if i >= 0:
        #     j = n - 1
        #     while j >= 0 and nums[i] >= nums[j]:
        #         j -= 1
        #     swap(i, j)
        # reverse(i + 1)

        # Single pass - O(n)T, O(1)S
        n = len(nums)
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = n - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j = n - 1
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

    # LC 32. Longest Valid Parentheses (Hard)
    # https://leetcode.com/problems/longest-valid-parentheses/
    def longestValidParentheses(self, s: str) -> int:
        # Stack - O(n)T, O(n)S
        maxans = 0
        stack = [-1]
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    maxans = max(maxans, i - stack[-1])
        return maxans

    # LC 33. Search in Rotated Sorted Array (Medium)
    # https://leetcode.com/problems/search-in-rotated-sorted-array/
    def search(self, nums: List[int], target: int) -> int:
        # Binary search - O(log n)T, O(1)S
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (r + l) // 2
            if nums[mid] == target:
                return mid
            # Check if left half is sorted
            if nums[l] <= nums[mid]:
                if nums[l] <= target <= nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            # Otherwise, right half is sorted
            else:
                if nums[mid] <= target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

    # LC 34. Find First and Last Position of Element in Sorted Array (Medium)
    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # # Binary search - O(log n)T, O(1)S
        # l, r = 0, len(nums) - 1
        # while l <= r:
        #     mid = (l + r) // 2
        #     if nums[mid] == target:
        #         l, r = mid - 1, mid + 1
        #         while l >= 0 and nums[l] == target:
        #             l -= 1
        #         while r < len(nums) and nums[r] == target:
        #             r += 1
        #         return [l+1, r-1]
        #     elif nums[mid] < target:
        #         l = mid + 1
        #     else:
        #         r = mid - 1
        # return [-1, -1]

        # # Two binary searches - O(log n)T, O(1)S
        # def search(n):
        #     l, r = 0, len(nums)
        #     while l < r:
        #         mid = (l + r) // 2
        #         if nums[mid] >= n:
        #             r = mid
        #         else:
        #             l = mid + 1
        #     return l
        # l = search(target)
        # if l == len(nums) or nums[l] != target:
        #     return [-1, -1]
        # return [l, search(target + 1) - 1]

        # # One-liner - O(log n)T, O(1)S
        # return [nums.index(target), nums.index(target)+nums.count(target)-1] if target in nums else [-1, -1]

        # With bisect module - O(log n)T, O(1)S
        l = bisect.bisect_left(nums, target)
        if l == len(nums) or nums[l] != target:
            return [-1, -1]
        return [l, bisect.bisect_right(nums, target) - 1]

    # LC 35. Search Insert Position (Easy)
    # https://leetcode.com/problems/search-insert-position/
    def searchInsert(self, nums: List[int], target: int) -> int:
        # # One-liner - O(log n)T, O(1)S
        # return bisect.bisect_left(nums, target)

        # Binary search (bisect_left) - O(log n)T, O(1)S
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return l

    # LC 36. Valid Sudoku (Medium)
    # https://leetcode.com/problems/valid-sudoku/
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # Brute force - O(1)T, O(1)S
        seen = set()
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    cur = board[i][j]
                    if (
                        (i, cur) in seen
                        or (cur, j) in seen
                        or (i // 3, j // 3, cur) in seen
                    ):
                        return False
                    seen.add((i, cur))
                    seen.add((cur, j))
                    seen.add((i // 3, j // 3, cur))
        return True

    # LC 37. Sudoku Solver (Hard)
    # https://leetcode.com/problems/sudoku-solver/
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        # Brute force - O(1)T, O(1)S
        def is_valid(row, col, num):
            for i in range(9):
                if board[row][i] == num:
                    return False
                if board[i][col] == num:
                    return False
                if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
                    return False
            return True

        def solve():
            for row in range(9):
                for col in range(9):
                    if board[row][col] == ".":
                        for num in "123456789":
                            if is_valid(row, col, num):
                                board[row][col] = num
                                if solve():
                                    return True
                                board[row][col] = "."
                        return False
            return True

        solve()

    # LC 38. Count and Say (Easy)
    # https://leetcode.com/problems/count-and-say/
    def countAndSay(self, n: int) -> str:
        # # With itertools.groupby - O(n*2^n)T, O(2^n)S
        # res = "1"
        # for _ in range(n-1):
        #     r = []
        #     for val, arr in itertools.groupby(res):
        #         r.append(str(len(list(arr))))
        #         r.append(val)
        #     res = "".join(r)
        # return res

        # With re - O(n*2^n)T, O(2^n)S
        res = "1"
        for _ in range(n - 1):
            res = re.sub(r"(.)\1*", lambda m: str(len(m.group(0))) + m.group(1), res)
        return res

    # LC 39. Combination Sum (Medium)
    # https://leetcode.com/problems/combination-sum/
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # Backtracking - O(2^n)T, O(n)S
        res = []

        def backtrack(remain, comb, start):
            if remain == 0:
                res.append(list(comb))
                return
            elif remain < 0:
                return
            for i in range(start, len(candidates)):
                comb.append(candidates[i])
                backtrack(remain - candidates[i], comb, i)
                comb.pop()

        backtrack(target, [], 0)
        return res

    # LC 40. Combination Sum II (Medium)
    # https://leetcode.com/problems/combination-sum-ii/
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # Backtracking - O((2^n)*k)T, O(k*x)S
        res = []
        candidates.sort()

        def backtrack(remain, comb, start):
            if remain == 0:
                res.append(list(comb))
                return
            elif remain < 0:
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                if candidates[i] > target:
                    break
                comb.append(candidates[i])
                backtrack(remain - candidates[i], comb, i + 1)
                comb.pop()

        backtrack(target, [], 0)
        return res

    # LC 41. First Missing Positive (Hard)
    # https://leetcode.com/problems/first-missing-positive/
    def firstMissingPositive(self, nums: List[int]) -> int:
        # == Smallest positive integer that is not in the array [1, len(nums)]

        # Set conversion - O(n*log(n))T, O(n)S
        # nums = set(nums)
        # for i in range(1, len(nums) + 1):
        #     if i not in nums:
        #         return i
        # return len(nums) + 1

        # Optimized solution with swapping - O(n)T, O(1)S
        n = len(nums)
        for i in range(n):
            correctPos = nums[i] - 1
            while 1 <= nums[i] <= n and nums[i] != nums[correctPos]:
                nums[i], nums[correctPos] = nums[correctPos], nums[i]
                correctPos = nums[i] - 1
        for i in range(n):
            if i + 1 != nums[i]:
                return i + 1
        return n + 1

    # LC 42. Trapping Rain Water (Hard)
    # https://leetcode.com/problems/trapping-rain-water/
    def trap(self, height: List[int]) -> int:
        # # Brute force - O(n^2)T, O(1)S
        # res = 0
        # for i in range(1, len(height) - 1):
        #     leftMax = max(height[:i])
        #     rightMax = max(height[i + 1 :])
        #     minHeight = min(leftMax, rightMax)
        #     if minHeight > height[i]:
        #         res += minHeight - height[i]
        # return res

        # # DP - O(n)T, O(n)S
        # if not height:
        #     return 0
        # n = len(height)
        # leftMax = [0] * n
        # rightMax = [0] * n
        # leftMax[0] = height[0]
        # rightMax[-1] = height[-1]
        # for i in range(1, n):
        #     leftMax[i] = max(height[i], leftMax[i - 1])
        # for i in range(n - 2, -1, -1):
        #     rightMax[i] = max(height[i], rightMax[i + 1])
        # res = 0
        # for i in range(1, n - 1):
        #     res += min(leftMax[i], rightMax[i]) - height[i]
        # return res

        # Two pointers - O(n)T, O(1)S
        if not height:
            return 0
        n = len(height)
        l, r = 0, n - 1
        leftMax = rightMax = res = 0
        while l < r:
            if height[l] < height[r]:
                if height[l] >= leftMax:
                    leftMax = height[l]
                else:
                    res += leftMax - height[l]
                l += 1
            else:
                if height[r] >= rightMax:
                    rightMax = height[r]
                else:
                    res += rightMax - height[r]
                r -= 1
        return res

    # LC 43. Multiply Strings (Medium)
    # https://leetcode.com/problems/multiply-strings/
    def multiply(self, num1: str, num2: str) -> str:
        # # One-liner - O(n*m)T, O(n+m)S
        # return str(int(num1) * int(num2))

        # Grade school algorithm - O(n*m)T, O(n+m)S
        if num1 == "0" or num2 == "0":
            return "0"
        res = [0] * (len(num1) + len(num2))
        for i, n1 in enumerate(reversed(num1)):
            for j, n2 in enumerate(reversed(num2)):
                res[i + j] += int(n1) * int(n2)
                res[i + j + 1] += res[i + j] // 10
                res[i + j] %= 10
        while res[-1] == 0:
            res.pop()
        return "".join(map(str, res[::-1]))

    # LC 44. Wildcard Matching (Hard)
    # https://leetcode.com/problems/wildcard-matching/
    def isMatch(self, s: str, p: str) -> bool:
        # # DP - O(n*m)T, O(n*m)S
        # n, m = len(s), len(p)
        # dp = [[False] * (m + 1) for _ in range(n + 1)]
        # # Base case
        # dp[0][0] = True
        # for j in range(1, m + 1):
        #     if p[j - 1] == "*":
        #         dp[0][j] = True
        #     else:
        #         break
        # # Main algorithm
        # for i in range(1, n + 1):
        #     for j in range(1, m + 1):
        #         if p[j - 1] in {s[i - 1], "?"}:
        #             dp[i][j] = dp[i - 1][j - 1]
        #         elif p[j - 1] == "*":
        #             dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
        # return dp[-1][-1]

        # DP with 1D array - O(n*m)T, O(m)S
        n, m = len(s), len(p)
        dp = [False] * (m + 1)
        # Base case
        dp[0] = True
        for j in range(1, m + 1):
            if p[j - 1] == "*":
                dp[j] = True
            else:
                break
        # Main algorithm
        for i in range(1, n + 1):
            new = [False] * (m + 1)
            for j in range(1, m + 1):
                if p[j - 1] in {s[i - 1], "?"}:
                    new[j] = dp[j - 1]
                elif p[j - 1] == "*":
                    new[j] = new[j - 1] or dp[j]
            dp = new
        return dp[-1]

    # LC 45. Jump Game II (Medium)
    # https://leetcode.com/problems/jump-game-ii/
    def jump(self, nums: List[int]) -> int:
        # # DP - O(n^2)T, O(n)S
        # n = len(nums)
        # dp = [float("inf")] * n
        # dp[0] = 0
        # for i in range(n):
        #     for j in range(i + 1, min(i + nums[i] + 1, n)):
        #         dp[j] = min(dp[j], dp[i] + 1)
        # return dp[-1]

        # Greedy (Implicit BFS) - O(n)T, O(1)S
        # Initialize reach (maximum reachable index), count (number of jumps), and last (rightmost index reached)
        reach, count, last = 0, 0, 0
        # Loop through the array excluding the last element
        for i in range(len(nums) - 1):
            # Update reach to the maximum between reach and i + nums[i]
            reach = max(reach, i + nums[i])
            # If i has reached the last index that can be reached with the current number of jumps
            if i == last:
                # Update last to the new maximum reachable index
                last = reach
                # Increment the number of jumps made so far
                count += 1
        # Return the minimum number of jumps required
        return count

    # LC 46. Permutations
    # https://leetcode.com/problems/permutations/
    def permute(self, nums: List[int]) -> List[List[int]]:
        # # One-liner - O(n!)T, O(n!)S
        # return itertools.permutations(nums)

        # Backtracking - O(n!)T, O(n!)S
        res = []

        def backtrack(nums, path):
            if not nums:
                res.append(path)
                return
            for i in range(len(nums)):
                backtrack(nums[:i] + nums[i + 1 :], path + [nums[i]])

        backtrack(nums, [])
        return res

    # LC 47. Permutations II
    # https://leetcode.com/problems/permutations-ii/
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # # One-liner - O(n!)T, O(n!)S
        # return set(itertools.permutations(nums))

        # Backtracking - O(n!)T, O(n!)S
        res = []

        def backtrack(nums, path):
            if not nums:
                res.append(path)
                return
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                backtrack(nums[:i] + nums[i + 1 :], path + [nums[i]])

        nums.sort()
        backtrack(nums, [])
        return res

    # LC 48. Rotate Image (Medium)
    # https://leetcode.com/problems/rotate-image/
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # # Transpose and reverse - O(n^2)T, O(1)S
        # for i in range(len(matrix)):
        #     for j in range(i, len(matrix)):
        #         matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
        #     matrix[i].reverse()

        # Rotate groups of 4 cells - O(n^2)T, O(1)S
        n = len(matrix)
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                tmp = matrix[n - j - 1][i]
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1]
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1]
                matrix[j][n - i - 1] = matrix[i][j]
                matrix[i][j] = tmp

    # LC 49. Group Anagrams (Medium)
    # https://leetcode.com/problems/group-anagrams/
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # # One-liner - O(n*k*log(k))T, O(n*k)S
        # return [list(g) for _, g in itertools.groupby(sorted(strs, key=sorted), sorted)]

        # Categorize by sorted string - O(n*k*log(k))T, O(n*k)S
        res = collections.defaultdict(list)
        for s in strs:
            res[tuple(sorted(s))].append(s)
        return res.values()

    # LC 50. Pow(x, n) (Medium)
    # https://leetcode.com/problems/powx-n/
    def myPow(self, x: float, n: int) -> float:
        # # One-liner - O(log n)T, O(1)S
        # return x ** n

        # # Brute force - O(n)T, O(1)S
        # res = 1
        # if n < 0:
        #     x = 1 / x
        #     n = -n
        # for _ in range(n):
        #     res *= x
        # return res

        # # Fast power algorithm - O(log n)T, O(log n)S
        # def fastPower(x, n):
        #     if n == 0:
        #         return 1.0
        #     half = fastPower(x, n // 2)
        #     if n % 2 == 0:
        #         return half * half
        #     else:
        #         return half * half * x
        # if n < 0:
        #     x = 1 / x
        #     n = -n
        # return fastPower(x, n)

        # Fast power algorithm (iterative) - O(log n)T, O(1)S
        if n < 0:
            x = 1 / x
            n = -n
        res = 1
        while n:
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res

    # LC 51. N-Queens (Hard)
    # https://leetcode.com/problems/n-queens/
    def solveNQueens(self, n: int) -> List[List[str]]:
        # Backtracking - O(n!)T, O(n)S
        def solve(col, board, ans, lr, ud, ld, n):
            if col == n:
                ans.append(["".join(row) for row in board])
                return
            for row in range(n):
                if lr[row] == 0 and ld[row - col] == 0 and ud[row + col] == 0:
                    board[row][col] = "Q"
                    lr[row] = 1
                    ld[row - col] = 1
                    ud[row + col] = 1
                    solve(col + 1, board, ans, lr, ud, ld, n)
                    board[row][col] = "."
                    lr[row] = 0
                    ld[row - col] = 0
                    ud[row + col] = 0

        ans = []
        board = [["." for _ in range(n)] for _ in range(n)]
        leftrow = [0] * n
        upperDiagonal = [0] * (2 * n - 1)
        lowerDiagonal = [0] * (2 * n - 1)
        solve(0, board, ans, leftrow, upperDiagonal, lowerDiagonal, n)
        return ans

    # LC 52. N-Queens II (Hard)
    # https://leetcode.com/problems/n-queens-ii/
    def totalNQueens(self, n: int) -> int:
        # Backtracking - O(n!)T, O(n)S
        def solve(col, board, ans, lr, ud, ld, n):
            if col == n:
                ans.append(1)
                return
            for row in range(n):
                if lr[row] == 0 and ld[row - col] == 0 and ud[row + col] == 0:
                    board[row][col] = "Q"
                    lr[row] = 1
                    ld[row - col] = 1
                    ud[row + col] = 1
                    solve(col + 1, board, ans, lr, ud, ld, n)
                    board[row][col] = "."
                    lr[row] = 0
                    ld[row - col] = 0
                    ud[row + col] = 0

        ans = []
        board = [["." for _ in range(n)] for _ in range(n)]
        leftrow = [0] * n
        upperDiagonal = [0] * (2 * n - 1)
        lowerDiagonal = [0] * (2 * n - 1)
        solve(0, board, ans, leftrow, upperDiagonal, lowerDiagonal, n)
        return len(ans)

    # LC 53. Maximum Subarray (Easy)
    # https://leetcode.com/problems/maximum-subarray/
    def maxSubArray(self, nums: List[int]) -> int:
        # # DP - O(n)T, O(1)S
        # dp = nums[0]
        # res = dp
        # for i in range(1, len(nums)):
        #     dp = max(nums[i], dp + nums[i])
        #     res = max(res, dp)
        # return res

        # Divide and conquer - O(n*log(n))T, O(1)S
        def divideAndConquer(nums, l, r):
            if l == r:
                return nums[l]
            mid = (l + r) // 2
            leftMax = divideAndConquer(nums, l, mid)
            rightMax = divideAndConquer(nums, mid + 1, r)
            crossMax = nums[mid]
            tmp = crossMax
            for i in range(mid - 1, l - 1, -1):
                tmp += nums[i]
                crossMax = max(crossMax, tmp)
            tmp = crossMax
            for i in range(mid + 1, r + 1):
                tmp += nums[i]
                crossMax = max(crossMax, tmp)
            return max(leftMax, rightMax, crossMax)

        return divideAndConquer(nums, 0, len(nums) - 1)

    # LC 54. Spiral Matrix (Medium)
    # https://leetcode.com/problems/spiral-matrix/
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # # One-liner - O(n*m)T, O(n*m)S
        # return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])

        # Layer-by-layer - O(n*m)T, O(n*m)S
        if not matrix:
            return []
        res = []
        rows, cols = len(matrix), len(matrix[0])
        left, right, top, bottom = 0, cols - 1, 0, rows - 1
        while left <= right and top <= bottom:
            for c in range(left, right + 1):
                res.append(matrix[top][c])
            for r in range(top + 1, bottom + 1):
                res.append(matrix[r][right])
            if left < right and top < bottom:
                for c in range(right - 1, left, -1):
                    res.append(matrix[bottom][c])
                for r in range(bottom, top, -1):
                    res.append(matrix[r][left])
            left, right, top, bottom = left + 1, right - 1, top + 1, bottom - 1
        return res

    # LC 55. Jump Game (Medium)
    # https://leetcode.com/problems/jump-game/
    def canJump(self, nums: List[int]) -> bool:
        # # DP - O(n^2)T, O(n)S
        # n = len(nums)
        # dp = [False] * n
        # dp[0] = True
        # for i in range(n):
        #     if dp[i]:
        #         for j in range(i + 1, min(i + nums[i] + 1, n)):
        #             if j < n:
        #                 dp[j] = True
        #             if j == n - 1:
        #                 return True
        # return dp[-1]

        # Greedy - O(n)T, O(1)S
        reachable = 0
        for i, num in enumerate(nums):
            if reachable < i:
                return False
            reachable = max(reachable, i + num)
        return True

    # LC 56. Merge Intervals (Medium)
    # https://leetcode.com/problems/merge-intervals/
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # Sort and merge - O(n*log(n))T, O(n)S
        intervals.sort(key=lambda x: x[0])
        res = []
        for interval in intervals:
            if not res or res[-1][1] < interval[0]:
                res.append(interval)
            else:
                res[-1][1] = max(res[-1][1], interval[1])  # Merge overlapping intervals
        return res

    # LC 57. Insert Interval (Medium)
    # https://leetcode.com/problems/insert-interval/
    def insert(
        self, intervals: List[List[int]], newInterval: List[int]
    ) -> List[List[int]]:
        # # Insert, sort, and merge - O(n*log(n))T, O(n)S
        # intervals.append(newInterval)
        # intervals.sort(key=lambda x: x[0])
        # res = []
        # for interval in intervals:
        #     if not res or res[-1][1] < interval[0]:
        #         res.append(interval)
        #     else:
        #         res[-1][1] = max(res[-1][1], interval[1])
        # return res

        # Insert and merge - O(n)T, O(n)S
        res = []
        i = 0
        while i < len(intervals) and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        while i < len(intervals) and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        res.append(newInterval)
        while i < len(intervals):
            res.append(intervals[i])
            i += 1
        return res

    # LC 58. Length of Last Word (Easy)
    # https://leetcode.com/problems/length-of-last-word/
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])

    # LC 59. Spiral Matrix II (Medium)
    # https://leetcode.com/problems/spiral-matrix-ii/
    def generateMatrix(self, n: int) -> List[List[int]]:
        # # Spiral filling - O(n^2)T, O(1)S
        # mat = [[0 for _ in range(n)] for _ in range(n)]  # matrix of n*n
        # el = 1  # element to be filled
        # dir = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # 4 directions
        # d = 0  # variable to refer direction
        # row = 0
        # col = 0
        # while el <= n * n:
        #     mat[row][col] = el
        #     r = (row + dir[d][0]) % n
        #     c = (col + dir[d][1]) % n
        #     if mat[r][c] != 0:
        #         # change in direction if cell already traveresed
        #         d = (d + 1) % 4
        #     row += dir[d][0]
        #     col += dir[d][1]
        #     el += 1
        # return mat

        # Spiral filling (alternative code) - O(n^2)T, O(n^2)S
        ans = [[0] * n for _ in range(n)]
        i = 0
        start_col, start_row, end_col, end_row = 0, 0, n, n
        while start_col < end_col or start_row < end_row:
            for c in range(start_col, end_col):
                i += 1
                ans[start_row][c] = i
            start_row += 1
            for r in range(start_row, end_row):
                i += 1
                ans[r][end_col - 1] = i
            end_col -= 1
            for c in range(end_col - 1, start_col - 1, -1):
                i += 1
                ans[end_row - 1][c] = i
            end_row -= 1
            for r in range(end_row - 1, start_row - 1, -1):
                i += 1
                ans[r][start_col] = i
            start_col += 1
        return ans

    # LC 60. Permutation Sequence (Hard)
    # https://leetcode.com/problems/permutation-sequence/
    def getPermutation(self, n: int, k: int) -> str:
        # # One-liner - O(n^2)T, O(n)S
        # return "".join(list(itertools.permutations([str(i) for i in range(1, n + 1)]))[k - 1])

        # # Backtracking (TLE) - O(n!)T, O(n)S
        # def backtrack(nums, path):
        #     if not nums:
        #         res.append(path)
        #         return
        #     for i in range(len(nums)):
        #         backtrack(nums[:i] + nums[i + 1 :], path + [nums[i]])
        # res = []
        # backtrack([str(i) for i in range(1, n + 1)], [])
        # return "".join(res[k - 1])

        # Optimal solution - O(n^2)T, O(n)S
        nums = [i for i in range(1, n + 1)]
        fact = [1] * n
        for i in range(1, n):
            fact[i] = fact[i - 1] * i
        k -= 1
        res = []
        for i in range(n - 1, -1, -1):
            idx = k // fact[i]
            k %= fact[i]
            res.append(str(nums.pop(idx)))
        return "".join(res)

    # LC 61. Rotate List (Medium)
    # https://leetcode.com/problems/rotate-list/
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # My solution - O(n)T, O(1)S
        if not head:
            return head
        # find last node and connect with head
        # also get list length (n)
        last, n = head, 1
        while last.next:
            last = last.next
            n += 1
        last.next = head
        # doing k or k%n rotations give the same results
        k = k % n
        # find how many nodes to skip based on k
        skip = n - k
        cur = head
        # get the new "last" node, save the next node (new "first" node)
        # set last.next to None and return the saved node
        for i in range(skip - 1):
            cur = cur.next
        new_head = cur.next
        cur.next = None
        return new_head

    # LC 62. Unique Paths (Medium)
    # https://leetcode.com/problems/unique-paths/
    def uniquePaths(self, m: int, n: int) -> int:
        # # Recursion with memoization - O(m*n)TS
        # memo = {}
        # def helper(i, j):
        #     if (i, j) in memo:
        #         return memo[(i, j)]
        #     if i==m or j==n:
        #         return 0
        #     if i==m-1 and j==n-1:
        #         return 1
        #     res = helper(i+1,j) + helper(i,j+1)
        #     memo[(i,j)] = res
        #     return res
        # return helper(0, 0)

        # # DP with hash matrix - O(m*n)TS
        # dp = [[1 if i==0 or j==0 else 0 for j in range(n)] for i in range(m)]
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = dp[i-1][j] + dp[i][j-1]
        # return dp[-1][-1]

        # # DP with only prev row and cur row - O(m*n)T, O(n)S
        # curr_row = [1] * n
        # prev_row = [1] * n
        # for i in range(1, m):
        #     for j in range(1, n):
        #         curr_row[j] = curr_row[j - 1] + prev_row[j]
        #     curr_row, prev_row = prev_row, curr_row
        # return prev_row[-1]

        # Math (comb) - O(1)T, O(1)S
        return math.comb(m + n - 2, m - 1)  # or n-1

    # LC 63. Unique Paths II (Medium)
    # https://leetcode.com/problems/unique-paths-ii/
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # Recursion with memoization - O(m*n)TS
        # m, n = len(obstacleGrid), len(obstacleGrid[0])
        # memo = {}
        # def helper(i, j):
        #     if (i, j) in memo:
        #         return memo[(i, j)]
        #     if i==m or j==n or obstacleGrid[i][j]==1:
        #         return 0
        #     if i==m-1 and j==n-1:
        #         return 1
        #     res = helper(i+1,j) + helper(i,j+1)
        #     memo[(i,j)] = res
        #     return res
        # return helper(0, 0)

        # # DP with hash matrix - O(m*n)TS
        # m, n = len(obstacleGrid), len(obstacleGrid[0])
        # dp = [[0 for _ in range(n)] for _ in range(m)]
        # for i in range(m):
        #     if obstacleGrid[i][0] == 1:
        #         break
        #     dp[i][0] = 1
        # for j in range(n):
        #     if obstacleGrid[0][j] == 1:
        #         break
        #     dp[0][j] = 1
        # for i in range(1, m):
        #     for j in range(1, n):
        #         if obstacleGrid[i][j] != 1:
        #             dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # return dp[-1][-1]

        # DP with only prev row and cur row - O(m*n)T, O(n)S
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        curr_row = [0] * n
        prev_row = [0] * n
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    curr_row[j] = 0
                elif i == 0 and j == 0:
                    curr_row[j] = 1
                elif i == 0:
                    curr_row[j] = curr_row[j - 1]
                elif j == 0:
                    curr_row[j] = prev_row[j]
                else:
                    curr_row[j] = curr_row[j - 1] + prev_row[j]
            curr_row, prev_row = prev_row, curr_row
        return prev_row[-1]

    # LC 64. Minimum Path Sum (Medium)
    # https://leetcode.com/problems/minimum-path-sum/
    def minPathSum(self, grid: List[List[int]]) -> int:
        # # Recursion with memoization - O(m*n)TS
        # m, n = len(grid), len(grid[0])
        # memo = {}
        # def helper(i, j):
        #     if (i, j) in memo:
        #         return memo[(i, j)]
        #     if i == m or j == n:
        #         return float("inf")
        #     if i == m - 1 and j == n - 1:
        #         return grid[i][j]
        #     res = grid[i][j] + min(helper(i + 1, j), helper(i, j + 1))
        #     memo[(i, j)] = res
        #     return res
        # return helper(0, 0)

        # # DP with hash matrix - O(m*n)TS
        # m, n = len(grid), len(grid[0])
        # dp = [[0 for _ in range(n)] for _ in range(m)]
        # dp[0][0] = grid[0][0]
        # for i in range(1, m):
        #     dp[i][0] = grid[i][0] + dp[i - 1][0]
        # for j in range(1, n):
        #     dp[0][j] = grid[0][j] + dp[0][j - 1]
        # for i in range(1, m):
        #     for j in range(1, n):
        #         dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
        # return dp[-1][-1]

        # DP with only prev row and cur row - O(m*n)T, O(n)S
        m, n = len(grid), len(grid[0])
        curr_row = [0] * n
        prev_row = [0] * n
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    curr_row[j] = grid[i][j]
                elif i == 0:
                    curr_row[j] = curr_row[j - 1] + grid[i][j]
                elif j == 0:
                    curr_row[j] = prev_row[j] + grid[i][j]
                else:
                    curr_row[j] = grid[i][j] + min(curr_row[j - 1], prev_row[j])
            curr_row, prev_row = prev_row, curr_row
        return prev_row[-1]

    # LC 65. Valid Number (Hard)
    # https://leetcode.com/problems/valid-number/
    def isNumber(self, s: str) -> bool:
        # # Minimal code - O(n)T, O(1)S
        # if "inf" in s or "Inf" in s or s == "nan":
        #     return False  # BS
        # try:
        #     float(s)
        #     return True
        # except:
        #     return False

        # DFA - O(n)T, O(1)S
        # Define DFA state transition tables
        states = [
            {},
            # State (1) - initial state (scan ahead thru blanks)
            {"blank": 1, "sign": 2, "digit": 3, ".": 4},
            # State (2) - found sign (expect digit/dot)
            {"digit": 3, ".": 4},
            # State (3) - digit consumer (loop until non-digit)
            {"digit": 3, ".": 5, "e": 6, "blank": 9},
            # State (4) - found dot (only a digit is valid)
            {"digit": 5},
            # State (5) - after dot (expect digits, e, or end of valid input)
            {"digit": 5, "e": 6, "blank": 9},
            # State (6) - found 'e' (only a sign or digit valid)
            {"sign": 7, "digit": 8},
            # State (7) - sign after 'e' (only digit)
            {"digit": 8},
            # State (8) - digit after 'e' (expect digits or end of valid input)
            {"digit": 8, "blank": 9},
            # State (9) - Terminal state (fail if non-blank found)
            {"blank": 9},
        ]
        currentState = 1
        for c in s:
            if c.isdigit():
                c = "digit"
            elif c in ["+", "-"]:
                c = "sign"
            elif c == " ":
                c = "blank"
            elif c in ["E", "e"]:
                c = "e"
            if c not in states[currentState]:
                return False
            currentState = states[currentState][c]
        return currentState in [3, 5, 8, 9]

    # LC 66. Plus One (Easy)
    # https://leetcode.com/problems/plus-one/
    def plusOne(self, digits: List[int]) -> List[int]:
        # # One-liner - O(n)T, O(n)S
        # return [int(c) for c in str(int("".join([str(d) for d in digits])) + 1)]

        # With carry - O(n)T, O(1)S
        carry = 1
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] + carry == 10:
                digits[i] = 0
                carry = 1
            else:
                digits[i] += carry
                carry = 0
        if carry:
            digits.insert(0, 1)
        return digits

    # LC 67. Add Binary (Easy)
    # https://leetcode.com/problems/add-binary/
    def addBinary(self, a: str, b: str) -> str:
        # # One-liner - O(n)T, O(1)S
        # return bin(int(a, 2) + int(b, 2))[2:]

        # Bit manipulation - O(n)T, O(1)S
        x, y = int(a, 2), int(b, 2)
        while y:
            answer = x ^ y
            carry = (x & y) << 1
            x, y = answer, carry
        return bin(x)[2:]

    # LC 68. Text Justification (Hard)
    # https://leetcode.com/problems/text-justification/
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # # Greedy (my solution) - O(n)T, O(n)S
        # def replace_last(string, old, new):
        #     return new.join(string.rsplit(old, 1))
        # res = []
        # while words:
        #     buffer = [words.pop(0)]
        #     while words and len(" ".join(buffer) + " " + words[0]) <= maxWidth:
        #         buffer.append(words.pop(0))
        #     if len(buffer) == 1 or len(words) == 0:
        #         res.append(" ".join(buffer).ljust(maxWidth, " "))
        #     else:
        #         r = " ".join(buffer)
        #         i = 1
        #         while len(r) < maxWidth:
        #             r = r.replace(" " * i, " " * (i + 1))
        #             i += 1
        #         while len(r) > maxWidth:
        #             r = replace_last(r, " " * i, " " * (i - 1))
        #         res.append(r)
        # return res

        # Greedy (alternative code) - O(n)T, O(n)S
        res = []
        cur = []
        cur_len = 0
        for word in words:
            if cur_len + len(word) + len(cur) > maxWidth:
                for i in range(maxWidth - cur_len):
                    cur[i % (len(cur) - 1 or 1)] += " "
                res.append("".join(cur))
                cur = []
                cur_len = 0
            cur.append(word)
            cur_len += len(word)
        res.append(" ".join(cur).ljust(maxWidth))
        return res

    # LC 69. Sqrt(x) (Easy)
    # https://leetcode.com/problems/sqrtx/
    def mySqrt(self, x: int) -> int:
        # # One-liner - O(1)T, O(1)S
        # return int(x ** 0.5)

        # Binary search - O(log n)T, O(1)S
        l, r = 0, x
        while l <= r:
            mid = (l + r) // 2
            if mid * mid <= x < (mid + 1) * (mid + 1):
                return mid
            elif x < mid * mid:
                r = mid - 1
            else:
                l = mid + 1

    # LC 70. Climbing Stairs (Easy)
    # https://leetcode.com/problems/climbing-stairs/
    def climbStairs(self, n: int) -> int:
        # # Recursion with memoization - O(n)T, O(n)S
        # memo = {}
        # def helper(i):
        #     if i in memo:
        #         return memo[i]
        #     if i > n:
        #         return 0
        #     if i == n:
        #         return 1
        #     res = helper(i+1) + helper(i+2)
        #     memo[i] = res
        #     return res
        # return helper(0)

        # DP - O(n)T, O(n)S
        # dp = [0] * (n + 1)
        # dp[0] = 1
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[-1]

        # # DP with only prev 2 values - O(n)T, O(1)S
        # if n <= 2:
        #     return n
        # prev1, prev2 = 1, 2
        # for _ in range(3, n + 1):
        #     prev1, prev2 = prev2, prev1 + prev2
        # return prev2

        # Math (fibonacci) - O(log n)T, O(1)S
        sqrt5 = 5**0.5
        fibn = ((1 + sqrt5) / 2) ** (n + 1) - ((1 - sqrt5) / 2) ** (n + 1)
        return int(fibn / sqrt5)

    # LC 71. Simplify Path (Medium)
    # https://leetcode.com/problems/simplify-path/
    def simplifyPath(self, path: str) -> str:
        # # My solution - O(n)T, O(n)S
        # path = re.sub(r'/+', '/', path)
        # path = re.sub(r'/\./', '/', path)
        # path = path.strip("/")
        # folders = path.split("/")
        # res = []
        # for f in folders:
        #     if f == ".":
        #         continue
        #     elif f == "..":
        #         if res:
        #             res.pop()
        #     else:
        #         res.append(f)
        # return "/" + "/".join(res)

        # Stack - O(n)T, O(n)S
        stack = []
        for folder in path.split("/"):
            if folder == "..":
                if stack:
                    stack.pop()
            elif folder and folder != ".":
                stack.append(folder)
        return "/" + "/".join(stack)

    # LC 72. Edit Distance (Hard)
    # https://leetcode.com/problems/edit-distance/
    def minDistance(self, word1: str, word2: str) -> int:
        # # Recursion with memoization - O(m*n)T, O(m*n)S
        # memo = {}
        # def helper(i, j):
        #     if (i, j) in memo:
        #         return memo[(i, j)]
        #     if i == len(word1):
        #         return len(word2) - j
        #     if j == len(word2):
        #         return len(word1) - i
        #     if word1[i] == word2[j]:
        #         res = helper(i + 1, j + 1)
        #     else:
        #         insert = helper(i, j + 1)
        #         delete = helper(i + 1, j)
        #         replace = helper(i + 1, j + 1)
        #         res = 1 + min(insert, delete, replace)
        #     memo[(i, j)] = res
        #     return res
        # return helper(0, 0)

        # DP with hash matrix - O(m*n)T, O(m*n)S
        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        # first row
        for j in range(n + 1):
            dp[0][j] = j
        # first column
        for i in range(m + 1):
            dp[i][0] = i
        # rest of the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    continue
                insert = dp[i][j - 1]
                delete = dp[i - 1][j]
                replace = dp[i - 1][j - 1]
                dp[i][j] = 1 + min(insert, delete, replace)
        return dp[-1][-1]

    # LC 73. Set Matrix Zeroes (Medium)
    # https://leetcode.com/problems/set-matrix-zeroes/
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # # Naive approach with memoization - O(m*n)T, O(m+n)S
        # m, n = len(matrix), len(matrix[0])
        # mset, nset = set(), set()
        # for i in range(m):
        #     for j in range(n):
        #         if matrix[i][j] == 0:
        #             mset.add(i)
        #             nset.add(j)
        # for i in mset:
        #     matrix[i] = [0] * n
        # for j in nset:
        #     for i in range(m):
        #         matrix[i][j] = 0

        # Constant extra space - O(m*n)T, O(1)S
        m, n = len(matrix), len(matrix[0])
        is_col = False
        for i in range(m):
            if matrix[i][0] == 0:
                is_col = True
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if not matrix[i][0] or not matrix[0][j]:
                    matrix[i][j] = 0
        if matrix[0][0] == 0:
            for j in range(n):
                matrix[0][j] = 0
        if is_col:
            for i in range(m):
                matrix[i][0] = 0

    # LC 74. Search a 2D Matrix (Medium)
    # https://leetcode.com/problems/search-a-2d-matrix/
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # # Double binary search - O(log(m*n))T, O(1)S
        # m, n = len(matrix), len(matrix[0])
        # row = None
        # l, r = 0, m - 1
        # while l <= r:
        #     mid = (l + r) // 2
        #     val = matrix[mid][0]
        #     if val == target:
        #         return True
        #     elif matrix[mid][0] <= target <= matrix[mid][-1]:
        #         row = mid
        #         l = float("inf")
        #     elif val < target:
        #         l = mid + 1
        #     else:
        #         r = mid - 1
        # if row is None:
        #     return False
        # l, r = 0, n - 1
        # while l <= r:
        #     mid = (l + r) // 2
        #     val = matrix[row][mid]
        #     if val == target:
        #         return True
        #     elif val < target:
        #         l = mid + 1
        #     else:
        #         r = mid - 1
        # return False

        # Single binary search - O(log(m*n))T, O(1)S
        m, n = len(matrix), len(matrix[0])
        l, r = 0, m * n - 1
        while l <= r:
            mid = (l + r) // 2
            val = matrix[mid // n][mid % n]
            if val == target:
                return True
            elif val < target:
                l = mid + 1
            else:
                r = mid - 1
        return False

    # LC 75. Sort Colors (Medium)
    # https://leetcode.com/problems/sort-colors/
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # Single pass in-place sorting - O(n)T, O(1)S
        l, r = 0, len(nums) - 1
        i = 0
        while i <= r:
            if nums[i] == 0:
                nums[i], nums[l] = nums[l], nums[i]
                l += 1
                i += 1
            elif nums[i] == 1:
                i += 1
            elif nums[i] == 2:
                nums[i], nums[r] = nums[r], nums[i]
                r -= 1

    # LC 76. Minimum Window Substring (Hard)
    # https://leetcode.com/problems/minimum-window-substring/
    def minWindow(self, s: str, t: str) -> str:
        # Sliding window - O(n)T, O(n)S
        if not t or not s:
            return ""
        dict_t = Counter(t)
        required = len(dict_t)
        l, r = 0, 0
        formed = 0
        window_counts = {}
        ans = (float("inf"), None, None)
        while r < len(s):
            char = s[r]
            window_counts[char] = window_counts.get(char, 0) + 1
            if char in dict_t and window_counts[char] == dict_t[char]:
                formed += 1
            while l <= r and formed == required:
                char = s[l]
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                window_counts[char] -= 1
                if char in dict_t and window_counts[char] < dict_t[char]:
                    formed -= 1
                l += 1
            r += 1
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]

    # LC 77. Combinations (Medium)
    # https://leetcode.com/problems/combinations/
    def combine(self, n: int, k: int) -> List[List[int]]:
        # One-liner - O(n^k)T, O(n^k)S
        # return itertools.combinations(range(1, n + 1), k)

        # # Backtracking - O(n^k)T, O(n^k)S
        # res = []
        # def backtrack(i, path):
        #     if len(path) == k:
        #         res.append(path)
        #         return
        #     for j in range(i, n + 1):
        #         backtrack(j + 1, path + [j])
        # backtrack(1, [])
        # return res

        # Lexicographic (binary sorted) combinations - O(n^k)T, O(n^k)S
        res = []
        nums = list(range(1, k + 1)) + [n + 1]
        j = 0
        while j < k:
            res.append(nums[:k])
            j = 0
            while j < k and nums[j + 1] == nums[j] + 1:
                nums[j] = j + 1
                j += 1
            nums[j] += 1
        return res

    # LC 78. Subsets (Medium)
    # https://leetcode.com/problems/subsets/
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # # My solution - O(n*2^n)T, O(n*2^n)S
        # res = set()
        # for i in range(len(nums) + 1):
        #     for el in itertools.combinations(nums, i):
        #         res.add(el)
        # return res

        # # One-liner - O(n*2^n)T, O(n*2^n)S
        # return itertools.chain.from_iterable(itertools.combinations(nums, r) for r in range(len(nums) + 1))

        # # Backtracking - O(n*2^n)T, O(n*2^n)S
        # res = []
        # def backtrack(i, path):
        #     res.append(path)
        #     for j in range(i, len(nums)):
        #         backtrack(j + 1, path + [nums[j]])
        # backtrack(0, [])
        # return res

        # Iterative - O(n*2^n)T, O(n*2^n)S
        res = [[]]
        for num in nums:
            res += [curr + [num] for curr in res]
        return res

    # LC 79. Word Search (Medium)
    # https://leetcode.com/problems/word-search/
    def exist(self, board: List[List[str]], word: str) -> bool:
        # # Backtracking - O(m*n*4^k)T, O(k)S
        # m, n = len(board), len(board[0])
        # visited = set()
        # def backtrack(i, j, idx):
        #     if idx == len(word):
        #         return True
        #     if (
        #         i < 0
        #         or i >= m
        #         or j < 0
        #         or j >= n
        #         or (i, j) in visited
        #         or board[i][j] != word[idx]
        #     ):
        #         return False
        #     visited.add((i, j))
        #     res = (
        #         backtrack(i + 1, j, idx + 1)
        #         or backtrack(i - 1, j, idx + 1)
        #         or backtrack(i, j + 1, idx + 1)
        #         or backtrack(i, j - 1, idx + 1)
        #     )
        #     visited.remove((i, j))
        #     return res
        # for i in range(m):
        #     for j in range(n):
        #         if backtrack(i, j, 0):
        #             return True
        # return False

        # Backtracking without set for visited cells - O(m*n*4^k)T, O(1)S
        m, n = len(board), len(board[0])

        def backtrack(i, j, idx):
            if idx == len(word):
                return True
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[idx]:
                return False
            board[i][j] = "#"
            res = (
                backtrack(i + 1, j, idx + 1)
                or backtrack(i - 1, j, idx + 1)
                or backtrack(i, j + 1, idx + 1)
                or backtrack(i, j - 1, idx + 1)
            )
            board[i][j] = word[idx]
            return res

        for i in range(m):
            for j in range(n):
                if backtrack(i, j, 0):
                    return True
        return False

    # LC 80. Remove Duplicates from Sorted Array II (Medium)
    def removeDuplicates(self, nums: List[int]) -> int:
        # Double pointers - O(n)T, O(1)S
        if len(nums) <= 2:
            return len(nums)
        i = 2
        for j in range(2, len(nums)):
            if nums[j] != nums[i - 2]:
                nums[i] = nums[j]
                i += 1
        return i

        # # One-liner - O(n)T, O(1)S
        # return len([nums[i] for i in range(len(nums)) if i < 2 or nums[i] != nums[i - 2]])

    # LC 81. Search in Rotated Sorted Array II (Medium)
    def search(self, nums: List[int], target: int) -> bool:
        # # One-liner - O(n)T, O(1)S
        # return target in nums

        # # Linear search - O(n)T, O(1)S
        # for num in nums:
        #     if num == target:
        #         return True

        # Binary search in rotated sorted array - O(log n)T, O(1)S
        l, r = 0, len(nums) - 1
        while l <= r:
            while l < r and nums[l] == nums[l + 1]:
                l += 1
            while l < r and nums[r] == nums[r - 1]:
                r -= 1
            mid = (l + r) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] >= nums[l]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return False

    # LC 82. Remove Duplicates from Sorted List II (Medium)
    # https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # One-pass solution - O(n)T, O(1)S
        dummy = prev = ListNode(0, head)
        while head:
            if head.next and head.val == head.next.val:
                while head.next and head.val == head.next.val:
                    head = head.next
                prev.next = head.next
            else:
                prev = prev.next
            head = head.next
        return dummy.next

    # LC 83. Remove Duplicates from Sorted List (Easy)
    # https://leetcode.com/problems/remove-duplicates-from-sorted-list/
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # # One-pass solution - O(n)T, O(1)S
        # dummy = prev = ListNode(0, head)
        # while head:
        #     if head.next and head.val == head.next.val:
        #         prev.next = head.next
        #     else:
        #         prev = prev.next
        #     head = head.next
        # return dummy.next

        # Optimized one-pass solution - O(n)T, O(1)S
        curr = head
        while curr and curr.next:
            if curr.val == curr.next.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head

    # LC 84. Largest Rectangle in Histogram (Hard)
    # https://leetcode.com/problems/largest-rectangle-in-histogram/
    def largestRectangleArea(self, heights: List[int]) -> int:
        # # Brute force - O(n^2)T, O(1)S
        # res = 0
        # for i in range(len(heights)):
        #     min_height = float("inf")
        #     for j in range(i, len(heights)):
        #         min_height = min(min_height, heights[j])
        #         res = max(res, min_height * (j - i + 1))
        # return res

        # # Divide and conquer - O(n log n)T, O(n)S
        # def divide_conquer(heights, l, r):
        #     if l > r:
        #         return 0
        #     min_idx = l
        #     for i in range(l, r + 1):
        #         if heights[i] < heights[min_idx]:
        #             min_idx = i
        #     return max(
        #         heights[min_idx] * (r - l + 1),
        #         divide_conquer(heights, l, min_idx - 1),
        #         divide_conquer(heights, min_idx + 1, r),
        #     )
        # return divide_conquer(heights, 0, len(heights) - 1)

        # Stack - O(n)T, O(n)S
        stack = [-1]
        res = 0
        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                res = max(res, heights[stack.pop()] * (i - stack[-1] - 1))
            stack.append(i)
        while stack[-1] != -1:
            res = max(res, heights[stack.pop()] * (len(heights) - stack[-1] - 1))
        return res

        # # DP - O(n)T, O(n)S
        # n = len(heights)
        # left = [0] * n
        # right = [0] * n
        # left[0] = -1
        # right[-1] = n
        # for i in range(1, n):
        #     p = i - 1
        #     while p >= 0 and heights[p] >= heights[i]:
        #         p = left[p]
        #     left[i] = p
        # for i in range(n - 2, -1, -1):
        #     p = i + 1
        #     while p < n and heights[p] >= heights[i]:
        #         p = right[p]
        #     right[i] = p
        # res = 0
        # for i in range(n):
        #     res = max(res, heights[i] * (right[i] - left[i] - 1))
        # return res

    # LC 85. Maximal Rectangle (Hard)
    # https://leetcode.com/problems/maximal-rectangle/
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # # Brute force - O(m^2 * n^2)T, O(1)S
        # if not matrix:
        #     return 0
        # m, n = len(matrix), len(matrix[0])
        # res = 0
        # for i in range(m):
        #     for j in range(n):
        #         if matrix[i][j] == "0":
        #             continue
        #         width = float("inf")
        #         for k in range(i, m):
        #             if matrix[k][j] == "0":
        #                 break
        #             width = min(width, k - i + 1)
        #             res = max(res, width * (j - i + 1))
        # return res

        # DP - O(m*n)T, O(n)S
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        left = [0] * n
        right = [n] * n
        height = [0] * n
        res = 0
        for i in range(m):
            curr_left, curr_right = 0, n
            for j in range(n):
                if matrix[i][j] == "1":
                    height[j] += 1
                else:
                    height[j] = 0
            for j in range(n):
                if matrix[i][j] == "1":
                    left[j] = max(left[j], curr_left)
                else:
                    left[j] = 0
                    curr_left = j + 1
            for j in range(n - 1, -1, -1):
                if matrix[i][j] == "1":
                    right[j] = min(right[j], curr_right)
                else:
                    right[j] = n
                    curr_right = j
            for j in range(n):
                res = max(res, height[j] * (right[j] - left[j]))
        return res

        # # Stack - O(m*n)T, O(n)S
        # if not matrix:
        #     return 0
        # m, n = len(matrix), len(matrix[0])
        # left = [0] * n
        # right = [n] * n
        # height = [0] * n
        # res = 0
        # for i in range(m):
        #     curr_left, curr_right = 0, n
        #     for j in range(n):
        #         if matrix[i][j] == "1":
        #             height[j] += 1
        #         else:
        #             height[j] = 0
        #     stack = []
        #     for j in range(n):
        #         while stack and height[stack[-1]] >= height[j]:
        #             curr_right = stack.pop()
        #         left[j] = stack[-1] if stack else -1
        #         stack.append(j)
        #     stack = []
        #     for j in range(n - 1, -1, -1):
        #         while stack and height[stack[-1]] >= height[j]:
        #             curr_left = stack.pop()
        #         right[j] = stack[-1] if stack else n
        #         stack.append(j)
        #     for j in range(n):
        #         res = max(res, height[j] * (right[j] - left[j] - 1))
        # return res

    # LC 86. Partition List (Medium)
    # https://leetcode.com/problems/partition-list/
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        # # One-pass solution - O(n)T, O(1)S
        # before_head = before = ListNode(0)
        # after_head = after = ListNode(0)
        # while head:
        #     if head.val < x:
        #         before.next = head
        #         before = before.next
        #     else:
        #         after.next = head
        #         after = after.next
        #     head = head.next
        # after.next = None
        # before.next = after_head.next
        # return before_head.next

        # LC solution - O(n)T, O(1)S
        # Initialize two dummy nodes for 'before' and 'after' lists.
        before, after = ListNode(0), ListNode(0)
        # Pointers to help in appending nodes to 'before' and 'after' lists.
        before_curr, after_curr = before, after
        # Traverse the original list.
        while head:
            # Compare current node's value with x and append to appropriate list.
            if head.val < x:
                before_curr.next, before_curr = head, head
            else:
                after_curr.next, after_curr = head, head
            head = head.next
        # Ensure 'after' list's end points to None.
        after_curr.next = None
        # Connect the end of 'before' list to the start of 'after' list.
        before_curr.next = after.next
        # Return the merged list.
        return before.next

    # LC 87. Scramble String (Hard)
    # https://leetcode.com/problems/scramble-string/
    def isScramble(self, s1: str, s2: str) -> bool:
        # # DP - O(n^4)T, O(n^4)S
        # if len(s1) != len(s2) or Counter(s1) != Counter(s2):
        #     return False
        # n = len(s1)
        # dp = [[[False for _ in range(n + 1)] for _ in range(n)] for _ in range(n)]
        # for i in range(n):
        #     for j in range(n):
        #         dp[i][j][1] = s1[i] == s2[j]
        # for l in range(2, n + 1):
        #     for i in range(n - l + 1):
        #         for j in range(n - l + 1):
        #             for k in range(1, l):
        #                 if (
        #                     dp[i][j][k] and dp[i + k][j + k][l - k]
        #                     or dp[i][j + l - k][k] and dp[i + k][j][l - k]
        #                 ):
        #                     dp[i][j][l] = True
        #                     break
        # return dp[0][0][n]

        # # Recursion with memoization - O(n^4)T, O(n^4)S
        # memo = {}
        # def helper(s1, s2):
        #     if (s1, s2) in memo:
        #         return memo[(s1, s2)]
        #     if s1 == s2:
        #         return True
        #     if Counter(s1) != Counter(s2):
        #         return False
        #     for i in range(1, len(s1)):
        #         if (
        #             helper(s1[:i], s2[:i])
        #             and helper(s1[i:], s2[i:])
        #             or helper(s1[:i], s2[-i:])
        #             and helper(s1[i:], s2[:-i])
        #         ):
        #             memo[(s1, s2)] = True
        #             return True
        #     memo[(s1, s2)] = False
        #     return False
        # return helper(s1, s2)

        # DFS - O(n^6)T but sped up with lru_cache, O(n^2)S
        @functools.lru_cache(None)
        def dfs(s1, s2):
            if s1 == s2:
                return True
            if Counter(s1) != Counter(s2):
                return False
            for i in range(1, len(s1)):
                if (
                    dfs(s1[:i], s2[:i])       # gr|eat
                    and dfs(s1[i:], s2[i:])   # rg|tea
                    or dfs(s1[:i], s2[-i:])   # gr|eat
                    and dfs(s1[i:], s2[:-i])  # tea|gr
                ):
                    return True
            return False
        return dfs(s1, s2)

    # LC 88. Merge Sorted Array (Easy)
    # https://leetcode.com/problems/merge-sorted-array/
    def merge88(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # # With built-in function - O((m+n)log(m+n))T, O(1)S
        # nums1[m:] = nums2
        # nums1.sort()

        # # Two pointers - O(m+n)T, O(m)S
        # nums1_copy = nums1[:m]
        # i, j = 0, 0
        # for k in range(m + n):
        #     if j >= n or i < m and nums1_copy[i] <= nums2[j]:
        #         nums1[k] = nums1_copy[i]
        #         i += 1
        #     else:
        #         nums1[k] = nums2[j]
        #         j += 1

        # Two pointers (start from the end) - O(m+n)T, O(1)S
        i, j = m - 1, n - 1
        for k in range(m + n - 1, -1, -1):
            if j < 0 or i >= 0 and nums1[i] >= nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1

    # LC 89. Gray Code (Medium)
    # https://leetcode.com/problems/gray-code/
    def grayCode(self, n: int) -> List[int]:
        # # Iterative - O(2^n)T, O(1)S
        # res = [0]
        # for i in range(n):
        #     for j in range(len(res) - 1, -1, -1):
        #         res.append(res[j] | 1 << i)
        # return res

        # # Recursive - O(2^n)T, O(1)S
        # if n == 0:
        #     return [0]
        # res = self.grayCode(n - 1)
        # return res + [x | 1 << n - 1 for x in reversed(res)]

        # # Mathematical - O(2^n)T, O(1)S
        # res = []
        # for i in range(2 ** n):
        #     res.append(i ^ (i >> 1))
        # return res

        # One-liner - O(2^n)T, O(1)S
        return [i ^ (i >> 1) for i in range(2 ** n)]

    # LC 90. Subsets II (Medium)
    # https://leetcode.com/problems/subsets-ii/
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        # # My solution - O(n*2^n)T, O(n*2^n)S
        # res = set()
        # for i in range(len(nums) + 1):
        #     for el in itertools.combinations(nums, i):
        #         res.add(tuple(sorted(el)))
        # return sorted(res)

        # # One-liner - O(n*2^n)T, O(n*2^n)S
        # return sorted(set(sum([sorted(itertools.combinations(sorted(nums), r)) for r in range(len(nums) + 1)], [])))

        # # Backtracking - O(n*2^n)T, O(n*2^n)S
        # res = []
        # nums.sort()
        # def backtrack(i, path):
        #     res.append(path)
        #     for j in range(i, len(nums)):
        #         if j > i and nums[j] == nums[j - 1]:
        #             continue
        #         backtrack(j + 1, path + [nums[j]])
        # backtrack(0, [])
        # return res

        # Iterative - O(n*2^n)T, O(n*2^n)S
        res = [[]]
        nums.sort()
        for num in nums:
            res += [curr + [num] for curr in res if curr + [num] not in res]
        return res

    # LC 91. Decode Ways (Medium)
    # https://leetcode.com/problems/decode-ways/
    def numDecodings(self, s: str) -> int:
        # # Recursion with memoization - O(n)T, O(n)S
        # memo = {}
        # def helper(i):
        #     if i in memo:
        #         return memo[i]
        #     if i == len(s):
        #         return 1
        #     if s[i] == "0":
        #         return 0
        #     res = helper(i + 1)
        #     if i < len(s) - 1 and (s[i] == "1" or s[i] == "2" and s[i + 1] <= "6"):
        #         res += helper(i + 2)
        #     memo[i] = res
        #     return res
        # return helper(0)

        # # DP - O(n)T, O(n)S
        # n = len(s)
        # dp = [0] * (n + 1)
        # dp[n] = 1
        # for i in range(n - 1, -1, -1):
        #     if s[i] == "0":
        #         dp[i] = 0
        #     else:
        #         dp[i] = dp[i + 1]
        #         if i < n - 1 and (s[i] == "1" or s[i] == "2" and s[i + 1] <= "6"):
        #             dp[i] += dp[i + 2]
        # return dp[0]

        # DP with constant space - O(n)T, O(1)S
        n = len(s)
        dp = [0] * 3
        dp[n % 3] = 1
        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                dp[i % 3] = 0
            else:
                dp[i % 3] = dp[(i + 1) % 3]
                if i < n - 1 and (s[i] == "1" or s[i] == "2" and s[i + 1] <= "6"):
                    dp[i % 3] += dp[(i + 2) % 3]
        return dp[0]

    # LC 92. Reverse Linked List II (Medium)
    # https://leetcode.com/problems/reverse-linked-list-ii/
    def reverseBetween(
        self, head: Optional[ListNode], left: int, right: int
    ) -> Optional[ListNode]:
        # # My solution (one-pass) - O(n)T, O(1)S
        # left -= 1
        # right -= 1
        # if right == left == 0:
        #     return head
        # prev_original, curr = None, head
        # i = 0
        # while i < left:
        #     prev_original, curr = curr, curr.next
        #     i += 1
        # last_to_place = curr
        # prev = prev_original
        # while i <= right:
        #     dummy = curr.next
        #     curr.next = prev
        #     prev = curr
        #     curr = dummy
        #     i += 1
        # if prev_original:
        #     prev_original.next = prev
        # last_to_place.next = curr
        # return head if left else prev

        # Only two pointers (one-pass) - O(n)T, O(1)S
        if not head or left == right:
            return head
        dummy = ListNode(0, head)
        prev = dummy
        for _ in range(left - 1):
            prev = prev.next
        current = prev.next
        for _ in range(right - left):
            next_node = current.next
            current.next, next_node.next, prev.next = (
                next_node.next,
                prev.next,
                next_node,
            )
        return dummy.next

    # LC 93. Restore IP Addresses (Medium)
    # https://leetcode.com/problems/restore-ip-addresses/
    def restoreIpAddresses(self, s: str) -> List[str]:
        # # Backtracking - O(4^4)T, O(|res|)S
        # res = []
        # def backtrack(i, path):
        #     if len(path) == 4 and i == len(s):
        #         res.append(".".join(path))
        #         return
        #     if len(path) == 4 or i == len(s):
        #         return
        #     for j in range(i, len(s)):
        #         if j > i and s[i] == "0":
        #             break
        #         if int(s[i : j + 1]) <= 255:
        #             backtrack(j + 1, path + [s[i : j + 1]])
        # backtrack(0, [])
        # return res

        # Iterative (Brute-force) - O(4^4)T, O(|res|)S
        res = []
        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(1, 4):
                    for d in range(1, 4):
                        if a + b + c + d == len(s):
                            A, B, C, D = (
                                int(s[:a]),
                                int(s[a : a + b]),
                                int(s[a + b : a + b + c]),
                                int(s[a + b + c :]),
                            )
                            if A <= 255 and B <= 255 and C <= 255 and D <= 255:
                                ip = f"{A}.{B}.{C}.{D}"
                                if len(ip) == len(s) + 3:
                                    res.append(ip)
        return res

    # LC 94. Binary Tree Inorder Traversal (Medium)
    # https://leetcode.com/problems/binary-tree-inorder-traversal/
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # # Recursive - O(n)T, O(n)S
        # res = []
        # def helper(node):
        #     if not node:
        #         return
        #     helper(node.left)
        #     res.append(node.val)
        #     helper(node.right)
        # helper(root)
        # return res

        # Iterative - O(n)T, O(n)S
        res = []
        stack = []
        curr = root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            res.append(curr.val)
            curr = curr.right
        return res

        # # Morris traversal - O(n)T, O(n)S
        # res = []
        # curr = root
        # while curr:
        #     if not curr.left:
        #         res.append(curr.val)
        #         curr = curr.right
        #     else:
        #         prev = curr.left
        #         while prev.right:
        #             prev = prev.right
        #         prev.right = curr
        #         temp = curr
        #         curr = curr.left
        #         temp.left = None
        # return res

        # One-liner - O(n)T, O(n)S
        # return (self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right) if root else [])

    # LC 95. Unique Binary Search Trees II (Medium)
    # https://leetcode.com/problems/unique-binary-search-trees-ii/
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        # # Recursion - O(4^n / n^(3/2))T, O(4^n / n^(3/2))S
        # def helper(start, end):
        #     if start > end:
        #         return [None]
        #     res = []
        #     for i in range(start, end + 1):
        #         left = helper(start, i - 1)
        #         right = helper(i + 1, end)
        #         for l in left:
        #             for r in right:
        #                 root = TreeNode(i)
        #                 root.left = l
        #                 root.right = r
        #                 res.append(root)
        #     return res
        # return helper(1, n)

        # # DP - O(4^n / n^(3/2))T, O(4^n / n^(3/2))S
        # if not n:
        #     return []
        # def clone(node, offset):
        #     if not node:
        #         return None
        #     root = TreeNode(node.val + offset)
        #     root.left = clone(node.left, offset)
        #     root.right = clone(node.right, offset)
        #     return root
        # dp = [[] for _ in range(n + 1)]
        # dp[0].append(None)
        # for i in range(1, n + 1):
        #     for j in range(i):
        #         for left in dp[j]:
        #             for right in dp[i - j - 1]:
        #                 root = TreeNode(j + 1)
        #                 root.left = left
        #                 root.right = clone(right, j + 1)
        #                 dp[i].append(root)
        # return dp[n]

        # DP with memoization - O(4^n / n^(3/2))T, O(4^n / n^(3/2))S
        memo = {}
        def helper(start, end):
            if start > end:
                return [None]
            if (start, end) in memo:
                return memo[(start, end)]
            res = []
            for i in range(start, end + 1):
                left = helper(start, i - 1)
                right = helper(i + 1, end)
                for l in left:
                    for r in right:
                        root = TreeNode(i)
                        root.left = l
                        root.right = r
                        res.append(root)
            memo[(start, end)] = res
            return res
        return helper(1, n)
