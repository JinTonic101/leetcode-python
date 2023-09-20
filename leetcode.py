import collections
import heapq
import itertools
import re
from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict, deque
from copy import deepcopy
from functools import reduce
from itertools import combinations_with_replacement, permutations
from math import comb, factorial, inf
from operator import xor
from typing import List, Optional
import bisect


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
