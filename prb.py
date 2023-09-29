import collections
import heapq
import math
import operator
import re
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from copy import deepcopy
from functools import reduce
from itertools import combinations_with_replacement, permutations
from math import comb, factorial, inf
from operator import xor
from typing import List, Optional


def manhattan_distance(p1: List[int], p2: List[int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for _ in range(n)]

    def find(self, u):
        if self.parent[u] == u:
            return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u == v:
            return False
        if self.rank[u] > self.rank[v]:
            u, v = v, u
        self.parent[u] = v
        if self.rank[u] == self.rank[v]:
            self.rank[v] += 1
        return True


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
    # LC 268. Missing Number (Easy)
    # https://leetcode.com/problems/missing-number/
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        # Using XOR
        # missing = len(nums)
        # for i, num in enumerate(nums):
        #     missing ^= i ^ num
        # return missing

        # Using Gauss' Formula (but not overflow-proof)
        # expected_sum = len(nums)*(len(nums)+1)//2
        # actual_sum = sum(nums)
        # return expected_sum - actual_sum

        # Using Gauss' Formula (overflow-proof)
        n = len(nums)
        res = 0 + n
        for i in range(n):
            res += i - nums[i]
        return res

    # LC 896. Monotonic Array (Easy)
    # https://leetcode.com/problems/monotonic-array/
    def isMonotonic(self, nums):
        # # My solution - O(n)T, O(1)S
        # if len(nums) == 1:
        #     return True
        # increasing = nums[0] < nums[-1]
        # for i in range(0, len(nums) - 1):
        #     if increasing and nums[i] > nums[i + 1] or not increasing and nums[i] < nums[i + 1]:
        #         return False
        # return True

        # Optimized solution - O(n)T, O(1)S
        direction = -1 if nums[0] <= nums[-1] else 1
        for i in range(0, len(nums) - 1):
            if (nums[i] - nums[i + 1]) * direction < 0:
                return False
        return True

        # # One-liner
        # return nums == sorted(nums) or nums == sorted(nums, reverse=True)

        # Optimized one-liner - O(n)T, O(1)S
        # return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)) or all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1))


    # LC 543. Diameter of Binary Tree (Easy)
    # https://leetcode.com/problems/diameter-of-binary-tree/
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.number_of_nodes = 1

        def depth(node):
            if not node:
                return 0
            L, R = depth(node.left), depth(node.right)
            self.number_of_nodes = max(self.number_of_nodes, L + R + 1)
            return max(L, R) + 1

        depth(root)
        return self.number_of_nodes - 1

    # LC 101. Symmetric Tree (Easy)
    # https://leetcode.com/problems/symmetric-tree/
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # # Iterative - O(n)T, O(n)S
        # stack = [(root, root)]
        # while stack:
        #     n1, n2 = stack.pop()
        #     if not n1 and not n2:
        #         continue
        #     if not n1 or not n2:
        #         return False
        #     if n1.val != n2.val:
        #         return False
        #     stack.append((n1.left, n2.right))
        #     stack.append((n1.right, n2.left))
        # return True

        # # Recursive - O(n)T, O(n)S
        # def helper(n1, n2):
        #     if not n1 and not n2:
        #         return True
        #     if not n1 or not n2:
        #         return False
        #     return (
        #         n1.val == n2.val
        #         and helper(n1.left, n2.right)
        #         and helper(n1.right, n2.left)
        #     )
        # return helper(root, root)

        # Recursive (optimized) - O(n)T, O(n)S
        def helper(n1, n2):
            if n1 and n2:
                return (
                    n1.val == n2.val
                    and helper(n1.left, n2.right)
                    and helper(n1.right, n2.left)
                )
            return n1 == n2

        return helper(root, root)

    # LC 448. Find All Numbers Disappeared in an Array (Easy)
    # https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        # # O(n²)
        # S = set(n)
        # return [x for x in range(1, len(n)+1) if x not in S]

        # # One liner
        # return set(range(1, len(nums)+1)).difference(set(nums))

        # Linear time & no extra space
        res = []
        i = 0
        l = len(nums)
        while i < l:
            cur = nums[i]
            while cur is not None:
                tmp = nums[cur - 1]
                nums[cur - 1] = None
                cur = tmp
            i += 1
        for i in range(l):
            if nums[i] is not None:
                res.append(i + 1)
        return res

    # LC 169. Majority Element (Easy)
    # https://leetcode.com/problems/majority-element/
    def majorityElement(self, nums: List[int]) -> int:
        # # Counters in dict
        # if len(nums) == 1:
        #     return nums[0]
        # dic = {}
        # half = len(nums) // 2
        # for i in nums:
        #     if i in dic.keys():
        #         if dic[i] == half:
        #             return i
        #         dic[i] += 1
        #     else:
        #         dic[i] = 1

        # # One liner
        # from collections import Counter
        # return Counter(nums).most_common(1)[0][0]

        # Boyer-Moore Voting Algorithm #WOW
        count = 0
        candidate = None
        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if candidate == num else -1
        return candidate

    # LC 283. Move Zeroes (Easy)
    # https://leetcode.com/problems/move-zeroes/
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # # Solution 1: one-pass naive approach
        # i = 0
        # l = len(nums)
        # while i < l:
        #     if nums[i] == 0:
        #         nums.pop(i)
        #         nums.append(0)
        #         i -= 1
        #         l -= 1
        #     i += 1

        # Solution 2: one-pass double pointers approach
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                # Swap non-zero element at beginning of list
                nums[fast], nums[slow] = nums[slow], nums[fast]
                slow += 1

    # LC 136. Single Number (Easy)
    # https://leetcode.com/problems/single-number/
    def singleNumber(self, nums: List[int]) -> int:
        # # Naive approach with set
        # s = set()
        # for v in nums:
        #     if v in s:
        #         s.remove(v)
        #     else:
        #         s.add(v)
        # return s.pop()

        # # XOR Approach (XOR same numbers = 0)
        # a = 0
        # for i in nums:
        #     a ^= i
        # return a

        # XOR one-liner (LeetCode imports operator.xor as xor) #WOW
        return reduce(xor, nums)

    # LC 1002. Find Common Characters (Easy)
    # https://leetcode.com/problems/find-common-characters/
    def commonChars(self, words: List[str]) -> List[str]:
        first_word = set(list(words[0]))
        res = []
        for char in first_word:
            min_count = min([word.count(char) for word in words])
            res += [char] * min_count
        return res

    # LC 557. Reverse Words in a String III (Easy)
    # https://leetcode.com/problems/reverse-words-in-a-string-iii/
    def reverseWords(self, s: str) -> str:
        return " ".join([word[::-1] for word in s.split(" ")])

    # LC 1122. Relative Sort Array (Easy)
    # https://leetcode.com/problems/relative-sort-array/
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        # # With dict
        # dic = {}
        # for val in arr1:
        #     if val in dic.keys():
        #         dic[val] += 1
        #     else:
        #         dic[val] = 1
        # res = []
        # for val in arr2:
        #     res += [val] * dic[val]
        #     del dic[val]
        # for val in sorted(dic.keys()):
        #     res += [val] * dic[val]
        # return res

        # Without dict
        res = []
        for num in arr2:
            while num in arr1:
                arr1.remove(num)
                res.append(num)
        res += sorted(arr1)
        return res

    # (WTF) LC 897. Increasing Order Search Tree (Easy)
    # https://leetcode.com/problems/increasing-order-search-tree/
    def increasingBST(self, root: TreeNode) -> TreeNode:
        # # O(n)T and O(n)S
        # def inorder(node):
        #     if node:
        #         yield from inorder(node.left)
        #         yield node.val
        #         yield from inorder(node.right)
        # ans = cur = TreeNode(None)
        # for v in inorder(root):
        #     cur.right = TreeNode(v)
        #     cur = cur.right
        # return ans.right

        # O(n)T and O(1)S with Morris in-order traversal #WOW
        # https://leetcode.com/problems/increasing-order-search-tree/solutions/958187/morris-in-order-traversal-python-3-o-n-time-o-1-space/
        dummy = tail = TreeNode()
        node = root
        while node is not None:
            if node.left is not None:
                predecessor = node.left
                while predecessor.right is not None:
                    predecessor = predecessor.right
                predecessor.right = node
                left, node.left = node.left, None
                node = left
            else:
                tail.right = node
                tail = node
                node = node.right
        return dummy.right

    # LC 1200. Minimum Absolute Difference (Easy)
    # https://leetcode.com/problems/minimum-absolute-difference/
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        # # Sort arr + two passes
        # sorted_arr = sorted(arr)
        # N = len(sorted_arr)
        # min_diff = sorted_arr[1] - sorted_arr[0]
        # for i in range(2, N):
        #     min_diff = min(min_diff, sorted_arr[i] - sorted_arr[i - 1])
        # res = []
        # for i in range(1, N):
        #     if sorted_arr[i] - sorted_arr[i - 1] == min_diff:
        #         res.append([sorted_arr[i - 1], sorted_arr[i]])
        # return res

        # Same but in 3 lines
        arr.sort()
        mn = min(b - a for a, b in zip(arr, arr[1:]))
        return [[a, b] for a, b in zip(arr, arr[1:]) if b - a == mn]

        # # Sort arr + one pass
        # arr = sorted(arr)
        # if len(arr) == 2:
        #     return [arr]
        # res = [[arr[0], arr[1]]]
        # min_diff = arr[1] - arr[0]
        # for i in range(2, len(arr)):
        #     cur_diff = arr[i] - arr[i - 1]
        #     if cur_diff == min_diff:
        #         res.append([arr[i - 1], arr[i]])
        #     elif cur_diff < min_diff:
        #         min_diff = cur_diff
        #         res = [[arr[i - 1], arr[i]]]
        # return res

    # (WTF) LC 883. Projection Area of 3D Shapes (Easy)
    # https://leetcode.com/problems/projection-area-of-3d-shapes/
    def projectionArea(self, grid: List[List[int]]) -> int:
        # # 3 passes
        # xy = sum(map(bool, sum(grid, [])))
        # xz = sum(map(max, grid))
        # yz = sum(map(max, zip(*grid)))
        # return xy + yz + zx

        # # 3 passes - one liner
        # return sum([1 for i in grid for j in i if j != 0]+[max(i) for i in grid]+[max(i) for i in list(zip(*grid))])

        # # Single pass
        a = 0
        for i in range(len(grid)):
            numMax, numMax1 = 0, 0
            for j in range(len(grid)):
                if grid[i][j] > 0:
                    a += 1
                if grid[i][j] > numMax:
                    numMax = grid[i][j]
                if grid[j][i] > numMax1:
                    numMax1 = grid[j][i]
            a += numMax + numMax1
        return a

    # LC 559. Maximum Depth of N-ary Tree (Easy)
    # https://leetcode.com/problems/maximum-depth-of-n-ary-tree/
    def maxDepth(self, root: TreeNode) -> int:
        # BFS (iterative)
        if not root:
            return 0
        nodes = deque()
        nodes.append((root, 1))
        maxx = 0
        while nodes:
            cur, val = nodes.popleft()
            maxx = val
            if cur.children:
                for child in cur.children:
                    nodes.append((child, val + 1))
        return maxx

        # # DFS (recursion with helper function)
        # if not root:
        #     return 0
        # self.max_level = 0
        # def dfs(node, level):
        #     self.max_level = max(self.max_level, level)
        #     if not node.children:
        #         return
        #     for child in node.children:
        #         dfs(child, level + 1)
        # dfs(root, 1)
        # return self.max_level

        # # DFS (recursion without helper function)
        # if not root:
        #     return 0
        # if root.children:
        #     return 1 + max([self.maxDepth(x) for x in root.children])
        # else:
        #     return 1

    # LC 811. Subdomain Visit Count (Easy)
    # https://leetcode.com/problems/subdomain-visit-count/
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        # My solution
        dic = defaultdict(int)
        c = "."
        for cpdomain in cpdomains:
            [count, domain] = cpdomain.split(" ")
            count = int(count)
            dic[domain] += count
            index = domain.index(c) + 1 if c in domain else -1
            while index != -1:
                print(domain)
                domain = domain[index:]
                dic[domain] += count
                index = domain.index(c) + 1 if c in domain else -1
        res = []
        # for domain in dic.keys():
        #     res.append(str(dic[domain]) + " " + domain)
        # return res
        for domain, count in dic.items():
            yield f"{count} {domain}"

        # # Leetcode solution
        # d = defaultdict(int)
        # for s in cpdomains:
        #     cnt, s = s.split()
        #     cnt = int(cnt)
        #     d[s] += cnt
        #     pos = s.find('.') + 1
        #     while pos > 0:
        #         d[s[pos:]] += cnt
        #         pos = s.find('.', pos) + 1
        # for x, i in d.items():
        #     yield f'{i} {x}'

    # LC 509. Fibonacci Number (Easy)
    # https://leetcode.com/problems/fibonacci-number/
    def fib(self, n: int) -> int:
        # # Recursive (exponential time) - one liner
        # return n if n < 2 else self.fib(n-1) + self.fib(n-2)

        # # Iterative (linear time)
        # a, b = 0, 1
        # for _ in range(n):
        #     a, b = b, a + b
        # return a

        # Using math #WOW
        return int((((1 + 5**0.5) / 2) ** n + 1) / 5**0.5)

    # LC 965. Univalued Binary Tree
    # https://leetcode.com/problems/univalued-binary-tree/
    def isUnivalTree(self, root: TreeNode) -> bool:
        # # DFS (recursive)
        # if not root:
        #     return True
        # if root.left and root.left.val != root.val:
        #     return False
        # if root.right and root.right.val != root.val:
        #     return False
        # return self.isUnivalTree(root.left) and self.isUnivalTree(root.right)

        # BFS (iterative)
        nodes = deque()
        nodes.append(root)
        while nodes:
            node = nodes.popleft()
            if node.left:
                if node.left.val != node.val:
                    return False
                nodes.append(node.left)
            if node.right:
                if node.right.val != node.val:
                    return False
                nodes.append(node.right)
        return True

    # LC 1222. Queens That Can Attack the King (Medium)
    # https://leetcode.com/problems/queens-that-can-attack-the-king/
    def queensAttacktheKing(
        self, queens: List[List[int]], king: List[int]
    ) -> List[List[int]]:
        res = []
        queens = {(x, y) for x, y in queens}
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in range(1, 8):
                    x, y = king[0] + i * k, king[1] + j * k
                    if (x, y) in queens:
                        res.append([x, y])
                        break
        return res

        # # Newbie solution
        # N = 8
        # board = [ [ 0 for i in range(N) ] for j in range(N) ]
        # for queen in queens:
        #     board[queen[0]][queen[1]] = 1
        # # queens_set = set((queen[0], queen[1]) for queen in queens)
        # [kx, ky] = king

        # res = []

        # # left side of the king
        # i = kx - 1
        # while (i >= 0):
        #     if (board[i][ky]):
        #         res.append([i, ky])
        #         break
        #     i -= 1

        # # right side of the king
        # i = kx + 1
        # while (i < N):
        #     if (board[i][ky]):
        #         res.append([i, ky])
        #         break
        #     i += 1

        # # upper side of the king
        # j = ky - 1
        # while (j >= 0):
        #     if (board[kx][j]):
        #         res.append([kx, j])
        #         break
        #     j -= 1

        # # lower side of the king
        # j = ky + 1
        # while (j < N):
        #     if (board[kx][j]):
        #         res.append([kx, j])
        #         break
        #     j += 1

        # # left up diagonal of the king, can ve refactored in the previous left side section tho
        # i = kx - 1
        # j = ky - 1
        # while (i >= 0 and j >= 0):
        #     if (board[i][j]):
        #         res.append([i, j])
        #         break
        #     i -= 1
        #     j -= 1

        # # right up diagonal of the king
        # i = kx + 1
        # j = ky - 1
        # while (i < N and j >= 0):
        #     if (board[i][j]):
        #         res.append([i, j])
        #         break
        #     i += 1
        #     j -= 1

        # # left down diagonal of the king
        # i = kx - 1
        # j = ky + 1
        # while (i >= 0 and j < N):
        #     if (board[i][j]):
        #         res.append([i, j])
        #         break
        #     i -= 1
        #     j += 1

        # # right down diagonal of the king
        # i = kx + 1
        # j = ky + 1
        # while (i < N and j < N):
        #     if (board[i][j]):
        #         res.append([i, j])
        #         break
        #     i += 1
        #     j += 1

        # # print (res)
        # return res

    # LC 1221. Split a String in Balanced Strings (Easy)
    # https://leetcode.com/problems/split-a-string-in-balanced-strings/
    def balancedStringSplit(self, s: str) -> int:
        # # Basic approach with "local" counter
        # counter = 0
        # res = 0
        # for char in s:
        #     counter += 1 if char == "R" else -1
        #     if counter == 0:
        #         res += 1
        # return res

        # 3-liner with dict and without if
        c, res, dic = 0, 0, {"L": -1, "R": 1}
        for char in s:
            c, res = c + dic[char], res + (c == 0)
        return res

    # LC 922. Sort Array By Parity II
    # https://leetcode.com/problems/sort-array-by-parity-ii/
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        # # In-place sorting solution, but I have no idea why I did in reverse order
        # e = len(nums)-1  # index of most left odd number
        # for i in range(len(nums)-1)[::-2]:
        #     while (nums[i] % 2): # while nums[i] is an odd number
        #         nums[i], nums[e] = nums[e], nums[i]
        #         e -= 2
        #         if e == -1:
        #             # We correctly placed every odd numbers, by induction the even numbers are correctly placed as well
        #             # So no need to continue the for loop
        #             break
        # return nums

        # In-place sorting with 2 pointers and increamenting them 2 by 2
        j = 1
        for i in range(0, len(nums), 2):
            if nums[i] % 2:
                while nums[j] % 2:
                    j += 2
                nums[i], nums[j] = nums[j], nums[i]
        return nums

    # LC 1160. Find Words That Can Be Formed by Characters (Easy)
    # https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/
    def countCharacters(self, words: List[str], chars: str) -> int:
        # # Initial solution
        # res = 0
        # for word in words:
        #     tmp_chars = list(chars)
        #     valid = True
        #     for c in word:
        #         index = tmp_chars.index(c) if c in tmp_chars else -1
        #         if index == -1:
        #             valid = False
        #             break
        #         tmp_chars.pop(index)
        #     if valid:
        #         res += len(word)
        # return res

        # # Initial solution simplified
        # res = 0
        # for word in words:
        #     valid = True
        #     for i in word:
        #         if word.count(i) > chars.count(i):
        #             valid = False
        #             break
        #     if valid:
        #         res += len(word)
        # return res

        # Saving chars frequencies in a dict to avoid recounting
        res = 0
        freq = defaultdict(lambda: 0)
        for c in chars:
            freq[c] += 1
        for word in words:
            for char in word:
                if freq[char] < word.count(char):
                    break
            else:
                res += len(word)
        return res

    # LC 1051. Height Checker (Easy)
    # https://leetcode.com/problems/height-checker/
    def heightChecker(self, heights: List[int]) -> int:
        heights_sort = sorted(heights)
        count = 0
        for i in range(len(heights)):
            if heights[i] != heights_sort[i]:
                count += 1
        return count

        # # 1-liner
        # return sum(a != b for a, b in zip(heights, sorted(heights)))

    # LC 929. Unique Email Addresses (Easy)
    # https://leetcode.com/problems/unique-email-addresses/
    def numUniqueEmails(self, emails: List[str]) -> int:
        # # Basic solution
        # res = set()
        # for email in emails:
        #     local_name, domain_name = email.split("@")
        #     local_name = local_name.split("+")[0].replace(".", "")
        #     res.add(f"{local_name}@{domain_name}")
        # return len(res)

        # Basic solution with single pass for each email
        res = set()
        for email in emails:
            local, domain = email.split("@")
            tmp = []
            for c in local:
                if c == ".":
                    continue
                elif c == "+":
                    break
                else:
                    tmp.append(c)
            res.add("".join(tmp + ["@", domain]))
        return len(res)

    # LC 590. N-ary Tree Postorder Traversal (Easy)
    # https://leetcode.com/problems/n-ary-tree-postorder-traversal/
    def postorder(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        res = []
        for s in root.children:
            res += self.postorder(s)
        res.append(root.val)
        return res

    # LC 589. N-ary Tree Preorder Traversal (Easy)
    # https://leetcode.com/problems/n-ary-tree-preorder-traversal/
    def preorder(self, root: TreeNode) -> List[int]:
        # # Recursive
        # if not root:
        #     return []
        # res = []
        # res.append(root.val)
        # for s in root.children:
        #     res += self.preorder(s)
        # return res

        # Iterative
        if not root:
            return []
        res = []
        nodes = deque()
        nodes.append(root)
        while nodes:
            node = nodes.popleft()
            res.append(node.val)
            for c in reversed(node.children):
                nodes.appendleft(c)
        return res

    # LC 700. Search in a Binary Search Tree (Easy)
    # https://leetcode.com/problems/search-in-a-binary-search-tree/
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        cur = root
        while cur:
            if cur.val == val:
                return cur
            else:
                if cur.val < val:
                    cur = cur.right
                else:
                    cur = cur.left
        # One liner
        # return root if not root or val == root.val else (self.searchBST(root.left, val) if val < root.val else self.searchBST(root.right, val))

    # LC 1028. Recover a Tree From Preorder Traversal (Hard)
    # https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/
    def recoverFromPreorder(self, S: str) -> TreeNode:
        # # First solution
        # hash_array = [[] for i in range(1000)]
        # splitted = re.findall(r"\d+|\-+", traversal)
        # root = TreeNode(splitted[0]) # Setting the depth 0
        # hash_array[0].append(root)
        # depth = 1
        # for val in splitted[1:]:
        #     if val.isdigit():
        #         node = TreeNode(val)
        #         parent = hash_array[depth-1][-1]
        #         if not parent.left:
        #             parent.left = node
        #         elif not parent.right:
        #             parent.right = node
        #         hash_array[depth].append(node)
        #     else:
        #         depth = len(val)
        # return root

        # Best solution (without hash array nor recursion)
        traversal = traversal.split("-")
        root = TreeNode(int(traversal[0]))
        parent = root
        for val in traversal[1:]:
            if val == "":
                parent = parent.right if parent.right else parent.left
            else:
                if not parent.left:
                    parent.left = TreeNode(int(val))
                else:
                    parent.right = TreeNode(int(val))
                parent = root  # Restart from top
        return root

    # LC 944. Delete Columns to Make Sorted (Easy)
    # https://leetcode.com/problems/delete-columns-to-make-sorted/
    def minDeletionSize(self, strs: List[str]) -> int:
        sum([sorted(s) != list(s) for s in zip(*strs)])

    # LC 942. DI String Match (Easy)
    # https://leetcode.com/problems/di-string-match/
    def diStringMatch(self, s: str) -> List[int]:
        per = []
        lower = 0
        upper = len(s)
        for i in s:
            if i == "I":
                per.append(lower)
                lower += 1
            else:
                per.append(upper)
                upper -= 1
        if s[len(s) - 1] == "I":
            per.append(upper)
        else:
            per.append(lower)
        return per

    # LC 561. Array Partition I (Easy)
    # https://leetcode.com/problems/array-partition/
    def arrayPairSum(self, nums: List[int]) -> int:
        return sum(sorted(nums)[::2])

    # LC 852. Peak Index in a Mountain Array (Medium)
    # https://leetcode.com/problems/peak-index-in-a-mountain-array/
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # # Basic solution - O(N)T
        # for i in range(len(arr)):
        #     if arr[i + 1] < arr[i]:
        #         return i

        # # 1-liner (takes more time) - O(N)T
        # index, _value = max(enumerate(arr), key=operator.itemgetter(1))
        # return index

        # # 1-liner - O(N)T
        # return arr.index(max(arr))

        # # Binary search - O(log(N))T
        # l, r = 1, len(arr)-2  # 0, len(arr)-1
        # while l<=r:
        #     m = (l+r)//2
        #     if arr[m+1] < arr[m] and arr[m-1] < arr[m]: return m
        #     if arr[m+1] > arr[m]: l = m+1
        #     else: r = m

        # 1-liner binary search - O(log(N))T
        return bisect_left(range(len(arr) - 1), 1, key=lambda x: arr[x + 1] < arr[x])

    # LC 1217. Minimum Cost to Move Chips to The Same Position (Easy)
    # https://leetcode.com/contest/weekly-contest-157/problems/play-with-chips/
    def minCostToMoveChips(self, position: List[int]) -> int:
        # # Greedy approach
        # even_numbers = 0
        # odd_numbers = 0
        # for pos in position:
        #     if pos % 2:
        #         odd_numbers += 1
        #     else:
        #         even_numbers += 1
        # if even_numbers == 0 or odd_numbers == 0:
        #     return 0
        # return min(even_numbers, odd_numbers)

        # # 2-liner - O(N)T O(N)S
        # d = collections.Counter([p % 2 for p in position])
        # return min(d[0], d[1])

        # O(N)T O(1)S
        dic = defaultdict(int)
        for n in position:
            dic[n % 2] += 1
        return min(dic[0], dic[1])

    # LC 5214. Longest Arithmetic Subsequence of Given Difference
    # https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/description/
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        # # See https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/solutions/3761478/python-it-s-all-about-expectations-explained/
        # subseqs = {}
        # for n in arr:
        #     cnt_prev = subseqs.get(n, 0)
        #     cnt_next = subseqs.get(n + difference, 0)
        #     subseqs[n + difference] = max(cnt_prev + 1, cnt_next)
        # return max(subseqs.values())

        # See https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/solutions/3762641/4-line-python-hashmap-dp-two-sum-variation/
        # Main Intuition comes from problem "Two Sum"
        # Simply Hash previously seen values and the count of ongoing AP Length
        map = {}
        for val in arr:
            map[val] = map.get(val - difference, 0) + 1
        return max(map.values())

    # LC 461. Hamming Distance (Easy)
    # https://leetcode.com/problems/hamming-distance/description/
    def hammingDistance(self, x: int, y: int) -> int:
        x_bin = str(bin(x))[2:]
        y_bin = str(bin(y))[2:]

        len_x = len(x_bin)
        len_y = len(y_bin)

        if len_x > len_y:
            y_bin = "0" * (len_x - len_y) + y_bin
        elif len_x < len_y:
            x_bin = "0" * (len_y - len_x) + x_bin

        count = 0

        for i in range(min(len(x_bin), len(y_bin))):
            if x_bin[i] != y_bin[i]:
                count += 1

        return count

        # One liner #WOW
        # return bin(x ^ y).count("1")

    # LC 728. Self Dividing Numbers (Easy)
    # https://leetcode.com/problems/self-dividing-numbers/
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        res = []
        for i in range(left, right + 1):
            for char in str(i):
                if char == "0" or i % int(char):
                    break
            else:
                res.append(i)
        return res

        # One liner (less elegant)
        # return [x for x in range(left, right+1) if '0' not in str(x) and all([x % int(digit)==0 for digit in str(x)])]

    # LC 977. Squares of a Sorted Array (Easy)
    # https://leetcode.com/problems/squares-of-a-sorted-array/
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # Time : O(N) with Two pointers or O(N log(N)) with sort
        # Space : O(N)
        res = [0] * len(nums)
        l, r = 0, len(nums) - 1
        while l <= r:
            left, right = abs(nums[l]), abs(nums[r])
            if left > right:
                res[r - l] = left**2
                l += 1
            else:
                res[r - l] = right**2
                r -= 1
        return res

        # Or one liner #EASY
        # return sorted([a * a for a in A])

    # LC 657. Robot Return to Origin (Easy)
    # https://leetcode.com/problems/robot-return-to-origin/
    def judgeCircle(self, moves: str) -> bool:
        dic = {"D": 0, "U": 0, "L": 0, "R": 0}
        for i in moves:
            dic[i] += 1
        return dic["U"] == dic["D"] and dic["L"] == dic["R"]

        # Or one liner #EASY
        # return moves.count('U') == moves.count('D') and moves.count('L') == moves.count('R')

    # LC 961. N-Repeated Element in Size 2N Array (Easy)
    # https://leetcode.com/problems/n-repeated-element-in-size-2n-array/
    def repeatedNTimes(self, nums: List[int]) -> int:
        # # With Counter
        # n = len(nums) / 2
        # for num, count in Counter(nums).items():
        #     if count == n:
        #         return num

        # With dict and early stop
        n = len(nums) / 2
        d = defaultdict(lambda: 0)
        for num in nums:
            d[num] += 1
            if d[num] == n:
                return num

    ########## LC 905. Sort Array By Parity (Easy)
    # https://leetcode.com/problems/sort-array-by-parity/

    # use custom sort
    def sortArrayByParity0(self, A: List[int]) -> List[int]:  # 96 ms
        return sorted(A, key=lambda x: x % 2 == 1)

    # Append to 2 different lists and join them when return.
    def sortArrayByParity3(self, A: List[int]) -> List[int]:  # 100 ms
        o = []
        e = []
        for num in A:
            if num % 2 == 0:
                e.append(num)
            else:
                o.append(num)
        return e + o

    # When left is odd, decrease right until right is even or right>left, then switch them
    def sortArrayByParity2(self, A: List[int]) -> List[int]:  # 88 ms
        left = 0
        right = len(A) - 1
        while right >= left:
            if A[left] % 2:
                while right > left and A[right] % 2:
                    right -= 1
                A[right], A[left] = A[left], A[right]
            left += 1
        return A

    # Same as above, however, slightly different logic.
    # Increase left if left is even
    # Decrease right if right is odd
    # Switch them if left is odd and right is even.
    # As you can imagine. Which ever stop increasing/decreasing first will have to wait for the other
    def sortArrayByParity1(self, A: List[int]) -> List[int]:  # 96ms
        left = 0
        right = len(A) - 1
        while right >= left:
            if A[left] % 2 == 1 and A[right] % 2 == 0:
                A[right], A[left] = A[left], A[right]
                left += 1
                right -= 1
            if A[left] % 2 == 0:
                left += 1
            if A[right] % 2 == 1:
                right -= 1
        return A

    ##########

    # LC 832. Flipping an Image (Easy)
    # https://leetcode.com/problems/flipping-an-image/submissions/
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        # # In-place update
        # for i in range(len(image)):
        #     image[i] = [b ^ 1 for b in reversed(image[i])]
        # return image

        # # One liner
        # return [[1-i for i in row[::-1]] for row in image] # WOW
        # return [[1 ^ i for i in row[::-1]] for row in image]

        for row in image:
            for i in range((len(row) + 1) // 2):
                row[i], row[~i] = 1 - row[~i], 1 - row[i]  # WOW tilde operator
        return image

    # LC 1207. Unique Number of Occurrences (Easy)
    # https://leetcode.com/problems/unique-number-of-occurrences/
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        occurence_values = Counter(arr).values()
        return len(occurence_values) == len(set(occurence_values))  # WOW

    # LC 1079. Letter Tile Possibilities (Medium)
    # https://leetcode.com/problems/letter-tile-possibilities/
    def numTilePossibilities(self, tiles: str) -> int:
        count = 0
        for i in range(1, len(tiles) + 1):
            p = permutations(list(tiles), i)
            count += len(set(p))
        return count

        # One-liner
        # return sum([len(set(permutations(tiles, i))) for i in range(1, len(tiles)+1)])

    # LC 1021 . Remove Outermost Parentheses (Easy)
    # https://leetcode.com/problems/remove-outermost-parentheses/
    def removeOuterParentheses(self, s: str) -> str:
        # # With 2 counters in a dict
        # res = tmp = ""
        # dic = defaultdict(lambda: 0)  # opening & closing counters
        # for c in s:
        #     dic[c] += 1
        #     if c == ")" and dic["("] == dic[")"]:
        #         res += tmp[1:]
        #         dic = defaultdict(lambda: 0)
        #         tmp = ""
        #         continue
        #     tmp += c
        # return res

        # With a single counter
        cnt, res = 0, []
        for c in s:
            if c == ")":
                cnt -= 1
            if cnt != 0:
                res.append(c)
            if c == "(":
                cnt += 1
        return "".join(res)

    # LC 804. Unique Morse Code Words
    # https://leetcode.com/problems/unique-morse-code-words/
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        char_map = {
            "a": ".-",
            "b": "-...",
            "c": "-.-.",
            "d": "-..",
            "e": ".",
            "f": "..-.",
            "g": "--.",
            "h": "....",
            "i": "..",
            "j": ".---",
            "k": "-.-",
            "l": ".-..",
            "m": "--",
            "n": "-.",
            "o": "---",
            "p": ".--.",
            "q": "--.-",
            "r": ".-.",
            "s": "...",
            "t": "-",
            "u": "..-",
            "v": "...-",
            "w": ".--",
            "x": "-..-",
            "y": "-.--",
            "z": "--..",
        }
        res = set()
        for word in words:
            tmp = ""
            for c in word:
                tmp += char_map[c]
            res.add(tmp)
        return len(res)

        # # Two-liner
        # a=[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        # return len(set(["".join([a[ord(i)-97] for i in j]) for j in words]))

    # LC 88. Merge Sorted Array (Easy)
    # https://leetcode.com/problems/merge-sorted-array/
    def merge88(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # with built-in function
        # nums1[m:] = nums2
        # nums1.sort()

        i, j = m - 1, n - 1
        for c in range(m + n - 1, -1, -1):
            if j < 0:
                break
            if nums1[i] > nums2[j] and i >= 0:
                nums1[c] = nums1[i]
                i -= 1
            else:
                nums1[c] = nums2[j]
                j -= 1

        return nums1

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

    # LC 118. Pascal's Triangle (Easy)
    # https://leetcode.com/problems/pascals-triangle/
    def generate(self, numRows: int) -> List[List[int]]:
        """Return the numRows first rows of Pascal's Triangle"""
        """Return the numRows first rows of Pascal's Triangle"""
        # # DP - O(n^2)T, O(n^2)S
        # res = []
        # for i in range(numRows):
        #     row = [1] * (i + 1)
        #     for j in range(1, i):
        #         row[j] = res[i-1][j-1] + res[i-1][j]
        #     res.append(row)
        # return res

        # # Recursion - O(n^2)T, O(n^2)S
        # if numRows == 0:
        #     return []
        # triangle = [[1]]
        # for i in range(1, numRows):
        #     prev_row = triangle[-1]
        #     new_row = [1]
        #     for j in range(1, len(prev_row)):
        #         new_row.append(prev_row[j-1] + prev_row[j])
        #     new_row.append(1)
        #     triangle.append(new_row)
        # return triangle

        # Math: Combinatorics - O(n^2)T, O(n^2)S
        # (n
        #  k)
        triangle = []
        for n in range(numRows):
            row = []
            for k in range(n + 1):
                row.append(math.comb(n, k))
            triangle.append(row)
        return triangle

    # LC 119. Pascal's Triangle II (Easy)
    # https://leetcode.com/problems/pascals-triangle-ii/
    def getRow(self, rowIndex: int) -> List[int]:
        # # DP - O(n^2)T, O(n^2)S
        # res = []
        # for i in range(numRows):
        #     row = [1] * (i + 1)
        #     for j in range(1, i):
        #         row[j] = res[i-1][j-1] + res[i-1][j]
        #     res.append(row)
        # return res[numRows]

        # # Math: Combinatorics - O(n)T, O(n)S
        # # (n
        # #  k)
        # return [math.comb(rowIndex, k) for k in range(rowIndex + 1)]

        # Math: Combinatorics without import - O(n)T, O(n)S
        row = [1] * (rowIndex + 1)
        for i in range(1, rowIndex):
            row[i] = row[i - 1] * (rowIndex - i + 1) // i
        return row

    # 377. Combination Sum IV
    # https://leetcode.com/problems/combination-sum-iv/
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # # Brute force TO #shame
        # count = 0
        # # Remove nums greater than target
        # nums = [n for n in nums if n <= target]
        # if not nums:
        #     return 0

        # for i in range(1, target + 1):
        #     for c in combinations_with_replacement(nums, i):
        #         if sum(c) == target:
        #             count += len(set(permutations(c)))
        # return count

        # # DP
        # dp = [0] * (target + 1)
        # dp[0] = 1
        # for i in range(1, target + 1):
        #     for num in nums:
        #         if i - num >= 0:
        #             dp[i] += dp[i - num]
        # return dp[target]

        # Memoization with recursion
        nums.sort()
        memo = {}

        def helper(n):
            if n in memo:
                return memo[n]
            if n == 0:
                return 1
            if n < nums[0]:
                return 0

            count = 0
            for num in nums:
                if n - num < 0:
                    break
                count += helper(n - num)

            memo[n] = count
            return count

        return helper(target)

    # LC 392. Is Subsequence (Easy)
    # https://leetcode.com/problems/is-subsequence/
    def isSubsequence(self, s: str, t: str) -> bool:
        # # One-liner - O(s*t)T, O(1)S
        # return all(s[i] in t[i:] for i in range(len(s)))

        # # My solution - O(s+t)T, O(1)S
        # ls, lt = len(s), len(t)
        # if s == 0 or t == 0 or ls > lt:
        #     return False
        # i, j = 0, 0
        # while i < ls and j < lt:
        #     if s[i] == t[j]:
        #         i += 1
        #     j += 1
        # return i == ls

        # Two pointers - O(s+t)T, O(1)S
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)

    # LC 1646. Get Maximum in Generated Array (Easy)
    # https://leetcode.com/problems/get-maximum-in-generated-array/
    def getMaximumGenerated(self, n: int) -> int:
        dp = [0, 1]
        if n <= 1:
            return dp[n]
        for i in range(2, n + 1):
            if i % 2 == 0:
                dp.append(dp[i // 2])
            else:
                dp.append(dp[i // 2] + dp[(i // 2) + 1])
        return max(dp)

        # # With & and >> operator
        # dp = [0, 1]
        # for i in range(2, n + 1):
        #     if i & 1:
        #         dp += dp[(i - 1) >> 1] + dp[(i + 1) >> 1]
        #     else:
        #         dp += dp[i >> 1]
        # return max(dp) if n else dp[n]

    # LC 1359. Count All Valid Pickup and Delivery Options (Hard)
    # https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/
    def countOrders(self, n: int) -> int:
        MOD = 10**9 + 7

        # # DP - O(n)T, O(1)S
        # res = 1
        # for i in range(2, n+1):
        #     res = (res * (2*i - 1) * i) % MOD
        # return res

        # # Resursion with memoization - O(n)T, O(n)S
        # memo = {}
        # def helper(i):
        #     if i == 1:
        #         return 1
        #     if i in memo:
        #         return memo[i]
        #     res = (helper(i-1)*(2*i-1)*i) % MOD
        #     memo[i] = res
        #     return res
        # return helper(n)

        # Math - O(n)T, O(1)S
        # res = (2n)! / 2**n
        return (factorial(2 * n) * pow(2, -n, MOD)) % MOD

    # LC 338. Counting Bits (Easy)
    # https://leetcode.com/problems/counting-bits/submissions/
    def countBits(self, n: int) -> List[int]:
        # # Naive solution - O(n*log(n))T, O(n)S
        # res = []
        # for i in range(n+1):
        #     res.append(sum([int(b) for b in bin(i)[2:]]))
        # return res

        # DP with bit operators (AND and SHIFT) - O(n)T, O(n)S
        res = [0] * (n + 1)
        for i in range(1, n + 1):
            res[i] = (i & 1) + res[i >> 1]
        return res

        # # DP with offset - O(n)T, O(n)S
        # res = [0] * (n + 1)
        # offset = 1
        # for i in range(1, n + 1):
        #     if offset * 2 == i:
        #         offset *= 2
        #     res[i] = res[i - offset] + 1
        # return res

    # LC 2707. Extra Characters in a String (Medium)
    # https://leetcode.com/problems/extra-characters-in-a-string/
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        # # DP (comparison based on words) - O(n*m)T, O(n)S
        # dp = [0] * (len(s) + 1)
        # dictionary = set(dictionary)  # test sets contains lots of duplicates
        # for i in range(1, len(s) + 1):
        #     dp[i] = dp[i - 1] + 1
        #     for w in dictionary:
        #         if i >= len(w) and s[i - len(w) : i] == w:
        #             dp[i] = min(dp[i], dp[i - len(w)])
        # return dp[-1]

        # DP (comparison based on s) - O(n*n)T, O(n)S
        dp = [inf] * (len(s) + 1)
        dp[0] = 0
        dictionary = set(dictionary)  # test sets contains lots of duplicates
        for i in range(1, len(s) + 1):
            dp[i] = dp[i - 1] + 1
            for j in range(1, i + 1):
                if s[i - j : i] in dictionary:
                    dp[i] = min(dp[i], dp[i - j])
        return dp[-1]

        # # DP with Trie - O(n*m)T, O(n+k)S
        # Trie = lambda: defaultdict(Trie)
        # trie = Trie()
        # for word in dictionary:
        #     t = trie
        #     for c in word: t = t[c]
        #     t[' ']
        # dp = defaultdict(lambda: inf)
        # dp[len(s)] = 0
        # for start in reversed(range(len(s))):
        #     dp[start] = dp[start + 1] + 1
        #     node = trie
        #     for i, end in enumerate(s[start:]):
        #         if end not in node:
        #             break
        #         node = node[end]
        #         if ' ' in node:
        #             dp[start] = min(dp[start], dp[start + i + 1])
        # return dp[0]

        # # =====
        # # Bad solution
        # fc = defaultdict(lambda: set())
        # for w in dictionary:
        #     fc[w[0]].add(w)
        # windows = []
        # ls = len(s)
        # for i in range(ls):
        #     if s[i] not in fc:
        #         continue
        #     for w in fc[s[i]]:
        #         if i + len(w) <= ls and s[i : i + len(w)] == w:
        #             windows.append([i, i + len(w) - 1])
        #             print(w, i, i + len(w) - 1)
        # windows.sort(key=lambda x: (x[0], x[1]))
        # # get biggest windows subsets without merging overlapping windows
        # biggest_windows = []
        # prev = [-1, -1]
        # for w in windows:
        #     print(w, prev)
        #     if w[0] > prev[1]:
        #         biggest_windows.append(deepcopy(w))
        #         prev = w
        #     elif w[0] <= prev[1] and w[1] - w[0] > prev[1] - prev[0]:
        #         biggest_windows[-1] = deepcopy(w)
        #         prev = w
        # # loop through biggest_windows, if a w from windows is strictly between two consecusive windows in biggest_windows, add w to biggest_windows
        # for i in range(len(biggest_windows) - 1):
        #     for w in windows:
        #         if biggest_windows[i][1] < w[0] and w[1] < biggest_windows[i + 1][0]:
        #             biggest_windows.insert(i + 1, deepcopy(w))
        # # count all indexes covered by merged windows
        # print("--")
        # print(windows)
        # print(biggest_windows)
        # count = 0
        # for w in biggest_windows:
        #     count += w[1] - w[0] + 1
        # #     print(w[0], w[1], "-", w[1] - w[0] + 1, count)
        # # print(ls, count, ls - count)
        # print("--------")
        # return ls - count

    # LC 1282. Group the People Given the Group Size They Belong To (Medium)
    # https://leetcode.com/problems/group-the-people-given-the-group-size-they-belong-to/
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        # Greedy solution
        d = defaultdict(lambda: [[]])
        for i, gs in enumerate(groupSizes):
            if len(d[gs][-1]) == gs:
                d[gs].append([i])
            else:
                d[gs][-1].append(i)
        return sum(d.values(), [])

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

        # Math (comb)
        return comb(m + n - 2, m - 1)  # = comb(m + n - 2, n - 1)

    # LC 141. Linked List Cycle (Easy)
    # https://leetcode.com/problems/linked-list-cycle/
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # # Hash table - O(n)TS
        # visited_nodes = set()
        # current_node = head
        # while current_node:
        #     if current_node in visited_nodes:
        #         return True
        #     visited_nodes.add(current_node)
        #     current_node = current_node.next
        # return False

        # Floyd's Tortoise and Hare (Cycle Detection with double cursors fast and slow) - O(n)T, O(1)S
        fast = head
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False

    # LC 138. Copy List with Random Pointer (Medium)
    # https://leetcode.com/problems/copy-list-with-random-pointer/
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        # # Cheating with built-in function
        # return deepcopy(head)

        # # Hash table - O(n)TS
        # if not head:
        #     return None
        # h = {}
        # curr = head
        # while curr:
        #     h[curr] = Node(curr.val)
        #     curr = curr.next
        # curr = head
        # while curr:
        #     h[curr].next = h.get(curr.next, None)
        #     h[curr].random = h.get(curr.random, None)
        #     curr = curr.next
        # return h[head]

        # Interweaving - O(n)T, O(1)S
        if not head:
            return None
        curr = head
        while curr:
            new_node = Node(curr.val, curr.next)
            curr.next = new_node
            curr = new_node.next
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
        old_head = head
        new_head = head.next
        curr_old = old_head
        curr_new = new_head
        while curr_old:
            curr_old.next = curr_old.next.next
            curr_new.next = curr_new.next.next if curr_new.next else None
            curr_old = curr_old.next
            curr_new = curr_new.next
        return new_head

    # LC 725. Split Linked List in Parts (Medium)
    # https://leetcode.com/problems/split-linked-list-in-parts/
    def splitListToParts(
        self, head: Optional[ListNode], k: int
    ) -> List[Optional[ListNode]]:
        # Two pass solution - O(n)T, O(k)S
        res = [None] * k
        count, curr = 0, head
        while curr:
            count += 1
            curr = curr.next
        q, rem = divmod(count, k)  # quotient, remainder
        prev = head
        curr = head
        for i in range(min(k, count)):
            res[i] = curr
            for _ in range(q + (rem > 0)):
                prev, curr = curr, curr.next
            prev.next = None
            rem -= 1
        return res

    # LC 206. Reverse Linked List (Easy)
    # https://leetcode.com/problems/reverse-linked-list
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # One pass solution - O(n)T, O(1)S
        prev = None
        curr = head
        while curr:
            dummy = curr.next
            curr.next = prev
            prev = curr
            curr = dummy
        return prev

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

    # LC 1647. Minimum Deletions to Make Character Frequencies Unique (Medium)
    # https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/
    def minDeletions(self, s: str) -> int:
        # # Initial solution - O(|s|)T, O(|s|)S
        # d = defaultdict(int)
        # for c in s:
        #     d[c] += 1
        # freq = [0] * (max(d.values()) + 1)
        # for c in d:
        #     freq[d[c]] += 1
        # count = 0
        # for i in range(len(freq) - 1, 0, -1):
        #     while freq[i] > 1:
        #         freq[i] -= 1
        #         freq[i - 1] += 1
        #         count += 1
        # return count

        # Same but with counters & less loops
        freq = Counter(Counter(s).values())
        count = 0
        for i in range(max(freq), 0, -1):
            if freq[i] > 1:
                diff = freq[i] - 1
                freq[i] = 1
                freq[i - 1] += diff
                count += diff
        return count

    # LC 135. Candy (Hard)
    # https://leetcode.com/problems/candy/
    def candy(self, ratings: List[int]) -> int:
        # # Greedy two pass - O(n)TS
        # n = len(ratings)
        # candies = [1] * n  # candies
        # for i in range(1, n):
        #     if ratings[i] > ratings[i - 1]:
        #         candies[i] = candies[i - 1] + 1
        # for i in range(n - 2, -1, -1):
        #     if ratings[i] > ratings[i + 1]:
        #         candies[i] = max(candies[i], candies[i + 1] + 1)
        # return sum(candies)

        # Greedy one pass - O(n)T, O(1)S
        if not ratings:
            return 0
        ret, up, down, peak = 1, 0, 0, 0
        for prev, curr in zip(ratings[:-1], ratings[1:]):
            if prev < curr:
                up, down, peak = up + 1, 0, up + 1
                ret += 1 + up
            elif prev == curr:
                up = down = peak = 0
                ret += 1
            else:
                up, down = 0, down + 1
                ret += 1 + down - int(peak >= down)
        return ret

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

        # # Solution with backtracking & bitmask
        # n = len(nums)
        # Mask = (1 << n) - 1
        # ans = []
        # def backtrack(mask, cur):
        #     if mask == Mask:
        #         ans.append(cur[:])
        #         return
        #     for i in range(n):
        #         if mask & (1 << i) == 0:
        #             cur.append(nums[i])
        #             backtrack(mask | (1 << i), cur[:])
        #             cur.pop()
        # cur = []
        # backtrack(0, cur)
        # return ans

    # LC 58. Length of Last Word (Easy)
    # https://leetcode.com/problems/length-of-last-word/
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])

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

    # LC 332. Reconstruct Itinerary
    # https://leetcode.com/problems/reconstruct-itinerary/
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # # DFS (Recursive) - O(n)T, O(n)S
        # graph = defaultdict(list)
        # for src, dst in sorted(tickets, reverse=True):
        #     graph[src].append(dst)
        # itinerary = []
        # def dfs(airport):
        #     while graph[airport]:
        #         dfs(graph[airport].pop())
        #     itinerary.append(airport)
        # dfs("JFK")
        # return itinerary[::-1]

        # DFS (Iterative) - O(n)T, O(n)S
        graph = defaultdict(list)
        for src, dst in sorted(tickets, reverse=True):
            graph[src].append(dst)
        stack = ["JFK"]
        itinerary = []
        while stack:
            while graph[stack[-1]]:
                stack.append(graph[stack[-1]].pop())
            itinerary.append(stack.pop())
        return itinerary[::-1]

    # LC 344. Reverse String (Easy)
    # https://leetcode.com/problems/reverse-string/
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        # # One-liner
        # s[:] = s[::-1]

        # In-place & stop at middle
        for i in range(len(s) // 2):
            s[i], s[-i - 1] = s[-i - 1], s[i]

    # LC 459. Repeated Substring Pattern (Easy)
    # https://leetcode.com/problems/repeated-substring-pattern/
    def repeatedSubstringPattern(self, s: str) -> bool:
        # # Brute force
        # n = len(s)
        # for i in range(1, n//2 + 1):
        #     if n % i == 0:
        #         substring = s[:i]
        #         if substring * (n // i) == s:
        #             return True
        # return False

        # String rotation
        return s in (s + s)[1:-1]

    # LC 1584. Min Cost to Connect All Points (Medium)
    # https://leetcode.com/problems/min-cost-to-connect-all-points/
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # # Prim's algorithm - O(n²log(⁡n)), O(n)S
        # n = len(points)
        # dist = [inf] * n
        # dist[0] = 0
        # visited = set()
        # res = 0
        # while len(visited) < n:
        #     min_dist = inf
        #     min_idx = -1
        #     for i in range(n):
        #         if i not in visited and dist[i] < min_dist:
        #             min_dist = dist[i]
        #             min_idx = i
        #     res += min_dist
        #     visited.add(min_idx)
        #     for i in range(n):
        #         if i not in visited:
        #             dist[i] = min(
        #                 dist[i],
        #                 abs(points[i][0] - points[min_idx][0])
        #                 + abs(points[i][1] - points[min_idx][1]),
        #             )
        # return res

        # Kruskal's algorithm - O(n²log(⁡n)), O(n²)S
        n = len(points)
        uf = UnionFind(n)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = manhattan_distance(points[i], points[j])
                heapq.heappush(edges, (distance, i, j))
        mst_weight = 0
        mst_edges = 0
        while edges:
            w, u, v = heapq.heappop(edges)
            if uf.union(u, v):
                mst_weight += w
                mst_edges += 1
                if mst_edges == n - 1:
                    break
        return mst_weight

    # LC 1631. Path With Minimum Effort (Medium)
    # https://leetcode.com/problems/path-with-minimum-effort/
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        # Dijkstra's algorithm - O(m*n*log(m*n)), O(m*n)S
        m, n = len(heights), len(heights[0])
        dist = [[inf] * n for _ in range(m)]
        dist[0][0] = 0
        visited = set()
        heap = [(0, 0, 0)]
        while heap:
            d, x, y = heapq.heappop(heap)
            if x == m - 1 and y == n - 1:
                return d
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    new_dist = max(d, abs(heights[nx][ny] - heights[x][y]))
                    if new_dist < dist[nx][ny]:
                        dist[nx][ny] = new_dist
                        heapq.heappush(heap, (new_dist, nx, ny))

    # LC 847. Shortest Path Visiting All Nodes (Hard)
    # https://leetcode.com/problems/shortest-path-visiting-all-nodes/
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        # # BFS - O(n*2^n), O(n*2^n)S
        # n = len(graph)
        # queue = deque((1 << i, i) for i in range(n))
        # dist = defaultdict(lambda: inf)
        # for i in range(n):
        #     dist[(1 << i, i)] = 0
        # while queue:
        #     cover, head = queue.popleft()
        #     d = dist[(cover, head)]
        #     if cover == 2 ** n - 1:
        #         return d
        #     for child in graph[head]:
        #         new_cover = cover | (1 << child)
        #         if d + 1 < dist[(new_cover, child)]:
        #             dist[(new_cover, child)] = d + 1
        #             queue.append((new_cover, child))

        # DP with bitmask - O(n*2^n), O(n*2^n)S
        n = len(graph)
        dp = [[inf] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 0
        for cover in range(1 << n):
            repeat = True
            while repeat:
                repeat = False
                for head, d in enumerate(dp[cover]):
                    for child in graph[head]:
                        new_cover = cover | (1 << child)
                        if d + 1 < dp[new_cover][child]:
                            dp[new_cover][child] = d + 1
                            if new_cover == cover:
                                repeat = True
        res = min(dp[-1])
        return res

    # LC 1337. The K Weakest Rows in a Matrix (Easy)
    # https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        # Counters - O(m*n)T, O(m)S
        counts = [(i, sum(mat[i])) for i in range(len(mat))]
        return sorted(counts, key=operator.itemgetter(0))[:k]

    # LC 219. Contains Duplicate II (Easy)
    # https://leetcode.com/problems/contains-duplicate-ii/
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        if len(nums) <= 1:
            return False
        d = defaultdict(lambda: [])
        for i, num in enumerate(nums):
            if num in d and i - d[num] <= k:
                return True
            d[num] = i
        return False

    # LC 287. Find the Duplicate Number (Medium)
    # https://leetcode.com/problems/find-the-duplicate-number/
    def findDuplicate(self, nums: List[int]) -> int:
        # Floyd's Tortoise and Hare (Cycle Detection with double pointers slow and fast) - O(n)T, O(1)S
        slow = fast = nums[0]
        # find cycle
        while True:
            slow, fast = nums[slow], nums[nums[fast]]
            if slow == fast:
                break
        # find cycle entry point
        slow = nums[0]
        while slow != fast:
            slow, fast = nums[slow], nums[fast]
        return slow

    # LC 205. Isomorphic Strings
    # https://leetcode.com/problems/isomorphic-strings/
    def isIsomorphic(self, s: str, t: str) -> bool:
        # One-liner - O(n)T, O(n)S
        return len(set(s)) == len(set(t)) == len(set(zip(s, t)))
        # return [*map(s.index, s)] == [*map(t.index, t)]

    # LC 414. Third Maximum Number (Easy)
    # https://leetcode.com/problems/third-maximum-number/
    def thirdMax(self, nums: List[int]) -> int:
        s = set(nums)
        if len(s) < 3:
            return max(s)
        return sorted(s, reverse=True)[2]  # Third max

    # LC 1658. Minimum Operations to Reduce X to Zero (Medium)
    # https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/
    def minOperations(self, nums: List[int], x: int) -> int:
        # Two pointers (sliding window) - O(n)T, O(1)S
        target, n = sum(nums) - x, len(nums)
        if target < 0:
            return -1
        if target == 0:
            return n
        max_len = cur_sum = left = 0
        for right, val in enumerate(nums):
            cur_sum += val
            while left <= right and cur_sum > target:
                cur_sum -= nums[left]
                left += 1
            if cur_sum == target:
                max_len = max(max_len, right - left + 1)
        return n - max_len if max_len else -1

    # LC 1048. Longest String Chain (Medium)
    # https://leetcode.com/problems/longest-string-chain/
    def longestStrChain(self, words: List[str]) -> int:
        # # DP - O(nlog(n) + n*m²), O(n)S
        # words.sort(key=len)
        # dp = collections.defaultdict(lambda: 1)
        # max_chain = 0
        # for word in words:
        #     for i in range(len(word)):
        #         prev_word = word[:i] + word[i + 1 :]
        #         if prev_word in dp:
        #             dp[word] = max(dp[word], dp[prev_word] + 1)
        #     max_chain = max(max_chain, dp[word])
        # return max_chain

        # DFS with memoization - O(n*m²), O(n)S
        def dfs(word):
            if word not in memo:
                memo[word] = 1
                for i in range(len(word)):
                    prev = word[:i] + word[i + 1 :]
                    if prev in words:
                        memo[word] = max(memo[word], dfs(prev) + 1)
            return memo[word]

        memo = {}
        words = set(words)
        return max(dfs(word) for word in words)

    # LC 799. Champagne Tower (Medium)
    # https://leetcode.com/problems/champagne-tower/
    def champagneTower(self, poured: int, query_row: int, query_glass: int) -> float:
        # # DP - O(n²)T, O(n²)S
        # dp = [[0] * (i + 1) for i in range(query_row + 2)]
        # dp[0][0] = poured
        # for i in range(query_row + 1):
        #     for j in range(i + 1):
        #         if dp[i][j] > 1:
        #             dp[i + 1][j] += (dp[i][j] - 1) / 2
        #             dp[i + 1][j + 1] += (dp[i][j] - 1) / 2
        #             dp[i][j] = 1
        # return dp[query_row][query_glass]

        # DP (optimized) - O(n²)T, O(n)S
        dp = [0] * (query_row + 1)
        dp[0] = poured
        for i in range(1, query_row + 1):
            for j in range(i, -1, -1):
                dp[j] = max(0, (dp[j] - 1) / 2) + max(0, (dp[j - 1] - 1) / 2)
        return min(1, dp[query_glass])

    # LC 389. Find the Difference (Easy)
    # https://leetcode.com/problems/find-the-difference/
    def findTheDifference(self, s: str, t: str) -> str:
        # # Hash table - O(n)T, O(1)S
        # d = defaultdict(int)
        # for c in s:
        #     d[c] += 1
        # for c in t:
        #     if d[c] == 0:
        #         return c
        #     d[c] -= 1

        # # Bit manipulation - O(n)T, O(1)S
        # res = 0
        # for c in s + t:
        #     res ^= ord(c)
        # return chr(res)

        # # One-liner (sum of unicodes) - O(n)T, O(1)S
        # return chr(sum(map(ord, t)) - sum(map(ord, s)))

        # # One-liner (Counter) - O(n)T, O(1)S
        # return list(Counter(t) - Counter(s))[0]

        # # One-liner (Counter and iter) - O(n)T, O(1)S
        return next(iter((Counter(t) - Counter(s)).keys()))

    # LC 1572. Matrix Diagonal Sum (Easy)
    # https://leetcode.com/problems/matrix-diagonal-sum/
    def diagonalSum(self, mat: List[List[int]]) -> int:
        # One-pass - O(n)T, O(1)S
        n, res = len(mat), 0
        for i in range(n):
            res += mat[i][i] + mat[i][-i - 1]
        if n % 2:
            res -= mat[n // 2][n // 2]
        return res

    # LC 316. Remove Duplicate Letters (Medium)
    # https://leetcode.com/problems/remove-duplicate-letters/
    def removeDuplicateLetters(self, s: str) -> str:
        # Greedy - O(n)T, O(1)S
        last_occurrence = {c: i for i, c in enumerate(s)}
        stack = []
        for i, c in enumerate(s):
            if c in stack:
                continue
            while stack and stack[-1] > c and i < last_occurrence[stack[-1]]:
                stack.pop()
            stack.append(c)
        return "".join(stack)

    # LC 880. Decoded String at Index (Medium)
    # https://leetcode.com/problems/decoded-string-at-index/
    def decodeAtIndex(self, s: str, k: int) -> str:
        # # Reverse traversal - O(n)T, O(1)S
        # n = len(s)
        # decoded_len = 0
        # for i in range(n):
        #     if s[i].isdigit():
        #         decoded_len *= int(s[i])
        #     else:
        #         decoded_len += 1
        # for i in range(n - 1, -1, -1):
        #     k %= decoded_len
        #     if k == 0 and s[i].isalpha():
        #         return s[i]
        #     if s[i].isdigit():
        #         decoded_len //= int(s[i])
        #     else:
        #         decoded_len -= 1

        # Reverse traversal (LC solution) - O(n)T, O(1)S
        length = 0
        i = 0
        while length < k:
            if s[i].isdigit():
                length *= int(s[i])
            else:
                length += 1
            i += 1
        for j in range(i - 1, -1, -1):
            char = s[j]
            if char.isdigit():
                length //= int(char)
                k %= length
            else:
                if k == 0 or k == length:
                    return char
                length -= 1

    # LC 905. Sort Array By Parity (Easy)
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        # # Two pointers - O(n)T, O(1)S
        # l, r = 0, len(nums) - 1
        # while l < r:
        #     while l < r and nums[l] % 2 == 0:
        #         l += 1
        #     while l < r and nums[r] % 2:
        #         r -= 1
        #     nums[l], nums[r] = nums[r], nums[l]
        # return nums

        # One-liner (two passes) - O(n)T, O(n)S
        return [x for x in nums if x % 2 == 0] + [x for x in nums if x % 2]
