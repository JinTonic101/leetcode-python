import bisect
import collections
import functools
import heapq
import itertools
import math
import operator
import re
import string
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

    # LC 102. Binary Tree Level Order Traversal (Medium)
    # https://leetcode.com/problems/binary-tree-level-order-traversal/
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Iterative - O(n)T, O(n)S
        if not root:
            return []
        res = []
        queue = collections.deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res

        # # Recursive - O(n)T, O(n)S
        # res = []
        # def helper(node, level):
        #     if not node:
        #         return
        #     if len(res) == level:
        #         res.append([])
        #     res[level].append(node.val)
        #     helper(node.left, level + 1)
        #     helper(node.right, level + 1)
        # helper(root, 0)
        # return res

    # LC 103. Binary Tree Zigzag Level Order Traversal (Medium)
    # https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Iterative - O(n)T, O(n)S
        if not root:
            return []
        res = []
        queue = collections.deque([root])
        l = 0
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level[::-1] if l % 2 else level)
            l += 1
        return res

        # # Recursive - O(n)T, O(n)S
        # res = []
        # def helper(node, level):
        #     if not node:
        #         return
        #     if len(res) == level:
        #         res.append([])
        #     if level % 2:
        #         res[level].insert(0, node.val)
        #     else:
        #         res[level].append(node.val)
        #     helper(node.left, level + 1)
        #     helper(node.right, level + 1)
        # helper(root, 0)
        # return res

    # LC 104. Maximum Depth of Binary Tree (Easy)
    # https://leetcode.com/problems/maximum-depth-of-binary-tree/
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # # Recursive (my solution) - O(n)T, O(n)S
        # depth = 0
        # def helper(node, level):
        #     if not node:
        #         return level
        #     return max(helper(node.left, level + 1), helper(node.right, level + 1))
        # return helper(root, 0)

        # Recursive - O(n)T, O(n)S
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

        # # Recursive (one-liner) - O(n)T, O(n)S
        # return 0 if not root else 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

        # # Iterative - O(n)T, O(n)S
        # if not root:
        #     return 0
        # depth = 0
        # queue = collections.deque([root])
        # while queue:
        #     depth += 1
        #     for _ in range(len(queue)):
        #         node = queue.popleft()
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        # return depth

    # LC 105. Construct Binary Tree from Preorder and Inorder Traversal (Medium)
    # https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # Recursion - O(n)T, O(n)S
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder.pop(0))
        idx = inorder.index(root.val)
        root.left = self.buildTree(preorder, inorder[:idx])
        root.right = self.buildTree(preorder, inorder[idx + 1 :])
        return root

    # LC 106. Construct Binary Tree from Inorder and Postorder Traversal (Medium)
    # https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        # # Iterative - O(n)T, O(n)S
        # if not inorder or not postorder:
        #     return None
        # root = TreeNode(postorder[-1])
        # stack = [root]
        # idx = len(inorder) - 1
        # for i in range(len(postorder) - 2, -1, -1):
        #     node = TreeNode(postorder[i])
        #     parent = None
        #     while stack and stack[-1].val == inorder[idx]:
        #         parent = stack.pop()
        #         idx -= 1
        #     if parent:
        #         parent.left = node
        #     else:
        #         stack[-1].right = node
        #     stack.append(node)
        # return root

        # Recursive - O(n)T, O(n)S
        if not inorder or not postorder:
            return None
        root = TreeNode(postorder.pop())
        idx = inorder.index(root.val)
        root.right = self.buildTree(inorder[idx + 1 :], postorder)
        root.left = self.buildTree(inorder[:idx], postorder)
        return root

    # LC 107. Binary Tree Level Order Traversal II (Easy)
    # https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Iterative - O(n)T, O(n)S
        if not root:
            return []
        res = []
        queue = collections.deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0, level)
        return res

        # # Recursive - O(n)T, O(n)S
        # res = []
        # def helper(node, level):
        #     if not node:
        #         return
        #     if len(res) == level:
        #         res.insert(0, [])
        #     res[-level - 1].append(node.val)
        #     helper(node.left, level + 1)
        #     helper(node.right, level + 1)
        # helper(root, 0)
        # return res

    # LC 108. Convert Sorted Array to Binary Search Tree (Easy)
    # https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        # Recursive - O(n)T, O(n)S
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid + 1 :])
        return root

        # # Iterative - O(n)T, O(n)S
        # if not nums:
        #     return None
        # root = TreeNode(0)
        # stack = [(root, 0, len(nums) - 1)]
        # while stack:
        #     node, left, right = stack.pop()
        #     mid = (left + right) // 2
        #     node.val = nums[mid]
        #     if left <= mid - 1:
        #         node.left = TreeNode(0)
        #         stack.append((node.left, left, mid - 1))
        #     if mid + 1 <= right:
        #         node.right = TreeNode(0)
        #         stack.append((node.right, mid + 1, right))
        # return root

    # LC 109. Convert Sorted List to Binary Search Tree (Medium)
    # https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        # # Double pass (recursive) - O(n)T, O(n)S
        # nums = []
        # while head:
        #     nums.append(head.val)
        #     head = head.next
        # if not nums:
        #     return None
        # def helper(node, nums):
        #     if not nums:
        #         return None
        #     mid = len(nums) // 2
        #     node.val = nums[mid]
        #     node.left = helper(TreeNode(0), nums[:mid])
        #     node.right = helper(TreeNode(0), nums[mid+1:])
        #     return node
        # return helper(TreeNode(0), nums)

        # One pass (recursive) - O(n)T, O(n)S
        def find_mid(head):
            slow = fast = head
            prev = None
            while fast and fast.next:
                prev = slow
                slow = slow.next
                fast = fast.next.next
            if prev:
                prev.next = None
            return slow

        if not head:
            return None
        mid = find_mid(head)
        root = TreeNode(mid.val)
        if head == mid:
            return root
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(mid.next)
        return root

    # LC 110. Balanced Binary Tree (Easy)
    # https://leetcode.com/problems/balanced-binary-tree/
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # # Recursive - O(n)T, O(n)S
        # def helper(node):
        #     if not node:
        #         return 0
        #     left = helper(node.left)
        #     right = helper(node.right)
        #     if left == -1 or right == -1 or abs(left - right) > 1:
        #         return -1
        #     return 1 + max(left, right)
        # return helper(root) != -1

        # Recursive (optimized) - O(n)T, O(n)S
        def helper(node):
            if not node:
                return 0
            left = helper(node.left)
            if left == -1:
                return -1
            right = helper(node.right)
            if right == -1 or abs(left - right) > 1:
                return -1
            return 1 + max(left, right)

        return helper(root) != -1

    # LC 111. Minimum Depth of Binary Tree (Easy)
    # https://leetcode.com/problems/minimum-depth-of-binary-tree/
    def minDepth(self, root: Optional[TreeNode]) -> int:
        # # Recursive - O(n)T, O(n)S
        # if not root:
        #     return 0
        # left = self.minDepth(root.left)
        # right = self.minDepth(root.right)
        # if left and right:
        #     return 1 + min(left, right)
        # return 1 + max(left, right)

        # Iterative - O(n)T, O(n)S
        if not root:
            return 0
        depth = 0
        queue = collections.deque([root])
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if not node.left and not node.right:
                    return depth
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return depth

    # LC 112. Path Sum (Easy)
    # https://leetcode.com/problems/path-sum/
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        # # Recursive - O(n)T, O(n)S
        # if not root:
        #     return False
        # v = targetSum - root.val
        # if not root.left and not root.right:
        #     return v == 0
        # return self.hasPathSum(root.left, v) or self.hasPathSum(root.right, v)

        # Iterative - O(n)T, O(n)S
        if not root:
            return False
        stack = [(root, targetSum)]
        while stack:
            node, target = stack.pop()
            if not node.left and not node.right and target == node.val:
                return True
            if node.left:
                stack.append((node.left, target - node.val))
            if node.right:
                stack.append((node.right, target - node.val))
        return False

    # LC 113. Path Sum II (Medium)
    # https://leetcode.com/problems/path-sum-ii/
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        # # Iterative - O(n)T, O(n)S
        # if not root:
        #     return []
        # res = []
        # stack = [(root, targetSum, [])]
        # while stack:
        #     node, target, path = stack.pop()
        #     if not node.left and not node.right and target == node.val:
        #         res.append(path + [node.val])
        #     if node.left:
        #         stack.append((node.left, target - node.val, path + [node.val]))
        #     if node.right:
        #         stack.append((node.right, target - node.val, path + [node.val]))
        # return res

        # # Recursive - O(n)T, O(n)S
        # res = []
        # def helper(node, target, path):
        #     if not node:
        #         return
        #     if not node.left and not node.right and target == node.val:
        #         res.append(path + [node.val])
        #         return
        #     helper(node.left, target - node.val, path + [node.val])
        #     helper(node.right, target - node.val, path + [node.val])
        # helper(root, targetSum, [])
        # return res

        # Recursive (optimized) - O(n)T, O(n)S
        if not root:
            return []
        if not root.left and not root.right and targetSum == root.val:
            return [[root.val]]
        left = self.pathSum(root.left, targetSum - root.val)
        right = self.pathSum(root.right, targetSum - root.val)
        return [[root.val] + path for path in left + right]

    # LC 114. Flatten Binary Tree to Linked List (Medium)
    # https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
    def flatten(self, root: Optional[TreeNode]) -> None:
        # # Iterative - O(n)T, O(n)S
        # if not root:
        #     return
        # stack = [root]
        # prev = None
        # while stack:
        #     node = stack.pop()
        #     if prev:
        #         prev.right = node
        #         prev.left = None
        #     if node.right:
        #         stack.append(node.right)
        #     if node.left:
        #         stack.append(node.left)
        #     prev = node

        # # Recursive (semi Morris traversal) - O(n)T, O(n)S
        # if not root:
        #     return
        # self.flatten(root.left)
        # self.flatten(root.right)
        # if root.left:
        #     node = root.left
        #     while node.right:
        #         node = node.right
        #     node.right = root.right
        #     root.right = root.left
        #     root.left = None

        # Morris traversal - O(n)T, O(1)S
        if not root:
            return
        node = root
        while node:
            if node.left:
                prev = node.left
                while prev.right:
                    prev = prev.right
                prev.right = node.right
                node.right = node.left
                node.left = None
            node = node.right

    # LC 115. Distinct Subsequences (Hard)
    # https://leetcode.com/problems/distinct-subsequences/
    def numDistinct(self, s: str, t: str) -> int:
        # # DP - O(mn)T, O(mn)S
        # m, n = len(s), len(t)
        # dp = [[0] * (n + 1) for _ in range(m + 1)]
        # for i in range(m + 1):
        #     dp[i][-1] = 1
        # for i in range(m - 1, -1, -1):
        #     for j in range(n - 1, -1, -1):
        #         dp[i][j] = dp[i + 1][j]
        #         if s[i] == t[j]:
        #             dp[i][j] += dp[i + 1][j + 1]
        # return dp[0][0]

        # DP (optimized) - O(mn)T, O(mn)S
        @functools.lru_cache(None)
        def dp(i, j):
            if j == len(t):
                return 1
            if i == len(s):
                return 0
            if len(s) - i < len(t) - j:
                return 0  # prunnig
            if s[i] == t[j]:
                return dp(i + 1, j + 1) + dp(i + 1, j)
            return dp(i + 1, j)

        return dp(0, 0)

    # LC 116. Populating Next Right Pointers in Each Node (Medium)
    # https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        # # Iterative (BFS with queues) - O(N)TS
        # if not root:
        #     return root
        # queue = collections.deque([root])
        # while queue:
        #     new_queue = collections.deque()
        #     while queue:
        #         node = queue.popleft()
        #         if queue:
        #             node.next = queue[0]
        #         if node.left and node.right:
        #             new_queue.append(node.left)
        #             new_queue.append(node.right)
        #     queue = new_queue
        # return root

        # # Iterative (BFS with constant space) - O(N)T, O(1)S
        # if not root:
        #     return root
        # node = root
        # while node.left:
        #     next_node = node.left
        #     while node:
        #         node.left.next = node.right
        #         node.right.next = node.next and node.next.left
        #         node = node.next
        #     node = next_node
        # return root

        # Recursive - O(N)T, O(N)S
        if not root:
            return root
        if root.left:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
        self.connect(root.left)
        self.connect(root.right)
        return root

    # LC 117. Populating Next Right Pointers in Each Node II (Medium)
    # https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/
    def connect(self, root: "Node") -> "Node":
        # # Iterative (BFS with queues) - O(N)TS
        # if not root:
        #     return root
        # queue = collections.deque([root])
        # while queue:
        #     new_queue = collections.deque()
        #     while queue:
        #         node = queue.popleft()
        #         if queue:
        #             node.next = queue[0]
        #         if node.left:
        #             new_queue.append(node.left)
        #         if node.right:
        #             new_queue.append(node.right)
        #     queue = new_queue
        # return root

        # # Iterative (BFS with constant space) - O(N)T, O(1)S
        # if not root:
        #     return root
        # curr = root
        # while curr:
        #     tmp = prev = Node(0)
        #     while curr:
        #         if curr.left:
        #             prev.next = curr.left
        #             prev = prev.next
        #         if curr.right:
        #             prev.next = curr.right
        #             prev = prev.next
        #         curr = curr.next
        #     curr = tmp.next
        # return root

        # Recursive - O(N)T, O(N)S
        if not root:
            return root
        if root.left:
            if root.right:
                root.left.next = root.right
            else:
                node = root.next
                while node:
                    if node.left:
                        root.left.next = node.left
                        break
                    elif node.right:
                        root.left.next = node.right
                        break
                    node = node.next
        if root.right:
            node = root.next
            while node:
                if node.left:
                    root.right.next = node.left
                    break
                elif node.right:
                    root.right.next = node.right
                    break
                node = node.next
        self.connect(root.right)
        self.connect(root.left)
        return root

    # LC 118. Pascal's Triangle (Easy)
    # https://leetcode.com/problems/pascals-triangle/
    def generate(self, numRows: int) -> List[List[int]]:
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

    # LC 120. Triangle (Medium)
    # https://leetcode.com/problems/triangle/
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # # DP (my solution) - O(n^2)T, O(n)S
        # dp = triangle[-1]
        # for row in range(len(triangle) - 2, -1, -1):
        #     new_dp = []
        #     for i in range(len(triangle[row])):
        #         new_dp.append(triangle[row][i] + min(dp[i], dp[i + 1]))
        #     dp = new_dp
        # return dp[0]

        # DP (optimized) - O(n^2)T, O(n)S
        dp = triangle[-1]
        for i in range(len(triangle) - 2, -1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
        return dp[0]

        # # Recursion (TLE) - O(2^n)T , O(n)S
        # def helper(row, col):
        #     if row == len(triangle):
        #         return 0
        #     return (
        #         triangle[row][col]
        #         + min(helper(row + 1, col), helper(row + 1, col + 1))
        #     )
        # return helper(0, 0)

    # LC 121. Best Time to Buy and Sell Stock (Easy)
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    def maxProfit(self, prices: List[int]) -> int:
        # # DP - O(n)T, O(n)S
        # if not prices:
        #     return 0
        # min_price = prices[0]
        # max_profit = 0
        # for price in prices:
        #     max_profit = max(max_profit, price - min_price)
        #     min_price = min(min_price, price)
        # return max_profit

        # Greedy - O(n)T, O(1)S
        min_price = float("inf")
        max_profit = 0
        for price in prices:
            if price < min_price:
                min_price = price
            else:
                max_profit = max(max_profit, price - min_price)
        return max_profit

    # LC 122. Best Time to Buy and Sell Stock II (Easy)
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    def maxProfit(self, prices: List[int]) -> int:
        # # DP - O(n)T, O(n)S
        # if not prices:
        #     return 0
        # dp = [0] * len(prices)
        # for i in range(1, len(prices)):
        #     dp[i] = max(dp[i - 1], dp[i - 1] + prices[i] - prices[i - 1])
        # return dp[-1]

        # Greedy - O(n)T, O(1)S
        max_profit = 0
        for i in range(1, len(prices)):
            max_profit += max(prices[i] - prices[i - 1], 0)
        return max_profit

        # # Greedy (one-liner) - O(n)T, O(1)S
        # return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices)))

    # LC 123. Best Time to Buy and Sell Stock III (Hard)
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    def maxProfit(self, prices: List[int]) -> int:
        # # DP - O(n)T, O(n)S
        # if not prices:
        #     return 0
        # n = len(prices)
        # left = [0] * n
        # right = [0] * n
        # min_price = prices[0]
        # for i in range(1, n):
        #     min_price = min(min_price, prices[i])
        #     left[i] = max(left[i - 1], prices[i] - min_price)
        # max_price = prices[-1]
        # for i in range(n - 2, -1, -1):
        #     max_price = max(max_price, prices[i])
        #     right[i] = max(right[i + 1], max_price - prices[i])
        # return max(left[i] + right[i] for i in range(n))

        # Greedy - O(n)T, O(1)S
        buy1 = profit1, float("inf"), 0  # 1st transaction
        buy2, profit2 = float("inf"), 0  # 2nd transaction
        for price in prices:
            buy1 = min(buy1, price)
            profit1 = max(profit1, price - buy1)
            buy2 = min(buy2, price - profit1)
            profit2 = max(profit2, price - buy2)
        return profit2

    # LC 124. Binary Tree Maximum Path Sum (Hard)
    # https://leetcode.com/problems/binary-tree-maximum-path-sum/
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # # Iterative - O(n)T, O(n)S
        # if not root:
        #     return 0
        # max_sum = float("-inf")
        # stack = [(root, False)]
        # while stack:
        #     node, visited = stack.pop()
        #     if not node:
        #         continue
        #     if visited:
        #         left = max(node.left.max_sum, 0) if node.left else 0
        #         right = max(node.right.max_sum, 0) if node.right else 0
        #         max_sum = max(max_sum, node.val + left + right)
        #         node.max_sum = node.val + max(left, right)
        #     else:
        #         stack.append((node, True))
        #         stack.append((node.right, False))
        #         stack.append((node.left, False))
        # return max_sum

        # Recursion - O(n)T, O(n)S
        max_sum = float("-inf")

        def helper(node):
            nonlocal max_sum
            if not node:
                return 0
            left = max(helper(node.left), 0)
            right = max(helper(node.right), 0)
            max_sum = max(max_sum, node.val + left + right)
            return node.val + max(left, right)

        helper(root)
        return max_sum

    # LC 125. Valid Palindrome (Easy)
    # https://leetcode.com/problems/valid-palindrome/
    def isPalindrome(self, s: str) -> bool:
        # # Basic solution - O(n)T, O(n)S
        # # s = [c.lower() for c in s if c.isalnum()]
        # s = "".join(c.lower() for c in s if c.isalnum())  # faster
        # return s == s[::-1]

        # Two pointers - O(n)T, O(1)S
        left = 0
        right = len(s) - 1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True

    # LC 126. Word Ladder II (Hard)
    # https://leetcode.com/problems/word-ladder-ii/
    def findLadders(
        self, beginWord: str, endWord: str, wordList: List[str]
    ) -> List[List[str]]:
        # TODO
        return

    # LC 127. Word Ladder (Hard)
    # https://leetcode.com/problems/word-ladder/
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # TODO
        return

    # LC 128. Longest Consecutive Sequence (Hard)
    # https://leetcode.com/problems/longest-consecutive-sequence/
    def longestConsecutive(self, nums: List[int]) -> int:
        # # Sorting - O(nlogn)T, O(1)S
        # if not nums:
        #     return 0
        # nums.sort()
        # longest = 1
        # curr_len = 1
        # for i in range(1, len(nums)):
        #     if nums[i] != nums[i - 1]:
        #         if nums[i] == nums[i - 1] + 1:
        #             curr_len += 1
        #         else:
        #             longest = max(longest, curr_len)
        #             curr_len = 1
        # return max(longest, curr_len)

        # # Hashset - O(n)T, O(n)S
        # nums = set(nums)
        # longest = 0
        # for num in nums:
        #     if num - 1 not in nums:
        #         curr = num
        #         curr_len = 1
        #         while curr + 1 in nums:
        #             curr += 1
        #             curr_len += 1
        #         longest = max(longest, curr_len)
        # return longest

        # Hashset (optimized) - O(n)T, O(n)S
        nums = set(nums)
        longest = 0
        for num in nums:
            if num - 1 not in nums:
                curr = num
                while curr + 1 in nums:
                    curr += 1
                longest = max(longest, curr - num + 1)
        return longest

    # LC 129. Sum Root to Leaf Numbers (Medium)
    # https://leetcode.com/problems/sum-root-to-leaf-numbers/
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        # # Recursive (my solution) - O(n)T, O(n)S
        # res = 0
        # def helper(node, num):
        #     nonlocal res
        #     if not node:
        #         return
        #     if not node.left and not node.right:
        #         res += int(num)
        #     if node.left:
        #         helper(node.left, num + str(node.left.val))
        #     if node.right:
        #         helper(node.right, num + str(node.right.val))
        # helper(root, str(root.val))
        # return res

        # # Recursive - O(n)T, O(n)S
        # if not root:
        #     return 0
        # self.total = 0
        # def helper(node, curr):
        #     if not node:
        #         return
        #     curr = curr * 10 + node.val
        #     if not node.left and not node.right:
        #         self.total += curr
        #     helper(node.left, curr)
        #     helper(node.right, curr)
        # helper(root, 0)
        # return self.total

        # Iterative - O(n)T, O(n)S
        if not root:
            return 0
        total = 0
        stack = [(root, root.val)]
        while stack:
            node, val = stack.pop()
            if not node.left and not node.right:
                total += val
            if node.left:
                stack.append((node.left, val * 10 + node.left.val))
            if node.right:
                stack.append((node.right, val * 10 + node.right.val))
        return total

    # LC 130. Surrounded Regions (Medium)
    # https://leetcode.com/problems/surrounded-regions/
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # DFS - O(mn)T, O(mn)S
        rows = len(board)
        columns = len(board[0])

        def mark_edge(row, col):
            if (
                row < 0
                or row > rows - 1
                or col < 0
                or col > columns - 1
                or board[row][col] != "O"
            ):
                return
            board[row][col] = "B"
            mark_edge(row - 1, col)
            mark_edge(row + 1, col)
            mark_edge(row, col - 1)
            mark_edge(row, col + 1)

        for row in range(rows):
            for col in range(columns):
                if board[row][col] == "O":
                    if row == 0 or row == rows - 1 or col == 0 or col == columns - 1:
                        mark_edge(row, col)
        for row in range(rows):
            for col in range(columns):
                if board[row][col] != "B":
                    board[row][col] = "X"
                if board[row][col] == "B":
                    board[row][col] = "O"

    # LC 131. Palindrome Partitioning (Medium)
    # https://leetcode.com/problems/palindrome-partitioning/
    def partition(self, s: str) -> List[List[str]]:
        # Backtracking - O(n*2^n)T, O(n)S
        def is_palindrome(s):
            return s == s[::-1]

        def helper(s, path):
            if not s:
                res.append(path)
                return
            for i in range(1, len(s) + 1):
                if is_palindrome(s[:i]):
                    helper(s[i:], path + [s[:i]])

        res = []
        helper(s, [])
        return res

    # LC 132. Palindrome Partitioning II (Hard)
    # https://leetcode.com/problems/palindrome-partitioning-ii/
    def minCut(self, s: str) -> int:
        # # DP - O(n^2)T, O(n^2)S
        # n = len(s)
        # dp = [[False] * n for _ in range(n)]
        # for i in range(n):
        #     dp[i][i] = True
        # for i in range(n - 1, -1, -1):
        #     for j in range(i + 1, n):
        #         dp[i][j] = s[i] == s[j] and (j - i < 3 or dp[i + 1][j - 1])
        # cuts = [float("inf")] * n
        # for i in range(n):
        #     if dp[0][i]:
        #         cuts[i] = 0
        #     else:
        #         for j in range(i):
        #             if dp[j + 1][i]:
        #                 cuts[i] = min(cuts[i], cuts[j] + 1)
        # return cuts[-1]

        # DP (optimized) - O(n^2)T, O(n)S
        n = len(s)
        dp = [float("inf")] * n
        for i in range(n):
            if s[: i + 1] == s[: i + 1][::-1]:
                dp[i] = 0
            else:
                for j in range(i):
                    if s[j + 1 : i + 1] == s[j + 1 : i + 1][::-1]:
                        dp[i] = min(dp[i], dp[j] + 1)
        return dp[-1]

    # LC 133. Clone Graph (Medium)
    # https://leetcode.com/problems/clone-graph/
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        # # DFS (recursive) - O(n)T, O(n)S
        # if not node:
        #     return None
        # visited = {}
        # def helper(node):
        #     if node in visited:
        #         return visited[node]
        #     clone = Node(node.val)
        #     visited[node] = clone
        #     for neighbor in node.neighbors:
        #         clone.neighbors.append(helper(neighbor))
        #     return clone
        # return helper(node)

        # BFS (iterative) - O(n)T, O(n)S
        if not node:
            return None
        visited = {}
        queue = collections.deque([node])
        visited[node] = Node(node.val)
        while queue:
            curr = queue.popleft()
            for neighbor in curr.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                visited[curr].neighbors.append(visited[neighbor])
        return visited[node]

    # LC 134. Gas Station (Medium)
    # https://leetcode.com/problems/gas-station/
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # # Brute force (TLE) - O(n^2)T, O(1)S
        # n = len(gas)
        # for i in range(n):
        #     tank = 0
        #     for j in range(n):
        #         k = (i + j) % n
        #         tank += gas[k] - cost[k]
        #         if tank < 0:
        #             break
        #     if tank >= 0:
        #         return i
        # return -1

        # Greedy - O(n)T, O(1)S
        n = len(gas)
        tank = 0
        total = 0
        start = 0
        for i in range(n):
            tank += gas[i] - cost[i]
            if tank < 0:
                start = i + 1
                total += tank
                tank = 0
        return start if total + tank >= 0 else -1

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

        # XOR one-liner #WOW
        return functools.reduce(operator.xor, nums)

    # LC 137. Single Number II (Medium)
    # https://leetcode.com/problems/single-number-ii/
    def singleNumber(self, nums: List[int]) -> int:
        # # Hashmap - O(n)T, O(n)S
        # d = {}
        # for num in nums:
        #     d[num] = d.get(num, 0) + 1
        # for k, v in d.items():
        #     if v == 1:
        #         return k

        # # One liner - O(n)T, O(n)S
        # return (3 * sum(set(nums)) - sum(nums)) // 2

        # Bit manipulation - O(n)T, O(1)S
        ones = twos = 0
        for num in nums:
            ones = (ones ^ num) & ~twos
            twos = (twos ^ num) & ~ones
        return ones

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

    # LC 139. Word Break (Medium)
    # https://leetcode.com/problems/word-break/
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # # Brute force (TLE) - O(n^2)T, O(n)S
        # def helper(s):
        #     if not s:
        #         return True
        #     for word in wordDict:
        #         if s.startswith(word) and helper(s[len(word) :]):
        #             return True
        #     return False
        # return helper(s)

        # # DP - O(n^2)T, O(n)S
        # n = len(s)
        # dp = [False] * (n + 1)
        # dp[0] = True
        # wordDict = set(wordDict)
        # for i in range(n + 1):
        #     for j in range(i):
        #         if dp[j] and s[j:i] in wordDict:
        #             dp[i] = True
        #             break
        # return dp[-1]

        # BFS - O(n^2)T, O(n)S
        n = len(s)
        wordDict = set(wordDict)
        visited = set()
        queue = collections.deque([0])
        while queue:
            start = queue.popleft()
            if start in visited:
                continue
            for end in range(start + 1, n + 1):
                if s[start:end] in wordDict:
                    queue.append(end)
                    if end == n:
                        return True
            visited.add(start)
        return False

    # LC 140. Word Break II (Hard)
    # https://leetcode.com/problems/word-break-ii/
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # # Brute force - O(n^2)T, O(n)S
        # def helper(s):
        #     if not s:
        #         return [[]]
        #     res = []
        #     for word in wordDict:
        #         if s.startswith(word):
        #             for sub in helper(s[len(word) :]):
        #                 res.append([word] + sub)
        #     return res
        # return [" ".join(words) for words in helper(s)]

        # # DP - O(n^2)T, O(n)S
        # n = len(s)
        # dp = [[] for _ in range(n + 1)]
        # dp[0] = [[]]
        # wordDict = set(wordDict)
        # for i in range(n + 1):
        #     for j in range(i):
        #         if dp[j] and s[j:i] in wordDict:
        #             for sub in dp[j]:
        #                 dp[i].append(sub + [s[j:i]])
        # return [" ".join(words) for words in dp[-1]]

        # DFS (with memoization)- O(n^2)T, O(n)S
        def helper(s):
            if not s:
                return [[]]
            if s in memo:
                return memo[s]
            res = []
            for word in wordDict:
                if s.startswith(word):
                    for sub in helper(s[len(word) :]):
                        res.append([word] + sub)
            memo[s] = res
            return res

        memo = {}
        return [" ".join(words) for words in helper(s)]

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

    # LC 142. Linked List Cycle II (Medium)
    # https://leetcode.com/problems/linked-list-cycle-ii/
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # # Hash table - O(n)TS
        # visited_nodes = set()
        # curr = head
        # while curr:
        #     if curr in visited_nodes:
        #         return curr
        #     visited_nodes.add(curr)
        #     curr = curr.next
        # return None

        # Floyd's Tortoise and Hare (Cycle Detection with double cursors fast and slow) - O(n)T, O(1)S
        fast = head
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            # Cycle detected
            if fast == slow:
                # Find the start of the cycle
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None

    # LC 143. Reorder List (Medium)
    # https://leetcode.com/problems/reorder-list/
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # # Hash table - O(n)T, O(n)S
        # if not head:
        #     return
        # nodes = []
        # curr = head
        # while curr:
        #     nodes.append(curr)
        #     curr = curr.next
        # n = len(nodes)
        # for i in range(n // 2):
        #     nodes[i].next = nodes[n - i - 1]
        #     nodes[n - i - 1].next = nodes[i + 1]
        # nodes[n // 2].next = None

        # Single pass - O(n)T, O(1)S
        if not head:
            return
        # Find the middle of the list
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # Reverse the second half of the list
        prev = None
        curr = slow
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        # Merge the two lists
        first = head
        second = prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next

    # LC 144. Binary Tree Preorder Traversal (Easy)
    # https://leetcode.com/problems/binary-tree-preorder-traversal/
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # Recursive - O(n)T, O(n)S
        if not root:
            return []
        return (
            [root.val]
            + self.preorderTraversal(root.left)
            + self.preorderTraversal(root.right)
        )

        # # Iterative - O(n)T, O(n)S
        # if not root:
        #     return []
        # res = []
        # stack = [root]
        # while stack:
        #     curr = stack.pop()
        #     res.append(curr.val)
        #     if curr.right:
        #         stack.append(curr.right)
        #     if curr.left:
        #         stack.append(curr.left)
        # return res

    # LC 145. Binary Tree Postorder Traversal (Easy)
    # https://leetcode.com/problems/binary-tree-postorder-traversal/
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # Recursive - O(n)T, O(n)S
        if not root:
            return []
        return (
            self.postorderTraversal(root.left)
            + self.postorderTraversal(root.right)
            + [root.val]
        )

        # # Iterative - O(n)T, O(n)S
        # if not root:
        #     return []
        # res = []
        # stack = [root]
        # while stack:
        #     node = stack.pop()
        #     res.insert(0, node.val)
        #     if node.left:
        #         stack.append(node.left)
        #     if node.right:
        #         stack.append(node.right)
        # return res

    # LC 146. LRU Cache (Medium)
    # https://leetcode.com/problems/lru-cache/
    class LRUCache:
        def __init__(self, capacity: int):
            self.capacity = capacity
            self.cache = {}
            self.queue = collections.deque()

        def get(self, key: int) -> int:
            if key not in self.cache:
                return -1
            self.queue.remove(key)
            self.queue.append(key)
            return self.cache[key]

        def put(self, key: int, value: int) -> None:
            if key in self.cache:
                self.queue.remove(key)
            elif len(self.cache) == self.capacity:
                del self.cache[self.queue.popleft()]
            self.cache[key] = value
            self.queue.append(key)

    # LC 147. Insertion Sort List (Medium)
    # https://leetcode.com/problems/insertion-sort-list/
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Brute force - O(n^2)T, O(1)S
        dummy = ListNode()
        curr = head
        while curr:
            prev = dummy
            while prev.next and prev.next.val < curr.val:
                prev = prev.next
            next_node = curr.next
            curr.next = prev.next
            prev.next = curr
            curr = next_node
        return dummy.next

    # LC 148. Sort List (Medium)
    # https://leetcode.com/problems/sort-list/
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # # Merge sort - O(nlogn)T, O(logn)S
        # def merge(left, right):
        #     dummy = ListNode()
        #     curr = dummy
        #     while left and right:
        #         if left.val < right.val:
        #             curr.next = left
        #             left = left.next
        #         else:
        #             curr.next = right
        #             right = right.next
        #         curr = curr.next
        #     curr.next = left or right
        #     return dummy.next
        # if not head or not head.next:
        #     return head
        # slow = fast = head
        # prev = None
        # while fast and fast.next:
        #     prev = slow
        #     slow = slow.next
        #     fast = fast.next.next
        # prev.next = None
        # left = self.sortList(head)
        # right = self.sortList(slow)
        # return self.merge(left, right)

        # Constant space merge sort - O(nlogn)T, O(1)S
        def split(head, n):
            for i in range(n - 1):
                if not head:
                    break
                head = head.next
            if not head:
                return None
            second = head.next
            head.next = None
            return second

        def merge(left, right, head):
            curr = head
            while left and right:
                if left.val < right.val:
                    curr.next = left
                    left = left.next
                else:
                    curr.next = right
                    right = right.next
                curr = curr.next
            curr.next = left or right
            while curr.next:
                curr = curr.next
            return curr

        if not head or not head.next:
            return head
        curr = head
        length = 0
        while curr:
            length += 1
            curr = curr.next
        dummy = ListNode()
        dummy.next = head
        step = 1
        while step < length:
            prev, curr = dummy, dummy.next
            while curr:
                left = curr
                right = split(left, step)
                curr = split(right, step)
                prev = merge(left, right, prev)
            step *= 2
        return dummy.next

    # LC 149. Max Points on a Line (Hard)
    # https://leetcode.com/problems/max-points-on-a-line/
    def maxPoints(self, points: List[List[int]]) -> int:
        # # Brute force - O(n^3)T, O(n)S
        # # TODO

        # GCD - O(n^2)T, O(n)S
        if len(points) < 3:
            return len(points)

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        res = 0
        for i in range(len(points)):
            d = {}
            overlap = 0
            curr_max = 0
            for j in range(i + 1, len(points)):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if dx == 0 and dy == 0:
                    overlap += 1
                    continue
                g = gcd(dx, dy)
                dx //= g
                dy //= g
                if (dx, dy) in d:
                    d[(dx, dy)] += 1
                else:
                    d[(dx, dy)] = 1
                curr_max = max(curr_max, d[(dx, dy)])
            res = max(res, curr_max + overlap + 1)
        return res

    # LC 150. Evaluate Reverse Polish Notation (Medium)
    # https://leetcode.com/problems/evaluate-reverse-polish-notation/
    def evalRPN(self, tokens: List[str]) -> int:
        # Stack - O(n)T, O(n)S
        stack = []
        for token in tokens:
            if token in "+-*/":
                b = stack.pop()
                a = stack.pop()
                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                elif token == "/":
                    stack.append(int(a / b))
            else:
                stack.append(int(token))
        return stack.pop()

    # LC 151. Reverse Words in a String (Medium)
    # https://leetcode.com/problems/reverse-words-in-a-string/
    def reverseWords(self, s: str) -> str:
        # # One-liner - O(n)T, O(n)S
        # return " ".join(s.split()[::-1])

        # Two pointers - O(n)T, O(n)S
        s = s.strip()
        res = []
        i = j = len(s) - 1
        while i >= 0:
            while i >= 0 and s[i] != " ":
                i -= 1
            res.append(s[i + 1 : j + 1])
            while s[i] == " ":
                i -= 1
            j = i
        return " ".join(res)

    # LC 152. Maximum Product Subarray (Medium)
    # https://leetcode.com/problems/maximum-product-subarray/
    def maxProduct(self, nums: List[int]) -> int:
        # min and max DP - O(n)T, O(1)S
        curr_max = curr_min = res = nums[0]
        for num in nums[1:]:
            temp = curr_max
            curr_max = max(num, curr_max * num, curr_min * num)
            curr_min = min(num, temp * num, curr_min * num)
            res = max(res, curr_max)
        return res

    # LC 153. Find Minimum in Rotated Sorted Array (Medium)
    # https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
    def findMin(self, nums: List[int]) -> int:
        # # One-liner - O(n)T, O(1)S
        # return min(nums)

        # Binary search - O(log(n))T, O(1)S
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2  # avoid overflow
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        return nums[l]

    # LC 154. Find Minimum in Rotated Sorted Array II (Hard)
    # https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/
    def findMin(self, nums: List[int]) -> int:
        # # One-liner - O(n)T, O(1)S
        # return min(nums)

        # Binary search - O(log(n))T, O(1)S
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2  # avoid overflow
            # If mid is greater than right, then the minimum is in the right half
            if nums[mid] > nums[r]:
                l = mid + 1
            # If mid is less than right, then the minimum is in the left half
            elif nums[mid] < nums[r]:
                r = mid
            # If mid is equal to right, then we don't know where the minimum is
            # But we can safely remove the right element
            else:
                r -= 1
        return nums[l]

    # LC 155. Min Stack (Easy)
    # https://leetcode.com/problems/min-stack/
    class MinStack:
        def __init__(self):
            """
            initialize your data structure here.
            """
            self.stack = []
            self.min_stack = [float("inf")]

        def push(self, val: int) -> None:
            self.stack.append(val)
            self.min_stack.append(min(self.min_stack[-1], val))

        def pop(self) -> None:
            self.stack.pop()
            self.min_stack.pop()

        def top(self) -> int:
            return self.stack[-1]

        def getMin(self) -> int:
            return self.min_stack[-1]

    # LC 156. Binary Tree Upside Down (Medium)
    # https://leetcode.com/problems/binary-tree-upside-down/
    def upsideDownBinaryTree(self, root: TreeNode) -> TreeNode:
        # In-place replacement - O(n)T, O(1)S
        curr, parent, parent_right = root, None, None
        while curr:
            left = curr.left
            curr.left = parent_right
            parent_right = curr.right
            curr.right = parent
            parent = curr
            curr = left
        return parent

    # LC 157. Read N Characters Given Read4 (Easy)
    # https://leetcode.com/problems/read-n-characters-given-read4/
    def read(self, buf: List[str], n: int) -> int:
        # Read 4 until reaching n or end of file - O(n)T, O(1)S
        i, size = 0, 4
        buf4 = [""] * 4
        while size == 4:
            size = read4(buf4)
            for j in range(size):
                buf[i] = buf4[j]
                i += 1
                if i == n:
                    break
        return i

    # LC 158. Read N Characters Given Read4 II - Call multiple times (Hard)
    # https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/
    def read(self, buf: List[str], n: int) -> int:
        # Basically the same as LC 157 but with some static class variables
        # that we reset in the init function of the Solution class
        # O(n)T, O(1)S

        # def __init__(self):
        #     self.buf4 = [""] * 4
        #     self.i = 0
        #     self.size = 0

        j = 0
        while j < n:
            if self.i == self.size:
                self.size = read4(self.buf4)
                self.i = 0
                if self.size == 0:
                    break
            while j < n and self.i < self.size:
                buf[j] = self.buf4[self.i]
                self.i += 1
                j += 1
        return j

    # LC 159. Longest Substring with At Most Two Distinct Characters (Medium)
    # https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        # Hashmap with max 3 characters as keys - O(n)T, O(1)S
        d = collections.defaultdict(int)
        res = i = 0
        for c in s:
            d[c] += 1
            while len(d) > 2:
                d[s[i]] -= 1
                if d[s[i]] == 0:
                    del d[s[i]]
                i += 1
            res = max(res, i - res)
        return res

    # LC 160. Intersection of Two Linked Lists (Easy)
    # https://leetcode.com/problems/intersection-of-two-linked-lists/
    def getIntersectionNode(
        self, headA: ListNode, headB: ListNode
    ) -> Optional[ListNode]:
        # # Naive approach: save all nodes from A then iterate through B - O(m+n)T, O(m)S
        # # Could be optimized with a hashset
        # a, nodesA = headA, []
        # while a:
        #     nodesA.append(a)
        #     a = a.next
        # b = headB
        # while b:
        #     if b in nodesA:
        #         return b
        #     b = b.next
        # return None

        # # Get both lengths and start at equal distance - O(m+n)T, O(1)S
        # a, lenA = headA, 0
        # while a:
        #     lenA += 1
        #     a = a.next
        # b, lenB = headB, 0
        # while b:
        #     lenB += 1
        #     b = b.next
        # while lenA > lenB:
        #     headA = headA.next
        #     lenA -= 1
        # while lenB > lenA:
        #     headB = headB.next
        #     lenB -= 1
        # while headA != headB:
        #     headA = headA.next
        #     headB = headB.next
        # return headA

        # Two pointers going though both linked lists - O(m+n)T, O(1)S
        a, b = headA, headB
        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a

    # LC 161. One Edit Distance (Medium)
    # https://leetcode.com/problems/one-edit-distance/
    def isOneEditDistance(self, s, t):
        # # Classic approach - O(m+n)T, O(1)S
        # m, n = len(s), len(t)
        # if abs(m - n) > 1:
        #     return False
        # if m == n:
        #     # There should be only one char diff
        #     return sum(c != t[i] for i, c in enumerate(s)) == 1
        # if m > n:
        #     # Swap to set s as the smaller word
        #     # so that deletion and insertion are the "same" operation
        #     # but we can also call self.getEditDistance(t, s) instead
        #     s, t = t, s
        #     m, n = n, m
        # for i in range(m):
        #     if s[i] != t[i]:
        #         # Test deletion/insertion
        #         return s[i:] == t[i + 1 :]
        # return False  # Same word

        # Classic approach (cleaner) - O(m+n)T, O(1)S
        m, n = len(s), len(t)
        if m > n:
            # Simplify code by arbitrarily setting s as the smaller word
            return self.isOneEditDistance(s, t)
        if n - m > 1:
            return False
        for i, c in enumerate(s):
            if c != t[i]:
                # if m == n:
                #     return s[i + 1 :] == t[i + 1 :]
                # else:
                #     return s[i:] == t[i + 1 :]
                return s[i + int(m == n) :] == t[i + 1 :]
        return False  # Same word

    # LC 162. Find Peak Element (Medium)
    # https://leetcode.com/problems/find-peak-element/
    def findPeakElement(self, nums: List[int]) -> int:
        # # Linear search - O(n)T, O(1)S
        # return nums.index(max(nums))

        # Binary search - O(log(n))T, O(1)S
        # this works because of the constraints:
        # 1) array borders are -inf
        # 2) no adjacent duplicated numbers
        # 3) we only need to return one peak (local maximum)
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] < nums[mid + 1]:
                l = mid + 1
            else:
                r = mid
        return l

    # LC 163. Missing Ranges (Medium)
    # https://leetcode.com/problems/missing-ranges/
    def findMissingRanges(self, nums, lower, upper):
        # Naive approach - O(n)T, O(1)S
        res = []
        nums = [lower - 1] + nums + [upper + 1]
        for i in range(1, len(nums)):
            lower, upper = nums[i - 1], nums[i]
            diff = upper - lower
            if diff == 0:
                continue
            if diff == 1:
                res.append(str(upper - 1))
            else:
                res.append(str(lower + 1) + "->" + str(upper - 1))
        return res

    # LC 164. Maximum Gap (Medium)
    # https://leetcode.com/problems/maximum-gap/
    def maximumGap(self, nums: List[int]) -> int:
        # # Sort and search - O(nlog(n))T, O(1)S
        # if len(nums) < 2:
        #     return 0
        # nums.sort()
        # res = 0
        # for i in range(1, len(nums)):
        #     res = max(res, nums[i] - nums[i - 1])
        # return res

        # Linear solution with bucket sort - O(n)T, O(n)S
        """
        First, find the maximum and minimum values of the array,
        and then determine the capacity of each bucket, which is (maximum-minimum) / number + 1.
        When determining the number of buckets, that is (maximum-minimum) / the capacity of the bucket + 1,

        Then you need to find the local maximum and minimum in each bucket,
        and the two numbers with the maximum distance will not be in the same bucket,
        but the distance between the minimum value of one bucket and the maximum value of another bucket.

        This is because all numbers should be evenly distributed to each bucket as much as possible,
        rather than crowded in one bucket, which ensures that the maximum and minimum values will not be in the same bucket
        """
        n = len(nums)
        if n < 2:
            return 0

        # find the maximum and minimum values
        maxi, mini = max(nums), min(nums)

        # compute the capacity (size) of each bucket
        b_size = max(1, (maxi - mini) // (n - 1))

        # compute the number of buckets
        n_buckets = (maxi - mini) // b_size + 1

        # initialize the buckets with maximum and minimum values
        buckets = [[float("inf"), float("-inf")] for _ in range(n_buckets)]
        for num in nums:
            i = (num - mini) // b_size
            buckets[i][0] = min(buckets[i][0], num)
            buckets[i][1] = max(buckets[i][1], num)

        # compute the maximum difference
        max_diff = 0
        prev_max = buckets[0][1]
        for i in range(1, n_buckets):
            if buckets[i][0] == float("inf"):
                continue
            max_diff = max(max_diff, buckets[i][0] - prev_max)
            prev_max = buckets[i][1]

        return max_diff

    # LC 165. Compare Version Numbers (Medium)
    # https://leetcode.com/problems/compare-version-numbers/
    def compareVersion(self, version1: str, version2: str) -> int:
        # # Trim and compare - O(m+n)T, O(m+n)S
        # def split_and_trim(version):
        #     revisions = version.split(".")
        #     while revisions and int(revisions[-1]) == 0:
        #         revisions.pop()
        #     return revisions
        # v1, v2 = split_and_trim(version1), split_and_trim(version2)
        # for i in range(min(len(v1), len(v2))):
        #     if int(v1[i]) < int(v2[i]):
        #         return -1
        #     if int(v1[i]) > int(v2[i]):
        #         return 1
        # res = 0
        # if len(v1) < len(v2):
        #     res = -1
        # elif len(v1) > len(v2):
        #     res = 1
        # return res

        # Linear comparison - O(m+n)T, O(1)S
        len1, len2 = len(version1), len(version2)
        i, j = 0, 0
        while i < len1 or j < len2:
            v1, v2 = 0, 0
            while i < len1 and version1[i] != ".":
                v1 = v1 * 10 + int(version1[i])
                i += 1
            while j < len2 and version2[j] != ".":
                v2 = v2 * 10 + int(version2[j])
                j += 1
            if v1 != v2:
                return 1 if v1 > v2 else -1
            i += 1
            j += 1
        return 0

    # LC 166. Fraction to Recurring Decimal (Medium)
    # https://leetcode.com/problems/fraction-to-recurring-decimal/
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        # Optimized integer & fractional division - O(logn)T, O(1)S
        # See: https://leetcode.com/problems/fraction-to-recurring-decimal/solutions/3208837/166-solution-with-step-by-step-explanation/
        """
        This solution also handles the edge cases of a zero numerator and a zero denominator,
        and also checks for the negative sign at the beginning.
        It then calculates the integer part of the result by doing an integer division of the numerator by the denominator,
        and checks if there is a fractional part by checking if the remainder of this division is zero. If there is a fractional part,
        it adds a decimal point to the result.

        The main optimization in this solution is the use of a dictionary to store the position of each remainder in the result.
        This way, we can easily check if a remainder has already appeared in the result, and if it has, we know that we have found a repeating part.
        We can then insert the opening and closing parentheses at the appropriate positions in the result.
        """
        # Handle edge cases
        if numerator == 0:
            return "0"
        if denominator == 0:
            return ""
        # Initialize result and check for negative sign
        result = ""
        if (numerator < 0) ^ (denominator < 0):
            result += "-"
        numerator, denominator = abs(numerator), abs(denominator)
        # Integer part of the result
        result += str(numerator // denominator)
        # Check if there is a fractional part
        if numerator % denominator == 0:
            return result
        result += "."
        # Use a dictionary to store the position of each remainder
        remainder_dict = {}
        remainder = numerator % denominator
        # Keep adding the remainder to the result until it repeats or the remainder becomes 0
        while remainder != 0 and remainder not in remainder_dict:
            remainder_dict[remainder] = len(result)
            remainder *= 10
            result += str(remainder // denominator)
            remainder %= denominator
        # Check if there is a repeating part
        if remainder in remainder_dict:
            result = (
                result[: remainder_dict[remainder]]
                + "("
                + result[remainder_dict[remainder] :]
                + ")"
            )
        return result

    # LC 167. Two Sum II - Input Array Is Sorted (Medium)
    # https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # Two pointers - O(n)T, O(1)S
        l, r = 0, len(numbers) - 1
        while l < r:
            _sum = numbers[l] + numbers[r]
            if _sum < target:
                l += 1
            elif _sum > target:
                r -= 1
            else:
                return [l + 1, r + 1]

    # LC 168. Excel Sheet Column Title (Easy)
    # https://leetcode.com/problems/excel-sheet-column-title/
    def convertToTitle(self, columnNumber: int) -> str:
        # # Iterative solution - O(log26(n))T, O(log26(n))S
        # res = []
        # while columnNumber > 0:
        #     columnNumber, remainder = divmod(columnNumber - 1, 26)
        #     res.append(chr(65 + remainder))  # 65 == ord("A")
        # return "".join(res[::-1])

        # Recursive solution - O(log26(n))T, O(log26(n))S
        if not columnNumber:
            return ""
        columnNumber, remainder = divmod(columnNumber - 1, 26)
        return self.convertToTitle(columnNumber) + chr(65 + remainder)  # 65 == ord("A")

    # LC 169. Majority Element (Medium)
    def majorityElement(self, nums: List[int]) -> int:
        # # Counter hashmap - O(n)T, O(n)S
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

        # # One liner - O(n)T, O(n)S
        # return collections.Counter(nums).most_common(1)[0][0]

        # Boyer-Moore Voting Algorithm #WOW - O(n)T, O(1)S
        count = 0
        candidate = None
        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if candidate == num else -1
        return candidate

    # LC 170. Two Sum III - Data structure design
    # https://leetcode.com/problems/two-sum-iii-data-structure-design/
    class TwoSum:
        def __init__(self):
            self.counter = collections.defaultdict(int)

        def add(self, number: int) -> None:
            self.counter[number] += 1

        def find(self, value: int) -> bool:
            for number in self.counter:
                diff = value - number
                if diff in self.counter:
                    # if number != diff or self.counter[number] > 1
                    if self.counter[diff] >= 1 + (number == diff):
                        return True
            return False
