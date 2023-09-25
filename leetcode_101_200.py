import bisect
import collections
import functools
import heapq
import itertools
import math
import re
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
