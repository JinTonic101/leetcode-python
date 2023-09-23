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


class Solution:  # LC 101. Symmetric Tree (Easy)
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
