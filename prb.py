# from itertools import permutations
# from collections import Counter
# import operator
import re
from typing import List

# <
# >


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.val) + ", " + str(self.left) + ", " + str(self.right)


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
        # if len(nums) == 1:
        #     return True
        # increasing = nums[0] < nums[-1]
        # for i in range(0, len(nums) - 1):
        #     if increasing and nums[i] > nums[i + 1] or not increasing and nums[i] < nums[i + 1]:
        #         return False
        # return True
        direction = -1 if nums[0] <= nums[-1] else 1
        for i in range(0, len(nums) - 1):
            if (nums[i] - nums[i + 1]) * direction < 0:
                return False
        return True

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
    def isSymmetric(self, root):
        # Recursive
        def sym(l, r):
            if l and r:
                return l.val == r.val and sym(l.left, r.right) and sym(l.right, r.left)
            else:
                return l == r

        return sym(root.left, root.right)

        # # Iterative
        # stack = [root.left, root.right]
        # while len(stack) > 0:
        #     r = stack.pop()
        #     l = stack.pop()
        #     if l and r:
        #         if l.val != r.val:
        #             return False
        #     else:
        #         if l is None and r is None:
        #             continue
        #         return False
        #     stack.append(l.left)
        #     stack.append(r.right)
        #     stack.append(r.left)
        #     stack.append(l.right)
        # return True

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
    def maxDepth(self, root: "Node") -> int:
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
        return int((((1 + 5 ** 0.5) / 2) ** n + 1) / 5 ** 0.5)

    # LC 965. Univalued Binary Tree
    # https://leetcode.com/problems/univalued-binary-tree/
    def isUnivalTree(self, root: TreeNode) -> bool:
        if not root:
            return True
        if root.left:
            if root.val != root.left.val:
                return False
        if root.right:
            if root.val != root.right.val:
                return False
        return self.isUnivalTree(root.left) and self.isUnivalTree(root.right)

    # LC 1222. Queens That Can Attack the King
    # https://leetcode.com/contest/weekly-contest-158/problems/queens-that-can-attack-the-king/
    def queensAttacktheKing(
        self, queens: List[List[int]], king: List[int]
    ) -> List[List[int]]:
        res = []
        queens = {(i, j) for i, j in queens}
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in range(1, 8):
                    x, y = king[0] + i * k, king[1] + j * k
                    if (x, y) in queens:
                        res.append([x, y])
                        break
        return res

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

    # LC 1221. Split a String in Balanced Strings
    # https://leetcode.com/contest/weekly-contest-158/problems/split-a-string-in-balanced-strings/
    def balancedStringSplit(self, s: str) -> int:
        total_count = 0
        l = 0
        for v in s:
            if v == "L":
                l += 1
            else:
                l -= 1
            if l == 0:
                total_count += 1
        return total_count

    # LC 922. Sort Array By Parity II
    # https://leetcode.com/problems/sort-array-by-parity-ii/
    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        # e = len(A)-1  # index of most left odd number
        # for i in range(len(A)-1)[::-2]:
        #     while (A[i] % 2): # while A[i] is an odd number
        #         A[i], A[e] = A[e], A[i]
        #         e -= 2
        #         if e == -1:
        #             # We correctly placed every odd numbers, by induction the even numbers are correctly placed as well
        #             # So no need to continue the for loop
        #             break
        # return A
        j = 1
        for i in range(0, len(A), 2):
            if A[i] % 2:
                while A[j] % 2:
                    j += 2  # Little improvement can be done here by breaking the fo loop if j is greater than len(A)
                A[i], A[j] = A[j], A[i]
        return A

    # LC 1160. Find Words That Can Be Formed by Characters
    # https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/
    def countCharacters(self, words: List[str], chars: str) -> int:
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
        res = 0
        for word in words:
            valid = True
            for i in word:
                if word.count(i) > chars.count(i):
                    valid = False
                    break
            if valid:
                res += len(word)
        return res

    # LC 1051. Height Checker
    # https://leetcode.com/problems/height-checker/
    def heightChecker(self, heights: List[int]) -> int:
        heights_sort = sorted(heights)
        count = 0
        for i in range(len(heights)):
            if heights[i] != heights_sort[i]:
                count += 1
        return count

    # LC 929. Unique Email Addresses
    # https://leetcode.com/problems/unique-email-addresses/
    def numUniqueEmails(self, emails: List[str]) -> int:
        email_set = set()
        for email in emails:
            [name, domain] = email.split("@")
            name = name.split("+")[0].replace(
                ".", ""
            )  # ignore what's after '+' and remove '.'
            email_set.add(name + "@" + domain)
        return len(email_set)

    # LC 590. N-ary Tree Postorder Traversal
    # https://leetcode.com/problems/n-ary-tree-postorder-traversal/
    def postorder(self, root: "Node") -> List[int]:
        if not root:
            return []
        res = []
        for s in root.children:
            res += self.postorder(s)
        res.append(root.val)
        return res

    # LC 589. N-ary Tree Preorder Traversal
    # https://leetcode.com/problems/n-ary-tree-preorder-traversal/
    def preorder(self, root: "Node") -> List[int]:
        if not root:
            return []
        res = []
        res.append(root.val)
        for s in root.children:
            res += self.preorder(s)
        return res

    # LC 700. Search in a Binary Search Tree
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

    # LC 1028. Recover a Tree From Preorder Traversal
    # https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/
    def recoverFromPreorder(self, S: str) -> TreeNode:
        hash_array = [[] for i in range(1000)]  # index = depth

        splitted = re.findall(r"\d+|\-+", S)
        print(splitted)

        # Setting the depth "0"
        root = TreeNode(splitted[0])
        hash_array[0].append(root)

        depth = 1
        for i in range(1, len(splitted)):
            val = splitted[i]
            if val.isdigit():
                node = TreeNode(val)
                parent = hash_array[depth - 1][-1]  # last one
                if not parent.left:
                    parent.left = node
                elif not parent.right:
                    parent.right = node
                else:
                    # shouldn't happend
                    print("WTF")

                hash_array[depth].append(node)
            else:
                depth = len(val)

        return root

    # LC 944. Delete Columns to Make Sorted
    # https://leetcode.com/problems/delete-columns-to-make-sorted/
    def minDeletionSize(self, A: List[str]) -> int:
        return sum([sorted(a) != list(a) for a in zip(*A)])

    # LC 942. DI String Match
    # https://leetcode.com/problems/di-string-match/
    def diStringMatch(self, S: str) -> List[int]:
        left, right = 0, len(S)
        arr = []
        for char in S:
            if char == "I":
                arr.append(left)
                left += 1
            else:
                arr.append(right)
                right -= 1
        return arr

    # LC 561. Array Partition I
    def arrayPairSum(self, nums: List[int]) -> int:
        return sum(sorted(nums)[::2])

    # LC 852. Peak Index in a Mountain Array
    def peakIndexInMountainArray(self, A: List[int]) -> int:
        for i in range(len(A)):
            if A[i + 1] < A[i]:
                return i

        # One liner (takes more time)
        # index, _value = max(enumerate(A), key=operator.itemgetter(1))
        # return index

    # LC 1217 https://leetcode.com/contest/weekly-contest-157/problems/play-with-chips/
    def minCostToMoveChips(self, chips: List[int]) -> int:
        even_numbers = 0
        odd_numbers = 0
        for number in chips:
            if number % 2 == 0:
                even_numbers += 1
            else:
                odd_numbers += 1

        if even_numbers == 0 or odd_numbers == 0:
            return 0

        return min(even_numbers, odd_numbers)

    # LC 5214 https://leetcode.com/contest/weekly-contest-157/problems/longest-arithmetic-subsequence-of-given-difference/
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        # Wrong solution :
        a, n, d = arr, len(arr), difference

        # key = starting element of an AP,
        # value = length of AP
        m = dict()

        # since the length of longest AP is at least
        # one i.e. the number itself.
        maxt = 1

        # if element a[i]'s starting element(i.e., a[i]-i*d)
        # is not in map then its value is 1 else there already
        # exists a starting element of an AP of which a[i]
        # can be a part.
        for i in range(n):
            if (a[i] - d) in m:
                m[a[i] - d] += 1
            else:
                m[a[i] - d] = 1

        # In this it variable will be
        # storing key value of dictionary.
        for it in m:
            if m[it] > maxt:
                # calculating the length of longest AP.
                maxt = m[it]

        return maxt

    # 461. Hamming Distance
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

    # LC 728. Self Dividing Numbers
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        res = []
        for i in range(left, right + 1):
            for char in str(i):
                if int(char) == 0 or i % int(char) != 0:
                    break
            else:
                res.append(i)
        return res

        # One liner (less elegant)
        # return [x for x in range(left, right+1) if '0' not in str(x) and all([x % int(digit)==0 for digit in str(x)])]

    # LC 977. Squares of a Sorted Array
    # Time : O(N) with Two pointers or O(N log(N)) with sort
    # Space : O(N)
    def sortedSquares(self, A: List[int]) -> List[int]:
        answer = [0] * len(A)
        l, r = 0, len(A) - 1
        while l <= r:
            left, right = abs(A[l]), abs(A[r])
            if left > right:
                answer[r - l] = left * left
                l += 1
            else:
                answer[r - l] = right * right
                r -= 1
        return answer

        # Or one liner #EASY
        # return sorted([a * a for a in A])

    # LC 657. Robot Return to Origin
    def judgeCircle(self, moves: str) -> bool:
        dic = {"D": 0, "U": 0, "L": 0, "R": 0}
        for i in moves:
            dic[i] += 1
            return dic["U"] == dic["D"] and dic["L"] == dic["R"]

        # Or one liner #EASY
        # return moves.count('U') == moves.count('D') and moves.count('L') == moves.count('R')

    # LC 961. N-Repeated Element in Size 2N Array
    def repeatedNTimes(self, A: List[int]) -> int:
        N = len(A) / 2
        keys = set()

        for i in A:
            if i in keys:
                return i
            keys.add(i)

    ########## LC 905. Sort Array By Parity

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

    # LC 832. Flipping an Image
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        # I, J, odd_size = len(A), len(A[0]), len(A[0]) % 2 == 1

        # for i in range(I):
        #     for j in range(int(J/2)):
        #         A[i][j], A[i][-j] = 1 - A[i][-j], 1 - A[i][j]
        #     if odd_size:
        #         A[i][J//2] = 1 - A[i][J/2]

        for row in A:
            for i in range((len(row) + 1) // 2):
                row[i], row[~i] = 1 - row[~i], 1 - row[i]  # WOW tilde operator
        return A

        # One liner
        # return [[1-i for i in row[::-1]] for row in A] # WOW
        # return [[1 ^ i for i in row[::-1]] for row in A]

    # LC 1207
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        occurence_values = Counter(arr).values()
        return len(occurence_values) == len(set(occurence_values))  # WOW

    # LC 1079
    def numTilePossibilities(self, tiles: str) -> int:
        count = 0
        for i in range(1, len(tiles) + 1):
            p = permutations(list(tiles), i)
            count += len(set(p))
        return count

    # LC 1021
    def removeOuterParentheses(self, S: str) -> str:
        cnt, res = 0, []
        for c in S:
            if c == ")":
                cnt -= 1
            if cnt != 0:
                res.append(c)
            if c == "(":
                cnt += 1
        return "".join(res)

    # LC
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        return

    # LC 88. Merge Sorted Array
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

    # LC 9. Palindrome Number
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
    def strStr(self, haystack: str, needle: str) -> int:
        # # One-liner
        # return haystack.find(needle)

        # # Without built-in function
        # # Double pointers
        i, j = 0, len(needle)
        while j <= len(haystack):
            if haystack[i:j] == needle:
                return i
            i += 1
            j += 1
        return -1

    # LC 20. Valid Parentheses (Easy)
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
