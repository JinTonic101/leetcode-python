import unittest

from leetcode import Solution as SolutionLeetCode
from prb import Solution, TreeNode


class Test(unittest.TestCase):
    def test_prb(self):
        solution = Solution()
        # self.assertEqual(solution.(), )

        # self.assertEqual(solution.missingNumber([3,0,1]), 2)
        # self.assertEqual(solution.missingNumber([9,6,4,2,3,5,7,0,1]), 8)

        # self.assertEqual(solution.isMonotonic([1,2,2,3]), True)
        # self.assertEqual(solution.isMonotonic([6,5,4,4]), True)
        # self.assertEqual(solution.isMonotonic([1,3,2]), False)
        # self.assertEqual(solution.isMonotonic([1,2,4,5]), True)
        # self.assertEqual(solution.isMonotonic([1,1,1]), True)
        # self.assertEqual(solution.isMonotonic([2,2,2,1,2]), False)
        # self.assertEqual(solution.isMonotonic([1,1,1,2,1]), False)

        # self.assertEqual(solution.findDisappearedNumbers([4,3,2,7,8,2,3,1]), [5,6])
        # self.assertEqual(solution.findDisappearedNumbers([1, 1]), [2])
        # self.assertEqual(solution.findDisappearedNumbers([1, 1]), [2])

        # self.assertEqual(solution.majorityElement([3,2,3]), 3)
        # self.assertEqual(solution.majorityElement([2,2,1,1,1,2,2]), 2)
        # self.assertEqual(solution.majorityElement([2]), 2)

        # self.assertEqual(solution.singleNumber([2,2,1]), 1)
        # self.assertEqual(solution.singleNumber([4,1,2,1,2]), 4)

        # self.assertEqual(solution.commonChars(["bella","label","roller"]), ['l', 'l', 'e'])
        # self.assertEqual(solution.commonChars(["cool","lock","cook"]), ["c","o"])

        # self.assertEqual(solution.reverseWords("Let's take LeetCode contest"), "s'teL ekat edoCteeL tsetnoc")

        # self.assertEqual(solution.relativeSortArray(arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]), [2,2,2,1,4,3,3,9,6,7,19])
        # self.assertEqual(solution.relativeSortArray([2,21,43,38,0,42,33,7,24,13,12,27,12,24,5,23,29,48,30,31], [2,42,38,0,43,21]), [2,42,38,0,43,21,5,7,12,12,13,23,24,24,27,29,30,31,33,48])

        # self.assertEqual(solution.minimumAbsDifference([4,2,1,3]), [[1,2],[2,3],[3,4]])
        # self.assertEqual(solution.minimumAbsDifference([1,3,6,10,15]), [[1,3]])
        # self.assertEqual(solution.minimumAbsDifference([3,8,-10,23,19,-4,-14,27]), [[-14,-10],[19,23],[23,27]])

        # self.assertEqual(solution.projectionArea([[2]]), 5)
        # self.assertEqual(solution.projectionArea([[1,2],[3,4]]), 17)
        # self.assertEqual(solution.projectionArea([[1,0],[0,2]]), 8)
        # self.assertEqual(solution.projectionArea([[1,1,1],[1,0,1],[1,1,1]]), 14)
        # self.assertEqual(solution.projectionArea([[2,2,2],[2,1,2],[2,2,2]]), 21)

        # self.assertEqual(solution.subdomainVisits(["9001 discuss.leetcode.com"]), ["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"])
        # self.assertEqual(solution.subdomainVisits(["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]), ["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"])

        # self.assertEqual(solution.queensAttacktheKing(queens = [[0,1],[1,0],[4,0],[0,4],[3,3],[2,4]], king = [0,0]), [[1,0],[0,1],[3,3]])
        # self.assertEqual(solution.queensAttacktheKing(queens = [[0,0],[1,1],[2,2],[3,4],[3,5],[4,4],[4,5]], king = [3,3]), [[2,2],[3,4],[4,4]])
        # self.assertEqual(solution.queensAttacktheKing(queens = [[5,6],[7,7],[2,1],[0,7],[1,6],[5,1],[3,7],[0,3],[4,0],[1,2],[6,3],[5,0],[0,4],[2,2],[1,1],[6,4],[5,4],[0,0],[2,6],[4,5],[5,2],[1,4],[7,5],[2,3],[0,5],[4,2],[1,0],[2,7],[0,1],[4,6],[6,1],[0,6],[4,3],[1,7]], king = [3,4]), [[2,3],[1,4],[1,6],[3,7],[4,3],[5,4],[4,5]])
        # self.assertEqual(solution.queensAttacktheKing(queens = [[4,7],[1,3],[6,6],[3,0],[0,2],[0,7],[6,2],[3,7],[5,1],[2,5],[0,3],[7,2],[4,0],[1,2],[3,3],[5,5],[4,4],[6,3],[1,5],[5,0],[0,4],[6,4],[5,4],[3,2],[0,0],[4,5],[0,5],[2,3],[1,0],[6,5],[5,3],[0,1],[7,0],[4,6],[5,7],[7,4],[2,0],[4,3],[3,4]], king = [2,4]), [[1,3],[0,4],[1,5],[2,3],[2,5],[3,3],[3,4],[4,6]])

        # self.assertEqual(solution.balancedStringSplit("RLRRLLRLRL"), 4)
        # self.assertEqual(solution.balancedStringSplit("RLLLLRRRLR"), 3)
        # self.assertEqual(solution.balancedStringSplit("LLLLRRRR"), 1)
        # self.assertEqual(solution.balancedStringSplit("RLLRRRLLLR"), 4)
        # self.assertEqual(solution.balancedStringSplit("RRLRRLRLLLRL"), 2)

        # self.assertIn(solution.sortArrayByParityII([4,2,5,7]), [[4,5,2,7], [4,7,2,5], [2,5,4,7], [2,7,4,5]])

        # self.assertEqual(solution.countCharacters(words = ["cat","bt","hat","tree"], chars = "atach"), 6)
        # self.assertEqual(solution.countCharacters(["hello","world","leetcode"], chars = "welldonehoneyr"), 10)

        # self.assertEqual(solution.heightChecker([1,1,4,2,1,3]), 3)
        # self.assertEqual(solution.heightChecker([2,1,2,1,1,2,2,1]), 4)
        # self.assertEqual(solution.heightChecker([1,2,1,2,1,1,1,2,1]), 4)

        # self.assertEqual(solution.numUniqueEmails(["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]), 2)

        # root = TreeNode(1)
        # root.left, root.right = TreeNode(2), TreeNode(5)
        # root.left.left, root.left.right = TreeNode(3), TreeNode(4)
        # root.right.left, root.right.right = TreeNode(6), TreeNode(7)
        # self.assertEqual(str(solution.recoverFromPreorder("1-2--3--4-5--6--7")), str(root))

        # self.assertEqual(solution.minDeletionSize(["cba","daf","ghi"]), 1)
        # self.assertEqual(solution.minDeletionSize(["a","b"]), 0)
        # self.assertEqual(solution.minDeletionSize(["zyx","wvu","tsr"]), 3)

        # self.assertEqual(solution.arrayPairSum([1,4,3,2]), 4)

        # self.assertEqual(solution.peakIndexInMountainArray([0,1,0]), 1)
        # self.assertEqual(solution.peakIndexInMountainArray([0,2,1,0]), 1)

        # self.assertEqual(solution.minCostToMoveChips([1,2,3]), 1)
        # self.assertEqual(solution.minCostToMoveChips([2,2,2,3,3]), 1)

        # self.assertEqual(solution.longestSubsequence(arr = [1,2,3,4], difference = 1), 4)
        # self.assertEqual(solution.longestSubsequence(arr = [1,3,5,7], difference = 1), 1)
        # self.assertEqual(solution.longestSubsequence(arr = [1, 4, 2, 5, 20, 11, 56, 100, 20, 23] , difference = 3), 5)
        # self.assertEqual(solution.longestSubsequence(arr = [7, 5], difference = -2), 2)
        # self.assertEqual(solution.longestSubsequence(arr = [1, 5, 3, 7, 6], difference = 3), 2)
        # self.assertEqual(solution.longestSubsequence(arr = [1, 7, 5, 3, 5, 1], difference = -2), 4)
        # self.assertEqual(solution.longestSubsequence(arr = [1,5,7,8,5,3,4,2,1], difference = -2), 4)

        # self.assertEqual(solution.hammingDistance(1, 4), 2)
        # self.assertEqual(solution.hammingDistance(93, 73), 2)

        # self.assertEqual(solution.selfDividingNumbers(1, 22), [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22])

        # self.assertEqual(solution.flipAndInvertImage([[1,1,0],[1,0,1],[0,0,0]]), [[1,0,0],[0,1,0],[1,1,1]])
        # self.assertEqual(solution.flipAndInvertImage([[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]), [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]])

        # self.assertEqual(solution.uniqueOccurrences([1,2,2,1,1,3]), True)
        # self.assertEqual(solution.uniqueOccurrences([1,2]), False)
        # self.assertEqual(solution.uniqueOccurrences([-3,0,1,-3,1,1,1,-3,10,0]), True)

        # self.assertEqual(solution.numTilePossibilities("AAB"), 8)
        # self.assertEqual(solution.numTilePossibilities("AAABBC"), 188)

        # self.assertEqual(solution.removeOuterParentheses(""), "")
        # self.assertEqual(solution.removeOuterParentheses("(()())(())"), "()()()")
        # self.assertEqual(solution.removeOuterParentheses("(()())(())(()(()))"), "()()()()(())")
        # self.assertEqual(solution.removeOuterParentheses("()()"), "")

        # self.assertEqual(solution.uniqueMorseRepresentations(["gin", "zen", "gig", "msg"]), 2)

        # self.assertEqual(solution.merge88([1,2,3,0,0,0], 3, [2,5,6], 3), [1,2,2,3,5,6])
        # self.assertEqual(solution.merge88([2,5,6,0,0,0], 3, [1,2,3], 3), [1,2,2,3,5,6])
        # self.assertEqual(solution.merge88([-1,0,3,0,0,0], 3, [1,2,3], 3), [-1,0,1,2,3,3])

        # self.assertEqual(solution.isPalindrome(1), True)
        # self.assertEqual(solution.isPalindrome(989), True)
        # self.assertEqual(solution.isPalindrome(9889), True)
        # self.assertEqual(solution.isPalindrome(-1), False)
        # self.assertEqual(solution.isPalindrome(10), False)

        # self.assertEqual(solution.strStr("a", "a"), 0)
        # self.assertEqual(solution.strStr("sadbutsad", "sad"), 0)
        # self.assertEqual(solution.strStr("sadbutsad", "but"), 3)
        # self.assertEqual(solution.strStr("leetcode", "leeto"), -1)

        # self.assertEqual(solution.isValidParentheses("()"), True)
        # self.assertEqual(solution.isValidParentheses("()[]{}"), True)
        # self.assertEqual(solution.isValidParentheses("(]"), False)
        # self.assertEqual(solution.isValidParentheses("([{}])"), True)
        # self.assertEqual(solution.isValidParentheses("{(}"), False)
        # self.assertEqual(solution.isValidParentheses("([)]"), False)

        # self.assertEqual(solution.combinationSum4([1, 2, 3], 4), 7)
        # self.assertEqual(solution.combinationSum4([9], 3), 0)
        # self.assertEqual(solution.combinationSum4([1, 2, 3], 9), 149)
        # self.assertEqual(solution.combinationSum4([5, 1, 8], 24), 982)

        # self.assertEqual(solution.getMaximumGenerated(7), 3)
        # self.assertEqual(solution.getMaximumGenerated(2), 1)
        # self.assertEqual(solution.getMaximumGenerated(3), 2)

        # fmt: off
        # self.assertEqual(solution.minExtraChar("abc", ["abc"]), 0)
        # self.assertEqual(solution.minExtraChar("abc", ["abcd"]), 3)
        # self.assertEqual(solution.minExtraChar("leetscode", ["leet", "code", "leetcode"]), 1)
        # self.assertEqual(solution.minExtraChar("sayhelloworld", ["hello", "world"]), 3)
        # self.assertEqual(solution.minExtraChar("dwmodizxvvbosxxw", ["ox","lb","diz","gu","v","ksv","o","nuq","r","txhe","e","wmo","cehy","tskz","ds","kzbu"]), 7) # noqa E501
        # self.assertEqual(solution.minExtraChar("ecolloycollotkvzqpdaumuqgs", ["flbri","uaaz","numy","laper","ioqyt","tkvz","ndjb","gmg","gdpbo","x","collo","vuh","qhozp","iwk","paqgn","m","mhx","jgren","qqshd","qr","qpdau","oeeuq","c","qkot","uxqvx","lhgid","vchsk","drqx","keaua","yaru","mla","shz","lby","vdxlv","xyai","lxtgl","inz","brhi","iukt","f","lbjou","vb","sz","ilkra","izwk","muqgs","gom","je"]), 2)
        # self.assertEqual(solution.minExtraChar("kevlplxozaizdhxoimmraiakbak", ["yv","bmab","hv","bnsll","mra","jjqf","g","aiyzi","ip","pfctr","flr","ybbcl","biu","ke","lpl","iak","pirua","ilhqd","zdhx","fux","xaw","pdfvt","xf","t","wq","r","cgmud","aokas","xv","jf","cyys","wcaz","rvegf","ysg","xo","uwb","lw","okgk","vbmi","v","mvo","fxyx","ad","e"]), 9)
        # self.assertEqual(solution.minExtraChar("voctvochpgutoywpnafylzelqsnzsbandjcqdciyoefi", ["tf","v","wadrya","a","cqdci","uqfg","voc","zelqsn","band","b","yoefi","utoywp","herqqn","umra","frfuyj","vczatj","sdww"]), 11)
        # self.assertEqual(solution.minExtraChar("nbxhpyyawmcsnuycfvoxhmxjclqadablucgikep", ["yaw","nbxhpy","arpqfg","bluc","thxpp","ox","a","zdaru","kmd","flckz","hnnn","dldal","yqssxn","ycf","lctpj","hmxjc","dv","cs","sxt","am","irfij","dbtg","cjnybn","ab","dngs","azbq","qa","mrx","mljbq","hphmy","b","hu","s","g"]), 10)
        # self.assertEqual(solution.minExtraChar("aakodubkrlauvfkzje", ["ix","qoqw","ax","ar","v","hxpl","nxcg","thr","kod","pns","cdo","euy","es","rf","bxcx","xe","ua","vws","vumr","zren","bzt","qwxn","ami","rrbk","ak","uan","g","vfk","jxmg","fhb","nqgd","fau","rl","h","r","jxvo","tv","smfp","lmck","od"]), 9)
        # fmt: on

        # self.assertEqual(solution.minDeletions("aab"), 0)
        # self.assertEqual(solution.minDeletions("aaabbbcc"), 2)
        # self.assertEqual(solution.minDeletions("ceabaacb"), 2)

        # self.assertEqual(
        #     [
        #         solution.candy([1]),
        #         solution.candy([1, 0, 2]),
        #         solution.candy([1, 2, 2]),
        #         solution.candy([1, 0, 2, 0, 0, 1, 1, 2, 1]),
        #         solution.candy([1, 1, 1]),
        #         solution.candy([1, 2, 87, 87, 87, 2, 1]),
        #     ],
        #     [
        #         1,
        #         5,
        #         4,
        #         13,
        #         3,
        #         13,
        #     ],
        # )

    def test_leetcode(self):
        s = SolutionLeetCode()
        # self.assertEqual(s.lengthOfLongestSubstring("abcabcbb"), 3)
        # self.assertEqual(s.lengthOfLongestSubstring("bbbbb"), 1)
        # self.assertEqual(s.lengthOfLongestSubstring("pwwkew"), 3)
        # self.assertEqual(s.lengthOfLongestSubstring(" "), 1)
        # self.assertEqual(s.lengthOfLongestSubstring("au"), 2)
        # self.assertEqual(s.lengthOfLongestSubstring("abba"), 2)
        # self.assertEqual(s.lengthOfLongestSubstring("aabaab!bb"), 3)

        # self.assertEqual(s.longestPalindrome("a"), "a")
        # self.assertEqual(s.longestPalindrome("aa"), "aa")
        # self.assertEqual(s.longestPalindrome("ab"), "a")
        # self.assertEqual(s.longestPalindrome("babad"), "bab")
        # self.assertEqual(s.longestPalindrome("cbbd"), "bb")
        # self.assertEqual(s.longestPalindrome("cbbc"), "cbbc")

        # self.assertEqual(
        #     [
        #         s.convert("PAYPALISHIRING", 4),
        #         s.convert("PAYPALISHIRING", 3),
        #         s.convert("A", 1),
        #         s.convert("AB", 1),
        #         s.convert("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 4),
        #         s.convert("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 5),
        #         s.convert("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10),
        #         s.convert("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 15),
        #     ],
        #     [
        #         "PINALSIGYAHRPI",
        #         "PAHNAPLSIIGYIR",
        #         "A",
        #         "AB",
        #         "AGMSYBFHLNRTXZCEIKOQUWDJPV",
        #         "AIQYBHJPRXZCGKOSWDFLNTVEMU",
        #         "ASBRTCQUDPVEOWFNXGMYHLZIKJ",
        #         "ABCDZEYFXGWHVIUJTKSLRMQNPO",
        #     ],
        # )

        # self.assertEqual(s.intToRoman(3), "III")
        # self.assertEqual(s.intToRoman(58), "LVIII")
        # self.assertEqual(s.intToRoman(1994), "MCMXCIV")

        # self.assertEqual(s.longestCommonPrefix(["flower", "flow", "flight"]), "fl")
        # self.assertEqual(s.longestCommonPrefix(["ab", "a"]), "a")
        # self.assertEqual(s.longestCommonPrefix(["a"]), "a")
        # self.assertEqual(s.longestCommonPrefix([""]), "")
        # self.assertEqual(s.longestCommonPrefix(["", "", ""]), "")
        # self.assertEqual(s.longestCommonPrefix(["cir", "car"]), "c")

        self.assertEqual(s.removeDuplicates([1, 1, 2]), 2)
        self.assertEqual(s.removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]), 5)


if __name__ == "__main__":
    unittest.main()
