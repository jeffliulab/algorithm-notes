# Leetcode 1 - 500

Currently no Chinese version.

※ Means the basic or foundational algorithm, that can be used in other problems. Such kind of foundations are very important.

(Mark) 就是还没做。

## No.1 - No.50

#### 1. ※ Two Sum

The difference between Two Sum and Sum2 is that Two Sum cannot sort the list, because we need the index.

```
traverse the nums list:
    sub = target - num  
    if num not in hashmap:
        store: {num: index of this num}
    else:
        return [index of num, hashmap[sub]]
```

...

...

#### 3. ※ Longest Substring Without Repeating Characters

经典滑动窗口题，这道题建议强行记忆。

这里注意，string不能用类似set或者hashmap的直接判断方法，所以需要记录有没有、存在不存在的题，都需要设置set或者hashmap来记录存在与否，而且这样也能保证查询的时间复杂度是O(1)。

这里要注意，左指针跳跃的条件是判断其最新位置是否在当前子字符串范围内，这个技巧非常聪明。

```
i是左边界，需要根据情况调整
for right in range(len(string)):
    检查当前字符是否在hashmap中:
        如果在，则判断其最新位置是否在[i,j]之间：
            如果在之间，则表明新字符是一个重复字符，不能记录
            移动i到该位置的下一个位置
    记录当前最大值max_len
```

最佳实践：

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        if not s:
            return 0
        occurd = {}
        left = 0
        max_len = 0
        for right in range(len(s)):
            cur_char = s[right]
            if cur_char in occurd and occurd[cur_char] >= left:
                left = occurd[cur_char] + 1

            occurd[cur_char] = right
            max_len = max(max_len, len(s[left:right+1]))
   
        return max_len
```

...

...

#### 6. Zigzag Conversion

这道题初见较难。如果考虑行列和形状，会导致解法变得复杂。正确思路应该是直接生成对应的行，然后分析当前遍历字符应当处于哪一行，按顺序添加进去。

难点：向上放置的时候，应放行数  = cycle_len - cycle_index，可以想象循环最终一定是在第一行，因此在向上部分，应放行数就等于循环终点 + 离终点的距离。

...

...

#### 11. Container With Most Water

初见：没想出来非暴力解法。

左右指针相向而行，高度更小的指针移动。这个解法如何保证不会遗漏更大的解呢？——这便是这道题非常重要的思想：**缩减搜索空间**。

![1752178970461](image/leetcode_interview_150.zh/1752178970461.png)

这是一种非常聪明的剪枝（Pruning）策略。

#### 12. Integer to Roman

The intuitive solution is to set a value list and a roman symbol list, and then perform // and % on the number num to be converted in turn, but it is difficult to handle special values here.

Based on the intuitive solution, make a slight modification, put the special value into the value list, and then subtract this value in turn to get the answer.

New python gramer: `value_symbol = [(1000, 'M), ...` This is the list of tuples. Equals to two lists. When you traverse it, use `for value, symbol in value_symbol:`.

...

#### 13. Roman to Integer

Use a dict/map store roman numbers.

Start from right, if left one is smaller, then sub; else sum.

New python grammer: `for s in reversed(str)` means traverse in reversed order.

#### 14. Longest Common Prefix

这道题初见感觉很简单，很容易想到直观解法（纵向扫描）：从每个单词的第一个字母开始对比，一直对比到出现不同的字母为止。coding的时候有一个技巧：总是把第一个单词作为基准去比较。

这道题初见直观解已经是最高效的了，时间效率O(N*P)，空间效率O(1)。

如果用字典序的方法去排序的话，会导致时间和空间复杂度提升，得不偿失。

#### 15. 3Sum

初见：没想出来。提示后想到把这道题分解为twosum，只是twosum的和的标准是和nums列表一样长的一个基准序列。换句话说，把3sum题转换为一个长度为N的2sum题。

注意：2sum需要先对列表进行排序，python自带的 `.sort()`的时间效率是

解法：

```pseudocode
已知nums数列，对nums排序
基准参数k：for k in range(len(nums)-2)：
    对于每个参数k所对应的有序列表，进行2sum经典解法（左右双指针）
    这里注意，2sum的范围是k后面的区域
    指针移动时跳过连续重复数字，因为nums已经排序，所以相当于防止重复结果（这是最佳去重的实践方法）
```

时间：O(N^2)，空间：O(N)

python coding注意：`.sort()`是原地修改。

这道题是一个经典的3SUM问题，长期以来一直被认为其复杂度下限就是O(N^2)，不过在科学家们的努力下，该问题如今被压缩到亚二次方（sub-quadratic）级别了；但是，优化改进仅推翻了强3SUM猜想，但是依然无法撼动弱3SUM猜想，即不存在时间复杂度为$O(N^{2-\epsilon})$的算法（其中epsilon为大于0的常数）：

![1752180976274](image/leetcode_interview_150.zh/1752180976274.png)

**4SUM问题（3SUM基础上提升）**

4SUM问题不能直接用同样的方式去降维，转而可以使用空间换时间的方法，让时间复杂度同样为O(N^2)，代价是牺牲空间，让空间复杂度为O(N^2)。4SUM详见下一道题。

#### 18 ※ 4Sum

**Solution1: Demension Reduction (This can pass leetcode)**

Reduce demensions, Time: O(N^3), Space: O(1)

```pseudocode
nums.sort()
for i in range(len(nums)-3):
    for j in range(i+1, len(nums)-2):
        left = j + 1
        right = len(nums) - 1
        use 2Sum to find the required pairs
```

More specifically:

```python
class Solution(object):
    def fourSum(self, nums, target):
        if len(nums) < 4:
            return []

        nums.sort()
        result = []
        for i in range(len(nums)-3):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            for j in range(i+1, len(nums)-2):
                if j > i+1 and  nums[j] == nums[j-1]:
                    continue
                left = j + 1
                right = len(nums) - 1
                while left < right:
                    if nums[left] + nums[right] + nums[i] + nums[j] < target:
                        left += 1
                    elif nums[left] + nums[right] + nums[i] + nums[j] > target:
                        right -= 1
                    else:
                        result.append([nums[i],nums[j],nums[left],nums[right]])
                        while left < right and nums[left] == nums[left+1]:
                            left += 1
                        while right > left and nums[right] == nums[right-1]:
                            right -= 1
                        left += 1
                        right -= 1
        return result


```

.

**Solution2: Hashmap (This cannot pass leetcode in extreme testcase)**

Time: O(N^2), Space: O(N^2)

```pseudocode
find sum of all two-numbers, make this problem into find 2 two-sum
traverse the dict, find two pairs that meets the requirement
lastly, eliminate the duplicates
```

More specifically:

```python
if len(nums) < 4:
    return []

from collections import defaultdict
sum_map = defaultdict(list)

for i in range(len(nums)-1):
    for j in range(i+1, len(nums)):
        sum_map[(nums[i]+nums[j])].append((i,j))

result_set = set()
for s1 in sum_map:
    s2 = target - s1
    if s2 in sum_map:
        pairs1 = sum_map[s1]
        pairs2 = sum_map[s2]

        for(a,b) in pairs1:
	    for(c,d) in pairs2:
		if b < c: # because when we create pairs, a < b and c < d
		    quad = tuple(sorted([nums[a],nums[b],nums[c],nums[d]]))
		    result_set.add(quad)

return [list(q) for q in result_set]
```

Why cannot pass leetcode: if the input nums is [2,2,2,2,2,2,2,2,2,2,2,...], then the hashmap will only have one key: 4. This cause the performance of the lookup phase to degenerate from near constant time to a staggering O(N^4)

..

...

...

...

...

#### 26. Remove Duplicates from Sorted Array

two pointers.

#### 27. Remove Element

这道题在循环设置上容易出错：

```python
while i <= j: # 注意这里是小于等于
    if nums[i] == val and nums[j] != val: ...
    elif nums[i] == val and nums[j] == val: ...
    else: ...
```

...

...

...

...

#### 30. Substring with Concatenation of All Words (Hard)

这道题题目看了半天硬是没明白想要我干啥。简单来说就是，给定一个string和一个list，寻找string中有几个包含list中给定的单词组成的字符串，并return这些字符串的开始位置。**并且，最重要的是，list中给定的单词长度一致，这个条件如果漏掉这道题就做不了了**。我一开始想了半天，最后才发现words里的候选词长度一样，难度直接降低了一百倍。

折腾了半天之后用下列算法做了出来：

```
set a hashmap, 记录words中单词和出现次数
设置一个长度相当于words中单词总长的window
对string进行遍历，从头start开始，到len(s)-window_size结束：
    设置一个临时的hashmap
    对当前窗口进行遍历，每步递进一个word的长度：
        获得当下的cur_word
        如果cur_word不在hashmap中，break
        将cur_word计入临时hashmap中
        如果cur_word出现次数超过hashmap中的记录次数，break
    对比临时的hashmap和hashmap：
        当且仅当两者完全一样的时候：
            记录当前的start
```

我的这个解法可以通过leetcode，但是效率有点低。最大的问题就是对每个窗口，都完全重新切分依次。

我的解法时间效率是O(N*M)，N是字符串长度，M是窗口大小。

优化思路是进行多个起始点，比如假设word长度是5，那么就分别以第一个字母、第二个字母、...、第五个字母为起始点。这样做的好处是，每次窗口移动的时候可以移动5个字符量，而不需要每个都移动。

这样修改后，外部循环的次数还是O(N)，但是内部不需要对每个窗口重新切了，因为不需要考虑类似'xtodayidkjsf'中把today给忽略掉的情况，因为五个出发点总有一个能考虑进去。所以内部循环每次只需要把左边踢出去的单词在临时hashmap中去掉，然后把右面新进入的加进来，就可以了。这样子内部的O(M)的平均时间复杂度就降低了，计算的话总共要O(N)次操作，摊销到每个子cur_word不过O(1)。

#### 30S. Substring with Uncertain Words (SuperHard)

给定一个字符串 `s` 和一个字符串数组 `words`，其中 `words` 中各单词长度 **不一定相同** 。定义一个 “连接子串” 为将 `words` 中每个字符串恰好出现一次、按任意顺序拼接而成的串。请返回所有 `s` 中恰好是某个“连接子串”的起始索引。答案可以按 任意顺序 返回。

...

#### 31. Next Permutation

注意：这道题跟回溯中的Permutation排列的解法没关系！这道题本质上是在考察对排列本质的理解。这道题直接记住思路就可以了。

限制条件：

* 必须原地修改
* constant extra memory，常数级别的额外空间

换句话说，就是不能创建一个新的result。

这道题有一个特别的解法，可以学习一下：

1. 以[1, 3, 5, 4, 2]为例，目标是找到比这个数字字典序大一位的下一个数
2. 从右往左看，[5, 4, 2]这个部分是降序的，无论如何都无法更大了
3. （追加判断）这里要注意，如果从头到尾都找不到较小数，或者说整个数列就是降序的，那么这个数字就是最大数；这个数字的下一个就是整个数组翻转过来的最小数。
4. 再往左看一个数字，3比5小，出现了爬坡的趋势，这个3就是我们要找的那个需要被变大的数，我们称之为“较小数”（smaller number）；**找到smaller number以后记得break循环，结束查找**。
5. 我们再看3的右边的数字，我们要从3的右边找到一个比3大的最少的那个数字，我们称为较大的数（larger number）
6. 从右向左遍历3右边的序列[5, 4, 2]，找到**第一个比3大的数**，这个数是4。
7. 交换smaller number 3和larger number 4
8. 交换后得到[1, 4, 5, 3, 2]
9. 交换后观察4右边的序列，发现不是最小的数，最小的序列应该是[2, 3, 5]
10. 对4右边的序列进行升序排序
11. 完成下一个字典序数字的寻找

...

#### 28. Find the Index of the First Occurence in a String

这个算是一个pyton技巧学习题，找到字符串中对应的单词的方法：`string.find(word)`，返回对应index。python底层在搜索字符上用的是Boyer-Moore-Horspool 算法，感兴趣的人可以看看相关资料。

#### 35. Valid Sudoku

First Meet: Pass with best practice:

```python
class Solution(object):
    def isValidSudoku(self, board):
        row = {}
        column = {}
        box = {}
        for i in range(9):
            for j in range(9):
                val = board[i][j] # This can improve a little bit of time efficiency

                # skip the '.'
                if val == '.':
                    continue

                # for each row, maintain a dict of set
                if i not in row:
                    row[i] = set() # row i's set
                if val not in row[i]:
                    row[i].add(val)
                else:
                    return False

                # for each column, maintain a dict of set
                if j not in column:
                    column[j] = set()
                if val not in column[j]:
                    column[j].add(val)
                else:
                    return False
  
                # for each box, maintain a dict of box
                box_row = i // 3
                box_col = j // 3
                if (box_row, box_col) not in box:
                    box[(box_row, box_col) ] = set()
                if val not in box[(box_row, box_col) ]:
                    box[(box_row, box_col) ].add(val)
                else:
                    return False
        return True
```

...

#### 37. ※ Sodoku Solver (HARD)

First Meet: I think I can make a set for each space, and delete the element when it meets the same exist one. But this idea cannot solve hard sodoku.

This is a CSP (Constraint Satisfaction Problem), the future value matters so cannot use greedy, but need use backtracking.

Basic Idea:

```
Step1: Iterative Constraint Propagation
    make a set of 1 - 9 for all spaces
    eliminate as much as possible for each space's set

Step2: Backtracking Search
    after step1, there might be many spaces with multiple numbers
    find a space (with least possibility), choose a number as a "guess"
    put this "guess" back to step1:
        if any space's set will be eliminated to None
        then we need to change a "guess"
```

More specifically:

...

...

...

...

...

...

#### 39. ※ Combination Sum

做这道题之前建议先做78 Subsets回溯基础题。

这道题和78相比，增加了一层难度：回溯的时候可以选择自己。

如果单纯只是把backtrack(start_index=i+1)这里的参数从i+1改成i，会导致无限死循环。为了避免这种无限死循环，需要引入一个新的概念：剪枝（pruning）。

Pruning的意思是：如果手里的资源已经用超了，那么就立刻掉头，不用等到走到尽头了。在这道题中，资源就是target。

同时还要注意，计算和的时候不要临时计算，而是把和作为参数去传递。临时计算的话会导致错误。

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        # a number can be chosen unlimited times
        # 可选无限次是和上一道78 Subsets回溯相比的最大区别
        # 可选无限次意味着backtrack到下一层后，自己还能再次被选
        result = []
        cur = []

        def backtrack(start_index, cur_sum):
            # Pruning一般写在开头（最佳实践）
            if cur_sum == target:
                result.append(list(cur)) # 这里需要加一个副本 ※※※
            if cur_sum > target:
                return
  
            for i in range(start_index, len(candidates)):
                cur.append(candidates[i])

                # 最佳实践的写法：直接在参数传递的时候加上
                backtrack(i, cur_sum + candidates[i])

                cur.pop()
  
        backtrack(0,0)

        return result
```

...

#### 40. ※ Combination Sum II

做这道题之前推荐先做39。这道题在39的基础上，增加了一个类型的剪枝：对于已经探索过的路径，在绝对重复的情况下，直接跳过。

这个pruning skill建议直接死记硬背地记住，因为这是一个基本功：

```
在同一层级的递归中，如果当前数字和它钱一个数字相等，那么就要跳过当前数字：
（1）当前元素candidates[i]是否和前一个元素candidates[i-1]相等（因此需要排序）
（2）当前元素不是我们在本轮for循环处理的第一个元素

代码：
if i > start_index and candidates[i-1] == candidates[i]:
    continue
```

...

#### 42. Trapping Rain Water

**Solution1: Dynamic Programming (Easy to Think)**

```
maintain a list max_left:
    max_left[i] means the heighest column to the left of column[i]
maintain a list max_right:
    max_right[i] means the heighest column to the right of column[i]
calculate each column[i]'s water:
    water's volume at column[i] = min(max_left[i], max_right[i]) - height[i]
```

Time/Space: O(N)

**Solution2: Use two pointers, optimize space to O(N)**

This idea comes from the formula `min(max_left, max_right) - height[i]`:

* when max_left is smaller, then the water volume depends on the max_left: `max_left - height[i]`
* else depends on the right

Therefore:

```pseudocode
init pointer i,j
init maxL, maxR as height[0], height[-1]
while i <= j: # The boundry here is very important
    if maxL <= maxR:
        calculate maxL - height[i]
        note: if height[i] >= maxL, then don't calculate, update maxL
    else:
        calculate maxR - height[j]
        note: if height[j] >= maxR, then don't calculate, update maxR
```

Time: O(N)

Space: O(1)

...

...

#### 45. ※ Jump Game II

```
set a cur_range
set a max_range
use greedy strategy:
    when i not reach cur_range:
        max(max_range, cur_max_range)
```

这道题思路简单但是写代码老写错：

```python
class Solution(object):
    def jump(self, nums):
        cur_range = 0
        max_range = 0
        count = 0
        for i in range(len(nums)-1):
            max_range = max(nums[i]+i, max_range)
            if i == cur_range:
                cur_range = max_range
                count += 1
        return count
```

...

#### 46. Permutations

78 Subset中首见回溯Backtracking问题的时候已经给出了解法。

重要提醒：

* path如果是list，加入result的时候一定要记得转换为list
* 无论进行了什么附加操作，都不要忘记backtrack之后的回溯步骤！

这道题的特别优化：用used[i]（存储True/False）比用set存储用过的数字更加高效

#### 47. Permutations II

和46的区别在于47的list中包含重复元素，因此有什么区别？不都可以用一个used list（带有index指针的）来pruning？

区别在于，如果有重复的列表，的确生成的结果是独一无二的，但是很多独一无二的结果是一样的结果，比如1,1,2,我们记作1a, 1b,2，其结果：

* 1a, 1b, 2
* 1b, 1a, 2

都是[1,1,2]，在出结果的时候只出一个[1, 1, 2]

这里的关键点就在于pruning：

* **规则：（必须先排序）如果发现当前的数和左邻居一样，并且左邻居还没用过，那么我们就必须跳过当前这个数字。**

为什么有这个规则呢？因为为了防止重复记录，我们必须遵从以下规则：

* **排序数组后，必须从左往右而不能跳过任何元素**

即，对于一堆相同的数字，必须严格从左到右按照顺序挑选。如果左邻居使用过，那么我们就符合从左到右规则，就可以继续使用当前的数字。

...

...

...

#### 49. Group Anagrams

The logic is easy, but notice the implementation in python:

```
create a hashmap
for string in strs:
    ls = sorted(string) # in python, this func will return a list
    key = tuple(ls) # make it in tuple, because list cannot be key
    hashmap[key].append(string)
now the values() can be returned use a python command:
return list(hashmap.values())
```

.

## No.51 - No.100

...

...

...

#### 55. Jump Game

记得先判断当下位置是不是在可达范围内

...

...

...

...

...

...

#### 56. Merge Intervals

First Meeting: Failed. Didn't handle sorting and chain merging issues.

Idea: Make a new list merged, compare merged[-1] and current interval.

solution:

```
sort the intervals (intervals.sort(key=lambda x:x[0]))
init list merged with the first interval
loop the rest of the intevals:
    merged.append(interval) if overlap
    merged.update else.
```

#### 57.

...

...

...

#### 58. Length of Last Word

太简单了，没什么说的。倒着traverse，从第一个字母开始，到最后一个字母结束。用 `.isalpha()`来判断是否是字母。

不过有个一行解法很有意思：`return len(s.strip().split(' ')[-1])`，但是因为要处理整个list，反而增加了时间复杂度和空间复杂度，属于没有什么意义的写法，但是作为趣味可供一乐。

...

...

...

#### 68. Text Justification

这道题直观思路简单，直接greedy一行一行往上填，难点是把代码coding出来（写了几次总是这儿那儿漏点东西），特别是分配带有多个可能不相等的空格的行的时候。

...

...

...

#### 76. Minimum Window Substring (Hard)

最小覆盖子串。

这里要注意，我第一遍做的时候逻辑是严格包含word中的字符数量，这样会导致错误。正确思路应该是，当子串中包含word中的字符的时候，无论是不是多包含了，都应该被视为包含。比如说，'abcgggggggggggood'这个string，如果要找good，那么用滑动窗口的时候，很明显在'ggggggggggood'的时候才找到了对应的字串，这个时候有非常多个g，需要慢慢收缩左指针left，进而找到最短的子串。

我在写的时候用两个hashmap比对，会导致每次比对的时候牺牲O(M)的时间，其中M是字符t中不同字符的数量，即最坏的情况下是len(t)。right指针循环一遍是O(N)，因此最坏情况是O(M*N)。

下面是我的初见算法：

```python
class Solution(object):
    def minWindow(self, s, t):
        if not s or not t:
            return ""
  
        hashmap = {}
        for char in t:
            hashmap[char] = hashmap.get(char, 0) + 1
  
        temp_hashmap = {}
        left = 0

        def compare(hashmap, temp_hashmap):
            # return True if hashmap is a subset of temp_hashmap
            for char, times in hashmap.items():
                if char not in temp_hashmap:
                    return False
                elif times > temp_hashmap[char]:
                    return False
            return True
  
        output = ''

        for right in range(len(s)):
            cur_char = s[right]
            temp_hashmap[cur_char] = temp_hashmap.get(cur_char, 0)+1
            if compare(hashmap, temp_hashmap) == False:
                continue
            while compare(hashmap, temp_hashmap) == True:
                cur_word = s[left:right+1]
                if output == '' or len(cur_word) < len(output):
                    output = cur_word
                temp_hashmap[s[left]] -= 1
                left += 1
  
  
        return output
```

优化思路是：我们可以引入一个 `match_count` 变量，用来记录当前窗口中满足 `t` 中字符需求（即 `temp_hashmap` 中字符数量大于等于 `hashmap` 中对应字符数量）的 **不同字符的数量** 。举例来说，对于gggggggggggood，如果要查找的是good，那么这里match_count的值为4，正好和hashmap的key长度一样。

1. 当 `temp_hashmap[char]` 增加时，如果 `temp_hashmap[char]` 首次达到或超过 `hashmap[char]`，就说明这个字符的需求被满足了，`match_count` 加 1。
2. 当 `temp_hashmap[char]` 减少时，如果 `temp_hashmap[char]` 刚好小于 `hashmap[char]`，就说明这个字符的需求不再满足，`match_count` 减 1。
3. 只有当 `match_count` 等于 `len(hashmap)`（即 `t` 中所有不同字符的需求都被满足时），当前窗口才是一个有效的覆盖窗口。

按照这个思路优化后：

```python
class Solution(object):
    def minWindow(self, s, t):
        if not s or not t:
            return ""
  
        hashmap = {}
        for char in t:
            hashmap[char] = hashmap.get(char, 0) + 1
  
        left = 0

        window_counts = {}
        match_count = 0
  
        output = ''
        min_len = float('inf')
        min_window_start = 0

        for right in range(len(s)):
            cur_char = s[right]
  
            # 新增一个window窗口中字符频率统计
            window_counts[cur_char] = window_counts.get(cur_char, 0) + 1

            # 新增对该字符出现频率的判定，只要超过hashmap中记录的值，才计入match_count
            if cur_char in hashmap and window_counts[cur_char] == hashmap[cur_char]:
                match_count += 1
  
            while match_count == len(hashmap):
                # 计算当前window长度
                cur_window_len = right - left + 1

                # 如果当前window长度比记录的min_window更小，则更新
                if cur_window_len < min_len:
                    min_len = cur_window_len
                    min_window_start = left
  
                # 减少window_counts中left字符的频率
                window_counts[s[left]] -= 1

                # 如果left字符是hashmap中记录的字符
                # 并且left字符在窗口中的数量减少后小于hashmap中记录的数量
                # 说明window中的字符不再符合需求
                # 才减少统计
                if s[left] in hashmap and window_counts[s[left]] < hashmap[s[left]]:
                    match_count -= 1
  
                left += 1

        # 如果min_len还是无穷大，说明没找到符合要求的窗口
        if min_len == float('inf'):
            return ""  
        else:
            return s[min_window_start:min_window_start+min_len]
```

这次优化主要集中在以下几点：

1. **移除 `compare` 函数** ：原先的 `compare` 函数在每次需要检查窗口是否满足条件时都会遍历 `hashmap`，导致时间复杂度较高。
2. **引入 `matched_chars` 计数器** ：这是一个关键的优化。它跟踪当前窗口中已经满足 `t` 中所有要求的**不同字符**的数量。这样，我们只需要检查 `matched_chars` 是否等于 `t` 中不同字符的总数（即 `len(hashmap)`），就可以 **O**(**1**) 时间判断窗口是否有效。
3. **优化窗口记录方式** ：不再频繁进行字符串切片（`s[left:right+1]`），因为切片操作本身会有额外的开销。而是通过记录最小窗口的**起始索引** (`min_window_start`) 和**长度** (`min_len`)，最后再进行一次切片来获取结果。

优化后时间复杂度从O(M*N)下降到O(M+N)

...

#### 78. ※ Subsets

Backtracking回溯算法入门题。

借这道题来讲一下回溯算法的概念和基本思想。

Backtracking可以被看作是一种聪明的暴力枚举法，他会尝试所有可能的选择；当发现当前的路走不通或者走完的时候，他会回溯，也就是退到上一步，尝试其他的选择。

这就像你在走一个迷宫：

* 你在一个路口，面前有几条路（ Choose **选择** ）。
* 你选择一条路往前走（ Explore/Forward **递进** ）。
* 如果这条路是死胡同，或者你已经走到了迷宫的尽头，你就需要**原路返回**到上一个路口（Backtrack **回溯** ），然后尝试你没走过的其他路。

对于这道题，我们的算法流程是这样的：

1. **开始** ：从一个空子集 `[]` 开始。
2. **决策** ：对于 `nums` 数组中的第一个数字（比如 `1`），我们做选择：

* **不选 `1`** ：继续考虑下一个数字 `2`。
* **选 `1`** ：把 `1` 加入当前子集，变成 `[1]`，然后继续考虑下一个数字 `2`。

1. **递进** ：对每一个后续数字都重复这个“选”与“不选”的决策过程。
2. **结束与回溯** ：当我们考虑完 `nums` 里的所有数字后，就得到了一个完整的子集，把它加入最终结果列表。然后，算法会“回溯”到上一个决策点，去探索其他的选择分支。

**举例 `nums = [1, 2, 3]`：**

* 从 `[]` 开始。
* **考虑 `1`** ：
* **不选 `1`** -> 接着考虑 `2`。
  * **不选 `2`** -> 接着考虑 `3`。
    * **不选 `3`** -> 得到子集 `[]`。
    * **选 `3`** -> 得到子集 `[3]`。
  * **选 `2`** -> 接着考虑 `3`。
    * **不选 `3`** -> 得到子集 `[2]`。
    * **选 `3`** -> 得到子集 `[2, 3]`。
* **选 `1`** -> 接着考虑 `2`。
  * **不选 `2`** -> 接着考虑 `3`。
    * **不选 `3`** -> 得到子集 `[1]`。
    * **选 `3`** -> 得到子集 `[1, 3]`。
  * **选 `2`** -> 接着考虑 `3`。
    * **不选 `3`** -> 得到子集 `[1, 2]`。
    * **选 `3`** -> 得到子集 `[1, 2, 3]`。

你看，通过这种系统性的决策和回溯，我们就能不重不漏地找出所有 `2^n`（n 是元素个数）个子集。

那么如何在计算机上实现回溯呢？下面举一个最简单的例子：

我们就用一个比求子集更简单的例子来彻底搞懂回溯的三个步骤： **生成所有长度为 2 的二进制字符串** 。

我们的目标是生成：`00`, `01`, `10`, `11`。

在这个问题里：

* **元素集合** ：`['0', '1']`
* **决策** ：在字符串的每一个位置，我们都要决定是放 '0' 还是 '1'。
* **目标** ：构建一个长度为 2 的字符串。

一个标准的回溯函数，通常包含下面三个部分：

1. **函数的参数** ：需要记录当前的状态。这里我们需要知道“当前正在构建的路径（path）”和“接下来要做的决策（决策列表）”。
2. **递归的出口（Base Case）** ：什么时候算找到一个完整的解了？对于本例，就是当我们的路径长度达到 2 的时候。这时我们就把路径保存下来，然后返回。
3. **循环和递归调用** ：遍历当前所有可做的选择，做出选择，然后带着这个选择进入下一层决策。决策完成后，一定要**撤销**刚才的选择，这样才能去尝试其他的选择。

用python来实现这个过程：

```python
def generate_binary_strings(n):
    """
    主函数，用来启动整个回溯过程
    """
    # result 用来存放所有找到的解
    result = []
    # path 用来记录当前正在走的路径（即正在构建的字符串）
    path = []

    def backtrack():
        # --- 2. 递归的出口 (Base Case) ---
        # 如果当前路径的长度已经达到 n，说明我们找到了一个完整的解
        if len(path) == n:
            # 将路径拼接成字符串，加入最终结果
            result.append("".join(path))
            # 找到一个解了，结束当前这条路的探索，返回
            return

        # --- 3. 循环和递归调用 ---
        # 对于当前位置，我们有哪些选择？就是 '0' 和 '1'
        possible_choices = ['0', '1']
  
        for choice in possible_choices:
            # 1. 选择 (Choose)
            # 做出选择，把 '0' 或者 '1' 加入到当前路径中
            path.append(choice)
  
            # 2. 递进 (Explore)
            # 进入下一层决策。因为path已经变长了，
            # 下一层的函数调用会去决定下一个位置的字符。
            backtrack()
  
            # 3. 回溯 (Backtrack)
            # **这是最关键的一步！**
            # 当上一行 backtrack() 函数返回时，
            # 说明基于当前 choice 的所有后续路径都探索完了。
            # 我们必须撤销刚才的选择（把加进去的字符删掉），
            # 这样 for 循环下一次迭代时，才能去尝试其他的选择。
            path.pop()

    # 从这里开始第一次调用，启动整个过程
    backtrack()
    return result

# 让我们来运行一下，生成长度为 2 的所有二进制字符串
final_result = generate_binary_strings(2)
print(final_result)
```

\

第一次接触backtracking的时候，最难理解的就是3. 回溯的部分。这里其实是利用了程序运行时function call的原理，即当backtrack()被调用后，这个时候程序在这里相当于暂停了，然后启动了一个新的子任务；当子任务结束后，程序就会回到这里来继续。

这背后是程序运行的一个基本概念： **函数调用栈 (Function Call Stack)** 。

你可以把函数调用想象成一个“暂停/继续”的游戏：

* 当你调用一个函数时，比如 `A` 调用 `B`，那么函数 `A` 就会在调用的地方 **暂停** ，等待 `B` 的结果。
* 函数 `B` 开始执行。如果 `B` 又调用了 `C`，那么 `B` 也会在那个地方 **暂停** ，等待 `C` 的结果。
* 当 `C` 执行完毕 `return` 时，它会把结果（或什么都不给）返回给 `B`。然后，函数 `B` 就会从它刚才暂停的地方 **继续往下执行** 。
* 同理，当 `B` 执行完毕 `return` 时，`A` 也会从它暂停的地方 **继续往下执行** 。

所以过程实际上如下：

1. backtrack()
   1. path.append(0)
   2. backtrack()
      1. path.append(0)
      2. backtrack()
         1. base case
         2. return
      3. pop
      4. path.append(1)
      5. backtrack()
         1. base case
         2. return
      6. pop
   3. pop
   4. path.append(1)
   5. backtrack()
      1. path.append(0)
      2. backtrack()
         1. basecase
         2. return
      3. pop
      4. path.append(1)
      5. backtrack()
         1. base case
         2. return
      6. pop
   6. pop
2. the function call end

可见，回溯表面上看起来是走回去，其实本质上还是在往下走，在这个过程中利用程序的暂停机制实现了看起来走过去又走回来的特性。真的是太美妙了！

现在回到这道题来，这道题和上面的不同之处在于：

* 选择列表是有限制的，不能每一次都在备选中随便选；一旦被选过，就不能再被选；只能从前往后选。

要实现上述功能，需要用到一个子集/组合问题回溯算法的标志性技巧：

* 给backtrack函数增加一个参数start_index，告诉他这次for循环应该从nums的哪个位置开始

```python
class Solution(object):
    def subsets(self, nums):
        # 正确的回溯不会产生重复的情况，所以这里直接用list存储result
        result = []
        path = []

        def backtrack(start_index):

            result.append(list(path))

            # 这里不需要这个base case
            # 在本题中，只要不进入for循环，就算是结束了
            # if len(path) == len(nums):
            #     return
  
            # start_index = len(path)
            # 不适用参数传递的方法会导致后续紊乱

            for i in range(start_index, len(nums)):

                path.append(nums[i])

                # 这里很关键，第一次写成start_index了
                # 但是这里如果写成start_index
                # 第一轮将正常进行
                # 但是第二轮将错误进行，因为1，2，3来看的话
                # 1完了2，2完了3，但是2和3的轮次完了后
                # 应该直接跟3，因为[2,3]和[3]在本题中是并列的可选子集  
                backtrack(i+1)

                path.pop()
  
        backtrack(0)

        return result
```

通过这道题，对回溯的思想进行了一个系统性的讲解。

...

#### 79. ※ Word Search

这是一道DFS+Backtracking的经典题目，建议牢记解法。

```
用DFS验证点是否匹配word[i]:
    如果找到，return True
    如果已经在路径中 或者 越界 或者 字符不匹配， return False
    path.add((r,c))
    向四个方向探索 ※ 如果为True，说明找到了
    path.remove((r,c))

遍历grid：
    if DFS(点，0)：
        return True
return False
```

这道题的代码一定要熟悉，是一种模板：

```python
class Solution(object):
    def exist(self, board, word):
        rows = len(board)
        cols = len(board[0])
        path = set() # 路径集合，用于防止重复访问

        def dfs(r, c, i):
            # 1. 递归成功的终止条件：i 越过了单词的最后一个字符，说明全部找到了
            if i == len(word):
                return True

            # 2. 递归失败的终止条件（把所有错误情况一次性处理）
            if (r < 0 or c < 0 or         # a. 越界检查
                r >= rows or c >= cols or
                board[r][c] != word[i] or # b. 字符不匹配检查 (你的代码缺失了这个!)
                (r, c) in path):          # c. 当前坐标是否已在路径中 (你的代码写错了!)
                return False

            # --- 到这里说明 (r, c) 这个点是有效的 ---
  
            # 3. 做出选择 (将当前点加入路径)
            path.add((r, c))

            # 4. 向四个方向进行下一层递归
            #    只要有一个方向成功 (返回True)，就立即停止并返回 True
            #    (这解决了你代码中 "found" 被覆盖的问题)
            for dr, dc in directions:
                if dfs(r+dr, c+dc, i+1):
                    found = True
                    break # 一旦找到，就没必要继续尝试其他方向了， 跳出循环

            # 5. 撤销选择 (回溯，为其他路径的探索让路)
            path.remove((r, c))
  
            return found

        # 主循环
        for r in range(rows):
            for c in range(cols):
                # 只需要在这里调用一次 dfs，所有逻辑都在 dfs 内部
                # 我们从 (r,c) 开始，尝试匹配 word 的第一个字符 (i=0)
                if dfs(r, c, 0):
                    return True
  
        return False
```

...

#### 80. ※ Remove Duplicates from Sorted Array II

题目要求是不超过2次重复，按照造轮子的指导方针，这道题现在应当直接给出一个通用解了。

设计一个通用解，关键在于写入条件：一个元素应当被保留，当且仅当：

* 它是一个新出现的数字
* 它是一个重复的数字，但是它的出现还不足K次

这个条件可以抽象概括为：

* **一个元素nums[j]可以被写入到nums[i]的位置，如果它和i左边第k个元素nums[i-k]不相等。**

特殊边界情况：

* **当i<k时，说明有效数组长度还不足k，因此前k个元素应该被无条件保留。**

```python
class Solution(object):
    def removeDuplicates(self, nums, k):
        # 一个通用的解决方案，允许每个元素最多出现 k 次。

        # i 是慢指针（或写指针/Writer）。
        # nums[0...i-1] 是处理好的部分。
        i = 0
  
        # 使用 for 循环遍历每个元素，num 充当快指针（或读指针）的角色。
        for num in nums:
            # 核心判断条件：
            # 1. 如果 i < k，说明当前处理好的数组长度还不足 k，
            #    所以任何元素都可以直接放进来。
            # 2. 如果 num > nums[i-k]，说明当前元素 num 和结果数组中倒数第 k 个元素不同。
            #    因为数组是排序的，这意味着我们还没有 k 个 num，所以可以把它放进来。
            if i < k or num > nums[i - k]:
                nums[i] = num
                i += 1
        return i
```

...

...

#### 88. Merge Sorted Array

Two pointers.

这道题逻辑简单，但是边界很容易写错：

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        if not nums2:
            return nums1
        i = m - 1
        j = n - 1
        for k in range(len(m+n-1,-1,-1):
            # 这里的边界判定很容易出错，导致index out of range error
            if j < 0:
                break
            if i >= 0 and nums1[i] >= nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
        return nums1

```

...

...

#### 90. ※ Subsets II

这道题找不重复的子集，非常经典，强烈全文背诵加深刻理解。

我们完全从零开始，忘记之前的代码，只看题目。

**目标：对于 `[1, 2, 2]`，找出所有不重复的子集。**

**第一步：直面“重复”的根源**

我们先把两个 `2` 区分开，记作 `2a` 和 `2b`。输入就是 `[1, 2a, 2b]`。
如果我们用最朴素的回溯法，会发生什么？

* 对于元素 `1`，我们可以选，也可以不选。
* 对于元素 `2a`，我们可以选，也可以不选。
* 对于元素 `2b`，我们可以选，也可以不选。

让我们看看子集中包含 `1` 的情况：

* 选 `1`，选 `2a`，不选 `2b`  =>  得到 `{1, 2a}`
* 选 `1`，不选 `2a`，选 `2b`  =>  得到 `{1, 2b}`
* 选 `1`，选 `2a`，选 `2b`   =>  得到 `{1, 2a, 2b}`

问题来了：在最终结果里，`{1, 2a}` 和 `{1, 2b}` 其实是同一个子集 `[1, 2]`。我们必须想办法 **只生成其中一个** 。

**第二步：如何制定一个“唯一”的选择规则？**

要避免重复，我们就得定个规矩，让所有重复的元素在选择时必须遵循同一种模式。
最容易想到的就是： **让相同的元素紧挨在一起** 。这样我们才能方便地比较它们。

* **思考** ：如何让相同元素挨在一起？
* **答案** ：**排序！** 这是解决所有包含重复元素的排列组合问题的标准起手式。

排序后，输入依然是 `[1, 2, 2]`。

**第三步：在决策树中发现重复模式**

我们画一个简化的决策树来模拟回溯过程。

```
                      []
                    /   \
                   /     \ (不选1)
                 [1]      ...
                 / \
      (选第一个2)/   \ (不选第一个2)
               /       \
            [1, 2]      [1]  <-- 此时路径是[1]，接下来要考虑第二个2
             /           \
   (选第二个2)/             \ (选第二个2)
           /                 \
      [1, 2, 2]             [1, 2]  <-- 问题出现！
```

观察上图右侧的分支：

1. 我们选择了 `1`。
2. 然后我们**跳过了**第一个 `2`。
3. 然后我们又**选择了**第二个 `2`。
   这导致我们生成了子集 `[1, 2]`。

再看上图左侧的分支：

1. 我们选择了 `1`。
2. 然后我们**选择了**第一个 `2`。
   这也导致我们生成了子集 `[1, 2]`。

 **重复的根源找到了** ：对于相同的元素（例如两个 `2`），我们既可以“选择第一个，跳过第二个”，也可以“跳过第一个，选择第二个”，这两种不同的选择路径，最终得到了相同的结果。

**第四步：制定剪枝规则，砍掉重复路径**

为了保证唯一性，我们必须在这两种路径中只选一种。哪种更自然？
**规定：对于一串重复的数字，我们只允许从左到右依次选择。我们不允许“跳过前面的重复数，反而去选择后面的”。**

这个规定如何翻译成代码逻辑？

当我们在 `for` 循环中遍历到 `nums[i]` 时：
我们要判断是否要“跳过”（`continue`）`nums[i]`。
根据我们的规定，跳过的条件是：
“我想要跳过前面的重复数，然后选择当前这个数”。—— 这是我们要禁止的行为。

所以，我们的剪枝规则应该是：
**如果当前数字 `nums[i]` 和它前一个数字 `nums[i-1]` 相同，并且我们是在“跳过了 `nums[i-1]`”的决策路径上，那么我们也必须跳过 `nums[i]`。**

在代码中，`for i in range(start_index, len(nums))` 这个循环本身就包含了“选择”和“跳过”的逻辑。当 `for` 循环从 `i-1` 迭代到 `i` 时，本身就意味着我们结束了对 `nums[i-1]` 的所有决策（即“回溯”了），相当于在当前层级“跳过”了 `nums[i-1]`。

因此，我们的规则可以精确地写成：
`if i > start_index and nums[i] == nums[i-1]`

* `nums[i] == nums[i-1]`：保证了我们正在处理重复元素。
* `i > start_index`：这是最精妙的部分！它保证了这个判断只在**同一层递归**的 `for` 循环中生效。如果 `i == start_index`，说明 `nums[i]` 是当前这层递归中我们遇到的第一个元素，我们无论如何都应该考虑它，所以不能剪枝。只有当 `i` 比 `start_index` 大时，才意味着我们正在横向遍历决策树的同一层，此时 `nums[i-1]` 就是我们刚刚考虑过的、位于同一层的兄弟节点。

**总结一下思考路径：**

1. **目标** ：去重。
2. **发现问题** ：不排序的话，相同的元素 `2a` 和 `2b` 会产生 `[1, 2a]` 和 `[1, 2b]` 这样的重复。
3. **初步解决方案** ：先 **排序** ，让 `[1, 2a, 2b]` 变成 `[1, 2, 2]`，把问题集中化。
4. **深入分析** ：画决策树，发现重复来源于“跳过第一个2，选择第二个2”这样的操作。
5. **制定规则** ：禁止这种操作。规定对于重复元素，只能从左到右选，不能跳着选。
6. **代码化规则** ：将规定翻译成 `if i > start_index and nums[i] == nums[i-1]: continue`，精准地砍掉产生重复的搜索分支。

代码中关键部分我已经专门写出，建议直接记忆，慢慢理解：

```python
...
nums.sort()
def backtrack(start_index):
    result.append(list(path))
    for i in range(start_index, len(nums)):
        if i > start_index and nums[i] == nums[i-1]:
            continue
        ...
```

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

## No.101 - No.150

...

...

...

#### 121. ※ Best Time to Buy and Sell Stock I

这道题的关键是关注于两件事：

* 截至目前为止最低的价格
* 截至目前为止最高的利润

这道题建议一定要记住，因为这道题中的左指针是min_price，而不是常见的left。

```python
min_price = float('inf')
max_profit = 0
for price in prices:
    if price < min_price:
        min_price = price
    max_profit = max(max_profit, price - min_price)
return max_profit
```

#### 122. Best Time to Buy and Sell Stock II

贪心直过，还没I难。

#### 123. Best Time to Buy and Sell Stock III (Hard)

这次**限制最多两次交易**，难度上来了。需要用DP动态规划的思路。

我们先抛开动态规划的概念，来回归到这道题上来：假设你是一个手机倒卖小商贩，本金无限。现在，突然出现了一个限量款的手机，每个人只能拥有一部。你的目标是**最多**在这个市场里倒卖两次，然后赚到最多的钱。现在核心规则如下：

* 你的手里最多只能持有一部手机
* 必须先卖掉第一部，才能买入第二部
* 你最多只能完成两轮“买入-卖出”的交易

四个核心状态（你的四本账）：想象你有四本账，分别记录不同阶段的最佳成果：

1. **账本1 (`buy1`)：记录“第一次买入”后的最佳状态**
   * **含义** ：只考虑第一次买入，你花掉的成本越少越好。这本账记录的是你“欠自己”最少的钱。
   * **每天的决策** ：看到今天的新价格，你问自己：“我是今天才第一次出手买呢，还是保持之前那个更便宜的买入价划算？”
   * **例子** ：第一天手机卖3000，你的账本1是 `-3000`。第二天手机降到2500，你就会想：“太好了，我应该按2500买才对！” 于是你把账本1更新为 `-2500`。你总是在寻找最低的买入点。
2. **账本2 (`sell1`)：记录“第一次卖出”后的最佳状态**
   * **含义** ：只考虑完成第一笔交易后，你手里赚到的最多现金。
   * **每天的决策** ：看到今天的新价格，你问自己：“我是今天把它卖掉呢，还是保持之前某天卖掉赚的钱更多？”
   * **例子** ：你之前在2500买的（账本1是-2500）。今天手机涨到5000，你一卖，就能净赚 `5000 - 2500 = 2500`。如果之前某次交易的最高纪录是赚了2200，那你现在就把账本2更新为 `2500`。
3. **账本3 (`buy2`)：记录“第二次买入”后的最佳状态**
   * **含义** ：在你完成第一笔交易赚到一笔钱后，你又出手买了第二部手机。这本账记录的是 **“第一笔赚的钱 - 第二次买入的成本”** 之后，你手里的“潜在总资产”。
   * **每天的决策** ：看到今天的新价格，你问自己：“我是在今天出手买第二部呢，还是保持之前那个‘买第二部后’的状态更有利？”
   * **例子** ：你第一笔交易净赚了2500（账本2是2500）。今天手机价格是4000，你决定买入第二部。那么你现在的总资产状态是 `2500 (第一笔赚的) - 4000 (第二笔成本) = -1500`。你把这个 `-1500` 记在账本3上。如果明天手机降到3800，你会更新账本3为 `2500 - 3800 = -1300`，因为这个状态更有利。
4. **账本4 (`sell2`)：记录“第二次卖出”后的最佳状态**
   * **含义** ：完成所有两次交易后，你手里最终持有的总现金。**这就是我们的终极目标！**
   * **每天的决策** ：看到今天的新价格，你问自己：“我是今天卖掉第二部手机来锁定总利润呢，还是之前某次完成两笔交易的利润更高？”
   * **例子** ：你买第二部手机后的状态是-1300（账本3是-1300）。今天手机价格涨到了6000，你立刻卖掉。你的最终总利润就是 `-1300 + 6000 = 4700`。你把 `4700` 记在账本4上，这就是你目前为止倒卖两轮能赚到的最多钱。

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        # 初始化四个状态的利润
        # buy1 和 buy2 初始化为一个极小值，确保第一次计算时会被当天的价格覆盖
        buy1 = float('-inf')
        sell1 = 0
        buy2 = float('-inf')
        sell2 = 0

        # 遍历每一天的价格
        for price in prices:
            # 更新第一次买卖的最大利润
            # max(保持前一天的buy1状态, 今天第一次买入)
            buy1 = max(buy1, -price)
            # max(保持前一天的sell1状态, 今天第一次卖出)
            sell1 = max(sell1, buy1 + price)

            # 更新第二次买卖的最大利润
            # max(保持前一天的buy2状态, 基于sell1的利润今天第二次买入)
            buy2 = max(buy2, sell1 - price)
            # max(保持前一天的sell2状态, 今天第二次卖出)
            sell2 = max(sell2, buy2 + price)

        # 最终的最大利润就是sell2
        # 如果不交易，sell2为0；如果只交易一次，sell2会等于sell1的最大值
        return sell2
```

...

#### 123S. Max Profit at Exactly Two Transactions

如果123的题目改成必须交易两次，那么在DP基础上稍作修改就可以了：

```python
def maxProfitExactlyTwoTransactions(prices: list[int]) -> int:
    # 检查价格序列是否足够长以完成两次交易
    # 买卖一次需要2天，两次需要4天。
    if len(prices) < 4:
        # 在这种情况下，无法完成两次交易，可以根据题意返回0或错误。
        # 但通常这类问题会保证输入长度足够。我们假设可以完成。
        # 如果非要返回一个数字，负无穷代表“不可能”。
        return -1 # 或者根据具体题目要求

    # 初始化四个状态，强制每个状态都必须发生
    buy1 = float('-inf')
    sell1 = float('-inf')  # 关键改动
    buy2 = float('-inf')
    sell2 = float('-inf')  # 关键改动

    for price in prices:
        # 状态转移方程不变
        buy1 = max(buy1, -price)
        sell1 = max(sell1, buy1 + price)
        buy2 = max(buy2, sell1 - price)
        sell2 = max(sell2, buy2 + price)

    # 最终的sell2就是必须完成两次交易后的最大利润
    # 如果市场一直下跌，这个结果可能是负数
    return sell2
```

...

#### 125. Valid Palindrome

初见：直接用python的 `[::-1]`暴力解。

时间：O(N), 空间：O(N)

用双指针法可以将空间优化到O(1)：左右指针同时出发，对比有效字符。

...

#### 128. ※ Longest Consecutive Sequence

这道题要求O(N)，也就是不能排序。

核心要点：对于一个数x，如何飞快地直到x+1在不在数组里？

解题方法：

```
将nums转换为一个set
traverse nums: # 这里注意要用hashset遍历否则会超时 ※
    判断（num-1）是否在set中，如果不在，说明num就是一个连续序列的起点：
        从num开始往后数，一直数到最后，记录连续数列的长度
        更新longest_consecutive_sequence的length或者具体的sequence
```

时间复杂度：

* 把nums转换为set需要O(N)
* 内部循环受限于数列长度也是O(N)

因此总的时间复杂度还是O(N)

空间复杂度：O(N)

注意，set中查找数字的平均情况是O(1)，最差情况是O(N)

...

...

#### 131. ※ Palindrome Partitioning

这是一道经典的分割/分段模型题。（Segmentation Model）

这道题有点难，需要对input string进行多种可能的切割，和之前遇到的其他backtracking题都有点不一样。常规的回溯题是从元素集合中选择元素，每次分叉面临的选择是：选择哪个元素？

而这道题需要换个思维方式，在每一次选择的时候，选的不是元素，而是切的那一刀。遍历的不是元素，而是字符和字符之间的缝隙，在每个缝隙处，都要面临一个问题：切不切。

在此基础上，其实没必要每一个字符都切割。这道题是组合类问题，所以组合类问题就用start_index；但是这道题又不是普通的组合问题，因为所有可能的切割方法都要考察。所以在此基础上，我们从start_index开始，到字符串结束，访问每一个以start_index为起点的子字符串substring。即，尝试的是以start_index未开始，i为结束子字符串。

如果substring是一个回文，那么就把这一段切出来，然后把start_index设置为切完后的下一个字母。比如说'ababa'，先找到了'a'，以'b'为下一个start_index；这一轮recursion结束后，再回溯恢复，以'aba'为下一个回文子序列，进而继续遍历下一个可能的以第二个'b'作为下一个start_index。

```python
class Solution(object):
    def partition(self, s):
        n = len(s)
        result = []  # 用于存放所有有效分割方案的最终结果
        path = []    # 用于存放当前正在探索的单个分割方案

        # --- 步骤 1: 创建一个辅助函数来判断回文 ---
        # 这个函数会按需被调用
        def is_palin(sub):
            # 一个非常简洁的 Python 写法，判断一个字符串是否和它的逆序相等
            return sub == sub[::-1]

        # --- 步骤 2: 核心回溯函数 ---
        # start_index 表示当前需要处理的子串的起始位置
        def backtrack(start_index):
            # 基线条件 (Base Case):
            # 如果起始位置已经到达了原字符串的末尾，
            # 说明我们已经成功地将整个字符串分割完毕。
            if start_index == n:
                # 将当前路径的一个副本加入到结果集中。
                # 必须是副本 (list(path))，否则后续 path 的修改会影响已存入 result 的结果。
                result.append(list(path))
                return

            # --- 决策循环 ---
            # 从 start_index 开始，尝试所有可能的切割终点 i
            for i in range(start_index, n):
                # 定义我们想要尝试切割的子串
                substring = s[start_index : i + 1]

                # 检查这个子串是否为回文
                if is_palin(substring):
                    # 如果是回文，这是一个有效的选择
    
                    # 1. 做出选择 (Choose)
                    # 将这个有效的回文子串加入当前路径
                    path.append(substring)
    
                    # 2. 继续探索 (Explore)
                    # 从 i + 1 的位置开始，继续对字符串的剩余部分进行分割
                    backtrack(i + 1)
    
                    # 3. 撤销选择 (Unchoose / Backtrack)
                    # 当从上一层的递归调用返回后，我们需要撤销当前的选择，
                    # 以便 for 循环可以继续尝试其他的切割点 (比如尝试更长的子串)。
                    path.pop()
                # 如果不是回文，for 循环会继续，i会增加，尝试一个更长的子串

        # --- 步骤 3: 启动回溯 ---
        # 从索引 0 开始，对整个字符串进行分割
        backtrack(0)
  
        return result
```

【最佳实践/最优解】在朴素的回溯中，我们会反复判断同一个子串是否是回文，比如处理'ababa'时，字串'aba'会被多次检查。为了避免这种冗余计算，我们可以预先计算出所有字串的回文属性，并将结果存储起来。这样在回溯的时候，判断一个字串是否是回文就从O(N)的复杂度变成了O(1)的查询操作。

```python
class Solution(object):
    def partition(self, s):
        n = len(s)
        # 如果字符串为空，直接返回空列表
        if n == 0:
            return []

        # --- 步骤 1: 动态规划预处理 ---
        # 创建一个 DP 表，dp[i][j] = True 表示子串 s[i..j] 是回文
        # 初始化为 False
        # 下面的部分等同于： dp = [[False] * n for _ in range(n)]
        dp = []  # 先创建一个空列表
        for _ in range(n):  # 循环 n 次
            # 每一次循环，都创建一个全新的行
            row = [False] * n
            # 把这一行加到 dp 列表中
            dp.append(row)

        # 填充 DP 表
        # i 从右往左遍历，j 从 i 往右遍历，这样可以保证计算 dp[i][j] 时,
        # 所依赖的 dp[i+1][j-1] 已经被计算过。
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                # 如果 s[i] 和 s[j] 相等，则s[i..j]是否为回文取决于 s[i+1..j-1]
                # 状态转移方程：dp[i][j] = (s[i] == s[j]) and (j - i < 2 or dp[i+1][j-1])
                # j - i < 2 是处理边界情况：
                #   - j == i: 单个字符，必是回文
                #   - j == i + 1: 两个字符，相等即是回文
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True

        # --- 步骤 2: 回溯搜索 ---
        result = []
        path = []

        def backtrack(start_index):
            if start_index == n:
                result.append(list(path))
                return

            for i in range(start_index, n):
                # [优化关键] 使用DP表进行 O(1) 的查询
                # 如果 s[start_index..i] 不是回文，则这条路是死路，直接跳过（剪枝）
                if not dp[start_index][i]:
                    continue

                path.append(s[start_index : i + 1])
  
                backtrack(i + 1)

                path.pop()

        backtrack(0)
        return result
```

...

#### 134. ※ Gas Station

这道题是一道经典的贪心greedy题，非常适合用来理解greedy思想。

本题要点：

* 设置一个总计数器，用来判断最终是否能符合条件
* 设置一个子计数器，用来判断哪个位置是合格的起点

...

#### 135. ※ Candy (Hard)

这道题可以记一下贪心策略，以后可能会用上类似的思想：

* 从左到右贪心
* 从右到左贪心

...

...

...

...

...

#### 146. ※ LRU Cache (Data Structure)

数据结构：双向链表 + HashMap(Node作为值)

面试中推荐手写链表，这样比较清晰一点

...

## No.151 - No.200

#### 151. Reverse Words in a String

直观解法：split后reverse。

python技巧：

* `.split()`可以把连续空格/换行符/制表符视作单个分隔符
* `.strip()`可以把开头和结尾的空格去掉。注意python的string是不可变的，strip操作是return了一个新的string。(英文翻译术语注意：delete leading or trailing spaces)

但是直观解法这里有一个坑要注意，就是python中在做字符串相加的时候，因为string是不可变的，所以每一次string相加都是一个O(N)的操作，这一特点直接会导致时间复杂度爆炸。因此，需要把直观写法 `string += char`改为 `' '.join(list of char)`。注意用法：'分隔符'.join(要用分隔符连接为string的可迭代对象，不一定仅限于list，tuple、iterator等都可以)

...

#### 167. Two Sum II - Input Array Is Sorted

初见：构造一个sub list，存储补数；然后只要list中存在补数，就说明有配对。

但是注意，这道题有个条件我没用到，就是数列是非递减的，那么其实就简单多了：左右双指针相向而行，对比两指针和和target，然后调整左右指针。

...

...

#### 169. Majority Element

摩尔投票法（Boyer-Moore Majority Vote Algorithm）：如果出现过就vote，没出现过就devote，最后剩下来的数字就是要找的数字。

**注意：必须确保存在过半的元素。**

如果没有过半的数字，找最多频率的数字，需要维护一个hashmap，然后记录频率，并即时更新最大值。

...

...

...

#### 189. ※ Rotate Array

原地翻转list，这里要注意，无论是切片还是.reverse()，都是生成了一个新的副本，如果你把新的副本再赋值给nums这个变量，那么原本的nums会丢失。所以在使用的时候，需要使用格外的技巧。

* new = nums[::-1]会创建新的副本，return翻转后的列表。
* **nums[:] = nums[::-1]是修改本身，return None。推荐使用。**
* nums.reverse()是修改本身，return None。但不推荐用这个，不够灵活。

注意: .reverse()是list的方法，修改本身；reversed()是python的内置函数，可用于任何可迭代对象，不修改原始对象，返回一个反向的迭代器iterator。

**Solution1. Slicing**

具体来说，如果使用切片，**需要使用nums[:]命令**，这个代码的意思是不要改变nums这个变量指向的对象，即原地操作；但是对nums[:k]进行修改的时候，这里本身就是切片赋值（slice assignment）：

```python
class SolutionSlicing:
    def rotate(self, nums: list[int], k: int) -> None:
        n = len(nums)
  
        # To handle cases where k is greater than the length of the array,
        # we take the modulo. The effect of rotating by k is the same as
        # rotating by k % n.
        k %= n
  
        if k == 0:
            return

        # Step 1: Reverse the entire list.
        # nums[:] = ... modifies the list in-place.
        # nums[::-1] creates a new reversed copy.
        nums[:] = nums[::-1]
  
        # Step 2: Reverse the first k elements.
        nums[:k] = nums[:k][::-1]
  
        # Step 3: Reverse the remaining n-k elements.
        nums[k:] = nums[k:][::-1]
```

**Solution2: 手撸代码**

```python
class Solution(object):
    def rotate(self, nums, k):
        n = len(nums)
        k = k % n  # 1. 处理 k 大于数组长度的情况

        if k == 0:
            return

        # 辅助函数，用于翻转数组的指定部分
        def reverse(arr, start, end):
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1

        # 2. 翻转整个数组
        reverse(nums, 0, n - 1)
        # 3. 翻转前 k 个元素
        reverse(nums, 0, k - 1)
        # 4. 翻转后 n-k 个元素
        reverse(nums, k, n - 1)
```

...

...

...

...

...

...

...

...

...

...

#### 200. ※ Number of Islands

这是一道入门知识学习题，主要学习DFS和BFS。

一般来说，求最短路径，BFS效率更好一点；求任意解或者所有解，一般用DFS。如果是本道题这种求连通性的，DFS更简洁。

（关键词：BFS/DFS/深度优先搜索/广度优先搜索/岛屿问题/图基础问题）

这道题的解题思路是：

```
遍历图：
    当出现岛屿的时候，记录该位置
    用DFS或者BFS把从该位置出发的所有岛屿全部淹没
    记录一次淹没，即一个岛屿
```

DFS用递归实现，BFS用队列实现。DFS有点类似于回溯，BFS则是用一个队列实现所有相关内容的遍历。

DFS算法思路：

```
def dfs(x,y):
    如果点(x,y)超出了边界或者位于水域：
        return
    将(x,y)的性质变更为水
    dfs(右边的一格)
    dfs(上边的一格)
    dfs(左边的一格)
    dfs(下边的一格)
```

DFS代码实现：

```python
class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        island_count = 0

        def dfs(r, c):
            # 1. 检查边界和是否是水域，如果是则返回
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
                return
  
            # 2. 将当前陆地 '1' 变为 '0'，表示已经访问过（沉岛）
            grid[r][c] = '0'

            # 3. 递归地访问四个方向的邻居
            dfs(r + 1, c) # 向下
            dfs(r - 1, c) # 向上
            dfs(r, c + 1) # 向右
            dfs(r, c - 1) # 向左

        # --- 主逻辑 ---
        # 遍历整个网格
        for r in range(rows):
            for c in range(cols):
                # 如果找到一块陆地
                if grid[r][c] == '1':
                    # 岛屿数量加一
                    island_count += 1
                    # 使用DFS将整个岛屿沉没
                    dfs(r, c)
  
        return island_count
```

BFS算法思路：

```
创建一个队列，一般用queue = collections.deque([(x,y)])
将该点标记为淹没 ※ 最佳实践：先淹没，再入队
当queue不为空的时候：
    取出队首元素
    找到队首元素的四个邻居格子a, b, c, d
    遍历a, b, c, d:
        如果是陆地且没有在界外：
            标记为淹没
            将该格子加入队列，等待未来处理
```

BFS代码实现：

```python
import collections

class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        island_count = 0
  
        # --- 主逻辑 ---
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    island_count += 1
                    # 使用BFS将整个岛屿沉没
  
                    # 1. 创建队列并加入起始点
                    q = collections.deque([(r, c)])
                    # 2. 标记起始点已访问 ※ 这里要在入队前就标记
                    grid[r][c] = '0'
  
                    while q:
                        # 3. 取出队首元素
                        row, col = q.popleft()
  
                        # 4. 探索四个方向
                        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        for dr, dc in directions:
                            nr, nc = row + dr, col + dc
  
                            # 检查新坐标是否有效且为陆地
                            if 0 <= nr < rows:
                                if 0 <= nc < cols
                                    if grid[nr][nc] == '1':
                                        # 加入队列并标记
                                        q.append((nr, nc))
                                        grid[nr][nc] = '0'

        return island_count
```

在最终的DFS/BFS处理上有一个细微的小的区别：

* DFS主要判断是否“不合格”，如果不合格就直接return
* BFS主要判断是否“合格”，如果合格就淹没然后入队

这种区别来自于DFS和BFS的理念上的不同：

* DFS是递归函数，因此第一要务是做好base case终止条件，否则程序会崩溃
* BFS

DFS中这种在函数一开头就立马检查不行就退出的风格，被称为卫语句（Guard Clauses）。Guard守护函数内部不会出现未被检查而导致程序崩溃的坏角色出现。因此DFS有点像夜店，开头的守卫就是夜店门口的保安。

...

## No.201 - No.250

...

...

...

...

#### 209. Minimum Size Subarray Sum

最佳实践：滑动窗口只用一个left指针，然后右指针通过遍历完成。

...

...

#### 215. ※ Kth Largest Element in an Array

这道题解法众多，需要了解每一种解法的优点和缺点

* 解法一：topK = heapq.nlargest(k,nums); return topK[-1]
  * 优点：在k值较小时相当高效。
  * 缺点：需要维护一个大小为k的heap存储k个元素，如果k很大会消耗很多空间。
  * 时间复杂度：O(N logK)
  * 空间复杂度：O(K) 即heap占用的空间
* 解法二：快速选择 Quick Select（快速排序的变体）
  * 优点：理论上平均时间复杂度最高
  * 缺点：最坏情况下算法性能会退化到O(N^2)
  * 时间复杂度：O(N), 最坏O(N^2)
  * 空间复杂度：O(logN)
  * 这里注意，快速排序的时间复杂度是O(NlogN)，快速选择是O(N)
* 解法三：python自带排序.sort，即Timsort
  * 优点：平均和最坏时间复杂度都为O(NlogN)
  * 缺点：平均时效性不如快速选择
  * 时间复杂度：O(NlogN)
  * 空间复杂度：O(N)

面试的时候可以先提出heapq和python自带sort的方法，作为参考baseline。然后说优化方法是Quick Select，因为这个方法的平均时间复杂度是O(N)，是这道题的最优解。最后说明最坏情况和原因。

这里先学习一下快速排序的方法：【知识点：快速排序 Quick Sort】

在快速排序中，有两个核心概念：

* Pivot：分界点，左边小于等于pivot，右边大于等于pivot；pivot一开始选择哪里都可以
* Partition：以pivot为界，array会被划分为两部分，这个划分步骤就是partition

具体的过程如下：

```
首先选择一个pivot：
    把pivot拿出来
    其他数字依次和pivot比：
        如果比pivot小，就放在pivot左边
        如果比pivot大，就放在pivot右边
        注意：这里只考虑往pivot左边还是右边放，不用考虑放了以后是否有序

比如：15. 22. 13. 27. 12. 10. 20. 25
选第一个15为pivot：
    13.12.10【15】22.27.20.25
    现在，左半边部分的第一个13和右半边的第一个22是pivot：
        12. 10.【13】
        （15）
        20.【22】27.25
        在此基础上继续：
            10.【12】
           （13）
           （15）
           （20）
           （22）
            25.【27】
            如此可见，最后只剩下10和25还没有被选中为pivot过
            最后将10和25选中为pivot，所有数字都被选中过，排序结束

上面的过程整理一下：
原始数组：15, 22, 13, 27, 12, 10, 20, 25
1. 以 15 为 pivot → [13, 12, 10]  15  [22, 27, 20, 25]
2. 左边以 13 为 pivot → [12, 10] 13 （相当于子问题A）
3. 左边以 12 为 pivot → [10] 12 （子问题A的子问题）
4. 右边以 22 为 pivot → [20] 22 [27, 25] （相当于子问题B）
5. 右边以 27 为 pivot → [25] 27 （子问题B的子问题）
最后组合：10, 12, 13, 15, 20, 22, 25, 27
```

写代码的时候，针对面试，最推荐双指针法，用两个函数partition和quick_sort_recursive分别负责divide和conquer。

快速排序算法的攻略如下：（检索关键词：Quick Sort/quick sort/quick_sort）

```python
import random
class Solution(object):
    def findKthLargest(self, nums, k):
        self._quick_sort_recursive(nums, 0, len(nums)-1)
        return nums[-k]
  
    def _quick_sort_recursive(self, arr, low, high):
        if low < high:
            pivot_index = self._partition(arr, low, high)
            self._quick_sort_recursive(arr, low, pivot_index-1)
            self._quick_sort_recursive(arr, pivot_index+1, high)

    def _partition(self, arr, low, high):
        # partition的作用是把总任务或者大的任务，分成三个子任务：（Divide）
        # pivot自己
        # pivot左边的，代表比pivot小
        # pivot右边的，代表比pivot大
        # 整理完后，return pivot的位置，即找到标杆的位置并返回
        # 因此，partition是物理意义上的重新排列
        # 并且要注意，重新排列是最重要的，否则返回的pivot没有任何意义

        # 那么我们怎么完成Divide呢？
        # 首先， 我们假设有两个左膀右臂的小兵配合我们工作
        # 他们分别是i和j
        # 现在，我们把pivot标杆放在最右边
        # 假设我们在排一个身高队伍
        # 然后助手j，负责从第一个人开始，一个一个检查身高，直到检查到标杆pivot前面的那个人
        # 而助手i，则一开始站在队伍外面，即i=-1的地方。
        # 每当助手j发现一个比pivot矮的人的时候，i就动一动：
        #     i首先向右挪一步，现在i到了0的地方
        #     0这里本身就有一个人，于是i把这里0的地方的人，和j发现的第一个矮个子交换位置
        #     以此类推，每当有一个矮个子被发现，i就往右走一步，然后和j发现的矮个子交换位置
        # 当j走到pivot标杆左边那个位置的时候，做最后一次判断后：
        #     现在可以确定，整个队伍的结构如下：
        #     首先，pivot在最后一个位置
        #     然后，i所处的位置及i左边的位置，是矮个子（比pivot矮的人）
        #     然后，i和pivot中间的，就是比pivot高的人。
        #     那么，把i+1和pivot换一下位置，就可以实现：
        #         这样的分区：【比pivot矮，到i】【pivot】【比pivot高】
        # 最后，完成重新排队后，return标杆pivot的位置

        random_pivot_index = random.randint(low, high)
        arr[random_pivot_index], arr[high] = arr[high], arr[random_pivot_index]
        # 默认方法是把最后一个数字作为pivot
        # 如果想optimize就加上面的随机置换就可以了
        # 这里之所以要随机，是因为考虑最差情况：数组已经排序，每次把pivot设为最后
        # 会导致每一次操作只能减少一个最后数字，然后前面的全部再重排
        # 这样时间效率会降低到O(N^2)
        pivot = arr[high]
        i = low - 1 # i指向最后一个小于等于pivot的元素的位置

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
  
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return i + 1
```

然而，这道题在这里无法通过leetcode测试，是因为leetcode使用了一个含有大量重复元素的testcase，这会导致快速排序效率恶化到O(N^2)。

当然，这道题要找的是最大的K，我们在这里学习一个基于快速排序的新方法：快速选择算法（关键词：Quick Select/quick select/quick_select/快速选择）

在快速排序中，有一个特点至关重要：那就是pivot总是在正确的位置上。因为pivot左边的都比pivot小，pivot右边的又都比pivot大，那么pivot自己一定是在正确的位置上。因此，只要pivot的index和目标索引K相同，那么直接返回这个pivot就求得了本题所寻的K。

算法思路如下：

```
基于快速排序法：
    如果pivot_index==target_index，return pivot
    如果pivot_index > target_index，那么继续找左半边，右半边就不用管了
    如果小于同理
```

如此一来，平均时间复杂度就从O(NlogN)优化到了O(N)：

```python
import random
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        low, high = 0, len(nums) - 1
        target_index = len(nums) - k

        while low < high:
            pivot_index = self._partition(nums, low, high)

            if pivot_index == target_index:
                return nums[pivot_index]
            elif pivot_index > target_index:
                # 如果pivot比target大，说明目标在左边
                # 把high移动到pivot左边，右边丢掉
                high = pivot_index - 1
            else:
                low = pivot_index + 1
  
        # 当循环结束时，low == high，此时唯一剩下的这个数字就是我们要找的
        return nums[low]

    def _partition(self, arr, low, high):
        random_pivot_index = random.randint(low, high)
        arr[random_pivot_index], arr[high] = arr[high], arr[random_pivot_index]

        pivot = arr[high]
        i = low - 1 

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
  
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return i + 1
```

优化后平均O(N), 最坏O(N^2)，这个testcase依然没能通过leetcode设置的一个极端重复测例。对于这种情况，可以返回使用heapq方法，其时间复杂度较为稳定在O(Nlogk)。

又或者，如果是高端面试或者竞赛，依然追求用快速选择法的话，可以引出三路分区法，作为robust的解决方案。三路分区法（3-Way Quicksort）是工业级解决方案：

```python
# 这是一个三路快排的完整实现，您可以作为参考
# findKthLargest 依然是排序后取 nums[-k]
# 但 _quick_sort_recursive 和 _partition 需要换成三路版本

def _quick_sort_3way(self, arr, low, high):
    if low >= high:
        return
    # lt: less than, gt: greater than
    lt, gt = low, high
    i = low + 1
    pivot = arr[low] # 为方便，这里选择第一个元素作基准

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
        else: # arr[i] == pivot
            i += 1
  
    self._quick_sort_3way(arr, low, lt - 1)
    self._quick_sort_3way(arr, gt + 1, high)
```

放到本题中，其实现如下：

```python
import random

class Solution(object):
    def findKthLargest(self, nums, k):
        low, high = 0, len(nums) - 1
        # 目标索引不变
        target_index = len(nums) - k

        # 循环条件可以保持不变，也可以用 while low <= high
        while low <= high:
            # ‼️ 步骤 1: 调用新的三路分区函数，它返回两个边界
            lt, gt = self._partition_3way(nums, low, high)

            # ‼️ 步骤 2: 修改判断逻辑
            if lt <= target_index <= gt:
                # 如果目标索引落在“等于pivot”的区间内，我们直接找到了答案
                return nums[lt]
            elif target_index < lt:
                # 目标在“小于pivot”的区间，更新 high
                high = lt - 1
            else: # target_index > gt
                # 目标在“大于pivot”的区间，更新 low
                low = gt + 1

        return -1 # 理论上，由于逻辑的完备性，循环内部一定会找到答案并返回


    def _partition_3way(self, arr, low, high):
        # 随机化对于避免最坏情况依然至关重要
        rand_idx = random.randint(low, high)
        arr[rand_idx], arr[low] = arr[low], arr[rand_idx]

        pivot = arr[low]
  
        # 初始化三个指针
        lt = low      # 小于 pivot 区间的右边界
        i = low + 1   # 当前考察元素的指针
        gt = high     # 大于 pivot 区间的左边界

        # i 指针从左向右扫描，直到与 gt 相遇
        while i <= gt:
            if arr[i] < pivot:
                # 当前元素 < pivot，把它换到 lt 的位置
                arr[lt], arr[i] = arr[i], arr[lt]
                # lt 和 i 指针都向右移动
                lt += 1
                i += 1
            elif arr[i] > pivot:
                # 当前元素 > pivot，把它换到 gt 的位置
                arr[gt], arr[i] = arr[i], arr[gt]
                # gt 指针向左移动，但 i 指针不动！
                # 因为从 gt 换过来的元素我们还没检查过
                gt -= 1
            else: # arr[i] == pivot
                # 当前元素 == pivot，它属于中间部分，i 指针直接跳过
                i += 1
  
        # 循环结束后，arr[low...lt-1] < pivot
        # arr[lt...gt] == pivot
        # arr[gt+1...high] > pivot
        # 返回“等于pivot”区间的左右边界
        return lt, gt
```

这里要注意，3-Way Quick Select的时间复杂度最坏情况依然是O(N^2)，但是进入最坏情况的可能性非常低，因为3-Way主要解决了以下常见问题：

* 有序数组
* 包含大量重复值的情况

为什么三路分区能解决大量重复值的情况呢？因为三路分区不仅扔掉没用的，对于由留下的部分，也进行了识别筛选，确保留下的都是有用的。比如说，对于大量重复的元素，三路分区直接把重复为同一个值的pivot全部选择。

...

...

...

...

...

...

...

#### 238. ※ Product of Array Ecept Self

Can't use division operation. Must run in O(n) time.

This question is about prefix and suffix product:

```
nums = [... , i , ...]
nums[i]'s product except self = prefix product [i-1] * suffix product [i+1]

eg.  nums = [1, 3, 2, 8, 0, 6]
prefix_ls = [1, 1, 3, 6, 48, 0]
suffix_ls = [0, 0, 0, 0, 6, 0]

so actually you only need one list to maintain, 
and one pointer to temprarily store the info of prefix or suffix
```

more specifically:

```python
class Solution(object):
    def productExceptSelf(self, nums):
        prefix = 1
        ls = []
        for i in range(len(nums)):
            ls.append(prefix)
            prefix *= nums[i]
        suffix = 1
        for i in range(len(nums)-1,-1,-1):
            ls[i] = ls[i] * suffix
            suffix *= nums[i]
        return ls
```

...

#### 242. Valid Anagram

First Meet: Misunderstood the meaning of "Anagram": Anagram means use same characters. I sorted two strings, and compare two lists. But this will cause O(NlogN) in time complexity.

Best Solution: Time Complexity O(N)

```python
if length of nums1 and nums2 are different: return False
count = [0] * 26 # Set a list
for ch1, ch2 in zip(nums1, nums2):
    count[ord(ch1) - ord('a')] += 1
    count[ord(ch2) - ord('a')] -= 1
return all(c == 0 for c in count) # determine if it is True for all element
```

## No.251 - No.300

...

#### 271. ※ Encode and Decode Strings

First Meet: Use a very special string to join and split.

But, to avoid any possibility that the string is contained in the original string, the length of the string should be considered. eg.  `word => '#4#word'`

...

...

...

#### 274. H-Index

这道题本身有一个特别的计算方法，可以记住：

* 逆序排列
* 记录list[i]>=i+1的数量，即h-index

...

...

...

...

...

## No.301 - No.350

...

#### 347. ※ Top K Frequent Elements

It is easy to make a hashmap and store all the frequency. The hard point is to extract k largest frequent elements.

```
use a hashmap to calculate each number's frequency
transfer the hashmap into a list, element is tuple (number, frequency)
use heapq.nlargest(k, list, key=lambda x:x[1]) # use frequency to sort
retrieve the number of heapq
```

...

## No.351 - No.400

...

...

...

...

...

#### 380. Insert Delete GetRandom O(1) (Data Structure)

这是一道【数据结构题】，即手写Set的数据结构

Set的关键点在于O(1)的平均时间复杂度

本题另一个关键点是如何实现O(1)的时间复杂度的getRandom，解法就是用list来存储，然后不进行删除操作：在删除的时候把要删除的内容和末尾的元素互换。

注意：本题不要求list也保持顺序，所以才能实现O(1)的随机取数操作。

本题关键点：

* hashmap存储value和list位置，这样可以根据key（即value）找到list对应位置
* list中存储value，这样就可以根据value找到对应的hashmap
* 使用del命令删除hashmap中的元素
* 使用.pop()删除list的最后一个元素

注意random用法：（关键词：random用法/随机数/随机生成/随机选择）

```
import random
random.randint(0, len(list)-1) # random.randint用的是闭区间, 生成的是整数
random.uniform(1.0, 5.5) # 生成闭区间内的随机小数
random.random() # 生成[0.0, 1.0)之间的浮点数
random.choice(sequence) # 在一个list，tuple或string中随机选一个元素
```

...

...

...

...

#### 392. Is Subsequence

初见看错题了，这道题不是 `.find()`的那种包含关系（子字符串），而是子序列打散后只要按顺序存在于主序列中，就认为是一种包含关系。

这里注意，我在写代码的时候习惯随时写随时优化（短路求值，一旦不满足就提前break或者return），但是Gemini提醒我：过早的优化是万恶之源（Premature optimization is the root of all evil），并推荐我分离循环和判断（最佳实践）。

...

...

## No.401 - No.450

...

...

...

#### 424. ※※ Longest Repeating Character Replacement

这道题巧妙绝伦，建议全文背诵！

注意：所有只有一种单调类型的字母的题都可以考虑是否用一个26个字母组成的数组来单调维护。这样可以避免频繁修改hashmap导致的效率低下。

核心思路：

* 当且仅当（window长度  - 窗口内出现的最多的字符的数量  <=  K）的时候，一个窗口是valid的。
* 维护一个非常巧妙的max_freq，当（right - left + 1 - max_freq > k）时，窗口收缩。这里要注意，窗口收缩的时候，max_freq不需要同步收缩，因为我们求的是最长窗口。换句话说，对于一个小于最高纪录的窗口，我们没有必要计算其精确的max_freq。一个窗口能够达到的最大有效长度，就是（max_freq + k）。要想让这个窗口变得更长，唯一的办法就是提升max_freq。

你可以把这个算法想象成一个 **只想着破纪录的跳高运动员** ：

* `longest` 是他已经跳过的最高高度。
* `max_freq` 是他的“力量值”。
* 他不会在每次助跑失败（窗口收缩）后都去重新评估自己的全部身体状态（重新计算 `max_freq`）。
* 他只会一直尝试，直到某一次他感觉“力量更足了”（`max_freq` 增加了），他才会去尝试冲击一个新的高度（让窗口长度 `longest` 增加）。

算法思路：

```
设置一个count，统计各个字符出现的频率
设置一个max_freq，记录在所有有效窗口中出现过的历史最高频率 ※
设置一个left指针
设置一个longest变量，存放最大
for right in range(len(s)):
    将right指针对应的字符加入窗口
    直接更新最大频率max_freq，这是一个乐观锁，不用回退
    如果窗口长度 - max_freq > k:
        从左边移出一个字符，即字典中对应字符的频率 -1
        同时left指针向右移动一格
    更新longest；每一轮都直接更新，不用担心错误更新
return longest
```

最佳实践：

```python
class Solution(object):
    def characterReplacement(self, s, k):
        count = {}  # 使用字典来存储窗口内各字符的频率，更通用
        left = 0
        longest = 0
        max_freq = 0  # 记录窗口内最频繁字符的出现次数

        for right in range(len(s)):
            # 1. 扩展窗口：将 right 指针的字符加入窗口
            char = s[right]
            count[char] = count.get(char, 0) + 1
    
            # 2. 更新窗口内的最大频率
            max_freq = max(max_freq, count[char])

            # 3. 检查有效性并收缩窗口
            # window_len = right - left + 1
            # replacements_needed = window_len - max_freq
            # 如果需要替换的字符数 > k，说明窗口无效，需要收缩
            if (right - left + 1) - max_freq > k:
                # 从左边移出一个字符
                left_char = s[left]
                count[left_char] -= 1
                left += 1
    
            # 4. 更新结果
            # 窗口在收缩后，长度不一定会减小，因为一进一出，长度可能不变
            # 所以我们可以在每次循环都计算当前有效窗口的长度
            longest = max(longest, right - left + 1)

        return longest
```

为什么不需要显示检查窗口是否valid呢？因为在检查有效性并收缩窗口后， 窗口的长度减少了一，因此它不可能比longest更长。这个解法是一个非常高级的滑动窗口解法，美丽无比。

...

...

...

...

## No.451 - No.500

...

#### 454. 4Sum II

初见：用18.4Sum的第二种解法。

First Meet: Use 18. 4Sum's Second Solution.

4Sum II is much easier than 18. 4Sum because in this question, four numbers are from four lists, so there is no need to eliminate duplicates:

```
traverse num1, num2 to get a hashmap
traverse num3, num4 to count if (num3 + num4) is in hashmap
```

.
