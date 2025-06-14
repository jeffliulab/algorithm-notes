##################################################################
########### 第一部分 Array / String 共24道题 ######################
##################################################################

#####################################################################
# (1) 88. Merge Sorted Array 
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None
        """

        # 初始化三个指针
        p1 = m - 1             # 指向 nums1 中最后一个有效元素
        p2 = n - 1             # 指向 nums2 中最后一个元素
        end = m + n - 1        # 指向 nums1 的最后一个位置（合并的尾部）

        # 从后往前进行合并，只要 nums2 还有元素就继续
        while p2 >= 0:
            # 情况一：nums1 还有值，并且当前 nums1 的值更大
            if p1 >= 0 and nums1[p1] > nums2[p2]: # 这里用p1 >=0 是因为当p1 < 0的时候直接比较会导致错误，需要做的其实就是继续把nums2得数组放进来
                nums1[end] = nums1[p1]  # 把较大值放在 nums1 的末尾
                p1 -= 1                # 移动指针
            else:
                # 情况二：nums2 的值更大，或 nums1 已用完（p1 < 0）
                nums1[end] = nums2[p2]  # 拷贝 nums2 的当前值
                p2 -= 1                # 移动指针
            end -= 1                   # 每次都要填一个位置，end 向前移动

        # 注意：当 p2 < 0 时，说明 nums2 合并完了，循环会自动停止
        # 若 p1 剩下还没合并的元素，它们本来就在 nums1 前面，位置已经对了，不需要动

#####################################################################
# (2) 27. Remove Element
"""
注意: 这道题不关心最后剩下的部分, 所以不用考虑最后再设置空值
"""
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        k = 0  # 指向写入新数组的位置

        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        
        # 不需要强行写0，因为题目说后面部分不关心
        return k  # 返回新数组的长度


#####################################################################
# (3) 26. Remove Duplicates from Sorted Array

"""
和上道题一样，最后剩下的不重要。
"""
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        k = 1  # 第一个元素一定是唯一的，从第二个位置开始写
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]
                k += 1
        return k

#####################################################################
# (4) 80. Remove Duplicates from Sorted Array II

class Solution(object):
    def removeDuplicates(self, nums):
        """
        删除有序数组中的重复项 II: 保留最多出现两次的元素
        
        算法思路：
        - 使用双指针技巧，一个指针(i)遍历原数组，另一个指针(k)维护新数组
        - 通过巧妙的条件判断，无需额外计数器即可限制每个元素最多出现两次
        
        时间复杂度: O(n) - 只需遍历一次数组
        空间复杂度: O(1) - 原地修改，不使用额外空间
        
        :type nums: List[int] - 输入的有序数组
        :rtype: int - 返回新数组的长度
        """
        # 如果数组为空，直接返回0
        if not nums:
            return 0
            
        # k是新数组的指针，指向下一个要填入的位置
        # 初始值为1是因为第一个元素必然保留（位置0）
        k = 1
        
        # 从位置1开始遍历原数组
        for i in range(1, len(nums)):
            # 判断当前元素是否应该保留：
            # 1. 当k=1时，第二个元素总是可以保留（任何元素至少可以出现一次）
            # 2. 或者当前元素与新数组中倒数第二个元素不同
            #    (这保证了相同元素最多只会出现两次)
            # 这里一个容易错的点是nums[k-2]容易写错
            # 这里要检查的一定是新的array, 因为新的array会修改元素内容，导致遍历指针往前数两个的内容改变，从而造成判断不准确
            if k == 1 or nums[i] != nums[k-2]:
                # 将当前元素复制到新数组的位置k
                nums[k] = nums[i]
                # 新数组指针前进
                k += 1
                
        # 返回新数组的长度
        return k
    
#####################################################################
# (5) 169. Majority Element1

def majorityElement(nums):
    """
    查找数组中的多数元素（出现次数超过 ⌊n/2⌋ 的元素）
    
    参数:
        nums: 整数数组
    返回:
        多数元素
    
    算法: 摩尔投票算法 (Boyer-Moore Voting Algorithm)
    时间复杂度: O(n) - 只需要遍历数组一次
    空间复杂度: O(1) - 只使用了常数额外空间
    """
    
    # 候选元素
    candidate = None
    # 计数器
    count = 0
    
    # 第一遍遍历: 找到可能的多数元素
    for num in nums:
        # 如果计数器为0，选择当前元素作为新的候选元素
        if count == 0:
            candidate = num
            count = 1
        # 如果当前元素等于候选元素，计数器加1
        elif candidate == num:
            count += 1
        # 如果当前元素不等于候选元素，计数器减1
        else:
            count -= 1
    
    """
    核心思想解释:
    
    1. 摩尔投票算法基于"抵消"的概念
    2. 把每个元素看作+1或-1票:
       - 如果元素等于候选元素，则投+1票
       - 如果元素不等于候选元素，则投-1票
    3. 当计数器为0时，之前的抵消已完成，选择一个新的候选元素
    4. 由于多数元素出现次数 > ⌊n/2⌋，所以它的"票数"一定会比其他所有元素的"票数"总和还多
    5. 最终留下的候选元素必然是多数元素
    
    形象理解:
    - 想象不同元素代表不同阵营的士兵
    - 不同阵营的士兵两两厮杀(抵消)
    - 由于多数元素的"士兵"数量超过总数的一半
    - 因此战斗结束后，战场上剩下的必然是多数元素的士兵
    """
    
    # 因为题目保证多数元素一定存在，所以直接返回候选元素
    # 如果题目不保证多数元素存在，还需要进行第二遍遍历来确认candidate确实出现超过⌊n/2⌋次
    return candidate

#####################################################################
# (6) 189. Rotate Array

def rotate(nums, k):
    """
    旋转数组: 将数组中的元素向右移动k个位置.
    
    使用三次翻转法实现:
    1. 先将整个数组翻转
    2. 再将前k个元素翻转
    3. 最后将剩余元素翻转
    
    例如: 对于数组[1,2,3,4,5,6,7], k=3:
    - 整体翻转后: [7,6,5,4,3,2,1]
    - 前k=3个元素翻转后: [5,6,7,4,3,2,1]
    - 剩余元素翻转后: [5,6,7,1,2,3,4]
    
    时间复杂度: O(n) - 其中n是数组的长度
                    - 我们最多遍历数组3次
    空间复杂度: O(1) - 只使用常数额外空间
                    - 只需要几个临时变量
    
    参数:
        nums: List[int] - 需要旋转的整数数组
        k: int - 向右旋转的步数, 非负整数
    
    返回:
        List[int] - 旋转后的数组
    """
    n = len(nums)
    k %= n  # 处理k大于数组长度的情况, 因为旋转n次等于没旋转
    
    if k == 0:  # 如果k是n的倍数, 等于不旋转, 直接返回
        return nums
    
    # 辅助函数: 翻转数组的指定部分
    # 注意: 这里不能用切片
    # 因为切片后会创建新的数组，而原题目要求in-place原地修改并且只占用O(1) extra space
    def reverse(arr, start, end):
        """
        翻转数组中从start到end的元素
        
        参数:
            arr: List[int] - 要操作的数组
            start: int - 起始索引(包含)
            end: int - 结束索引(包含)
        """
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]  # 交换元素
            start += 1
            end -= 1
    
    # 三次翻转法
    reverse(nums, 0, n-1)    # 步骤1: 翻转整个数组
    reverse(nums, 0, k-1)    # 步骤2: 翻转前k个元素
    reverse(nums, k, n-1)    # 步骤3: 翻转剩余元素
    
    return nums

#####################################################################
# (7) 121. Best Time to Buy and Sell Stock

class Solution(object):
    def maxProfit(self, prices):
        """
        寻找买卖股票的最大利润
        
        讲题思路：
        1. 这道题本质上是求价格数组中的最大差值（后面的元素减前面的元素）
        2. 关键点在于，我们可以通过记录历史最低价格，并计算当前价格与之差值来找到最大利润
        3. 只需遍历一次数组，同时更新两个值：
           - 历史最低价格
           - 最大可能利润
        
        算法策略：一次遍历
        - 对于第i天的价格, 假设在这天卖出
        - 那么应该在前i-1天价格最低的那天买入
        - 通过跟踪到目前为止的最低价格, 我们可以在O(n)时间内找到最优解
        
        参数:
            prices: List[int] - 股票在各天的价格数组
        返回:
            int - 可获得的最大利润, 如果无法获利则返回0
        
        时间复杂度: O(n) - 只需遍历价格数组一次
        空间复杂度: O(1) - 只使用常数额外空间
        """
        # 边界条件检查：空数组或只有一天价格无法交易
        if not prices or len(prices) < 2:
            return 0
        
        # 初始化变量
        min_price = prices[0]  # 记录到目前为止的最低价格
        max_profit = 0        # 记录最大可能利润
        
        # 遍历价格数组
        for price in prices:
            # 计算当前价格卖出的最大利润（当前价格减去历史最低价格）
            max_profit = max(max_profit, price - min_price)
            
            # 更新历史最低价格
            min_price = min(min_price, price)
        
        return max_profit

#####################################################################
# (8) 122. Best Time to Buy and Sell Stock II

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int

        Greedy解法: 忽视全局, 只考虑隔日利润
        这道题可能出的有点无脑, 可以当日买卖而且也不考虑手续费
        那么题目就可以无脑转变为: 只考虑今天和明天, 如果有得赚就今天买明天卖, 没得赚就算了
        """
        profit=0
        for i in range(1,len(prices)):
            if prices[i]>prices[i-1]:
                profit+=prices[i]-prices[i-1]
        return profit
    
# 这道题稍微优化一下: 每次交易有手续费, 所以需要尽可能减少交易次数
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int

        该解法仅在上升趋势结束时一次性交易
        """
        if not prices or len(prices) < 2:
            return 0
        
        max_profit = 0
        buy_index = 0  # 当前买入点的索引
        
        for i in range(1, len(prices)):
            # 如果当前是下降趋势，更新买入点
            if prices[i] < prices[i-1]:
                # 如果之前有上涨区间，计算并添加利润
                if i-1 > buy_index:
                    max_profit += prices[i-1] - prices[buy_index]
                # 更新新的买入点
                buy_index = i
        
        # 处理最后一段上涨区间（如果存在）
        if buy_index < len(prices) - 1:
            max_profit += prices[-1] - prices[buy_index]
            
        return max_profit

#####################################################################
# (9) 55. Jump Game 

class Solution(object):
   def canJump(self, nums):
       """
       :type nums: List[int]
       :rtype: bool
       
       问题：判断是否能从数组第一个位置跳到最后一个位置
       算法：贪心策略，维护一个"最远可达位置"变量
       因为不在乎可能的路径, 只在乎最远可达距离, 所以属于一种贪心策略(非最优路径)

       时间复杂度: O(n) - 只需要遍历数组一次
       空间复杂度: O(1) - 只使用常数级别的额外空间
       """
       # 初始化最远可达位置为0（即只能到达起点）
       maxReach = 0
       
       # 遍历数组中的每个位置
       for i in range(len(nums)):
           # 如果当前位置已经超过了能到达的最远位置，说明无法继续前进
           if i > maxReach:
               return False
           
           # 更新最远可达位置
           # 当前位置i加上该位置允许的最大跳跃长度nums[i]，与之前的最远可达位置比较取较大值
           maxReach = max(i + nums[i], maxReach)
           
           # 如果最远可达位置已经能够到达或超过最后一个索引，可以提前返回成功
           if maxReach >= len(nums) - 1:
               return True
       
       # 遍历结束后的最终检查：能否到达最后一个位置
       # 实际上，由于上面的提前返回，代码执行到这里时一定是可以到达终点的
       # 保留这行是为了确保逻辑完整性
       return maxReach >= len(nums) - 1
   

#####################################################################
# (10) 45. Jump Game II

def jump(nums):
   """
   计算从数组第一个位置跳到最后一个位置所需的最小跳跃次数.
   
   算法: 贪心策略, 通过不断扩展可达范围的边界来最小化跳跃次数.
   
   工作原理:
   1. 维护当前跳跃可达的最远边界(current_max)
   2. 在移动过程中不断计算下一跳可达的最远边界(next_max)
   3. 只有在到达当前边界时才执行跳跃, 并扩展边界
   4. 这个算法只记录跳跃次数, 不记录具体的跳跃路径
   
   参数:
       nums: 整数数组, 每个元素表示在该位置可以跳跃的最大长度
   返回:
       到达最后一个位置的最小跳跃次数
   """
   # 如果数组长度为1或更小, 已经在终点位置, 无需跳跃
   if len(nums) <= 1:
       return 0
   
   jumps = 0          # 记录跳跃次数
   current_max = 0    # 当前跳跃能到达的最远边界
   next_max = 0       # 下一跳能到达的最远边界
   
   # 遍历数组(注意只需要考虑到倒数第二个位置)
   for i in range(len(nums) - 1):
       # 更新下一跳可能到达的最远边界
       # 在移动过程中不断探索更远的可能性
       next_max = max(next_max, i + nums[i])
       
       # 如果已经到达当前跳跃能到达的边界
       # 这意味着必须进行下一次跳跃才能继续前进
       if i == current_max:
           # 跳跃次数加1
           jumps += 1
           
           # 更新当前能到达的最远边界为下一步能到达的最远边界
           # 这相当于扩展了可达范围
           current_max = next_max
           
           # 如果更新后的边界已经能到达或超过终点, 可以提前返回结果
           if current_max >= len(nums) - 1:
               return jumps
   
   # 循环结束后返回总跳跃次数
   return jumps

#####################################################################
# (11) 274. H-Index

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        
        算法名称: 普通排序法计算h-index

        时间复杂度: O(n log n) - 主要来自排序操作
        空间复杂度: O(n) 或 O(1) - 取决于排序算法的实现
        """
        # 对论文引用次数进行降序排序：
        # 这样排在前面的论文引用次数更多，便于计算 h-index
        # 排序后index索引将代表第几篇论文, 遍历的时候数到第几个就相当于一共有多少篇
        # 又因为是倒序, 所以索引更大的对应的那个值(被引用次数)要把索引更小的部分计算进去
        # 比如在统计引用次数4的时候, 所有引用次数大于等于4的论文都要被统计进来, 所以这里才要倒序
        ls = sorted(citations, reverse=True)
        
        # 初始化 h-index 为 0
        h = 0
        
        # 遍历排序后的引用次数数组
        for i in range(len(ls)):
            # 如果当前论文的引用次数 >= 当前位置 + 1
            # 则说明这篇论文满足 h-index 的条件
            if ls[i] >= i+1:
                h += 1
            # 当遇到不满足条件的论文时，不更新 h 值
            # 由于数组已降序排序，之后的论文引用次数只会更少
        
        # 返回最终计算出的 h-index 值
        return h
    

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        
        算法名称: 计数排序法计算h-index
        
        h-index的定义: 若一位科研人员的h-index为h，则表示他有h篇论文分别被引用了至少h次。
        
        算法思路:
        1. 创建一个计数数组，索引表示引用次数，值表示该引用次数的论文数量
        2. 从最大可能的h值(即论文总数)开始向下检查，找到第一个满足条件的值
        
        具体实现:
        - 由于h-index最大为论文总数n，所以只需要n+1大小的数组
        - 所有引用次数超过n的论文都计入counts[n]中(因为h不可能大于n)
        - 从高到低遍历可能的h值，累计引用次数大于等于当前值的论文总数
        - 当"引用次数≥h的论文数量"首次大于等于h时，找到了最大的h-index
        
        时间复杂度: O(n) - 只需要遍历一次输入数组和一次计数数组
        空间复杂度: O(n) - 需要一个额外的计数数组
        """
        n = len(citations)
        
        # 创建计数数组，大小为n+1(包含引用次数为0到n的所有可能值)
        counts = [0] * (n + 1)
        
        # 遍历每篇论文的引用次数并计数
        # 引用次数超过n的论文计入counts[n]，因为h-index不可能超过论文总数
        for c in citations:
            if c >= n:
                counts[n] += 1
            else:
                counts[c] += 1
        
        # 从高到低检查每个可能的h值
        # total表示引用次数大于等于h的论文总数
        total = 0
        
        for i in range(n, -1, -1):
            # 累加引用次数为i的论文到total中
            # 因为引用5的文章在计算到引用大于等于4的这个指标时候也会被计算在内
            # 所以这里需要累加(本题的核心关键点)
            total += counts[i]
            
            # h-index的核心判断:
            # 如果有total篇论文的引用次数大于等于i，且total大于等于i
            # 则找到了符合定义的h-index
            if total >= i:
                return i
        
        # 如果没有满足条件的h值(即所有论文引用次数都为0)
        return 0


#####################################################################
# (12) 380. Insert Delete GetRandom O(1)

import random

class RandomizedSet(object):

    def __init__(self):
        # 使用一个字典 val_to_index 存储元素 -> 索引
        # 使用一个列表 values 存储当前的所有元素（支持 O(1) getRandom）
        self.val_to_index = {}
        self.values = []

    def insert(self, val):
        """
        插入元素 val。如果 val 不在集合中，则插入并返回 True, 否则返回 False。
        时间复杂度: O(1)
        """
        if val in self.val_to_index:
            return False  # 元素已存在，不能插入
        self.val_to_index[val] = len(self.values)  # 记录 val 在数组中的索引
        self.values.append(val)  # 添加到数组末尾
        return True

    def remove(self, val):
        """
        删除元素 val。如果 val 在集合中，则删除并返回 True, 否则返回 False。
        时间复杂度: O(1) 平均时间
        """
        if val not in self.val_to_index:
            return False  # 元素不存在，不能删除

        # 获取要删除元素的索引
        idx_to_remove = self.val_to_index[val]
        last_val = self.values[-1]

        # 将数组最后一个元素移动到要删除的位置
        self.values[idx_to_remove] = last_val
        self.val_to_index[last_val] = idx_to_remove

        # 删除最后一个元素（现在是重复的）
        self.values.pop()
        del self.val_to_index[val]

        # 不使用 None，是为了保持数组紧凑且不影响 getRandom 的 O(1) 实现
        return True

    def getRandom(self):
        """
        从当前集合中随机返回一个元素。
        时间复杂度: O(1)
        """
        return random.choice(self.values)



#####################################################################
# (13) 238. Product of Array Except Self

class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        
        解法策略：使用前缀积和后缀积，完成 O(n) 时间、O(1) 空间构建（输出数组不计入空间复杂度）。
        """

        # 解法说明（现实意义）：
        # 本题禁止使用除法，除了为了增加算法难度外，这个限制在现实中也有工程意义：
        # 1. 精度问题：除法涉及浮点运算，容易因精度误差导致错误结果，尤其在金融/科学计算中更明显。
        # 2. 安全性：某些系统（如密码学或安全领域）中，除法操作可能泄露信息或引发边界错误（如除0）。
        # 3. 性能成本：在嵌入式系统、图形硬件（GPU）等平台中，除法远比乘法慢，影响运行效率。
        # 4. 多个0处理困难：若数组中有多个0，除法方案逻辑复杂化（要特判0，逻辑分支多，代码不优雅）。        
        # 因此，面试中出此题不仅考察你的算法思维，也考查你面对限制条件时的建模和应变能力。

        n = len(nums)
        res = [1] * n  # 初始化结果数组，每个位置默认乘积为1

        # 第一遍：构建每个位置左侧所有元素的乘积
        left = 1
        for i in range(n):
            res[i] = left
            left *= nums[i]

        # 第二遍：从右往左乘上右侧所有元素的乘积
        right = 1
        for i in range(n - 1, -1, -1):
            res[i] *= right
            right *= nums[i]

        return res

#####################################################################
# (14) 134. Gas Station

class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """

        # 如果总gas小于总cost，那么可以直接判定没有任何一个地方出发可以跑完一圈
        if sum(gas) < sum(cost):
            return -1
        
        # 在确定能跑完一圈的情况下，我们只需要排除不能作为起点的选项，就可以找到合法的出发点
        tank = 0
        start = 0

        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            if tank < 0:
                start = i + 1 # 排除掉当前的出发点选项
                tank = 0
        
        return start
    
#####################################################################
# (15) 135. Candy

class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int

        策略: 本题只涉及邻居对比, 不涉及全局路径调整, greedy显然就足够了
        """
        candy = len(ratings) * [1]

        # 从左往右，保证右边的高分学生总是比左边的低分学生多
        for i in range(1,len(ratings)):
            if ratings[i] > ratings[i-1]:
                candy[i] = candy[i-1] + 1
        
        # 从右往左，保证左边的高分学生总是比右边的低分学生多
        for i in range(len(ratings)-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                candy[i]  = max(candy[i], candy[i+1] + 1) # 关键点，保留了第一轮此的发糖结果

        return sum(candy)
        