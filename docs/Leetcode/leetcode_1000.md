# LeetCode 500 - 1000

🚫表示过于简单，不易展示。

## 501 - 550

...

## 551 - 600

...

...

...

#### 560. ※ Subarray Sum Equals K

知识点：前缀和/prefix sum

这道题的关键：找到subarray，说明不能对原本的array进行排序操作。因此对于完全无序的array，需要使用prefix sum。

prefix sum的核心思想是记账和查账：

* nums: 每天的收入（正数）或支出（负数）
* current_sum: 当前的前缀和：到今天为止，账户总余额
* k: 想要查找的某笔交易金额，比如你想找到某段时间，净收入正好是10元
* prefix_sum_freq: 一个完美的记账本，记录了历史上每一天结束的时候，你的账户余额是多少，并且这个余额出现过几次

关键公式：第i-j天的净收入 = 第j天的总余额 - 第i-1天的总余额

即： sum(i,j) = prefix_sum(j) - prefix_sum(i-1)

**前缀和字典哈希表的结构：{前缀和：出现次数}**

```
设立一个hashmap:{前缀和:出现次数}
初始化hashmap = {0: 1}，这一步非常关键，保证里程计/账本有开头，即0出现了一次
目的：查找和为k的连续区段
traverse 列表：
    如果（当前总余额-k）在hashmap中出现过：
        查找其出现次数count
        总计数器 += count
    计算当前总余额，记录在前缀和hashmap中
```

为什么不会重复？不会多算？因为你可以想象我们开着车前进，然后不断记录总里程。在每个时刻，我们记录的都是当下的总里程，找的都是以当下的地点为结束、往前数区段长度为k的子数组。换句话说，当遍历到nums[i]的时候，我们能找到的所有的子数组，他们的终点必然都是nums[i]，因此不存在重复的问题。

在每一个当下时刻，我们都以当前位置为固定的终点，探寻有多少个不同的大于等于0的起点可以满足：从该起点到当下位置的区段长度为K的这个条件。

...

...

...

...

...

...

#### 567. Permutation in String

这道题很简单，就是要注意在删除hashmap的字母对应的频率的时候，如果频率降低到0，一定要记得把key也删掉，否则后面的判断两个hashmap是否相等就没有意义了。

...

...

...

...

...

## 601 - 650

...

## 651 - 700

...

#### 692.※ Top K Frequent Words

这道题关键点在于构建min-heap以及理解python中的排序机制。

这道题的关键在于将(-freq, word)以tuple形式存入heap中，这样在排序的时候，会先排列-freq，然后排列word。

...

## 701 - 750

...

...

...

#### 739. Daily Temperatures

单调栈教学题，非常经典与优雅的思维方式。

...

## 751 - 800

...

...

#### 771.🚫Jewels and Stones

🚫过于简单，不易展示。

...

...

...

...

## 801 - 850

...

...

...

...

...

...

# 851 - 900

...

...

...

#### 853. ※ Car Fleet

这本质上就是一个单调递增stack的问题，把到达时间作为value算出来，然后维护一个后面的值必须比前面的值小的stack。因为如果前面的value大于后面的，那么前面的车就必须合并到后面的车队中。

这里要注意，处理的时候要按照位置信息来处理。因为原始的position是无序的。所以要把位置信息和速度打包为(position, speed)，然后按照position从大到小排序，即从离终点最近的地方开始看。（单调递增栈）

如果从初始出发点开始看的话，则需要维护一个单调递减栈，并调整top节点的数值，容易出错。所以这道题比较适合用单调递增栈，然后从终点开始往回看。

不过这道题的最佳实践是greedy贪心。但是本质上贪心解法就是把这个单调递增栈的使用进行了优化。

mono stack solution:

```python
cars = sorted(zip(position, speed), key=lambda x: -x[0])

stack = []
for pos, spd in cars: 
    arrival_time = float(target - pos) / spd
    if not stack or arrival_time > stack[-1]:
        stack.append(arrival_time)

return len(stack)
```

greedy其实就是在此基础上，把stack改为一个单独的变量，比如leader_time。为什么能这么改呢？因为这道题我们只用到了最顶上的值，而不关注top下面的值。**换句话说：因为不需要追溯历史状态，只依赖于最近的/最重要的一个历史状态，所以可以用一个变量来记录这个状态，从而把这道题转换为一个贪心问题。**

```python
cars = sorted(zip(position, speed), key=lambda x: -x[0])

leader_time = 0
count = 0
for pos, spd in cars: 
    arrival_time = float(target - pos) / spd
    if arrival_time > leader_time:
        count += 1
        leader_time = arrival_time

return count
```

...

...

## 901 - 950

...

...

...

...

...

...

## 951 - 1000

...

...

...

...

...
