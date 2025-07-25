# Leetcode 1001 - 1500

## 1001 - 1050

...

...

#### 1011. Capacity To Ship Packages

这道题是875 koko吃香蕉的变体。注意，设置left最小边界的时候，一定要注意，不能设置为1，否则会导致错误。为什么呢？假如要运送的包裹是2，那么你的船载重是1的话，根本就无法运送，但是计算却会继续进行，导致最终拿到一个错误的答案。所以最低的left左边界要设置成最重的包裹重量，确保每一个包裹都起码能被运走！

...

...

...

...

#### 1046. Last Stone Weight

最大堆，水题一个。

...

...

...

...

## 1051 - 1100

...

...

#### 1091. Shortest Path in Binary Matrix

常规的BFS题

...

## 1101 - 1150

...

...

...

## 1151 - 1200

...

...

...

## 1201 - 1250

...

...

...

## 1251 - 1300

...

...

...

## 1301 - 1350

...

...

#### 1343. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold

这道题比较简单，算是一个简易版本的滑动窗口（不需要调整窗口大小）

...

## 1351 - 1400

...

...

...

## 1401 - 1450

...

...

#### 1448. Count Good Nodes in Binary Tree

这道题的意思是：对于任何一个非root的节点X，如果在从根节点到该节点X（包括该节点本身）的路径上，没有任何一个节点的值大于当前节点的值，那么这个节点就被认为是一个好的节点（good node）。找到good nodes的数量并return。

普通dfs加一个cur_max参数就可以。做这道题的时候才发现自己在leetcode一直用的是python2，所以nonlocal用不了。

...

## 1451 - 1500

...

...

#### 1462. Course Schedule IV

这道题本质上是一道transitive closure问题。

这道题最简洁的解法是用Floyd-Warshall算法，借着这道题我们一起来学习一下这个巧妙的算法：

```
1、初始化一个邻接矩阵

2、把题目中给的先修关系填充进去

3、floyd warshall算法，然后这张表就填好了

算法内容是:
for k in range(nodes):
    for i in range(nodes):
	for j in range(nodes):
	    if [i][k] is valid and [k][j] is valid:
		[i][j] = valid

4、想要知道任意(u,v)的关系，直接查表
```

**目标：从“直达”到“可换乘”**

想象一下，你是一家航空公司的路线规划师。

* **第1、2步之后** ：你拿到了一张表（`is_prereq` 矩阵），上面只记录了所有**直飞**的航班。例如，`is_prereq[北京][上海] = True` 表示有从北京到上海的直飞航班。但此时，如果你想查“从北京能否到三亚”，而航线是“北京->上海->三亚”，这张表会告诉你 `is_prereq[北京][三亚] = False`，因为它只知道直飞。
* **第3步的目标** ：我们要把这张“直飞表”升级成一张“ **可换乘表** ”。这张新表需要能回答“从城市 `i` 到城市 `j` 是否存在一条由任意多次换乘组成的航线？”

**核心思想：逐一开放“中转站”**

Floyd-Warshall 算法想出了一个天才的主意：我们不一次性考虑所有复杂的换乘，而是 **逐个地把城市开放为“中转站”** 。

这正是那三层 `for` 循环的含义：

**Python**

```
for k in range(numCourses):      # 第1层：我们本次开放的中转站是 k
    for i in range(numCourses):  # 第2层：检查每一个可能的起点 i
        for j in range(numCourses): # 第3层：检查每一个可能的终点 j
            # ... 在这里施展“魔法” ...
```

* **`for k in range(numCourses)`** ： 这是算法的 **引擎** 。它的意思是：“我们来按顺序检查，如果允许通过 `城市0` 作为中转站，我们的航线表会增加哪些线路？……如果允许通过 `城市1` 作为中转站，又会怎样？……直到所有城市都作为中转站被考虑过。”

**“魔法”详解：一步一步看懂 `k` 的作用**

让我们用一个简单的例子来走一遍流程：`A -> B -> C`

* **课程/城市** : A(0), B(1), C(2)
* **先修关系** : `[A, B]` 和 `[B, C]`

**初始状态 (第1、2步之后)**

我们的 `is_prereq` 矩阵看起来是这样的（只显示True的格子）：

* `is_prereq[A][B] = True`
* `is_prereq[B][C] = True`
  此时，`is_prereq[A][C]` 是 `False`，因为没有从A到C的直达路径。

**第3步开始：**

**1. 当 `k = 0` (也就是城市 A) 时：**
算法会检查所有 `(i, j)` 对。我们来看一下 `(A, C)` 这个例子。
`is_prereq[A][C] = is_prereq[A][C] or (is_prereq[A][A] and is_prereq[A][C])`
这显然不会改变什么。实际上，因为A是起点，没有航线指向它，所以把A作为中转站不会创造任何新的航线。 **矩阵不变** 。

**2. 当 `k = 1` (也就是城市 B) 时：<-- 见证奇迹的时刻**
现在，我们允许所有航线都可以通过B来中转。算法会继续检查所有 `(i, j)` 对。当它检查到 `i=A`, `j=C` 时，会执行下面的逻辑：

`is_prereq[A][C] = is_prereq[A][C] or (is_prereq[A][B] and is_prereq[B][C])`

我们把已知的值代入进去：
`is_prereq[A][C] = False or (True and True)`
`is_prereq[A][C] = False or True`
`is_prereq[A][C] = True`

**看！** 因为我们现在允许通过城市B (`k=1`)，算法发现了“从A到B” (`is_prereq[A][B]`) 和“从B到C” (`is_prereq[B][C]`) 这两条已知的路径。于是，它成功地**推断**出了一条新的间接路径“从A可以到达C”，并立刻在我们的表中将 `is_prereq[A][C]` 更新为 `True`！

**3. 当 `k = 2` (也就是城市 C) 时：**
算法继续检查。因为C是终点，没有从C出发的航线，所以把C作为中转站也不会创造任何新的航线。 **矩阵不变** 。

最终结果：

当三层循环全部结束后，`is_prereq` 矩阵已经被完全填充。它不再是一张“直飞表”，而是一张包含了所有直接和间接路径的、完整的“ **可换乘总表** ”！

`is_prereq[A][C]` 现在是 `True` 了，我们成功地计算出了这条间接的依赖关系。

这就是第三步发生的“魔法”：它通过一个简单而强大的动态规划思想，逐个引入中转节点，系统性地、无遗漏地发现了图中所有隐藏的间接路径，并记录下来。

```python
from typing import List

class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        # --- 解法一：Floyd-Warshall ---

        # 1. 初始化一个邻接矩阵来表示可达性
        # is_prereq[i][j] = True 表示 i 是 j 的先修课
        is_prereq = [[False] * numCourses for _ in range(numCourses)]
    
        # 根据给定的直接先修关系填充矩阵
        for u, v in prerequisites:
            is_prereq[u][v] = True
        
        # 2. 预计算 - 运行 Floyd-Warshall 算法
        # 复杂度为 O(N^3)
        for k in range(numCourses):  # 尝试通过中间节点 k
            for i in range(numCourses):  # 遍历所有起点 i
                for j in range(numCourses):  # 遍历所有终点 j
                    # 如果 i->k 且 k->j，那么 i->j
                    if is_prereq[i][k] and is_prereq[k][j]:
                        is_prereq[i][j] = True
                    
        # 3. 查询 - O(1) 时间复杂度
        answer = []
        for u, v in queries:
            answer.append(is_prereq[u][v])
        
        return answer
```

...
