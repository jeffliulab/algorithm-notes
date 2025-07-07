# Leetcode Interview 150

## Array/String (24)

题1到题15需要更新为伪代码。

#### (16) 42. Trapping Rain Water

给定一个数组 `height`，存储每个柱子的高度。

**解法 1：动态规划（易于想到）**

- 先计算一个数组 `max_left`，其中 `max_left[i]` 表示位置 `i` 左侧（包括 `i`）的最高柱子高度。  
- 再计算一个数组 `max_right`，其中 `max_right[i]` 表示位置 `i` 右侧（包括 `i`）的最高柱子高度。  
- 最后，每个位置 `i` 能蓄的水量就是 `min(max_left[i], max_right[i]) - height[i]`。  

时间复杂度：O(n)  
空间复杂度：O(n)

```pseudocode
get a list of max_left to i
get a list of max_right to i
calculate each i's water as [min of (max_left, max_right) - height[i]]
```

**解法 2：双指针，空间优化到 O(1)**

思路来源于公式 `min(max_left, max_right) - height[i]`：  
- 当 `max_left <= max_right` 时，蓄水量由左侧决定，即 `max_left - height[i]`；  
- 否则由右侧决定，即 `max_right - height[i]`。

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

时间复杂度：O(n)  
空间复杂度：O(1)
