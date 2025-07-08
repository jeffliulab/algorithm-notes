# Leetcode Interview 150

## Array/String (24)

(1) to (15) needs modified into pseudocode, instead of codes.

#### (16) 42. Trapping Rain Water

list height stores the information of height of blocks.

**Solution 1. Dynamic Programming (Easy to Come Up With)**

```pseudocode
get a list of max_left to i
get a list of max_right to i
calculate each i's water as [min of (max_left, max_right) - height[i]]
```

Time: O(n)

Space: O(n)

**Solution 2. Optimize Space to O(1), use two pointers.**

I think this gonna be a simplification of math equation `[min of (max_left, max_right) - height[i]]`.

Because we always caculate this, this can be considered as:

* max_left - height[i], if max_left <= max_right
* max_right - height[i], if max_left > max_right

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

Time: O(n)

Space: O(1)
