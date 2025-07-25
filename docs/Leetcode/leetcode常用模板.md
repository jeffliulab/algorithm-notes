## Binary Search

...

这两个模板一定要区分好

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1,-1]
  
        # 两种模板

        # 第一种： 
        left = 0
        right = len(nums)-1  # 这里是 -1 
        while left <= right:
            mid = (left+right)//2
            if mid == target:
                # find the target
            elif mid < target:
                left = mid + 1
            else:
                # 缩上界，因为右边是闭区间
                right = mid - 1
        # 循环结束后
        # 这个时候left和right正好差1，且left = right + 1
  

        # 第二种：
        left = 0
        right = len(nums)
        while left < right:
            mid = (left+right)//2
            if mid == target:
                # find the target
            elif mid < target:
                left = mid + 1
            else:
                # 缩上界，因为右边是开区间
                right = mid
        # 循环结束后
        # 这个时候left和right正好相等









```

...

...

...

## Sliding Window

...

```
        string, list:
        window: left, right
        设置一个dict记录内容
        移动右边的话就增加，移动左边的话就减少
  
        while 移动右边
            while 移动左边（一般是条件，结合题目的条件）

```

...

### 单调递减队列

```
while R < len(nums):
    # 入队，保持队列单调性
    while queue and nums[q[-1]] < nums[R]:
	queue.pop()
    queue.append(R)

    # 清理过期队头
    if L > queue[0]:
	queue.popleft()

    # 滑动窗口
    if (R + 1) >= k:
	result.append(nums[q[0]])
	L += 1
    R += 1
```

...

## Graph

### DFS模板

...

### BFS模板

...

### Kahn's模板

```python
indegree = {}
neighbors = {}

for i in range(numCourses):
    indegree[i] = 0
    neighbors[i] = []

for course, pre_course in prerequisites:
    indegree[course] += 1
    neighbors[pre_course].append(course)

queue = deque()
for node in indegree:
    if indegree[node] == 0:
	queue.append(node)

result = []
while queue:
    node = queue.popleft()
    result.append(node)
  
    node_neighbors = neighbors[node]
    for neighbor in node_neighbors:
	indegree[neighbor] -= 1
	if indegree[neighbor] == 0:
	    queue.append(neighbor)

if len(result) == numCourses:
    return result # or return True, the DAG is valid
else:
    return False # the DAG has circle

```

...
