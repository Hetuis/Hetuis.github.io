#### [剑指 Offer 03. 数组中重复的数字](/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

**输入：** 
\[2, 3, 1, 0, 2, 5, 3\]
**输出：** 2 或 3

```python
def findRepeatNumber(self, nums: List[int]) -> int:
    n = len(nums)
    for i in range(n):
        while nums[nums[i]]!=nums[i]:
            # 交换顺序很重要
            nums[nums[i]],nums[i] = nums[i],nums[nums[i]]
        if nums[i]!=i:
            return nums[i]
    return -1
```

#### [剑指 Offer 04. 二维数组中的查找](/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

在一个 n \* m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

现有矩阵 matrix 如下：

\[
  \[1,   4,  7, 11, 15\],
  \[2,   5,  8, 12, 19\],
  \[3,   6,  9, 16, 22\],
  \[10, 13, 14, 17, 24\],
  \[18, 21, 23, 26, 30\]
\]

给定 target = `5`，返回 `true`。

给定 target = `20`，返回 `false`。

```python
# 右上角开始判断，每次删除一行/列
def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix:
        return False
    m,n = len(matrix),len(matrix[0])
    i,j = 0,n-1
    while i<m and j>=0:
        curVal = matrix[i][j]
        if curVal == target:
            return True
        elif curVal < target:
            i += 1
        else:
            j -= 1
    return False
```

#### ==[剑指 Offer 12. 矩阵中的路径](/problems/ju-zhen-zhong-de-lu-jing-lcof/)==

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。

![](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

**示例 1：** 

**输入：** board = \[\["A","B","C","E"\],\["S","F","C","S"\],\["A","D","E","E"\]\], word = "ABCCED"
**输出：** true

```python
def exist(self, board: List[List[str]], word: str) -> bool:
    def dfs(board,word,u,x,y):
        if board[x][y] != word[u]:
            return False
        # u==len(word) ['a']过不去
        if u==len(word)-1:
            return True
        # y总，用来保存matrix[x][y]值的临时变量t是不是可以不用定义？因为程序进行到了这里，matrix[x][y]==str[u]是成立的，还原matrix[x][y]的时候直接用str[u]是不是就可以啦
        t = board[x][y]
        board[x][y] = '*'
        dx = [-1,0,1,0]
        dy = [0,1,0,-1]
        for i in range(4):
            a = x + dx[i]
            b = y + dy[i]
            if a>=0 and a<m and b>=0 and b<n :
                if dfs(board,word,u+1,a,b):
                    return True
        board[x][y] = t
        return False
    
    m,n = len(board),len(board[0])
    for i in range(m):
        for j in range(n):
            if dfs(board,word,0,i,j):
                return True
    return False
```

#### ==[剑指 Offer 13. 机器人的运动范围](/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)==

地上有一个m行n列的方格，从坐标 `[0,0]` 到坐标 `[m-1,n-1]` 。一个机器人从坐标 `[0, 0]` 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 \[35, 37\] ，因为3+5+3+7=18。但它不能进入方格 \[35, 38\]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

**输入：** m = 2, n = 3, k = 1
**输出：** 3

```python
def movingCount(self, m: int, n: int, k: int) -> int:
    def get_single(n):
        s = 0
        while n:
            s += n%10
            n //= 10
        return s
    def get_sum(nums):
        return get_single(nums[0])+get_single(nums[1])
    
    ans = 0
    visited = [[0 for col in range(n)] for row in range(m)]
    if not m or not n:
        return 0
    q = []
    q.append([0,0])
    dirs =[[-1,0],[0,1],[1,0],[-1,0]]
    
    while q:
        t = q[0]
        del q[0]

        if get_sum(t) > k or visited[t[0]][t[1]]:
            continue
        ans += 1
        visited[t[0]][t[1]] = 1
        
        for i in range(4):
            x = t[0] + dirs[i][0]
            y = t[1] + dirs[i][1]
            if x>=0 and x<m and y>=0 and y<n:
                q.append([x,y])
    
    return ans
```

#### [剑指 Offer 14- I. 剪绳子](/problems/jian-sheng-zi-lcof/)

给你一根长度为 `n` 的绳子，请把绳子剪成整数长度的 `m` 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m-1]` 。请问 `k[0]*k[1]*...*k[m-1]` 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

**输入:** 10
**输出:** 36
**解释:** 10 = 3 + 3 + 4, 3 × 3 × 4 = 36

![image-20220218161458391](/home/aimee/.config/Typora/typora-user-images/image-20220218161458391.png)

```python
def cuttingRope(self, n: int) -> int:
    if n<=3:
        return n-1
    ans = 1
    if n%3==1:
        ans *= 4
        n -= 4
    if n%3==2:
        ans *= 2
        n -= 2
    while n:
        ans *= 3
        n -= 3
    return ans
```

dp(没看)

```java
class Solution {
    public int integerBreak(int n) {
        int[] dp = new int[n+1];
        for(int i = 2; i <= n; i++)
            for(int j = 1; j < i; j++)
                dp[i] = Math.max(dp[i], Math.max(j * (i-j), j * dp[i-j]));
        return dp[n];
    }
}
```

#### [剑指 Offer 15. 二进制中1的个数](/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 [汉明重量](http://en.wikipedia.org/wiki/Hamming_weight)).）。

**输入：** n = 11 (控制台输入 00000000000000000000000000001011)
**输出：** 3
**解释：** 输入的二进制串 `**00000000000000000000000000001011** 中，共有三位为 '1'。`

```python
# 菜狗子
def hammingWeight(self, n: int) -> int:
    ans = 0
    while n:
        ans += 1
        n &= (n-1)
    return ans
# yxc
def hammingWeight(self, n: int) -> int:
    ans = 0
    while n:
        ans += n&1 # 1. 如果 nn 在二进制表示下末尾是1，则在答案中加1；
        n >>= 1 # 2. 将 nn 右移一位，也就是将 nn 在二进制表示下的最后一位删掉；
    return ans
```

#### [剑指 Offer 16. 数值的整数次方](/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

实现 [pow(_x_, _n_)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

**输入：** x = 2.00000, n = -2
**输出：** 0.25000

```python
def myPow(self, x: float, n: int) -> float:
    flag = True if n<0 else False
    ans = 1
    n = abs(n)
    while n:
        if n&1:
            ans *= x
        x *= x
        n >>= 1
    if flag:
        return 1/ans
    return ans
```

#### [剑指 Offer 19. 二叉树的下一个节点](/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

给定一棵二叉树的其中一个节点，请找出中序遍历序列的下一个节点。

**注意：** 

*   如果给定的节点是中序遍历序列的最后一个，则返回空节点;
*   二叉树一定不为空，且给定的节点一定不是空节点；

```
假定二叉树是：[2, 1, 3, null, null, null, null]， 给出的是值等于2的节点。

则应返回值等于3的节点。

解释：该二叉树的结构如下，2的后继节点是3。
  2
 / \
1   3 
```

```python
def inorderSuccessor(self, p):
    """
    :type p: TreeNode
    :rtype: TreeNode
    """
    # 如果有右子树，找右子树最左边的节点
    if p.right:
        p = p.right
        while p.left:
            p = p.left
        return p
    # 没有右子树，找第一个满足的父节点a(其中a是其父节点的左儿子)
    while p.father and p == p.father.right:
        p = p.father
    return p.father
```
