```python

if not p or not q:
                return (not p) and (not q)
```



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

#### ==[剑指 Offer 26. 树的子结构](/problems/shu-de-zi-jie-gou-lcof/)==

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:  
给定的树 A:

`     3  
    / \  
   4   5  
  / \  
 1   2`  
给定的树 B：

`   4   
  /  
 1`  
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

**输入：** A = \[1,2,3\], B = \[3,1\]
**输出：** false

```python
def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
    def isPart(p1,p2):
        if not p2:
            return True
        if not p1 or p1.val != p2.val:
            return False
        return isPart(p1.left,p2.left) and isPart(p1.right,p2.right)
    if not A or not B:
        return False
    if isPart(A,B):
        return True
    return self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B)
```

#### [剑指 Offer 27. 二叉树的镜像](/problems/er-cha-shu-de-jing-xiang-lcof/)

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

`     4  
   /   \  
  2     7  
 / \   / \  
1   3 6   9`  
镜像输出：

`     4  
   /   \  
  7     2  
 / \   / \  
9   6 3   1`

**输入：** root = \[4,2,7,1,3,6,9\]
**输出：** \[4,7,2,9,6,3,1\]

```python
def mirrorTree(self, root: TreeNode) -> TreeNode:
    if not root:
        return root
    self.mirrorTree(root.left)
    self.mirrorTree(root.right)
    root.left,root.right = root.right,root.left
    return root
```

#### [剑指 Offer 29. 顺时针打印矩阵](/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

**输入：** matrix = \[\[1,2,3\],\[4,5,6\],\[7,8,9\]\]
**输出：** \[1,2,3,6,9,8,7,4,5\]

```python
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    m,n = len(matrix),len(matrix[0])
    visited = [[0 for col in range(n)] for row in range(m)]
    ans = []
    dx = [-1,0,1,0]
    dy = [0,1,0,-1]
    x = y = 0
    d = 1
    for i in range(m*n):
        ans.append(matrix[x][y])
        visited[x][y] = 1
        # a,b用来查找下一个合法的移动位置
        a,b = x+dx[d],y+dy[d]
        if a<0 or a>=m or b<0 or b>=n or visited[a][b]:
            d = (d+1)%4
            a,b = x+dx[d],y+dy[d]
        x,y = a,b
    return ans
```

#### ==[剑指 Offer 30. 包含min函数的栈](/problems/bao-han-minhan-shu-de-zhan-lcof/)==

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stk,self.min_stk = [],[]

    def push(self, x: int) -> None:
        self.stk.append(x)
        # >= x 避免了重复最小值被弹出(如0-1-0)
        if not self.min_stk or self.min_stk[-1]>=x:
            self.min_stk.append(x)

    def pop(self) -> None:
        if self.stk.pop()==self.min_stk[-1]:
            self.min_stk.pop()

    def top(self) -> int:
        return self.stk[-1]

    def min(self) -> int:
        return self.min_stk[-1]

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```

单变量，连续存最小值， 存差值法

*   每次存入：原来值 - 当前最小值
*   存入的是：非负数 -> 栈中的值 + 当前最小值 (还原)
*   存入的是：负数 -> 当前最小值 - 栈顶 (还原)， 且更新最小值

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stk,self.minVal = [],float('inf')

    def push(self, x: int) -> None:
        if not self.stk:
            self.stk.append(0)
            self.minVal = x
        else:
            self.stk.append(x-self.minVal)
            self.minVal = min(self.minVal,x)

    def pop(self) -> None:
        peek = self.stk.pop()
        if peek < 0:
            self.minVal -= peek

    def top(self) -> int:
        peek = self.stk[-1]
        return peek+self.minVal if peek>=0 else self.minVal


    def min(self) -> int:
        return self.minVal
```

#### ==[剑指 Offer 33. 二叉搜索树的后序遍历序列](/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)==

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同。

参考以下这颗二叉搜索树：

     5
    / \\
   2   6
  / \\
 1   3

**输入:** \[1,6,3,2,5\]
**输出:** false

```python
def verifyPostorder(self, postorder: List[int]) -> bool:
    def dfs(l,r):
        if l>r:
            return True
        root = postorder[r]
        k = l
        # 左子树的值都小于根节点
        while k<r and postorder[k]<root:
            k += 1
        # 右子树的值都大于根结点
        while k<r:
            if postorder[k]<root:
                return False
            k += 1
        return dfs(l,k-1) and dfs(k,r-1)
    if not postorder:
        return True
    return dfs(0,len(postorder)-1)
```

#### ==[剑指 Offer 34. 二叉树中和为某一值的路径](/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)==

给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出所有 **从根节点到叶子节点** 路径总和等于给定目标和的路径。

**叶子节点** 是指没有子节点的节点。

**示例 1：** 

![](https://pic.leetcode-cn.com/1644934303-RqPpsT-fe4925baab7f15819719d0bf0ba28eb.jpg)

**输入：** root = \[5,4,8,11,null,13,4,7,2,null,null,5,1\], targetSum = 22
**输出：** \[\[5,4,11,2\],\[5,8,4,5\]\]

```python
def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
    ans = []
    res = []
    def dfs(root,target):
        if not root: return
        res.append(root.val)
        target -= root.val
        if not(root.left or root.right or target):
            # 如果是ans.append(res)，那么后续res改变之后，ans也会改变，所以要添加拷贝。
            # ans.append(res)不行。 如果直接传入res，当后续res改变时，加入到ans的res也会改变，所以要对res进行了复制在插入ans.
            # 此处递归最终都会被这个代码 res.pop()，移除掉。所以保存结果时需要新开辟一个res内存。
            ans.append(res[:])
        dfs(root.left,target)
        dfs(root.right,target)
        
        ' ******回溯不理解****** '
        # 细节：找到后不能直接return，需要在path中移除最后一个元素，
        # 因为，即使你到根节点找到或找不到，该节点不能影响其他搜索
        res.pop()

    dfs(root,target)
    return ans
```

#### [剑指 Offer 35. 复杂链表的复制](/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

请实现 `copyRandomList` 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 `next` 指针指向下一个节点，还有一个 `random` 指针指向链表中的任意节点或者 `null`。

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

**输入：** head = \[\[7,null\],\[13,0\],\[11,4\],\[10,2\],\[1,0\]\]
**输出：** \[\[7,null\],\[13,0\],\[11,4\],\[10,2\],\[1,0\]\]

```python
def copyRandomList(self, head: 'Node') -> 'Node':
    if not head:
        return None
    # 在每个结点后添加一个他的复制
    p = head
    while p:
        np = Node(p.val)
        np.next = p.next
        p.next = np
        p = np.next

    # 将有random结点的复制指向对应random结点的复制
    p = head
    while p:
        if p.random:
            p.next.random = p.random.next
        p = p.next.next

    # 将复制的结点连接起来
    dummy = Node(0,head)
    p = dummy
    while p.next:
        q = p.next
        p.next = q.next
        p = p.next
    return dummy.next
```

#### ==[剑指 Offer 36. 二叉搜索树与双向链表](/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)==

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

为了让您更好地理解问题，以下面的二叉搜索树为例：

![Picture1.png](https://pic.leetcode-cn.com/1599401091-PKIjds-Picture1.png)

```python
def treeToDoublyList(self, root: 'Node') -> 'Node':
    def dfs(root):
        # 如果是叶节点，那么返回当前的节点
        if not (root.left or root.right ):
            return root,root
        # 如果左右子树都有    
        elif root.left  and root.right :
            #那么递归处理一下左右子树
            lsides, rsides = dfs(root.left), dfs(root.right)
            #然后拼接起来
            root.left, lsides[1].right = lsides[1], root
            root.right,rsides[0].left = rsides[0],root

            return lsides[0], rsides[1]

        elif root.left :
            #那么递归处理一下左子树
            lsides = dfs(root.left)
            root.left, lsides[1].right = lsides[1], root
            return lsides[0], root

        elif root.right :
            #那么递归处理一下右子树
            rsides = dfs(root.right)
            root.right,rsides[0].left = rsides[0],root
            return root, rsides[1]

    if not root: return root
    sides = dfs(root)
    # 将返回的双向链表的首尾相连
    sides[0].left, sides[-1].right = sides[-1], sides[0]
    return sides[0]
```

#### https://www.acwing.com/problem/1/

https://www.acwing.com/solution/content/67203/

#### ==[全排列](/https://www.acwing.com/problem/content/description/47/)==

给定一个不含重复数字的数组 `nums` ，返回其 _所有可能的全排列_ 。你可以 **按任意顺序** 返回答案。

**输入：** nums = \[1,2,3\]
**输出：** \[\[1,2,3\],\[1,3,2\],\[2,1,3\],\[2,3,1\],\[3,1,2\],\[3,2,1\]\]

由于有重复元素的存在，这道题的枚举顺序和 Permutations 不同。

1.  先将所有数从小到大排序，这样相同的数会排在一起；
2.  从左到右依次枚举每个数，每次将它放在一个空位上；
3.  对于相同数，我们人为定序，就可以避免重复计算：我们在dfs时记录一个额外的状态，记录上一个相同数存放的位置 start，我们在枚举当前数时，只枚举 start+1,start+2,…,n 这些位置。
4.  不要忘记递归前和回溯时，对状态进行更新。

```python
def permutation(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    nums.sort()
    ans = []
    path = [0]*len(nums)
    def dfs(nums,u,start,state):
        if u==len(nums):
            ans.append(path[:])
            return 
        if u==0 or nums[u]!=nums[u-1]: start = 0
        for i in range(start,len(nums)):
            # state表示第i位是否被占用
            if not (state>>i&1):
                path[i] = nums[u]
                dfs(nums,u+1,i+1,state+(1<<i))
    dfs(nums,0,0,0)
    return ans
# state没懂
```

#### [剑指 Offer 39. 数组中出现次数超过一半的数字](/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**输入:** \[1, 2, 3, 2, 2, 2, 5, 4, 2\]
**输出:** 2

如果一个数大于一半，这个数最后的数量肯定大于其他所有数的数量之和。

先记录数组中的第一个元素，并用count记录元素出现的次数 
依次遍历数组元素

*   若val!=num\[i\]时 
    当count<0时，需要更新val为当前元素值 
    否则count–
    
*   若val==num\[i\]时 
    count++

```python
def moreThanHalfNum_Solution(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    val,cnt = nums[0],1
    for i in range(1,n):
        if nums[i]!=val:
            cnt -= 1
        else:
            cnt += 1
        if cnt < 0:
            val,cnt = nums[i],1
    return val
# 
```

#### [剑指 Offer 41. 数据流中的中位数](/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

```
输入：1, 2, 3, 4

输出：1,1.5,2,2.5

解释：每当数据流读入一个数据，就进行一次判断并输出当前的中位数。
```

直接构造大小堆，`大顶堆放小于中位数的集合`，`小顶堆放大于中位数的集合`，
取得时候，如果是奇数，就取前半部分的最大数，如果是偶数，就取前半部分的最大数和后半部分的最小数的平均数

```python
class Solution:
    def __init__(self):
        self.min_heap = []
        self.max_heap = []
    # 先往大根堆里放
    # 如果大根堆顶元素》小根堆顶元素，则 交换堆顶元素
    # 若大根堆元素数 》 小根堆元素数+1，则弹出大根堆堆顶元素加入小根堆
    def insert(self, num):
        """
        :type num: int
        :rtype: void
        """
        heapq.heappush(self.max_heap,-num)
        if len(self.min_heap) and -self.max_heap[0] > self.min_heap[0]:
            a,b = -self.max_heap[0],self.min_heap[0]
            heapq.heappop(self.max_heap)
            heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap,-b)
            heapq.heappush(self.min_heap,a)
        if len(self.max_heap) > len(self.min_heap)+1:
            heapq.heappush(self.min_heap,-heapq.heappop(self.max_heap))
        
    def getMedian(self):
        """
        :rtype: float
        """
        if len(self.max_heap) + len(self.min_heap) & 1: return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2
```

