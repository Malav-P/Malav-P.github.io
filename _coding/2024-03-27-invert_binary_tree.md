---
layout: post
title: "Invert Binary Tree"
katex: False
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

Given the root of a binary tree, invert the tree, and return its root.

## Solution

#### Python

```python

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """

        if root:
            root.right, root.left = self.invertTree(root.left), self.invertTree(root.right)

        return root
```

#### C++

```c++

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
 
class Solution {
    public:
        TreeNode* invertTree(TreeNode* root) {

            if (root != nullptr) {

                TreeNode* tmp = root->right;
                root->right = root->left;
                root->left = tmp;

                invertTree(root->left);
                invertTree(root->right);
            }

            return root;

        }
};

```