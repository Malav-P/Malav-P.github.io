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

## Description

Given the root of a binary tree, invert the tree, and return its root.

## Solution

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