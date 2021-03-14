# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: test_quicksort.py
   Description : 
   Author : ericdoug
   date：2021/3/9
-------------------------------------------------
   Change Activity:
         2021/3/9: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import os

# third packages


# my packages

def quick_sort(ilist):

    if len(ilist) <= 1:
        return ilist

    left = []
    right = []
    for i in ilist[1:]:
        if i <= ilist[0]:
            left.append(i)
        else:
            right.append(i)
    return quick_sort(left) + [ilist[0]] + quick_sort(right)

if __name__ == '__main__':
    ilist = [1, 5, 7, 3, 2, 9]
    print(quick_sort(ilist))

