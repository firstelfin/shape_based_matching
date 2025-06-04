# shape_based_matching
模板匹配库：shape-based-matching, 来源于没趣啦项目https://github.com/meiqua/shape_based_matching/tree/python_binding


论文：
    1. 作者：https://campar.cs.tum.edu/Main/StefanHinterstoisser
    2. 论文：https://mediatum.ub.tum.de/1629981?style=full_standard
    3. 论文：https://github.com/meiqua/shape_based_matching/tree/python_binding


# 项目简介

此项目是shape-based-matching的学习项目，添加了如下内容：
1. C++版本的match函数；
2. python版本的match函数；
3. C++代码的解读

# quick start

编译C++

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```

python安装

```shell
# 进入setup.py所在目录
$ cd ..
$ pip3 install ./
$ python3 shape_base_match.py
```

# debug
1. libstdc++ ImportError
参考：[GLIBCXX_3.4.30 not found](https://blog.csdn.net/L1481333167/article/details/137919464?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-137919464-blog-129650003.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-137919464-blog-129650003.235%5Ev43%5Epc_blog_bottom_relevance_base5)

# 文档

1. [项目源码说明](source_doc/shape-based-matching代码梳理.md)