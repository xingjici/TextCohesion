# ICDAR_PAPER
A PyTorch implement of **TextCohesion: A Accurate Detector for Detecting Text of Arbitrary Shapes
TextCohesion element:

- center point
- Pixel centripetal force
- text region

## Description

已改进：
        1、resume
        2、validation和evaluation
        3、dataset ：remove negative cycle , vastly saving time
        1、数据增强
        2、tr内的方向四分类
        3、方向权重
        4、预处理


指标：


            修改后处理操作


Experence:

    将total text和icdar2019相混合进行训练

     修改了后处理，速度提升了6倍
    tcl_thresh: 0.54
    max_area : 50
    precision:  0.881 , recall:   0.814
    f_score:  0.846