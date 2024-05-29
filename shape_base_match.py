#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2024/05/28 17:27:40

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional
warnings.filterwarnings('ignore')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import cv2
import time
import numpy as np
import shape_based_matching_py


def img_padding(img: np.ndarray, mask: Optional[np.ndarray], padding: int=100) -> np.ndarray:
    """同步padding图片及其对应的mask, 一般用于旋转放缩预处理

    :param np.ndarray img: 待处理的图片
    :param Optional[np.ndarray] mask: 图片关注区域mask
    :param int padding: 每个边padding的尺寸, defaults to 100
    :return np.ndarray: padding后的图像
    """
    imgh, imgw, imgc = img.shape
    padded_templ = np.zeros(shape=(imgh+2*padding, imgw+2*padding, imgc), dtype=np.uint8)
    padded_templ[padding:padding+imgh, padding:padding+imgw, :] = img[...]
    if mask is None: padded_mask = None
    else:
        padded_mask  = np.zeros(shape=(imgh+2*padding, imgw+2*padding), dtype=np.uint8)
        padded_mask[padding:padding+imgh, padding:padding+imgw] = mask[...]
    return padded_templ, padded_mask


def mask_gen(img: np.ndarray, contours: List=[]) -> np.ndarray:
    """mask生成函数

    :param np.ndarray img: 源图片
    :param List contours: 多边形候选区域列表, defaults to []
    :return np.ndarray: roi的区域
    """
    if not contours:
        mask = np.ones(shape=img.shape[:2], dtype=np.uint8)
        mask *= 255
    else:
        mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
    return mask


def shape_base_match(
        search_img: np.ndarray, 
        templ: np.ndarray, 
        contours: List[np.ndarray] = [],
        num_features: int = 128, 
        padding: List = [100, 250],
        pyramid_levels: List = [4, 8], 
        angle_range: List = [-40, 40],
        use_rot: bool = True,
        show=True
    ) -> float:
    """基于边缘梯度方向的模板匹配算法

    :param np.ndarray search_img: 搜索图
    :param np.ndarray templ: 模板图
    :param List[np.ndarray] contours: 关注区域多边形列表, 注意多边形的坐标要锚定templ, defaults to []
    :param int num_features: 需要配置的特征数量, defaults to 128
    :param List padding: 分别表示模板和搜索图的padding大小, defaults to [100, 250]
    :param List pyramid_levels: line memory的T, 即模板分块的边长, defaults to [4, 8]
    :param List angle_range: 模板匹配旋转角度, defaults to [-40, 40]
    :param bool use_rot: _description_, defaults to True
    :param bool show: 是否展示模板关键点匹配图, defaults to True
    :return float: 最佳匹配得分
    """
    start_time = time.perf_counter()
    ssim = 0

    detector = shape_based_matching_py.Detector(num_features, pyramid_levels)
    templ_mask = mask_gen(templ, contours=contours)
    padded_templ, padded_mask = img_padding(templ, templ_mask, padding[0])

    shapes = shape_based_matching_py.shapeInfo_producer(padded_templ, padded_mask)
    shapes.angle_range = angle_range
    shapes.angle_step = 1
    shapes.scale_range = [1]
    shapes.produce_infos()

    infos_have_templ = []
    class_id = "test"
    is_first = True
    first_id = 0
    first_angle = 0

    for info in shapes.infos:
        templ_id = 0
        if is_first:
            templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info))
            first_id = templ_id
            first_angle = info.angle
            if use_rot: is_first = False
        else:
            templ_id = detector.addTemplate_rotate(
                class_id, 
                first_id,
                info.angle-first_angle,
                shape_based_matching_py.CV_Point2f(padded_templ.shape[1]/2.0, padded_templ.shape[0]/2.0)
            )
        if templ_id != -1:
            infos_have_templ.append(info)
    
    middle_time = time.perf_counter()

    # 开始搜索图的匹配计算
    ids = []
    ids.append('test')
    padded_img, _ = img_padding(search_img, None, padding[1])

    stride = 16
    img_rows = int(padded_img.shape[0] / stride) * stride
    img_cols = int(padded_img.shape[1] / stride) * stride
    img = np.zeros((img_rows, img_cols, padded_img.shape[2]), np.uint8)
    img[:, :, :] = padded_img[0:img_rows, 0:img_cols, :]
    matches = detector.match(img, 90, ids)
    top5 = 1
    if top5 > len(matches): top5 = 1
    if len(matches): ssim = matches[0].similarity
    end_time = time.perf_counter()
    print(f"execTime of Match: {end_time-middle_time}, execTime of addTempl: {middle_time - start_time}")
    if show and len(matches):
        for i in range(top5):
            match = matches[i]
            templ = detector.getTemplates("test", match.template_id)
            
            for feat in templ[0].features:
                img = cv2.circle(img, (feat.x+match.x, feat.y+match.y), 3, (0, 0, 255), -1)

            # cv2 have no RotatedRect constructor?
            print('match.template_id: {}'.format(match.template_id))
            print('match.similarity: {}'.format(match.similarity))
        cv2.imshow("img", img)
        cv2.waitKey(0)

    
    return ssim


if __name__ == "__main__":
    import json
    with open("test/case1/templ.json", "r+", encoding="utf-8") as f:
        contour_data = json.load(f)
    contour = np.array(contour_data["shapes"][0]["points"], dtype=np.int32).reshape(-1, 1, 2)
    s_img = cv2.imread("test/case1/test.png")
    m_img = cv2.imread("test/case1/templ.png")
    sbm = shape_base_match(s_img, m_img, contours=[contour])
