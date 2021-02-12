import cv2
import torch
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

cfg = get_config()
cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def draw_bboxes(image, bboxes, line_thickness):
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    list_pts = []
    point_radius = 4

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)

        # 撞线的点
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
        list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

        ndarray_pts = np.array(list_pts, np.int32)

        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

        list_pts.clear()

    return image


def update(bboxes, image):
    bbox_xywh = []
    confs = []
    bboxes2draw = []

    if len(bboxes) > 0:
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) / 2), int((y1 + y2) / 2),
                x2 - x1, y2 - y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, image)

        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append((x1, y1, x2, y2, '', track_id))
        pass
    pass

    return bboxes2draw
