import os

import numpy as np
import traffic.detect.tracker as tracker
from traffic.detect.detector import Detector
import cv2
import json
import yaml
import pandas as pd
from tqdm import tqdm
print(os.getcwd())

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def create_mask_image(config):
    mask_image_temp = np.zeros((540, 960), dtype=np.uint8)
    lanes = {
        1: config["list_pts_blue"],
        2: config["list_pts_yellow"],
        3: config["list_pts_green"],
    }
    for lane_id, points in lanes.items():
        ndarray_pts = np.array(points, np.int32)
        cv2.fillPoly(mask_image_temp, [ndarray_pts], color=lane_id)
    return mask_image_temp[:, :, np.newaxis]


def create_color_polygons_image(config):
    color_polygons_image = np.zeros((540, 960, 3), dtype=np.uint8)
    lanes = {
        (255, 0, 0): config["list_pts_blue"],
        (0, 255, 255): config["list_pts_yellow"],
        (0, 255, 0): config["list_pts_green"],
    }
    for color, points in lanes.items():
        ndarray_pts = np.array(points, np.int32)
        cv2.fillPoly(color_polygons_image, [ndarray_pts], color)
    return color_polygons_image


def update_lane_sets(track_id, value, lane_sets):
    if value == 1:
        lane_sets["current_lane1"].add(track_id)
        lane_sets["total_lane1"].add(track_id)
    elif value == 2:
        lane_sets["current_lane2"].add(track_id)
        lane_sets["total_lane2"].add(track_id)
    elif value == 3:
        lane_sets["current_lane3"].add(track_id)


def remove_left_ids(set_ids_in_frame, lane_sets):
    for lane in ["lane1", "lane2", "lane3"]:
        current_set = lane_sets[f"current_{lane}"]
        for track_id in current_set.copy():
            if track_id not in set_ids_in_frame:
                current_set.remove(track_id)


def write_data_to_file(f, capture, lane_sets):
    current_frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    data_to_write = {
        "lane_data": [
            {"current_frame_number": current_frame_number},
            {
                "lane_id": 1,
                "current_ids": list(lane_sets["current_lane1"]),
                "total_ids": list(lane_sets["total_lane1"]),
                "current_count": len(lane_sets["current_lane1"]),
                "total_count": len(lane_sets["total_lane1"]),
            },
            {
                "lane_id": 2,
                "current_ids": list(lane_sets["current_lane2"]),
                "total_ids": list(lane_sets["total_lane2"]),
                "current_count": len(lane_sets["current_lane2"]),
                "total_count": len(lane_sets["total_lane2"]),
            },
            {
                "lane_id": 3,
                "current_ids": list(lane_sets["current_lane3"]),
                "total_ids": list(lane_sets["total_lane3"]),
                "current_count": len(lane_sets["current_lane3"]),
                "total_count": len(lane_sets["total_lane3"]),
            },
        ]
    }
    print(data_to_write)
    f.write(json.dumps(data_to_write) + "\n")


def draw_text_on_frame(output_image_frame, lane_sets, draw_text_postion, font_draw_number):
    text_draw = (
        f"Lane 1: Current {len(lane_sets['current_lane1'])}, Total {len(lane_sets['total_lane1'])} | "
        f"Lane 2: Current {len(lane_sets['current_lane2'])}, Total {len(lane_sets['total_lane2'])} | "
        f"Lane 3: Current {len(lane_sets['current_lane3'])}, Total {len(lane_sets['total_lane3'])}"
    )
    return cv2.putText(
        img=output_image_frame,
        text=text_draw,
        org=draw_text_postion,
        fontFace=font_draw_number,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2,
    )


def detect_video(config_path):
    config = load_config(config_path)
    polygon_mask = create_mask_image(config)
    color_polygons_image = create_color_polygons_image(config)

    lane_sets = {
        "current_lane1": set(),
        "current_lane2": set(),
        "current_lane3": set(),
        "total_lane1": set(),
        "total_lane2": set(),
        "total_lane3": set(),
    }

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    detector = Detector()
    capture = cv2.VideoCapture(config["input_video_path"])
    f = open(config["json_path"], "w", encoding="utf-8")
    id_label_dict = {}
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        _, im = capture.read()
        if im is None:
            break
        im = cv2.resize(im, (960, 540))
        bboxes = detector.detect(im)

        if bboxes:
            list_bboxs = tracker.update(bboxes, im)
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
        else:
            output_image_frame = im

        output_image_frame = cv2.addWeighted(output_image_frame, 1, color_polygons_image, 0.5, 0)
        set_ids_in_frame = set()

        if bboxes:
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox
                id_label_dict[track_id] = label
                x_center = int(x1 + (x2 - x1) * config["offset"])
                y_center = y2
                cv2.circle(output_image_frame, (x_center, y_center), 5, (0, 0, 255), -1)
                x_center = max(0, min(x_center, polygon_mask.shape[1] - 1))
                y_center = max(0, min(y_center, polygon_mask.shape[0] - 1))
                value = polygon_mask[y_center, x_center, 0]
                set_ids_in_frame.add(track_id)
                update_lane_sets(track_id, value, lane_sets)

            remove_left_ids(set_ids_in_frame, lane_sets)
            write_data_to_file(f, capture, lane_sets)
        else:
            lane_sets["current_lane1"].clear()
            lane_sets["current_lane2"].clear()
            lane_sets["current_lane3"].clear()

        output_image_frame = draw_text_on_frame(output_image_frame, lane_sets, draw_text_postion, font_draw_number)
        cv2.waitKey(1)

    id_label_df = pd.DataFrame(list(id_label_dict.items()), columns=["track_id", "label"])
    id_label_df.to_csv(config["id_label_path"], index=False)
    capture.release()
    cv2.destroyAllWindows()
    return id_label_df


if __name__=='__main__':
    detect_video('')
