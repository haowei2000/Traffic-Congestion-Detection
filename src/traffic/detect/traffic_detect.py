import json
from typing import Any, Dict, List, Set, Tuple
import warnings

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import traffic.detect.tracker as tracker
from traffic.detect.detector import Detector
from traffic.my_path import results_dir,video_dir


def _create_mask_image(config: Dict[str, Any]) -> np.ndarray:
    """
    Creates a mask image based on the provided configuration.
    Args:
        config (Dict[str, Any]): A dictionary containing the configuration for the mask image.
            Expected keys are:
                - "list_pts_blue": List of points defining the blue lane.
                - "list_pts_yellow": List of points defining the yellow lane.
                - "list_pts_green": List of points defining the green lane.
    Returns:
        np.ndarray: A 3D numpy array representing the mask image with lanes filled with different colors.
    """

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


def _create_color_polygons_image(config: Dict[str, Any]) -> np.ndarray:
    """
    Creates an image with colored polygons based on the provided configuration.

    Args:
        config (Dict[str, Any]): A dictionary containing lists of points for different colored polygons.
            The dictionary should have the following keys:
                - "list_pts_blue": List of points for the blue polygon.
                - "list_pts_yellow": List of points for the yellow polygon.
                - "list_pts_green": List of points for the green polygon.

    Returns:
        np.ndarray: An image (numpy array) with the colored polygons drawn on it.
    """
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


def _remove_left_ids(
    set_ids_in_frame: Set[int], lane_sets: Dict[str, Set[int]]
) -> None:
    """
    Removes track IDs from the lane sets if they are not present in the current frame.
    Args:
        set_ids_in_frame (Set[int]): A set of track IDs that are currently in the frame.
        lane_sets (Dict[str, Set[int]]): A dictionary where keys are lane identifiers
                                         (e.g., "current_lane1", "current_lane2", "current_lane3")
                                         and values are sets of track IDs for each lane.
    Returns:
        None: This function modifies the lane_sets dictionary in place.
    """

    for lane in ["lane1", "lane2", "lane3"]:
        current_set = lane_sets[f"current_{lane}"]
        for track_id in current_set.copy():
            if track_id not in set_ids_in_frame:
                current_set.remove(track_id)


def _update_lane_sets(
    track_id: int, value: int, lane_sets: Dict[str, Set[int]]
) -> None:
    """
    Updates the lane sets with the given track ID based on the value provided.

    Args:
        track_id (int): The ID of the track to be updated.
        value (int): The lane value indicating which lane sets to update.
                     - 1: Updates "current_lane1" and "total_lane1"
                     - 2: Updates "current_lane2" and "total_lane2"
                     - 3: Updates "current_lane3" and "total_lane3"
        lane_sets (Dict[str, Set[int]]): A dictionary containing sets of track IDs for different lanes.

    Returns:
        None
    """
    if value == 1:
        lane_sets["current_lane1"].add(track_id)
        lane_sets["total_lane1"].add(track_id)
    elif value == 2:
        lane_sets["current_lane2"].add(track_id)
        lane_sets["total_lane2"].add(track_id)
    elif value == 3:
        lane_sets["current_lane3"].add(track_id)
        lane_sets["total_lane3"].add(track_id)


def _draw_text_on_frame(
    output_image_frame: np.ndarray,
    lane_sets: Dict[str, Set[int]],
    draw_text_position: Tuple[int, int],
    font_draw_number: int,
) -> np.ndarray:
    """
    Draws text on a given image frame indicating the current and total number of vehicles in each lane.
    Args:
        output_image_frame (np.ndarray): The image frame on which the text will be drawn.
        lane_sets (Dict[str, Set[int]]): A dictionary containing sets of vehicle IDs for each lane.
            Expected keys are 'current_lane1', 'total_lane1', 'current_lane2', 'total_lane2',
            'current_lane3', and 'total_lane3'.
        draw_text_position (Tuple[int, int]): The (x, y) coordinates where the text will be drawn on the image.
        font_draw_number (int): The font type to be used for drawing the text.
    Returns:
        np.ndarray: The image frame with the drawn text.
    """

    text_draw = (
        f"Lane 1: Current {len(lane_sets['current_lane1'])}, Total {len(lane_sets['total_lane1'])} | "
        f"Lane 2: Current {len(lane_sets['current_lane2'])}, Total {len(lane_sets['total_lane2'])} | "
        f"Lane 3: Current {len(lane_sets['current_lane3'])}, Total {len(lane_sets['total_lane3'])}"
    )
    return cv2.putText(
        img=output_image_frame,
        text=text_draw,
        org=draw_text_position,
        fontFace=font_draw_number,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2,
    )


def _initialize_lane_sets() -> Dict[str, Set[int]]:
    """
    Initializes and returns a dictionary containing empty sets for different lanes.

    The dictionary contains the following keys:
    - "current_lane1": Set for current lane 1.
    - "current_lane2": Set for current lane 2.
    - "current_lane3": Set for current lane 3.
    - "total_lane1": Set for total lane 1.
    - "total_lane2": Set for total lane 2.
    - "total_lane3": Set for total lane 3.

    Returns:
        Dict[str, Set[int]]: A dictionary with keys representing lane names and values as empty sets.
    """
    return {
        "current_lane1": set(),
        "current_lane2": set(),
        "current_lane3": set(),
        "total_lane1": set(),
        "total_lane2": set(),
        "total_lane3": set(),
    }


def _process_frame(
    im: np.ndarray,
    bboxes: List[Tuple[int, int, int, int, int, int]],
    polygon_mask: np.ndarray,
    color_polygons_image: np.ndarray,
    lane_sets: Dict[str, Set[int]],
    id_label_dict: Dict[int, int],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Set[int]]:
    """
    Processes a single frame of the video to detect and track objects, draw bounding boxes,
    and update lane sets.

    Args:
        im (np.ndarray): The input image frame.
        bboxes (List[Tuple[int, int, int, int, int, int]]): List of bounding boxes with
            each tuple containing (x1, y1, x2, y2, label, track_id).
        polygon_mask (np.ndarray): Mask image used to determine the lane of detected objects.
        color_polygons_image (np.ndarray): Image with colored polygons to overlay on the frame.
        lane_sets (Dict[str, Set[int]]): Dictionary mapping lane identifiers to sets of track IDs.
        id_label_dict (Dict[int, int]): Dictionary mapping track IDs to labels.
        config (Dict[str, Any]): Configuration dictionary with additional parameters.

    Returns:
        Tuple[np.ndarray, Set[int]]: The processed image frame with drawn bounding boxes and
        a set of track IDs present in the frame.
    """
    if bboxes:
        list_bboxs = tracker.update(bboxes, im)
        output_image_frame = tracker.draw_bboxes(
            im, list_bboxs, line_thickness=None
        )
    else:
        output_image_frame = im

    output_image_frame = cv2.addWeighted(
        output_image_frame, 1, color_polygons_image, 0.5, 0
    )
    set_ids_in_frame = set()

    if bboxes:
        for item_bbox in list_bboxs:
            x1, y1, x2, y2, label, track_id = item_bbox
            id_label_dict[track_id] = label
            x_center, y_center = _get_center_coordinates(x1, x2, y2, config)
            cv2.circle(
                output_image_frame, (x_center, y_center), 5, (0, 0, 255), -1
            )
            value = polygon_mask[y_center, x_center, 0]
            set_ids_in_frame.add(track_id)
            _update_lane_sets(track_id, value, lane_sets)

    return output_image_frame, set_ids_in_frame


def _get_center_coordinates(
    x1: int, x2: int, y2: int, config: Dict[str, Any]
) -> Tuple[int, int]:
    """
    Calculates the center coordinates based on the provided x and y boundaries. The function adjusts the x-coordinate using an offset from the configuration and ensures the coordinates remain within specified bounds.

    Args:
        x1 (int): The starting x-coordinate.
        x2 (int): The ending x-coordinate.
        y2 (int): The y-coordinate to be used for the center.
        config (Dict[str, Any]): A configuration dictionary containing the offset value.

    Returns:
        Tuple[int, int]: The calculated center coordinates as a tuple (x_center, y_center).
    """
    x_center = int(x1 + (x2 - x1) * config["offset"])
    y_center = y2
    x_center = max(0, min(x_center, 959))  # 960 - 1
    y_center = max(0, min(y_center, 539))  # 540 - 1
    return x_center, y_center


def _clear_current_lane_sets(lane_sets: Dict[str, Set[int]]) -> None:
    """
    Clears the sets of current lanes in the provided lane_sets dictionary.

    This function takes a dictionary with keys "current_lane1", "current_lane2",
    and "current_lane3", each associated with a set of integers. It clears the
    sets associated with these keys.

    Args:
        lane_sets (Dict[str, Set[int]]): A dictionary containing sets of lane
        identifiers to be cleared. The dictionary must have the keys
        "current_lane1", "current_lane2", and "current_lane3".
    """
    lane_sets["current_lane1"].clear()
    lane_sets["current_lane2"].clear()
    lane_sets["current_lane3"].clear()


def _save_output_video(
    config: Dict[str, Any], output_image_frame: np.ndarray
) -> None:
    """
    Writes a frame to the output video file.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing the output video path.
        output_image_frame (np.ndarray): The image frame to be written to the video.

    Returns:
        None
    """
    if "video_writer" not in locals():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            results_dir/config["output_video_path"], fourcc, 30, (960, 540)
        )
    video_writer.write(output_image_frame)


def _save_lane_set(
    f: Any, capture: cv2.VideoCapture, lane_sets: Dict[str, Set[int]]
) -> None:
    """
    Save the current lane set data to a file.

    Args:
        f (Any): A file-like object to write the lane set data.
        capture (cv2.VideoCapture): A video capture object to get the current frame number.
        lane_sets (Dict[str, Set[int]]): A dictionary containing the current and total IDs for each lane.

    Returns:
        None
    """
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
    f.write(json.dumps(data_to_write) + "\n")


def _save_id_label_dict(
    id_label_dict: Dict[int, int], config: Dict[str, Any]
) -> None:
    """
    Save a dictionary of track IDs and labels to a CSV file.

    Args:
        id_label_dict (Dict[int, int]): A dictionary where keys are track IDs and values are labels.
        config (Dict[str, Any]): Configuration dictionary containing the path to save the CSV file.
                                 Expected key is "id_label_path".

    Returns:
        None
    """
    id_label_df = pd.DataFrame(
        list(id_label_dict.items()), columns=["track_id", "label"]
    )
    output_path = results_dir/config["id_label_path"]
    id_label_df.to_csv( output_path, index=False)


def detect_video(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Detects objects in a video and processes each frame to track and label detected objects.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing the following keys:
            - "input_video_path" (str): Path to the input video file.
            - "json_path" (str): Path to save the JSON file with lane set information.
            - "output_video_path" (Optional[str]): Path to save the output video with annotations.

    Returns:
        pd.DataFrame: A DataFrame containing track IDs and their corresponding labels.

    The function performs the following steps:
        1. Creates a mask image and a color polygons image based on the configuration.
        2. Initializes lane sets and sets up text drawing parameters.
        3. Initializes the object detector and opens the video capture.
        4. Iterates through each frame of the video:
            - Resizes the frame.
            - Detects bounding boxes of objects in the frame.
            - Processes the frame to update lane sets and draw annotations.
            - Saves lane set information to a JSON file.
            - Optionally saves the annotated frame to an output video.
        5. Saves the ID-label dictionary to a file.
        6. Releases the video capture and destroys all OpenCV windows.

    Note:
        - The function uses OpenCV for video processing and drawing annotations.
        - The tqdm library is used to display a progress bar for frame processing.
    """
    warnings.filterwarnings("ignore")
    polygon_mask = _create_mask_image(config)
    color_polygons_image = _create_color_polygons_image(config)

    lane_sets = _initialize_lane_sets()
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_position = (int(960 * 0.01), int(540 * 0.05))

    detector = Detector()
    video_path = video_dir/config["input_video_path"]
    if video_path.is_file():
        capture = cv2.VideoCapture(video_path.as_posix())
    else:
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with open(results_dir/config["json_path"], "w", encoding="utf-8") as f:
        id_label_dict = {}
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            _, im = capture.read()
            if im is None:
                break
            else:
                im = cv2.resize(im, (960, 540))
                bboxes = detector.detect(im)

                output_image_frame, set_ids_in_frame = _process_frame(
                    im,
                    bboxes,
                    polygon_mask,
                    color_polygons_image,
                    lane_sets,
                    id_label_dict,
                    config,
                )

                if bboxes:
                    _remove_left_ids(set_ids_in_frame, lane_sets)
                    _save_lane_set(f, capture, lane_sets)
                else:
                    _clear_current_lane_sets(lane_sets)

                output_image_frame = _draw_text_on_frame(
                    output_image_frame,
                    lane_sets,
                    draw_text_position,
                    font_draw_number,
                )
                cv2.waitKey(1)
                if config.get("output_video_path"):
                    _save_output_video(config, output_image_frame)

    _save_id_label_dict(id_label_dict, config)
    capture.release()
    cv2.destroyAllWindows()
    return pd.DataFrame(
        list(id_label_dict.items()), columns=["track_id", "label"]
    )
