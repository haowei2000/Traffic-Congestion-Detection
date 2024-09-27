import pandas as pd
from datetime import datetime, timedelta
import jsonlines
from joblib import Parallel, delayed


def parse_time_precision(time_str):
    # 获取最后一个字符作为时间单位
    unit = time_str[-1]
    # 获取数字部分，默认时间单位为1（如 "D" 表示1天）
    value = int(time_str[:-1]) if time_str[:-1] else 1

    # 定义单位与秒数的映射
    unit_to_seconds = {
        "S": 1,  # 秒
        "M": 60,  # 分钟
        "H": 3600,  # 小时
        "D": 86400,  # 天
    }

    # 计算总秒数
    if unit in unit_to_seconds:
        return value * unit_to_seconds[unit]
    else:
        raise ValueError(f"不支持的时间单位: {unit}")


def count_per_time(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    grouped = df.groupby("timestamp")["current_ids"].apply(
        lambda x: set().union(*x)
    )
    return grouped.apply(len).reset_index(name="count")


def new_count_per_sec(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    grouped = df.groupby("timestamp")["current_ids"].apply(
        lambda x: set().union(*x)
    )
    result = []
    previous_ids = set()
    for timestamp, current_ids in grouped.items():
        new_ids = current_ids - previous_ids
        removed_ids = previous_ids - current_ids
        result.append(
            {
                "timestamp": timestamp,
                "new_count": len(new_ids),
                "removed_count": len(removed_ids),
            }
        )
        previous_ids = current_ids  # 更新 previous_ids
    return pd.DataFrame(result)


def filter_label(df, id_to_label, label):
    def filter_non_car_rows(row, label):
        row["current_ids"] = [
            car_id
            for car_id in row["current_ids"]
            if id_to_label.get(str(car_id)) == label
        ]
        row["total_ids"] = [
            car_id
            for car_id in row["total_ids"]
            if id_to_label.get(str(car_id)) == label
        ]

    if label is not None and label != "all":
        df.apply(filter_non_car_rows, axis=1, label=label)
    return df


def assign_timestamps(df, start_time, sec1frame, setAccuracy="30S"):
    df["timestamp"] = [
        start_time + timedelta(seconds=i * sec1frame) for i in range(len(df))
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor(setAccuracy)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def get_time(video_path):
    times = video_path.split("_")
    start_time = datetime.strptime(times[1], "%Y%m%d%H%M%S")
    end_time = datetime.strptime(times[2], "%Y%m%d%H%M%S")
    return start_time, end_time


def score_qkv(df, frames, video_path, distance, floor):
    start_time, end_time = get_time(video_path)
    total_time = (end_time - start_time).total_seconds()
    total_frames = len(df)
    sec1frame = total_time / total_frames
    frames = assign_timestamps(frames, start_time, sec1frame, floor)
    df = pd.concat(
        [frames.reset_index(drop=True), df.reset_index(drop=True)], axis=1
    )
    total_df = count_per_time(df)
    change_df = new_count_per_sec(df)
    merged_df = pd.merge(total_df, change_df, on="timestamp", how="outer")

    merged_df["q"] = merged_df["new_count"] / parse_time_precision(floor)
    merged_df["k"] = merged_df["count"] / distance
    merged_df["v"] = merged_df["q"] / merged_df["k"]
    return merged_df


def read_json(json_path):
    frames, lane0, lane1, lane2, lane3 = [], [], [], [], []
    with jsonlines.open(json_path) as reader:
        for item in reader:
            obj = item["lane_data"]
            frames.append(obj[0])
            lane1.append(obj[1])
            lane2.append(obj[2])
            lane3.append(obj[3])
            lane0.append(
                {
                    "lane_id": 3,
                    "current_ids": obj[1]["current_ids"]
                    + obj[2]["current_ids"]
                    + obj[3]["current_ids"],
                    "total_ids": obj[1]["total_ids"]
                    + obj[2]["total_ids"]
                    + obj[3]["total_ids"],
                    "current_count": obj[1]["current_count"]
                    + obj[2]["current_count"]
                    + obj[3]["current_count"],
                    "total_count": obj[1]["total_count"]
                    + obj[2]["total_count"]
                    + obj[3]["total_count"],
                }
            )
    return (
        pd.DataFrame(frames),
        pd.DataFrame(lane0),
        pd.DataFrame(lane1),
        pd.DataFrame(lane2),
        pd.DataFrame(lane3),
    )


def process_lane(df, frames, config, id_to_label, lane_index):
    results = None
    for label in ["car", "bus", "truck", "all"]:
        print(f"Processing {label} in lane {lane_index}")
        filtered_df = filter_label(df.copy(), id_to_label, label)
        result = score_qkv(
            filtered_df,
            frames,
            config["input_video_path"],
            distance=config["distance"],
            floor=config["floor"],
        )
        result.columns = [
            f"{col}_{label}_{lane_index}" if col != "timestamp" else col
            for col in result.columns
        ]
        if results is not None:
            results = pd.merge(result, results, on="timestamp")
        else:
            results = result
    return results


def calculate(config):
    frames, lane0, lane1, lane2, lane3 = read_json(config["json_path"])

    # 读取 id_label_path 文件并检查列名
    id_label_df = pd.read_csv(config["id_label_path"])
    id_label_df.columns = ["id", "label"]
    id_to_label = id_label_df.set_index("id").to_dict()["label"]
    results = Parallel(n_jobs=4)(
        delayed(process_lane)(df, frames, config, id_to_label, i)
        for i, df in enumerate([lane0, lane1, lane2, lane3])
    )
    results_df = pd.concat(results, axis=1)
    results_df.to_csv(config["time_series"], index=False)
    return results_df