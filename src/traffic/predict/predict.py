from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def predict_level(data, item_id, output_path):
    # 删除所有包含 timestamp 的列
    data = data.drop(
        columns=[col for col in data.columns if "timestamp" in col.lower()]
    )

    # 计算划分点
    total_samples = len(data)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)

    # 划分数据集
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size : train_size + val_size]
    test_data = data.iloc[train_size + val_size :]

    # 对训练集进行子采样（如果需要的话）
    subsample_size = 500  # 可以根据需要调整这个值
    if len(train_data) > subsample_size:
        train_data = train_data.sample(n=subsample_size, random_state=0)

    label = "congestion_label"
    print(f"唯一类别: {list(train_data[label].unique())}")

    predictor = TabularPredictor(label=label).fit(train_data)
    y_true = test_data[label]

    # 存储每个模型的评分
    scores = []

    for model_name in predictor.model_names():
        # 预测结果
        y_pred = predictor.predict(test_data, model=model_name)

        # 计算评价指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        # 将每个模型的指标加入字典
        score = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        scores.append(score)
    # 判断目录是否存在
    output_dir = output_path / str(item_id)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # 将所有模型的指标保存为CSV文件
    pd.DataFrame(scores).to_csv(
        output_dir / "level_classify_model_metrics.csv",
        index=False,
    )

    # 生成并保存排行榜
    leaderboard = predictor.leaderboard(test_data)
    leaderboard.to_csv(
        output_dir / "level_classify_leaderboard.csv", index=False
    )

    return train_data, val_data, test_data


def predict_float(
    data,
    item_id,
    poi_df=None,
    split_timestamp="2024-05-01 13:45:00",
    perdict_length=20,
    output_path="",
):
    output_dir = output_path / str(item_id)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    train_data = data[data["timestamp"] < split_timestamp]
    test_data = data
    train_data = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column="item_id",
        timestamp_column="timestamp",
        static_features_df=poi_df,
    )
    test_data = TimeSeriesDataFrame.from_data_frame(
        test_data,
        id_column="item_id",
        timestamp_column="timestamp",
        static_features_df=poi_df,
    )
    # 创建并训练预测器
    predictor = TimeSeriesPredictor(
        prediction_length=perdict_length,
        target="overall_congestion",
        freq="T",
    )
    predictor.fit(train_data, time_limit=15)

    # 获取并打印排行榜
    leaderboard = predictor.leaderboard()
    best_score = leaderboard.iloc[0, 1]
    # 追加写入步长和最佳得分到txt文件
    with open(output_dir / "best_score.txt", "a") as f:
        f.write(f"{perdict_length},{str(best_score)}\n")
    return predictor


# 主预测函数
def predict_congestion():
    mode = "global"
    source = "raw"
    result_dir = Path(__file__).parents[2] / "results"
    poi_df = pd.read_csv(result_dir / "poi.csv")
    split_timestamp = "2024-05-01 13:45:00"
    for perdict_length in range(1, 50):
        if source == "raw":
            data = pd.read_csv(result_dir / "combined_data.csv")
        elif source == "level":
            data = pd.read_csv(
                result_dir / "labeled_congestion_data_final.csv"
            )
            data.drop(columns=["overall_congestion"], inplace=True)
        else:
            raise ValueError("数据源必须是raw或level")
        if mode == "global" and source == "raw":
            data = data.dropna()
            predict_float(
                data,
                item_id=100,
                poi_df=poi_df,
                split_timestamp=split_timestamp,
                perdict_length=perdict_length,
                output_path=result_dir,
            )
        elif mode == "local" and source == "raw":
            data = data.dropna()
            for item_id in [105]:  #
                data = data[data["item_id"] == item_id]
                predict_float(data, item_id, output_path=result_dir)
        elif mode == "global" and source == "level":
            data = data.dropna()
            predict_level(data, 100)
        elif mode == "local" and source == "level":
            for item_id in [107, 105, 108, 103]:  # 107,105,108,
                data = data[data["item_id"] == item_id]
                data = data.dropna()  # 删除有任何空值的行
                predict_level(data, item_id)


if __name__ == "__main__":
    predict_congestion()
