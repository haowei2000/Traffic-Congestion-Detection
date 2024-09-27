import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def predict_classify(data, item_id, output_dir="results"):
    """
    Predict congestion using classification data.

    Parameters:
    data (DataFrame): The input data containing congestion information.
    item_id (int): The ID of the item to predict.

    Returns:
    Predictor: The classify predictor.    """
    data = data.drop(
        columns=[col for col in data.columns if "timestamp" in col.lower()]
    )
    label = "congestion_label"
    features = data.drop(columns=[label])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    data[features.columns] = scaled_features
    total_samples = len(data)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size: train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    subsample_size = 500  # 可以根据需要调整这个值
    if len(train_data) > subsample_size:
        train_data = train_data.sample(n=subsample_size, random_state=0)

    predictor = TabularPredictor(label=label).fit(train_data)
    y_true = test_data[label]

    # 存储每个模型的评分
    def get_score4model(model_name: str) -> dict:
        """
        根据模型名称获取得分
        """
        y_pred = predictor.predict(test_data, model_name)
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
        return score

    scores = [get_score4model(model_name) for model_name in predictor.get_model_names()]
    pd.DataFrame(scores).to_csv(
        f"{output_dir}/{item_id}/level_classify_model_metrics.csv",
        index=False,
    )
    leaderboard = predictor.leaderboard(test_data)
    leaderboard.to_csv(
        f"{output_dir}/{item_id}/level_classify_leaderboard.csv", index=False
    )
    return predictor


def predict_float(
        data,
        item_id,
        poi_df=pd.read_csv("results/poi.csv"),
        split_timestamp="2024-05-01 13:45:00",
        perdict_length=20,
        output_dir="results",
):
    """
        Predict congestion using float data.

        Parameters:
        data (DataFrame): The input data containing congestion information.
        item_id (int): The ID of the item to predict.
        poi_df (DataFrame): The Points of Interest (POI) data. Default is read from 'results/poi.csv'.
        split_timestamp (str): The timestamp to split the training and testing data. Default is '2024-05-01 13:45:00'.
        perdict_length (int): The length of the prediction. Default is 20.

        Returns:
        TimeSeriesPredictor: The trained time series predictor.
    """
    train_data = data[data["timestamp"] < split_timestamp]
    test_data = data
    # 对除了timestamp外的数据进行正则化
    features = data.drop(columns=["timestamp"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    data[features.columns] = scaled_features
    # 转换为时间序列数据格式
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
    with open(f"{output_dir}/{item_id}/best_score.txt", "a") as f:
        f.write(f"{perdict_length},{str(best_score)}\n")
    return predictor


# 主预测函数
def predict_congestion(mode="global", source="float", poi_path: str = "results/poi.csv",
                       split_timestamp="2024-05-01 13:45:00", output_dir="results"):
    """
    Main function to predict congestion based on the specified mode and source.

    Parameters:
    mode (str): The mode of prediction, either "global" or "local".
    source (str): The source of data, either "float" or "classify".
    poi_path (str): The file path to the Points of Interest (POI) data.
    split_timestamp (str): The timestamp to split the training and testing data.

    Raises:
    ValueError: If the source is not "float" or "classify".
    """
    poi_df = pd.read_csv(poi_path)
    if source == "float":
        data = pd.read_csv("results/combined_data.csv")
    elif source == "classify":
        data = pd.read_csv("results/labeled_congestion_data_final.csv")
        data.drop(columns=["overall_congestion"], inplace=True)
    else:
        raise ValueError("数据源必须是float或classify")
    data = data.dropna()
    for perdict_length in range(1, 50):
        if mode == "global" and source == "raw":
            predict_float(
                data,
                item_id=100,
                poi_df=poi_df,
                split_timestamp=split_timestamp,
                perdict_length=perdict_length,
                output_dir=output_dir
            )
        elif mode == "local" and source == "raw":
            for item_id in [107, 105, 108, 103]:
                data = data[data["item_id"] == item_id]
                predict_float(data, item_id, poi_df, split_timestamp, perdict_length, output_dir)
        elif mode == "global" and source == "classify":
            predict_classify(data, 100, output_dir)
        elif mode == "local" and source == "classify":
            for item_id in [107, 105, 108, 103]:  # 107,105,108,
                data = data[data["item_id"] == item_id]
                predict_classify(data, item_id, output_dir)


if __name__ == "__main__":
    predict_congestion()
