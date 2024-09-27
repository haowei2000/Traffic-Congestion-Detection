# Python函数实现，用于计算基于多车道的交通拥挤度，并考虑应急车道的影响。

import pandas as pd
import os
import yaml  # 新增导入yaml模块


# k_max: 2.5
# v_max: 2
# q_threshold: 0.5
# v_threshold: 0.5
def calculate_one_congestion(
    data: pd.Series, k_max, v_max, q_threshold, v_threshold
):
    total_weighted_q,total_weighted_congestion = 0,0
    # 遍历每个车道的拥挤度
    for lane in range(4):
        q = data[f"q_all_{lane}"]
        k = data[f"k_all_{lane}"]
        v = data[f"v_all_{lane}"]

        # 计算每个车道的拥挤度
        congestion = (k / k_max) * (1 - v / v_max)

        # 检查是否是应急车道 (lane_0)
        if lane == 3 and (q < q_threshold and v > v_threshold):
            continue
        # 计算加权拥挤度
        total_weighted_congestion += q * congestion
        total_weighted_q += q
    # 计算整体拥挤度
    if total_weighted_q == 0:
        return 0  # 避免除以0的情况
    return total_weighted_congestion / total_weighted_q


def apply_calculate_congestion(
    df, port, k_max, v_max, q_threshold, v_threshold
):
    df["overall_congestion"] = df.apply(
        lambda record: calculate_one_congestion(
            data=record,
            k_max=k_max,
            v_max=v_max,
            q_threshold=q_threshold,
            v_threshold=v_threshold,
        ),
        axis=1,
    )
    df["item_id"] = port
    return df


if __name__ == "__main__":
    with open(file="configs/congestion.yaml", mode="r") as file:
        congestion_config = yaml.safe_load(stream=file)
    all_data = []
    for port in [103, 105, 107, 108]:
        for j in range(3):
            if not os.path.exists(f"configs/config_{port}_{j}.yaml"):
                continue
            with open(f"configs/config_{port}_{j}.yaml", "r") as file:
                config = yaml.safe_load(file)
            path = config["time_series"]
            df = pd.read_csv(path)
            df = apply_calculate_congestion(
                df=df.copy(), port=port, **congestion_config
            )
            df.to_csv(path, index=False)
            all_data.append(df)
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)  # 纵向合并
    combined_data.reset_index(drop=True, inplace=True)
    combined_data = combined_data.loc[
        :, ~combined_data.columns.str.contains("^timestamp.")
    ]  # 删除timestamp.1类似的列

    combined_data = combined_data.loc[
        :, ~combined_data.columns.str.contains("^Unnamed:")
    ]  # 删除'Unnamed: 0'列
    combined_data = combined_data.sort_values(by="timestamp")
    combined_data.to_csv("results/combined_data2.csv", index=False)
