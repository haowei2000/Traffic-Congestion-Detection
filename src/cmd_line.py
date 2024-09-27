import click
from sympy import im
from traffic.detect.traffic_detect import detect_video
from traffic.analysis.traffic_info import get_traffic_infos
from traffic.analysis.congestion import apply_calculate_congestion
from traffic.chart.chart import draw_hist,draw_line
from pathlib import Path
import os
import yaml


@click.command()
def main():
    repo_path = Path(__file__).parent.parent
    os.environ['REPO_PATH'] = str(repo_path)
    config_path = repo_path / "res/configs/config_105_0.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    detect_video(config)
    infos_df=get_traffic_infos(config)
    with open(repo_path / "res/configs/congestion_config.yaml", 'r') as file:
        congestion_config = yaml.safe_load(file)
    congestion_df=apply_calculate_congestion(infos_df,**congestion_config)
    draw_line(congestion_df)
    draw_hist(congestion_df)




if __name__ == "__main__":
    main()