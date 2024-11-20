import logging
import os
from pathlib import Path

import click
import pandas as pd
import yaml

from traffic.analysis.calculate_congestion import calculate_infos
from traffic.analysis.traffic_info import get_traffic_infos
from traffic.chart.chart import draw_hist, draw_line
from traffic.detect.traffic_detect import detect_video
from traffic.my_path import results_dir
from traffic.predict.predict import predict_congestion


@click.command()
def detect():
    '''
    Detect traffic information from videos.
    '''
    logging.basicConfig(level=logging.WARNING)
    repo_path = Path(__file__).parent.parent
    os.environ['REPO_PATH'] = str(repo_path)
    for port in [105, 107, 108, 103]:
        for part in range(4):
            config_path = repo_path / f"res/configs/config_{port}_{part}.yaml"
            if config_path.exists():
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                detect_video(config)
                get_traffic_infos(config)
            else:
                print(f"config_{port}_{part}.yaml not found.")

@click.command()
def calculate():
    """
    Calculate congestion information from traffic data.
    """
    calculate_infos()

@click.command()
def draw():
    """Draw charts from traffic data.
    """
    logging.basicConfig(level=logging.WARNING)
    congestion_df=pd.read_csv(results_dir / "combined_data.csv")
    draw_line(congestion_df,output_dir=results_dir)
    draw_hist(congestion_df,output_dir=results_dir)

@click.command()
def predict():
    """
    Predict congestion information."""
    predict_congestion()


if __name__ == "__main__":
    detect()
    draw()
    predict()