import click
from traffic.detect.traffic_detect import detect_video
from traffic.analysis.traffic_info import calculate
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
    calculate(config)



if __name__ == "__main__":
    main()
