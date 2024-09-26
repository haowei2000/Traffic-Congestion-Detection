import os
import yaml
from detect.detect_video import detect_video

if __name__ == "__main__":
    config_path = 'configs/config_103_0.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    if 'json_path' in config and os.path.exists(config['json_path']):
        print(f"{config['json_path']} exists, skipping detect_video.")
    else:
        detect_video(config_path)

