"""
Filename: my_path.py
Description: This file contains the path setting for project.
"""

from pathlib import Path

# Path to the project root directory
project_root = Path(__file__).parents[2]
src_dir = Path(__file__).parents[1]
config_dir = project_root / "res" / "configs"
results_dir = project_root / "src" / "results"
video_dir = project_root / "src" / "video" / "transport"
package_dir = Path(__file__).parents[0]
