import os
import sys

# add project root to sys.path
THIS_FILE = os.path.abspath(__file__)
TEST_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(TEST_DIR)
sys.path.append(PROJECT_ROOT)

from vlm import CFG, collect_dataset, train_clip_goal_vlm

if __name__ == "__main__":
    # 1) Generate dataset if not already present
    collect_dataset(CFG)

    # 2) Train CLIP + goal-head, prints loss + sample predictions
    train_clip_goal_vlm(CFG, lambda_clip=1.0, lambda_reg=1.0)
