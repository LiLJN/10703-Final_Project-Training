import os
import sys

# Make project root importable
THIS_FILE = os.path.abspath(__file__)
TEST_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from vlm import CFG, collect_dataset, train_clip_two_policies, evaluate_retrieval


if __name__ == "__main__":
    collect_dataset(CFG)

    model, tokenizer = train_clip_two_policies(CFG)

    evaluate_retrieval(model, tokenizer, CFG, num_samples=32)