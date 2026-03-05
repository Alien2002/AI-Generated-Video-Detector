import argparse
import json
import os

import numpy as np

from utils.calibration import fit_platt_scaling, save_calibrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="Path to a .npy of raw probabilities")
    ap.add_argument("--labels", required=True, help="Path to a .npy of labels (0/1, 1=AI)")
    ap.add_argument("--out", default="checkpoints/calibration/platt_fusion.json")
    args = ap.parse_args()

    probs = np.load(args.probs)
    labels = np.load(args.labels)

    cal = fit_platt_scaling(probs, labels)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_calibrator(args.out, cal)
    print(json.dumps(cal.to_dict(), indent=2))


if __name__ == "__main__":
    main()
