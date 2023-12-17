import torch
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model-path-1", type=str, required=True)
parser.add_argument("--model-path-2", type=str, required=True)
parser.add_argument("--megatron-path", type=str, required=True)
args = parser.parse_args()

sys.path.insert(0, args.megatron_path)

checkpoint_1 = torch.load(args.model_path_1, map_location="cpu")
checkpoint_2 = torch.load(args.model_path_2, map_location="cpu")


def compare_values(value1, value2):
    if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return value1.shape == value2.shape and torch.equal(value1, value2)
    elif isinstance(value1, argparse.Namespace) and isinstance(value2, argparse.Namespace):
        for subkey, subval in vars(value1).items():
            if subval != vars(value2).get(subkey, None):
                print(f"Different in Namespace: {subkey} = {subval} vs {vars(value2).get(subkey, None)}")
                return False
    elif isinstance(value1, list) or isinstance(value2, list):
        for v1, v2 in zip(value1, value2):  # type: ignore
            if not compare_values(v1, v2):
                print(f"Different: Values: {value1} vs {value2}")
                return False
    elif isinstance(value1, dict) or isinstance(value2, dict):
        for k in value1.keys():  # type: ignore
            if k not in value2:
                return False
            if not compare_values(value1[k], value2[k]):  # type: ignore
                print(f"Different Value: {k}, Values: {value1[k]} vs {value2[k]}")  # type: ignore
                return False
    elif isinstance(value1, tuple) or isinstance(value2, tuple):
        for v1, v2 in zip(value1, value2):  # type: ignore
            if not compare_values(v1, v2):
                print(f"Different: Values: {value1} vs {value2}")
                return False
    else:
        if value1 != value2:
            print(f"Different: Values: {value1} vs {value2}")
            return False
        return True


is_same = True

for key in checkpoint_1.keys():
    if key not in checkpoint_2:
        print(f"Only in checkpoint_1: {key}")
        is_same = False
        continue

    if not compare_values(checkpoint_1[key], checkpoint_2[key]):
        print(f"Different: {key}")
        is_same = False
        continue

for key in checkpoint_2.keys():
    if key not in checkpoint_1:
        print(f"Only in checkpoint_2: {key}")
        is_same = False
        continue

if is_same:
    print("Same")
