import argparse
import os
import numpy as np
import glob

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--transitions-dir', default='dropo_datasets/transitions', type=str, help='Path to the transitions directory')

	return parser.parse_args()

args = parse_args()
transitions_dir = args.transitions_dir

collection_types = [
    "observations",
    "next_observations",
    "actions",
    "terminals",
]

for ct in collection_types:
    collected = []
    
    matching_filenames = glob.glob(os.path.join(transitions_dir, "*_" + ct + ".npy"))
    
    for filename in matching_filenames:
        collected.append(np.load(filename))
    
    collection = np.concatenate(collected)

    np.save(os.path.join(transitions_dir, ct), collection)