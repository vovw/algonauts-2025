# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .defaults import SAVEDIR

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def select_diverse_subset(C, k):
    """
    Greedy algorithm to select k most diverse indices.
    """
    import numpy as np
    n = C.shape[0]
    selected = [int(np.argmin(np.sum(np.abs(C), axis=0)))]  # Start with the least correlated overall

    while len(selected) < k:
        candidates = list(set(range(n)) - set(selected))
        scores = []
        for c in candidates:
            total_corr = sum(abs(C[c, s]) for s in selected)
            scores.append((c, total_corr))
        best = min(scores, key=lambda x: x[1])[0]
        selected.append(best)
    
    return selected

def get_k_most_diverse_indices(predictions, k):
    """
    Get the indices of the k most diverse predictors
    """
    preds = []
    for sub in predictions[0].keys():
        for chunk in tqdm(predictions[0][sub].keys(), desc="Gathering predictions for diverse subset estimation"):
            preds.append(np.array([data[sub][chunk] for data in predictions])) 
        break # only the first subject
    preds = np.concatenate(preds, axis=1)
    preds = preds.reshape(preds.shape[0], -1)
    assert preds.shape[0] == len(predictions)
    
    corr_matrix = np.corrcoef(preds)
    indices = select_diverse_subset(corr_matrix, k)
    return np.array(indices)

def average_submissions(grid_path: Path, weigh_by_score: bool = False, per_voxel_weights: bool = False, temperature: float = 1.0, max_runs: int | None = None, k_most_diverse: int | None = None):
    """
    Average the submissions of a grid.
    """
    checkpoint_paths = []
    # find all folders in grid_path and get the best.ckpt file
    print("Found the following submissions:")
    for folder in os.listdir(grid_path):
        if max_runs is not None and len(checkpoint_paths) == max_runs:
            break
        if os.path.isdir(os.path.join(grid_path, folder)):
            submission_path = os.path.join(grid_path, folder, "submission.zip")
            to_remove = os.path.join(grid_path, folder, "submission.npy")
            if os.path.exists(submission_path):
                checkpoint_paths.append(submission_path)
                print(submission_path)
            if os.path.exists(to_remove):
                os.remove(to_remove)
    print(f"Found {len(checkpoint_paths)} submissions")

    predictions = []
    scores = []
    pearsons = []

    def load_submission(path):
        try: 
            submission = np.load(path, allow_pickle=True)["submission"].item()
        except: 
            print(f"Error loading submission from {path}")
            return None
        metrics = pd.read_csv(path.replace("submission.zip", "metrics.csv"))
        if os.path.exists(path.replace("submission.zip", "pearson.npy")):
            pearson = np.load(path.replace("submission.zip", "pearson.npy"))
        else:
            pearson = None
        return submission, metrics, pearson

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_path = {executor.submit(load_submission, path): path for path in checkpoint_paths}
        
        for future in tqdm(as_completed(future_to_path), total=len(checkpoint_paths), desc="Loading submissions"):
            output = future.result()
            if output is None:
                continue
            predictions.append(output[0])
            scores.append(output[1])
            pearsons.append(output[2])

    if k_most_diverse is not None:
        indices = get_k_most_diverse_indices(predictions, k_most_diverse)
        predictions, scores = [predictions[i] for i in indices], [scores[i] for i in indices]

    if per_voxel_weights:
        pearsons = torch.Tensor(pearsons) / temperature
        weights = pearsons.softmax(dim=1) # n_submissions x n_voxels
        weights = np.array(weights.unsqueeze(1))
    else:
        scores = np.array([score["val/pearson"].item() for score in scores])
        weights = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
        weights = weights[:, None, None]
    print(weights.min(), weights.max())

    averaged_predictions = defaultdict(dict)
    for sub in tqdm(predictions[0].keys(), desc="Averaging submissions"):
        for chunk in predictions[0][sub].keys():
            preds = np.array([data[sub][chunk] for data in predictions]) # n_submissions x n_timepoints x n_voxels
            if weigh_by_score:
                avg_preds = np.sum(preds * weights, axis=0)
            else:
                avg_preds = np.mean(preds, axis=0)
            averaged_predictions[sub][chunk] = avg_preds

    submission_path = grid_path / "submission.npy"
    np.save(submission_path, averaged_predictions)
    with zipfile.ZipFile(submission_path.with_suffix(".zip"), "w") as zipf:
        zipf.write(submission_path, arcname = submission_path.name)
    print(f"Saved average submission to {submission_path.with_suffix('.zip')}")

if __name__ == "__main__":
    grid_path = Path(SAVEDIR) / "model_soup"
    weigh_by_score = True
    per_voxel_weights = True
    temperature = 0.3
    max_runs = None
    k_most_diverse = None
    average_submissions(grid_path=grid_path, weigh_by_score=weigh_by_score, per_voxel_weights=per_voxel_weights, temperature=temperature, max_runs=max_runs, k_most_diverse=k_most_diverse)