import logging
import os
import subprocess

import numpy as np
import torch
import torcheval
from torcheval import metrics as _


def atoi(s):
    try:
        i = int(s)
    except Exception as e:
        return -1, str(e)
    
    return i, None


def get_checkpoint_path(checkpoint_dir, epoch):
    return os.path.join(checkpoint_dir, "checkpoint_{:06d}.pth".format(epoch))


def get_all_checkpoint_paths(checkpoint_dir):
    err = None
    try:
        fnames = os.listdir(checkpoint_dir)
    except Exception as e:
        err = str(e)
    if err:
        return []
    
    path_ms = []
    for fname in fnames:
        fpath = os.path.join(checkpoint_dir, fname)
        noext = os.path.splitext(os.path.basename(fpath))[0]
        splitted = noext.split("_")
        if not (len(splitted) == 2 and splitted[0] == "checkpoint"):
            continue
        epoch, err = atoi(splitted[1])
        if err:
            continue
        
        path_ms.append({"path": fpath, "epoch": epoch})
    
    path_ms.sort(key=lambda d: d["epoch"], reverse=True)
    
    return path_ms


def load_checkpoint(checkpoint_dir):
    # Torch being stupid by refusing to run numpy when loading checkpoints.
    torch.serialization.add_safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
    ])
    
    paths = get_all_checkpoint_paths(checkpoint_dir)
    if len(paths) == 0:
        return None, ""
    
    for path_d in paths:
        fpath = path_d["path"]
        checkpoint= None
        err = None
        try:
            checkpoint = torch.load(fpath, weights_only=True)
        except Exception as e:
            err = str(e)
            logging.info("%s %s", fpath, e)
        if not err:
            return checkpoint, fpath
    
    return None, ""


def delete_old_checkpoints(checkpoint_dir):
    paths = get_all_checkpoint_paths(checkpoint_dir)
    
    keep_num = 2
    if len(paths) <= keep_num:
        return

    for path_d in paths[keep_num:]:
        fpath = path_d["path"]
        subprocess.run(["rm", "-r", fpath])


def update_metric(metrics, k, v):
    if not(k in metrics):
        metrics[k] = torcheval.metrics.Mean().to(v.device)
    metrics[k].update(v)
    return metrics
