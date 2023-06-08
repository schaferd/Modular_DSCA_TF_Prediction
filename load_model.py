import torch
import os
import sys


def load_resave(model_path):
    model = torch.load(model_path)
    print(model.state_dict())
    torch.save(model.state_dict(),model_path)

def resave_model_files(run_path):
    activity_paths = [run_path+'/'+f for f in os.listdir(run_path) if "fold" in f and "cycle" in f and os.path.isfile(run_path+f) == False]
    for i, f in enumerate(activity_paths):
        paths = os.listdir(f)
        for p in paths:
            if "pth" in p:
                load_resave(f+'/'+p)

