import os
import importlib
import os.path as osp
import torch

print(importlib.machinery.PathFinder().find_spec('_scatter_cuda', [osp.dirname(__file__)]))
print(importlib.machinery.PathFinder().find_spec('_version_cuda', [osp.dirname(__file__)]))
