from multiprocessing.dummy import freeze_support
from utils import *
from config import cfg
from data import *
import torch
from PIL import Image
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm


# parser arguments to be added

if __name__ == '__main__':
	# to resolve some child processs error 
	freeze_support()
	# run main task
	run_retrieval(cfg)