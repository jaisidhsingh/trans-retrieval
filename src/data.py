from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from config import cfg


def preprocess_datafolder(datafolder, label_file):
	paths = os.listdir(datafolder)	
	classes = np.unique([filename[:-11] for filename in paths])
	classes = list(classes)
	for path in paths:
		label = path[:-11]
		if label not in classes:
			os.remove(os.path.join(datafolder, path))
		else:
			with open(label_file, 'a') as f:
				f.write(f'{datafolder+"/"+path}---{classes.index(label)}\n')
	
	print("Labels and paths stored in", label_file)


class OxfordBuildingsDataset(Dataset):
	def __init__(self, cfg, mode, transforms):
		self.mode = mode
		self.transforms = transforms

		# mdoel training function not written
		# however training dataset code done
		if self.mode == 'train':
			self.img_paths = []
			self.img_labels = []

			with open(cfg.label_file, 'r') as f:
				for line in f.readlines()[:cfg.train_split]:
					line = line.strip()
					img_path, label = line.split('---')
					self.img_paths.append(img_path)
					self.img_labels.append(int(label))

		# similar for validation data code
		if self.mode == 'val':
			self.img_paths = []
			self.img_labels = []

			with open(cfg.label_file, 'r') as f:
				for line in f.readlines()[cfg.train_split:cfg.train_split+cfg.val_split]:
					line = line.strip()
					img_path, label = line.split('---')
					self.img_paths.append(img_path)
					self.img_labels.append(int(label))

		# similar for testing data code
		if self.mode == 'test':
			self.img_paths = []
			self.img_labels = []

			with open(cfg.label_file, 'r') as f:
				for line in f.readlines()[cfg.train_split+cfg.val_split:cfg.train_split+cfg.val_split+cfg.test_split]:
					line = line.strip()
					img_path, label = line.split('---')
					self.img_paths.append(img_path)
					self.img_labels.append(int(label))

		# to query or retrieve data
		if self.mode == 'query':
			self.img_paths = []
			for path in os.listdir(cfg.query_dir):
				self.img_paths.append(os.path.join(cfg.query_dir, path))					
			self.img_labels = cfg.query_labels

		# to create the database to match the query
		if self.mode == 'database':
			self.img_paths = []
			self.img_labels = []

			with open(cfg.label_file, 'r') as f:
				for line in f.readlines():
					line = line.strip()
					img_path, label = line.split('---')
					self.img_paths.append(img_path)
					self.img_labels.append(int(label))

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, idx):
		img_path = self.img_paths[idx]
		img = Image.open(img_path).convert('RGB')

		if self.img_labels is not None:
			label = self.img_labels[idx]
		else:
			label = None

		if self.transforms is not None:
			img = self.transforms(img)
		
		return {'img_path': img_path, 'img': img, 'label': label}
