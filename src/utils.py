import json
import numpy as np
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import matplotlib.pyplot as plt
from data import *
from config import cfg
from torch.utils.data import DataLoader
from tqdm import tqdm

# load in pretrained timm model
def get_model(model_name):
	model = timm.create_model(model_name, pretrained=True)
	model.eval()
	return model

# get timm model config for transforms
def get_model_config(model):
	model_config = resolve_data_config({}, model=model)
	return model_config

# get timm model transforms for data loading
def get_model_transforms(model_config):
	model_transforms = create_transform(**model_config)
	return model_transforms

# get image feature descriptor from timm transformer model
def get_descriptor(model, x, method='cls'):
	with torch.no_grad():
		x = model.patch_embed(x)
		x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
		x = model.pos_drop(x + model.pos_embed)
		x = model.blocks(x)
		x = model.norm(x)

	if method == 'cls':
		y = x[0][0]
		y = y.view((1, 1, y.shape[0]))
		return y
	
	elif method == 'all':
		return x

# compute cosine similarity between the descriptors
def get_similarity(tensor1, tensor2):
	t1 = tensor1.flatten()
	t2 = tensor2.flatten()
	return torch.nn.functional.cosine_similarity(t1, t2, dim=0).cpu().numpy()

# retrieve top "k" most similar images from the database
def get_topk_similar(query, database_config, k):
	scores = []
	for db_entry in database_config['descriptors']:
		database_tensor = np.load(db_entry['path'])
		database_tensor = torch.from_numpy(database_tensor)
		score = get_similarity(query, database_tensor)
		scores.append({
			'score':round(float(score), 4), 
			'path': db_entry['path'][:-4], 
			'label': db_entry['label']
		})

	scores.sort(key=lambda x: x['score'], reverse=True)	
	topk = scores[:k]
	return topk

# plot the topk matches, to be written
def plot_topk(img, topk):
	pass

# run the retrieval using a query folder
def run_retrieval(cfg):
	model = get_model(cfg.model_name)
	model.to(cfg.device)
	model_config = get_model_config(model)
	model_transforms = get_model_transforms(model_config)

	query_dataset = OxfordBuildingsDataset(
		cfg=cfg,
		mode='query',
		transforms=model_transforms
	)
	dataloader = DataLoader(query_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

	with open(cfg.db_file) as f:
		database_config = json.load(f)

	for data in tqdm(dataloader):
		images = data['img'].to(cfg.device)
		descriptors = get_descriptor(model, images, method=cfg.method)
		descriptors = descriptors.detach().cpu()

		for query in descriptors:
			topk = get_topk_similar(query, database_config, k=cfg.k)
	
	print('Top matches are: \n')
	for item in topk:
		print(item)

	print('Done')

# make the database to cross-reference later 
def make_database(cfg):
	model = get_model(cfg.model_name)
	model.to(cfg.device)
	model_config = get_model_config(model)
	model_transforms = get_model_transforms(model_config)
	
	dataset = OxfordBuildingsDataset(
		cfg=cfg,
		mode='database',
		transforms=model_transforms
	)
	
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

	for data in tqdm(dataloader):
		images = data['img'].to(cfg.device)
		img_path = str(data['img_path'])
		target_path = cfg.db_dir+'/'+img_path.split('/')[2][:-6]
		label = data['label']
		out_desc = get_descriptor(model, images, method=cfg.method)
		out_desc = out_desc.detach().cpu().numpy()
		np.save(f'{target_path}.npy', out_desc)

		db_entry = {'path': f'{target_path}.npy', 'label': int(label)}

		with open(cfg.db_file, 'r+') as f:
			db = json.load(f)
			db['descriptors'].append(db_entry)
			f.seek(0)
			json.dump(db, f, indent=3)