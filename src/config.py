from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# data loading configs
cfg.device = 'cuda'
cfg.batch_size = 8
cfg.num_workers = 1

# model configs
cfg.vanilla_vit = "vit_base_patch16_224_in21k"
cfg.hybrid_vit = "vit_base_r50_s16_224_in21k"
cfg.deit = "deit_base_distilled_patch16_224"
cfg.swin = "swin_base_patch4_window7_224_in22k"

# model and method names to use
cfg.model_name = cfg.vanilla_vit
cfg.method = 'cls'

# number of highest matches to retrieve
cfg.k = 4

# dataset configs
cfg.dataset = 'oxford_buildings'
cfg.data_dir = '../oxbuild_images-v1'
cfg.label_file = './labels.txt'

with open(cfg.label_file, 'r') as f:
	cfg.data_len = len(f.readlines())

# query configs
cfg.query_dir = '../queries'
cfg.query_labels = [0]

# database configs
cfg.db_file = './db.json'
cfg.db_dir = '../oxbuild_desc'

# dataset splits
cfg.train_split = round(0.7*cfg.data_len)
cfg.val_split = round(0.2*cfg.data_len)
cfg.test_split = round(0.1*cfg.data_len)