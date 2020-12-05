import torch
from configparser import ConfigParser


parser = ConfigParser()
parser.read('./config.ini', encoding='utf-8')
model_config = parser['model_config']
max_seq_len = model_config.getint('max_seq_len')
pad_idx = model_config.getint('pad_idx')
sos_idx = model_config.getint('sos_idx')
eos_idx = model_config.getint('eos_idx')
unk_idx = model_config.getint('unk_idx')
vocab_size = model_config.getint('vocab_size')
device = torch.device('cuda:0')