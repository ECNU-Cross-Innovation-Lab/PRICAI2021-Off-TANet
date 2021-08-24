from models.dual_inception import DualInception
from models.off_apexnet import OffApexNet
from models.ststnet import STSTNet
from models.macnn import MACNN
from models.micro_attention import MicroAttention
from models.off_tanet import OffTANet

from torch import nn,optim
import torch
from data import CASMECombinedDataset

model_names = ['dual-inception','off-apexnet','ststnet','macnn','micro-attention','off-tanet']
model_name = model_names[-1]

from_file = 1
data_filename = 'dataset_112*112_strain.pkl'
data_class = CASMECombinedDataset
data_args = {'path':'/home/qwq/Datasets/CASMEs','img_sz' : 112,'calculate_strain' : True,'raw_img' : False}

n_epochs = 201
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
observed_epochs = set([i for i in range(0,n_epochs,20)])

batch_size = 64
model_class = OffTANet
model_args = {'net_type' : 'ta'}

optimizer_class = optim.Adam
optimizer_args = {'lr' : 1e-3}

scheduler_class = optim.lr_scheduler.StepLR
scheduler_args = {'step_size' : 60,'gamma' : 0.3}

print_debug_info = 1
train_process_filename = 'off-tanet.pkl'

if model_name == 'dual-inception':
    data_filename = 'dataset_28*28.pkl'
    data_args['img_sz'] = 28
    data_args['calculate_strain'] = False
    model_class = DualInception
    n_epochs = 501
    observed_epochs = set([i for i in range(0,n_epochs,20)])
    optimizer_args = {'lr' : 1e-3}
    scheduler_args = {'step_size' : 9999,'gamma' : 1}
    train_process_filename = 'dual-inception-result.pkl'
elif model_name == 'off-apexnet':
    data_filename = 'dataset_28*28.pkl'
    data_args['img_sz'] = 28
    data_args['calculate_strain'] = False
    model_class = OffApexNet
    n_epochs = 3000
    observed_epochs = set([i for i in range(0,n_epochs,200)])
    optimizer_args = {'lr' : 1e-4}
    scheduler_args = {'step_size' : 9999,'gamma' : 1}
    train_process_filename = 'off-apexnet-result.pkl'
elif model_name == 'ststnet':
    data_filename = 'dataset_28*28.pkl'
    data_args['img_sz'] = 28
    data_args['calculate_strain'] = False
    model_class = STSTNet
    n_epochs = 1001
    observed_epochs = set([i for i in range(0,n_epochs,20)])
    optimizer_args = {'lr' : 5e-5}
    scheduler_args = {'step_size' : 9999,'gamma' : 1}
    train_process_filename = 'ststnet-result.pkl'
elif model_name == 'macnn':
    data_filename = 'dataset_112*112_strain.pkl'
    data_args['img_sz'] = 112
    data_args['calculate_strain'] = True
    model_class = MACNN
    model_args = {}
    n_epochs = 501
    observed_epochs = set([i for i in range(0,n_epochs,20)])
    scheduler_args = {'step_size' : 50,'gamma' : 0.9}
    train_process_filename = 'macnn-result.pkl'
elif model_name == 'micro-attention':
    data_filename = 'dataset_224*224_strain.pkl'
    data_args['img_sz'] = 224
    data_args['calculate_strain'] = True
    model_class = MicroAttention
    model_args = {}
    n_epochs = 51
    observed_epochs = set([i for i in range(0,n_epochs,10)])
    optimizer_class = optim.SGD
    optimizer_args = {'lr':1e-3,'weight_decay':5e-4,'momentum':0.9}
    scheduler_args = {'step_size' : 10,'gamma' : 0.1}
    train_process_filename = 'micro-attention-result.pkl'
    batch_size = 10
