import os
import argparse
import random
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import BertModel

from model.probe import *
from utils.evaluate import test
from utils.cuda import get_max_available_gpu
from utils.dataset import SequenceDatasetMultiRank

import warnings
warnings.filterwarnings('ignore')

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)
    
    data_path = "./PlantSeqs/dnabert"
    bert_path = os.path.join("./checkpoints", f'{args.model_name}/bert')
    probe_path = os.path.join("./checkpoints", f'{args.model_name}/hlps.pt')
    
    results_path = os.path.join("./results", f'{args.model_name}/{time.strftime("%y%m%d-%H%M%S")}')
    
    if torch.cuda.is_available():
        try:
            device_id, _ = get_max_available_gpu()
        except:
            device_id = 0
        print(f"Using GPU: {device_id}")
    else:
        print("Using CPU")
    
    device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
    
    # load dataset
    print(f'############# Loading dataset #############')
        
    id_test_dataset = torch.load(os.path.join(data_path, "id_test_dataset.pt"))
    ood_test_dataset = torch.load(os.path.join(data_path, "ood_test_dataset.pt"))
    assert id_test_dataset.labels_maps == ood_test_dataset.labels_maps, f'The labels_map in ID test set and OOD test set are inconsistent. Please check.'
    
    id_test_data_loader = DataLoader(id_test_dataset, batch_size=args.batch_size, shuffle=False)
    ood_test_data_loader = DataLoader(ood_test_dataset, batch_size=args.batch_size, shuffle=False)
     
    # get labels_maps and tax_ranks
    labels_maps = id_test_dataset.labels_maps
    tax_ranks = id_test_dataset.tax_ranks
    print(f'------ Label information -----')
    for tax_rank in tax_ranks:
        print(f'{tax_rank}: {list(labels_maps[tax_rank].values())}')
    print('---------------')
    
    print(f'Number of sequences in test set: {len(id_test_dataset)} (ID), {len(ood_test_dataset)} (OOD)')
    
    # load model
    print(f'############# Loading {args.model_name} #############')
    ## load BERT module
    bert = BertModel.from_pretrained(bert_path)
    bert.to(device)
    for param in bert.parameters():
        param.requires_grad = False
    
    ## load hierarchical linear probes
    hlp_config_dict = {'device': device, 
                       'default_dtype': default_dtype,
                       'model_name': args.model_name,
                       'tax_ranks': tax_ranks,
                       'labels_maps': labels_maps,
                       } 
    hlp = LinearProbeMultiRank(**hlp_config_dict)
    hlp.to(device)
    hlp.load_state_dict(torch.load(probe_path, map_location=device))
    
    print(f'############# Testing #############')
    test(id_test_data_loader, ood_test_data_loader, bert, hlp, results_path=results_path, 
         exp_name=f'{args.model_name}_{args.suffix}', confusion_matrix=True)

    print('Done!')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Phylogeny Task')

    parser.add_argument('--model_name', '-m', type=str, default='bertphylo', choices=['bertphylo', 'dnabert'], help='Set model name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--suffix', type=str, default='test', required=False)
    
    args = parser.parse_args()
    
    main(args)
