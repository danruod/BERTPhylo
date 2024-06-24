import torch

class SequenceDatasetMultiRank(torch.utils.data.Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, tax_ranks, labels_id, labels_name, 
                 labels_taxid, labels_maps, seq_tag):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.tax_ranks = tax_ranks
        self.labels_id = labels_id
        self.labels_name = labels_name
        self.labels_taxid = labels_taxid
        self.labels_maps = labels_maps
        self.seq_tag = seq_tag
        
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        token_type_id = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels_id[idx]
        
        tag = self.seq_tag[idx]
        return input_id, token_type_id, attention_mask, label, tag
    
    def __len__(self):
        return len(self.labels_id)
    
    