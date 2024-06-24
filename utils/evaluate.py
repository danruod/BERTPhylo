import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib

from utils.utils import get_model_output
from utils.plot import visulization

figsize = [15, 15, 30, 50]

# OOD detection taxo
def anomaly_detection(output, ood_output, save_path=None): 
    p = torch.softmax(output, dim=-1)
    scores = p.max(-1)[0].cpu().detach().numpy()

    ood_p = torch.softmax(ood_output, dim=-1)
    ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()

    corrects = np.concatenate([np.ones(output.size(0)), np.zeros(ood_output.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)
    
    if save_path is not None:
        results = np.concatenate([corrects.reshape(-1, 1), scores.reshape(-1, 1)], axis=-1)
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)
        
    fpr, tpr, _ = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
 
    return auroc, aupr


def plot_confusion_matrix(figsize, labels, preds, labels_id, labels_name, save_path=None):
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    cm = confusion_matrix(labels, preds, labels=labels_id)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)

    disp.plot(ax=ax, xticks_rotation='vertical')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return


def calculate_metric(labels, logits, tags, labels_maps, tax_ranks, confusion_matrix=False, results_path=None):
    taxo = dict()
    taxo_tag = dict()
    
    ood = dict()
    ood_tag = dict()
    
    for i, tax_rank in enumerate(tax_ranks):
        # split id and ood sequences for each tax rank
        ood_idx = list(labels_maps[tax_rank].values()).index('ood')       
        is_ood = (labels[:, i] == ood_idx)
        
        # Evaluate the performance of taxonomic classfication
        if (~is_ood).sum() == 0:
            taxo[tax_rank] = None
        else:            
            if confusion_matrix and results_path is not None:
                if not os.path.exists(f'{results_path}/confusion_matrix/'):
                    os.makedirs(f'{results_path}/confusion_matrix/')
                plot_confusion_matrix(figsize[i], labels[~is_ood, i].cpu().numpy(), logits[i][~is_ood].argmax(-1).cpu().numpy(), 
                                      list(labels_maps[tax_rank].keys())[:-1], list(labels_maps[tax_rank].values())[:-1], 
                                      save_path=f'{results_path}/confusion_matrix/{tax_rank}.png')
            
            # classification_report
            taxo[tax_rank] = pd.DataFrame(classification_report(labels[~is_ood, i].cpu().numpy(), 
                                                                logits[i][~is_ood].argmax(-1).cpu().numpy(), 
                                                                labels=list(labels_maps[tax_rank].keys())[:-1], 
                                                                target_names=list(labels_maps[tax_rank].values())[:-1],
                                                                output_dict=True,
                                                                zero_division=np.nan))
            
             # Evaluate the performance of taxonomic classfication for each marker
            for tag in np.unique(tags):
                idx = (~is_ood.cpu().numpy()) * (tags == tag)
                # classification_report
                taxo_tag[f'{tax_rank}-{tag}'] = pd.DataFrame(classification_report(labels[idx, i].cpu().numpy(), 
                                                                                   logits[i][idx].argmax(-1).cpu().numpy(), 
                                                                                   labels=list(labels_maps[tax_rank].keys())[:-1], 
                                                                                   target_names=list(labels_maps[tax_rank].values())[:-1],
                                                                                   output_dict=True, 
                                                                                   zero_division=np.nan))
        
        # Evaluate the performance of novelty detection
        if sum(is_ood) > 0: 
            ood[tax_rank] = dict()
            ood_tag[tax_rank] = dict()
        
            if results_path is not None:
                if not os.path.exists(f'{results_path}/ood_scores/'):
                    os.makedirs(f'{results_path}/ood_scores/') 
                save_path=f'{results_path}/ood_scores/{tax_rank}.csv'
                
            ood[tax_rank]['auroc'], ood[tax_rank]['apr'] = anomaly_detection(logits[i][~is_ood], logits[i][is_ood], save_path=save_path)
            
            # Evaluate the performance of novelty detection for each marker
            for tag in np.unique(tags):
                id_idx = (~is_ood.cpu().numpy()) * (tags == tag)
                ood_idx = (is_ood.cpu().numpy()) * (tags == tag)
                ood_tag[tax_rank][f'{tag}-auroc'], ood_tag[tax_rank][f'{tag}-apr'] = anomaly_detection(logits[i][id_idx], logits[i][ood_idx])
                    
    return (taxo, ood, taxo_tag, ood_tag)


def evaluate(id_data_loader, ood_data_loader, bert, hlp, confusion_matrix=False, results_path=None):
    hlp.eval()
    bert.eval()
    
    id_acc, ood_acc = dict(), dict()
    id_logits_all, ood_logits_all, logits_all = dict(), dict(), dict()
    id_labels_all, ood_labels_all = [], []
    id_tags_all, ood_tags_all = [], []
        
    for i, tax_rank in enumerate(hlp.tax_ranks):
        id_acc[tax_rank], ood_acc[tax_rank] = 0.0, 0.0
        id_logits_all[i], ood_logits_all[i] = [], []
 
    # get logits, labels, markers for ID test set 
    for id_batch in tqdm(id_data_loader, desc="[Evaluate-ID dataset]"):
        id_logits, id_label, id_tags = get_model_output(bert, hlp, id_batch, eval=True, return_tag=True)
        
        for i, tax_rank in enumerate(hlp.tax_ranks):
            id_acc[tax_rank] += (id_logits[i].argmax(-1) == id_label[:, i].view(-1)).sum().item()
            id_logits_all[i].append(id_logits[i])

        id_labels_all.append(id_label)
        id_tags_all.append(id_tags)
        
        # break
    
    for i, tax_rank in enumerate(hlp.tax_ranks):
        id_logits_all[i] = torch.vstack(id_logits_all[i])
    id_labels_all = torch.vstack(id_labels_all)
    id_tags_all = np.hstack(id_tags_all)
     
    # get logits, labels, markers for OOD test set  
    for ood_batch in tqdm(ood_data_loader, desc="[Evaluate-OOD dataset]"):
        ood_logits, ood_label, ood_tags = get_model_output(bert, hlp, ood_batch, eval=True, return_tag=True)
        
        for i, tax_rank in enumerate(hlp.tax_ranks):
            ood_acc[tax_rank] += (ood_logits[i].argmax(-1) == ood_label[:, i].view(-1)).sum().item()
            ood_logits_all[i].append(ood_logits[i])

        ood_labels_all.append(ood_label)
        ood_tags_all.append(ood_tags)
        
        # break
    
    ood_labels_all = torch.vstack(ood_labels_all)
    ood_tags_all = np.hstack(ood_tags_all)
    
    # get logits, labels, markers for ID+OOD test set 
    for i, tax_rank in enumerate(hlp.tax_ranks):
        ood_logits_all[i] = torch.vstack(ood_logits_all[i])
        logits_all[i] = torch.vstack([id_logits_all[i], ood_logits_all[i]])
    labels_all = torch.vstack([id_labels_all, ood_labels_all])
    tags_all = np.hstack([id_tags_all, ood_tags_all])
    
    # evaluate
    metric = calculate_metric(labels_all, logits_all, tags_all, hlp.labels_maps, hlp.tax_ranks, 
                              confusion_matrix=confusion_matrix, results_path=results_path)
    taxo, ood, taxo_tag, ood_tag = metric
    
    if not os.path.exists(f'{results_path}/pr_roc'):
        os.makedirs(f'{results_path}/pr_roc') 
    visulization(labels_all, logits_all, hlp, fig_path=f'{results_path}/pr_roc')
        
    results_taxo = {
        'Taxo': taxo,
        'Tag-Taxo': taxo_tag,
    }
    
    results_ood = {
        'Novelty': ood,
        'Tag-Novelty': ood_tag,
    }
    
    return results_taxo, results_ood


def test(id_test_data_loader, ood_test_data_loader, bert, hlp, results_path, exp_name=None, confusion_matrix=False):
    results_taxo, results_ood = evaluate(id_test_data_loader, ood_test_data_loader, bert, hlp, confusion_matrix, results_path=results_path)
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
             
    with pd.ExcelWriter(os.path.join(results_path, f'{exp_name}.xlsx')) as writer:
        print('Novelty detection: ') 
        for key, value in results_ood.items():
            pd.DataFrame(value).to_excel(writer, sheet_name=key)
            print(f'{key}: ', pd.DataFrame(value))
        
        print('Taxonomic classification: ')
        for key, _ in results_taxo.items():
            results_taxo_all = None
            for tax_rank, value in results_taxo[key].items():
                if results_taxo_all is None:
                    results_taxo_all = value.iloc[:,-3:].rename(index=lambda x: tax_rank+'_'+x)
                else:
                    results_taxo_all = pd.concat([results_taxo_all, value.iloc[:,-3:].rename(index=lambda x: tax_rank+'_'+x)])
            
            results_taxo_all.to_excel(writer, sheet_name=f'{key}') 
            print(f'{key}: ', results_taxo_all) 

        for key, value in results_taxo.items():
            for name, tab in value.items():
                tab.to_excel(writer, sheet_name=f'{key}-{name}')

    return