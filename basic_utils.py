import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from load_data import NLPCustomDataset, NLPCollateFn, get_corpus
import torch
import numpy as np
from transformers import AutoTokenizer,BertTokenizer
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def get_dataloader(file_path, max_length, bs, neg_sample_num, init_PLM):
    
    data = get_corpus(file_path)
    dataset = NLPCustomDataset(data, neg_sample_num)
    
    if init_PLM == 'Prot_Bert':
        tokenizer = BertTokenizer.from_pretrained("./"+init_PLM)
        pad_id=tokenizer.pad_token_id 
        
    elif init_PLM == 'Transformer':
        pass
    else:
        tokenizer = AutoTokenizer.from_pretrained("./"+init_PLM)
        pad_id=tokenizer.pad_token_id 

        
    data_number = len(dataset)
    collator = NLPCollateFn(tokenizer, max_length)
    batch_size = bs # Set your desired batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    return dataloader, data_number,pad_id

def batch_sinkhorn(batch_matrices, epsilon=0.1, max_iters=10):
    batch_size, num_rows, num_cols = batch_matrices.shape
    #batch_matrices=torch.exp(batch_matrices)

    u = torch.ones((batch_size, num_rows), dtype=torch.float32, requires_grad=False).to(batch_matrices.device)
    v = torch.ones((batch_size, num_cols), dtype=torch.float32, requires_grad=False).to(batch_matrices.device)

    for _ in range(max_iters):
        u = 1.0 / (batch_matrices @ v.unsqueeze(-1)).squeeze(-1)
        v = (1.0 / (batch_matrices.transpose(1, 2) @ u.unsqueeze(-1))).squeeze(-1)
        #u = 1.0 / (batch_matrices @ v.unsqueeze(-1)).squeeze(-1)
        #v = 1.0 / (batch_matrices.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)

    normalized_batch_matrices = batch_matrices.clone()  
    for i in range(batch_size):
        normalized_batch_matrices[i] = torch.diag(u[i]) @ batch_matrices[i] @ torch.diag(v[i])
    return normalized_batch_matrices

def stable_softmax(input_data, dim):
    max_values, _ = torch.max(input_data, dim=dim, keepdim=True)
    input_data_exp = torch.exp(input_data - max_values)
    sum_exp = torch.sum(input_data_exp, dim=dim, keepdim=True)
    softmax_result = input_data_exp / sum_exp
    return softmax_result


def get_entropy(att, d):
    bs, m, n = att.size()

    if d == 2:  # softmax along dim 2 

        softmax_tensor = F.softmax(att, dim=2) #torch.exp(torch.log_softmax(att,dim=2))#F.softmax(att, dim=2)



    else:  # softmax along dim 1
        softmax_tensor =  F.softmax(att, dim=1)#torch.exp(torch.log_softmax(att,dim=1))#F.softmax(att, dim=1)
        
    log_softmax_tensor = torch.log(softmax_tensor)
    multiplied_tensor = softmax_tensor * log_softmax_tensor


    sum_tensor = torch.sum(multiplied_tensor, dim=2 if d == 2 else 1)

    if d == 2:
        entropy_tensor = sum_tensor.view(bs, m)
        total_entropy = torch.sum(entropy_tensor, dim=1)
        negative_entropy_tensor = -1 * total_entropy / m
    else:
        entropy_tensor = sum_tensor.view(bs, n)
        total_entropy = torch.sum(entropy_tensor, dim=1)
        negative_entropy_tensor = -1 * total_entropy / n

    return negative_entropy_tensor

def loss_fn(att_pos, att_neg): #parameter margin
    #threshold=0.3
    margin= 0.3
    e_pos = get_entropy(att_pos, 1) + get_entropy(att_pos, 2)
    e_neg = get_entropy(att_neg, 1) + get_entropy(att_neg, 2)
    #loss =  e_pos - e_neg 
    #e_pos[e_pos < threshold] = threshold
    loss = torch.max(e_pos - e_neg + margin, torch.zeros_like(e_pos))
    #loss =  e_pos
    loss_mean = torch.mean(loss)

    return loss_mean, torch.mean(e_pos) ,torch.mean(e_neg)

import torch
import torch.nn.functional as F

def masked_mae_loss(attention_scores, uniform_matrix, threshold):
    # print(attention_scores.shape)
    mask = attention_scores > threshold

    masked_attention_scores = torch.where(mask, attention_scores, torch.zeros_like(attention_scores))
    masked_uniform_matrix = torch.where(mask, uniform_matrix, torch.zeros_like(uniform_matrix))
    num_nonzero_elements = torch.sum(mask)

    if num_nonzero_elements > 0:
        mae_loss = F.l1_loss(masked_attention_scores, masked_uniform_matrix,reduction='mean') * masked_uniform_matrix.shape[-1]
        #average_mae_loss = mae_loss / num_nonzero_elements
    else:
        mae_loss = torch.tensor(0.0) 
        
    return mae_loss