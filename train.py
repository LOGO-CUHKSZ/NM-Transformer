import sys
import os

sys.path.append('./')
import torch
from att import *
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from basic_utils import *
# import wandb
import argparse
import datetime
#from geomloss import SamplesLoss
from sup_test import test


os.environ['CURL_CA_BUNDLE'] = ''


import torch
loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
mse_loss = nn.MSELoss(reduction='none')


def train_batch(model, dataloader_train, loss_type, optimizer, device):
    running_loss = 0.0
    running_loss_sup = 0.0
    for batch_idx, data in enumerate(dataloader_train):
        emb_pos = data['input_ids_pos'].to(device)
        emb_gt = data['input_ids_gt'].to(device)
        emb_neg = data['input_ids_neg'].to(device)
        label=data['label'].to(device) #-1
        negative_label=data['negative_label'].to(device) #-1
        gt_att_mask=data['gt_att_mask'].to(device)
        pos_att_mask= data['pos_att_mask'].to(device)
        neg_att_mask=data['neg_att_mask'].to(device)
        optimizer.zero_grad()


        att_pos, emb_out_pos, emb_out_gt, _, _ = model.forward(emb_pos, emb_gt,pos_att_mask, gt_att_mask)
        # att_neg, emb_out_neg, _, mask_xy,gt_len= model.forward(emb_neg, emb_gt, neg_att_mask ,gt_att_mask)
        bs = emb_out_pos.shape[0]
        #batch_data=emb_out_pos 
        logits = emb_out_gt
        loss_sup = torch.mean(loss_fct(logits.view(-1, logits.size(-1)), label.view(-1)).view(label.shape))
        loss_neg=0
        #start_memory = torch.cuda.memory_allocated()


        if loss_type == 's':
            loss = loss_sup


        elif loss_type == 's_p':
            att_pos=F.softmax(att_pos, dim=1)
            l1_regularization = torch.sum(torch.abs(att_pos)) / (att_pos.size(0)* 
                                                                 att_pos.size(1) * att_pos.size(2))
            loss =loss_sup + l1_regularization
            
        elif loss_type == 's_n':
            att_neg, emb_out_neg, _, mask_xy,gt_len= model.forward(emb_neg, emb_gt, neg_att_mask ,gt_att_mask)
            threshold = 1/(att_neg.shape[-1])
            neg_num, batch_size, num = att_neg.shape[0],att_neg.shape[1], att_neg.shape[2]
            att_neg=att_neg*mask_xy
            uniform_attention_matrix = torch.ones((neg_num, batch_size, num, num), dtype=torch.float).to(device) / gt_len.unsqueeze(-1).unsqueeze(-1)
            # uniform_attention_matrix = mask_xy * torch.full((neg_num, batch_size,  num, num), 1.0 / (num)).to(device)
            num_ones = mask_xy.sum(dim=(2, 3))#,keepdim=True)
            loss_neg=mse_loss(att_neg, uniform_attention_matrix)
            loss_neg=torch.mean(loss_neg.sum(dim=(2,3))/torch.sqrt(num_ones)) 
            loss = loss_sup + loss_neg*args.nl_coe
        

        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        if loss_type=='s_n' :#or loss_type=='s':
            running_loss_sup += loss_neg.item()
        else:
            running_loss_sup += loss_sup.item()

    return running_loss/len(dataloader_train), running_loss_sup/len(dataloader_train)


def validate_batch(model, dataloader_val, loss_type, device):
    running_loss = 0.0
    running_loss_sup = 0.0

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient tracking for validation
        for batch_idx, data in enumerate(dataloader_val):
            emb_pos = data['input_ids_pos'].to(device)
            emb_gt = data['input_ids_gt'].to(device)
            emb_neg = data['input_ids_neg'].to(device)
            label=data['label'].to(device) #-1
            negative_label=data['negative_label'].to(device) #-1
            gt_att_mask=data['gt_att_mask'].to(device)
            pos_att_mask= data['pos_att_mask'].to(device)
            neg_att_mask=data['neg_att_mask'].to(device)
            att_pos, emb_out_pos, emb_out_gt,_,_ = model.forward(emb_pos, emb_gt,pos_att_mask, gt_att_mask)
            att_neg, emb_out_neg, _, mask_xy,_ = model.forward(emb_neg, emb_gt, neg_att_mask ,gt_att_mask)
            bs = emb_out_pos.shape[0]
            batch_data=emb_out_gt
            batch_input = batch_data#batch_data[perm_indices]
            logits = batch_input
            loss_sup = torch.mean(loss_fct(logits.view(-1, logits.size(-1)), label.view(-1)).view(label.shape))
            running_loss += loss_sup.item()
            running_loss_sup += loss_sup.item()

    return running_loss/len(dataloader_val), running_loss_sup/len(dataloader_val)

# To use this function in your validation loop:

def train_with_early_stopping(model, dataloader_train, dataloader_val, loss_type, optimizer,
                              device, num_epochs=15, patience=4,filename=None,hidden_size=128,num_class=1195):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    accuracy_list=[]
    loss_list=[]
    loss_neg_list=[]
    train_loss_list=[]
    best_acc=0
    for epoch in range(num_epochs):
        running_loss, running_loss_neg = train_batch(model, dataloader_train, loss_type, optimizer, device)
        val_running_loss, val_running_loss_sup = validate_batch(model, dataloader_val, loss_type, device)
        loss_neg_list.append(running_loss_neg)
        train_loss_list.append(running_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss}, Validation Loss: {val_running_loss}")
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss_neg}")

        # Check if validation loss improved
        if  val_running_loss < best_val_loss :
            best_val_loss = val_running_loss
            out_dir = './checkpoints'
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            out_dir = os.path.join(out_dir, filename)
            torch.save(model.state_dict(), out_dir)
            accuracy=test(args.test_file, out_dir, args.max_sequence_length, args.batch_size,args.esm_grad, args.neg_sample_num,args.model,init_PLM=args.init_PLM,hidden_size=hidden_size,num_class=num_class)
            accuracy_list.append(accuracy)
            loss_list.append(val_running_loss)
            print(accuracy_list)
            best_acc=accuracy

            #print(loss_list)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs without improvement in validation loss.")
            # return best_acc
            break

    print("Training complete.")
    return best_acc

    



def main(args):
    set_seed(args.seed)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(current_time)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_file_train = args.input_file_path + f"train.json"
    input_file_valid = args.input_file_path + f"val.json"
    max_seq_len = args.max_sequence_length
    bs = args.batch_size
    hidden_size = args.hidden_size
    esm_grad = args.esm_grad
    init_PLM=args.init_PLM
    esm_coe=args.esm_coe
    cls_coe=args.cls_coe
    num_class=args.num_class
 
    ################
    dataloader_train, num,pad_id = get_dataloader(input_file_train, max_seq_len, bs, args.neg_sample_num,init_PLM)
    dataloader_valid, num,pad_id = get_dataloader(input_file_valid, max_seq_len, bs, args.neg_sample_num,init_PLM)
    
    
    if args.model=='T':
       print(args.init_PLM)
       model = CustomTransformerModel(hidden_size,args.esm_grad,args.neg_sample_num,init_PLM=args.init_PLM,num_class=num_class).to(device)
    elif args.model=='M':
        model= CustomMLP(hidden_size,args.esm_grad,args.neg_sample_num,init_PLM=args.init_PLM,num_class=num_class).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print('The parameter count is :{}'.format(pytorch_total_params))
    num_epochs = args.epochs
    lr = args.learning_rate
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.model=='T':
        optimizer = optim.Adam([
                                {'params':model.encoder.parameters(),'lr':esm_coe*args.learning_rate,'weight_decay':1e-5}, #0}, 5
                                {'params':model.custom_attention.parameters(),'lr':cls_coe*args.learning_rate,'weight_decay':1e-5},
                                #{'params':model.readout.parameters(),'lr':args.learning_rate,'weight_decay':1e-5} # 1
                                {'params':list(model.readout.parameters()) + list(model.classifier.parameters()),'lr':cls_coe*args.learning_rate,'weight_decay':1e-5}
                                ])
    elif args.model=='M':
        optimizer = optim.Adam([
                                #{'params':model.readout.parameters(),'lr':10*args.learning_rate} #ali2 1 10
                                {'params':list(model.readout.parameters()) + list(model.classifier.parameters()),'lr':cls_coe*args.learning_rate} #ali2 1 10                                
                                ])

    filename = f"_checkpoints_{current_time}_dataset_{args.dataset}_init_PLM{init_PLM}_lr{lr}_bs{bs}_seq{max_seq_len}_loss_{args.loss}_neg_num{args.neg_sample_num}_esmgrad{args.esm_grad}_seed{args.seed}_bs{bs}_nl_{args.nl_coe}_att_emb.pt"

    accuracy=train_with_early_stopping(model, dataloader_train, dataloader_valid, args.loss, optimizer, device, num_epochs,filename=filename,hidden_size=hidden_size,num_class=num_class)

    
    

    
    #####################
    filename = './result.txt'
    with open(filename, 'a') as file:
        file.write(f"Seed: {args.seed}\n")
        np.savetxt(file, [accuracy], fmt='%.6f')




    #####################


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train the alignment')
    parser.add_argument('--learning_rate', type=float, default = 0.0001, help='Learning rate for the model')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')
    parser.add_argument('--max_sequence_length', type=int, default=40, help='Maximum sequence length')
    parser.add_argument('--input_file_path', type=str, help='Path to the input file')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of input word embedding')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--esm_grad', type = int, default = 0, help='Determine to use pretrained model or not')
    parser.add_argument('--loss', type = str, default = 's', help='entropy loss, only_neg, only_pos, s:supervise, s_e,s_n,s_p')
    parser.add_argument('--seed', type=int, default=42, help='seeds')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test file')
    parser.add_argument('--neg_sample_num', type=int, default=2, help='Path to the test file')
    parser.add_argument('--model', type=str, default='T', help='transformer or MLP')
    parser.add_argument('--init_PLM', type=str, default='esm1b', help='transformer or MLP')
    parser.add_argument('--esm_coe', type=float, default=10, help='coefficient')
    parser.add_argument('--cls_coe', type=float, default=1, help='coefficient')
    parser.add_argument('--dataset', type=str, default='fold', help='coefficient')
    parser.add_argument('--num_class', type=int, default=1195, help='coefficient')
    parser.add_argument('--nl_coe', type=float, default=0.1, help='coefficient')

    args = parser.parse_args()
    
    main(args)
