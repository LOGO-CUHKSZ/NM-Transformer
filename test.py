import torch
import torch.nn as nn
import torch.nn.functional as F
from att import CustomTransformerModel, CustomMLP
import torch.optim as optim
from basic_utils import get_dataloader, set_seed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

os.environ['CURL_CA_BUNDLE'] = ''

set_seed(520)
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        super(LinearClassifier, self).__init__()
        self.linear = nn.Sequential(nn.Linear(self.input_size, 128),
                                               nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        out = self.linear(x)
        return out


def compute_T_SNE(input_tensor): #the input tensor has a dim to be [total_data_num*max_seq_len, hidden_dim]
# Convert tensor to numpy array
    print(input_tensor.shape)
    flattened_array =input_tensor.cpu()
    flattened_array = flattened_array.detach().numpy()
    dataset_num = len(flattened_array)

# Compute t-SNE
    tsne = TSNE(n_components=2)  # Set the desired number of components
    tSNE_representation = tsne.fit_transform(flattened_array)


# Reshape t-SNE representation
    tSNE_representation = tSNE_representation.reshape(dataset_num, 2)
    split_idx = dataset_num/3

# Visualize t-SNE embeddings
    for idx in range(dataset_num):
        if idx < split_idx:
            c= 'b'
        elif split_idx<= idx< 2*split_idx:
            c = 'g'
        else:
            c = 'r'
        plt.scatter(tSNE_representation[idx, 0], tSNE_representation[idx, 1], c = c).set

    plt.savefig('finetune.png')
    plt.show()





def test(data_file, model_file, max_seq_len, bs,esm_grad,neg_sample_num,model='T',init_PLM='esm1b',hidden_size=128,num_class=1195):
    dataloader, total_data_num,_ = get_dataloader(data_file, max_seq_len, bs,neg_sample_num,init_PLM)
    print('The total data number is {}'.format(total_data_num))
    if model=='T':
       model = CustomTransformerModel(hidden_size,esm_grad,neg_sample_num,init_PLM=init_PLM,num_class=num_class)#.to(device)
    elif model=='M':
        model= CustomMLP(hidden_size,esm_grad,neg_sample_num,init_PLM=init_PLM,num_class=num_class)#.to(device)
    model.load_state_dict(torch.load(model_file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  
        for batch_idx, data in enumerate(dataloader):
            emb_pos = data['input_ids_pos'].to(device)
            emb_gt = data['input_ids_gt'].to(device)
            label=data['label'].to(device) #-1
            gt_att_mask=data['gt_att_mask'].to(device)
            pos_att_mask= data['pos_att_mask'].to(device)
            neg_att_mask=data['neg_att_mask'].to(device)
            _, emb_encode_pos, emb_encode_gt,_ ,_= model.forward(emb_pos, emb_gt,pos_att_mask, gt_att_mask)


            bs= emb_encode_gt.shape[0]
            batch_data=emb_encode_gt 
            logits = batch_data

            _, predicted = torch.max(logits, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total

        print(f'Test Accuracy: {accuracy:.2f}%')

    return accuracy



def main(args):
    data_file = args.input_file_path
    test_data_file= args.test_file
    model_file =args.Pretrain_model_path 
    MLP_save_path = args.MLP_save_path
    max_seq_len = args.max_sequence_length
    bs = args.batch_size
    num_epoch = args.epochs
    esm_grad=args.esm_grad
    
    test(test_data_file, model_file, max_seq_len, bs,esm_grad)
    print(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune')
    parser.add_argument('--learning_rate', type=float, default = 0.0001, help='Learning rate for the Finetune model')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for Finetunning')
    parser.add_argument('--max_sequence_length', type=int, default=40, help='Maximum sequence length')
    parser.add_argument('--input_file_path', type=str, help='Path to the input file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test file')
    parser.add_argument('--MLP_save_path', type=str, required=True, help='Path to save finetune model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Finetune epochs')
    parser.add_argument('--Pretrain_model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--esm_grad', type = int, default = 0, help='Determine to use pretrained model or not')
   
    args = parser.parse_args()
    main(args)

