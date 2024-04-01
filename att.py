import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import math
import os
from transformers import AutoModel, AutoTokenizer
from transformers import EsmModel,EsmConfig,BertModel

os.environ['CURL_CA_BUNDLE'] = ''
import torch
import torch.nn as nn
from torch.nn.functional import softmax

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, input_dim,neg_sample_num):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.neg_sample_num=neg_sample_num
        self.W_q = nn.Linear(input_dim, input_dim, bias=False)  # 
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)  # 
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)  # 

    def forward(self, x, y, mask_x, mask_y): # x:pos/neg y:gt

        
        Q_x = self.W_q(x)  
        K_x = self.W_k(x)  
        Q_y = self.W_q(y) 
        K_y = self.W_k(y) 
        V_x = self.W_v(x)    
        V_y = self.W_v(y)    
         
        
        b, m, d = Q_x.size() 
        bs, m, d =K_y.size() 
        attention_scores_x = torch.einsum('bid,bjd->bij', Q_x, K_x)
        attention_scores_y = torch.einsum('bid,bjd->bij', Q_y, K_y)
        if bs==b:
            attention_scores_xy = torch.einsum('bid,bjd->bij', Q_x, K_y)
        else:
                K_y_reshaped = K_y.repeat(1, self.neg_sample_num, 1, 1).view(b//bs,bs,m,d)#.transpose(0,1)

                Q_x_expanded = (Q_x.view(bs, b//bs, m, d)).transpose(0, 1)# (b//bs, b, m, d) 


                attention_scores_xy = torch.matmul(Q_x_expanded, K_y_reshaped.transpose(2, 3))# (b//bs, bs, m, d)  * (b//bs, bs, d, m) 


        
        mask_x = mask_x.unsqueeze(2).bool()
        mask_y = mask_y.unsqueeze(2).bool()
        mask_x_fill=mask_x * mask_x.transpose(1, 2)


        mask_y_fill=mask_y * mask_y.transpose(1, 2)
        y_len=0#torch.sum(mask_y.repeat(self.neg_sample_num,1,1).squeeze(2),dim=-1)#.view(bs,b//bs)

        attention_scores_x=attention_scores_x.clone().masked_fill_(~mask_x_fill, -1e10)
        attention_scores_y=attention_scores_y.clone().masked_fill_(~mask_y_fill,-1e10)
        if bs==b:
            mask_xy = mask_x * mask_y.transpose(1, 2)
        else:

            mask_xy = mask_x.view(bs, b//bs, m,1).transpose(0, 1) * mask_y.repeat(self.neg_sample_num, 1, 1, 1).transpose(2, 3)

            # y_len =torch.sum(torch.cat([mask_y]*self.neg_sample_num,dim=0),dim=1)#.#.transpose(0, 1)
           
            y_len =torch.sum(mask_y.repeat(1,self.neg_sample_num,1).view(bs,b//bs,-1),dim=2).transpose(0, 1)

        attention_scores_xy=attention_scores_xy.clone().masked_fill_(~mask_xy,-1e10)#float('-inf'))


        attention_probs_x = softmax(attention_scores_x, dim=-1)
        # #print(attention_scores_x)

        attention_probs_y = softmax(attention_scores_y, dim=-1)
        attention_probs_xy = softmax(attention_scores_xy, dim=-1)

        
        contextual_representation_x = torch.einsum('bij,bjd->bid', attention_probs_x, V_x)
        contextual_representation_y = torch.einsum('bij,bjd->bid', attention_probs_y, V_y)

        return contextual_representation_x, contextual_representation_y, attention_scores_x, attention_scores_y, attention_probs_xy, mask_xy, y_len

class CustomTransformerModel(nn.Module):
    def __init__(self, hidden_size, use_pretrained=1,neg_sample_num=2, num_attention_heads=1,init_PLM='esm1b',num_class=1195):
        super(CustomTransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.neg_sample_num=neg_sample_num
        self.num_class=num_class
        if init_PLM=='Prot_Bert':
             print("./"+init_PLM)
             self.encoder = BertModel.from_pretrained("./"+init_PLM)
        else:  
            config=EsmConfig.from_pretrained("."+init_PLM)
            self.encoder = EsmModel.from_pretrained("./"+init_PLM, config = config, proxies = {'https':"xx:xx"})#AutoModel.from_pretrained(model_name)
        if use_pretrained:
            print('param.requires_grad =  True')
            for param in self.encoder.parameters():
                    param.requires_grad = True
        else:
            print('param.requires_grad = False')
            for param in self.encoder.parameters():
                    param.requires_grad = False
        self.custom_attention = CustomMultiHeadAttention(num_attention_heads, self.encoder.config.hidden_size,self.neg_sample_num)

        self.readout=nn.Sequential(nn.Linear(self.encoder.config.hidden_size, self.hidden_size), 
                                                nn.ReLU())  #num_class
        self.classifier=nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size+num_class))  #num_class

    def forward(self, x_inputs, y_inputs,mask_x,mask_y):

        x_outputs = self.encoder(x_inputs,mask_x)#.last_hidden_state
        y_outputs = self.encoder(y_inputs,mask_y)#.last_hidden_state
        #print(x_inputs[mask_x])
        contextual_representation_x, contextual_representation_y, attention_scores_x, attention_scores_y, attention_scores_xy,mask_xy,y_len = self.custom_attention(x_outputs.last_hidden_state, 
        y_outputs.last_hidden_state,mask_x,mask_y)
        
        representation_x =contextual_representation_x* mask_x.unsqueeze(2)
        valid_tokens_x = torch.sum(mask_x, dim=1)
        representation_x = torch.sum(representation_x, dim=1) / valid_tokens_x.unsqueeze(1)


        representation_y = contextual_representation_y * mask_y.unsqueeze(2)
        valid_tokens_y = torch.sum(mask_y, dim=1)
        representation_y = torch.sum(representation_y, dim=1) / valid_tokens_y.unsqueeze(1)


        representation_x= self.readout(representation_x) 
        representation_y= self.readout(representation_y) 
        representation_x= self.classifier(representation_x) 
        representation_y= self.classifier(representation_y) 
        
        return attention_scores_xy, representation_x[:,-self.num_class:], representation_y[:,-self.num_class:], mask_xy, y_len #, attention_scores_x, attention_scores_y, attention_scores_xy
    
    def get_emb(self, x_inputs, y_inputs, mask_x,mask_y):
        x_outputs = self.encoder(x_inputs,mask_x)#.last_hidden_state
        y_outputs = self.encoder(y_inputs,mask_y)#.last_hidden_state
        
        #print(x_inputs[mask_x])

        contextual_representation_x, contextual_representation_y, attention_scores_x, attention_scores_y, attention_scores_xy,mask_xy,y_len = self.custom_attention(x_outputs.last_hidden_state, 
        y_outputs.last_hidden_state,mask_x,mask_y)
        
        representation_x =contextual_representation_x* mask_x.unsqueeze(2)
        valid_tokens_x = torch.sum(mask_x, dim=1)
        representation_x = torch.sum(representation_x, dim=1) / valid_tokens_x.unsqueeze(1)


        representation_y = contextual_representation_y * mask_y.unsqueeze(2)
        valid_tokens_y = torch.sum(mask_y, dim=1)
        representation_y = torch.sum(representation_y, dim=1) / valid_tokens_y.unsqueeze(1)
        representation_x= self.readout(representation_x) 
        representation_y= self.readout(representation_y) 
        return representation_x, representation_y
        



class CustomMLP(nn.Module):
    def __init__(self, hidden_size, use_pretrained=1,neg_sample_num=2, num_attention_heads=1,init_PLM='esm1b',num_class=1195,pad_id=0):
        super(CustomMLP, self).__init__()
        self.hidden_size = hidden_size
        self.neg_sample_num=neg_sample_num
        self.num_class=num_class
        self.pad_id=pad_id
        config=EsmConfig.from_pretrained("./"+init_PLM)
        print("./"+init_PLM)
        if init_PLM =='Prot_Bert':
            self.encoder = BertModel.from_pretrained("./"+init_PLM)
            print("using Prot_Bert")
            
        elif init_PLM =='Transformer':
            pass
        
        else:
            config=EsmConfig.from_pretrained("./"+init_PLM)
            self.encoder = EsmModel.from_pretrained("./"+init_PLM, config = config)

        if use_pretrained:
            print('param.requires_grad =  True')
            for param in self.encoder.parameters():
                    param.requires_grad = True
        else:
            print('param.requires_grad = False')
            #with torch.no_grad():
            for param in self.encoder.parameters():
                    param.requires_grad = False
        self.readout=nn.Sequential(nn.Linear(self.encoder.config.hidden_size, self.hidden_size), 
                                                nn.ReLU())  #num_class
        self.classifier=nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size+num_class))  #num_class

    def forward(self, x_inputs, y_inputs,mask_x,mask_y):

        x_outputs = self.encoder(x_inputs,attention_mask=mask_x)#.last_hidden_state
        y_outputs = self.encoder(y_inputs,attention_mask=mask_y)#.last_hidden_state



        representation_x = x_outputs.last_hidden_state* mask_x.unsqueeze(2)
        valid_tokens_x = torch.sum(mask_x, dim=1)
        representation_x = torch.sum(representation_x, dim=1) / valid_tokens_x.unsqueeze(1)


        representation_y = y_outputs.last_hidden_state* mask_y.unsqueeze(2)
        valid_tokens_y = torch.sum(mask_y, dim=1)
        representation_y = torch.sum(representation_y, dim=1) / valid_tokens_y.unsqueeze(1)

        representation_x=self.readout(representation_x)
        representation_y=self.readout(representation_y)
        representation_x= self.classifier(representation_x) 
        representation_y= self.classifier(representation_y) 

        return 0, representation_x[:,-self.num_class:], representation_y[:,-self.num_class:],0,0
    
    
    def get_emb(self, x_inputs, y_inputs, mask_x,mask_y):
        x_outputs = self.encoder(x_inputs,attention_mask=mask_x)#.last_hidden_state
        y_outputs = self.encoder(y_inputs,attention_mask=mask_y)#.last_hidden_state
        #print(x_inputs[mask_x])


        representation_x = x_outputs.last_hidden_state* mask_x.unsqueeze(2)
        valid_tokens_x = torch.sum(mask_x, dim=1)
        representation_x = torch.sum(representation_x, dim=1) / valid_tokens_x.unsqueeze(1)


        representation_y = y_outputs.last_hidden_state* mask_y.unsqueeze(2)
        valid_tokens_y = torch.sum(mask_y, dim=1)
        representation_y = torch.sum(representation_y, dim=1) / valid_tokens_y.unsqueeze(1)
        representation_x=self.readout(representation_x)
        representation_y=self.readout(representation_y)
        return  representation_x, representation_y
