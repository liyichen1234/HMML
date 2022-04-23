import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
dtype = torch.FloatTensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class BiLSTM_Attention(nn.Module):
    def __init__(self,vocab_size,embedding_dim,n_hidden,num_classes):
        super(BiLSTM_Attention,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=n_hidden,dropout=0.5,bidirectional=True)
        self.out = nn.Linear(n_hidden*2,num_classes)
        self.n_hidden = n_hidden
    
    def attention_net(self,lstm_output,final_state):
        hidden = final_state.view(-1,self.n_hidden*2,1)
        attn_weights = torch.bmm(lstm_output,hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights)
        context = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context,soft_attn_weights

    def forward(self,X,device="cuda:0"):
        input = self.embedding(X)
        input = input.permute(1,0,2).to(device)
        hidden_state = Variable(torch.zeros(1*2,len(X),self.n_hidden)).to(device)
        cell_state = Variable(torch.zeros(1*2,len(X),self.n_hidden)).to(device)
        output,(final_hidden_state,final_cell_state) = self.lstm(input,(hidden_state,cell_state))
        output = output.permute(1,0,2)
        attn_output ,attention = self.attention_net(output,final_hidden_state)
        last = F.sigmoid(self.out(attn_output))
        return last,attention


class Bilstm_att(nn.Module):
    def __init__(self,vocab_size,embedding_dim,n_hidden,num_classes):
        super(Bilstm_att,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,n_hidden,num_layers=2,bidirectional=True,dropout=0.6)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(n_hidden*2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(n_hidden*2,64)
        self.fc = nn.Linear(64,num_classes)
        self.apply(_init_model_weights)
    
    def forward(self,X):
        emb = self.embeddings(X)
        H,_ = self.lstm(emb)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M,self.w),dim=1).unsqueeze(-1)
        out = H*alpha
        out = torch.sum(out,1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)
        out = F.sigmoid(out)
        return out,_
    

class GCN(nn.Module):
    def __init__(self, vocab_size,embedding_dim,hidden_channels,multi_classes):
        super(GCN, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, multi_classes)
        self.apply(_init_model_weights)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding(x).squeeze(1)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)   
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x).squeeze(0)
        x = torch.sigmoid(x)
        return x


def _init_model_weights(module):
   
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)        

    elif isinstance(module, nn.LSTM):
        nn.init.kaiming_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.kaiming_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
