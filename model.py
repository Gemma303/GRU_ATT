import torch
import torch.nn as nn
import torch.nn.functional as F

class gru_gnn(nn.Module):
    def __init__(self,gru_lengths=[126,20,60],k_dim=8,dropout_rate=0.2,device='cuda'):
        super(gru_gnn,self).__init__()
        
        self.num_gru=len(gru_lengths)
        self.gru_lengths=gru_lengths
        self.k_dim=k_dim
        self.device=device
        
        self.weight_key_1=nn.Parameter(torch.zeros(size=(self.num_gru, self.k_dim)))
        nn.init.normal_(self.weight_key_1.data, mean=0, std=1/(self.num_gru ** 0.5))

        self.weight_query_1=nn.Parameter(torch.zeros(size=(self.num_gru, self.k_dim)))
        nn.init.normal_(self.weight_query_1.data, mean=0, std=1/(self.num_gru ** 0.5))
        
        self.W = nn.Parameter(torch.Tensor(self.num_gru*2, 1))
        nn.init.xavier_normal_(self.W)
        
        self.dropout = nn.Dropout(p=dropout_rate)  
        self.bn1=nn.BatchNorm1d(self.num_gru)
        self.bn2=nn.BatchNorm1d(self.num_gru*2)
        self.grus=nn.Sequential()
        for i in range(self.num_gru):
            self.grus.append(nn.GRU(1,1,1))
    
    def get_adjacency_matrix(self,X):
        
        key = torch.matmul(X.T,self.weight_key_1)
        query = torch.matmul(X.T,self.weight_query_1)
        
        attention=F.softmax(torch.matmul(key,query.T)/torch.sqrt(torch.tensor(self.k_dim)),dim=-1)
        attention = self.dropout(attention)
        attention=0.5*(attention+attention.T)
        
        return attention
    
    def forward(self,X):
        gru_results=[self.grus[i](X.unsqueeze(-1)[-self.gru_lengths[i]:])[-1][-1] for i in range(self.num_gru)]
        gru_results=self.bn1(torch.concat(gru_results,axis=-1)).T

        A=self.get_adjacency_matrix(gru_results)
        A_hat=torch.stack([torch.eye(A.shape[0]).to(self.device),A])
        
        gnn_results=torch.matmul(A_hat, gru_results.T).permute(1,0,2).contiguous().view(-1,self.num_gru*2)
        gnn_results=self.bn2(gnn_results)
        gnn_results = torch.matmul(gnn_results,self.W).squeeze()
        
        return gnn_results


class gru(nn.Module):
    def __init__(self,gru_lengths=[126,20,60],device='cuda'):
        super(gru,self).__init__()
        
        self.num_gru=len(gru_lengths)
        self.gru_lengths=gru_lengths
        self.device=device
        
        self.W = nn.Parameter(torch.Tensor(self.num_gru, 1))
        nn.init.xavier_normal_(self.W)
        
        self.bn1=nn.BatchNorm1d(self.num_gru)
        
        self.grus=nn.Sequential()
        for i in range(self.num_gru):
            self.grus.append(nn.GRU(1,1,1))
    
    def forward(self,X):
        gru_results=[self.grus[i](X.unsqueeze(-1)[-self.gru_lengths[i]:])[-1][-1] for i in range(self.num_gru)]
        gru_results=self.bn1(torch.concat(gru_results,axis=-1))
        final_results = torch.matmul(gru_results,self.W).squeeze()
        
        return final_results