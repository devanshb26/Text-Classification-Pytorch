import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNN1d(nn.Module):
    def __init__(self,vocab_size,word_embeddings,INPUT_DIM,HIDDEN_DIM,EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM,DROPOUT,Dropout_2,weights):
        
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = EMBEDDING_DIM, 
                                              out_channels = N_FILTERS, 
                                              kernel_size = fs)
                                    for fs in FILTER_SIZES
                                    ])
        
        self.fc1 = nn.Linear(len(FILTER_SIZES) * N_FILTERS, 364)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(364,162)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(162,50)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(50,OUTPUT_DIM)
        nn.init.kaiming_normal_(self.fc4.weight)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout_2)
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
        
        #text = [batch size, sent len]
        
        embedded = self.embeddings(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        embedded=self.dropout(embedded)
        #embedded = [batch size, emb dim, sent len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = torch.cat(pooled, dim = 1)
        cat=self.dropout_2(cat)
        out=self.dropout_2(self.relu(self.fc1(cat)))
        out=self.dropout_2(self.relu(self.fc2(out)))
        out=self.relu(self.fc3(out))
#         out=self.dropout(out)
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc4(out)
