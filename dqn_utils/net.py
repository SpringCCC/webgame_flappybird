
import torch.nn as nn

class DQNet(nn.Module):
    
    def __init__(self, env:Env, hidden_dim=32):
        super(DQNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.row_table = nn.Embedding(env.n_row,hidden_dim)
        self.col_table = nn.Embedding(env.n_col,hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), 
            nn.ReLU(), 
            nn.Linear(2*hidden_dim, hidden_dim*2), 
            nn.ReLU(),
            
            )
        self.v_head = nn.Linear(2*hidden_dim, 1)
        self.a_head = nn.Linear(2*hidden_dim, env.n_action)

    def forward(self, r, c):
        x = self.row_table(r) + self.col_table(c)
        x = self.net(x)
        v = self.v_head(x)
        a = self.a_head(x)
        y = v + a - a.mean(dim=-1, keepdim=True)
        return y