
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    带有attention计算的网络层
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        '''
        参数：in_features 输入节点的特征数F: 特征数怎么理解? 是指输入的dims?
        参数：out_features 输出的节点的特征数F'
        参数：dropout
        参数：alpha LeakyRelu激活函数的斜率
        参数：concat
        '''
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout      
        self.in_features = in_features                # 输入特征数
        self.out_features = out_features              # 输出特征数
        self.alpha = alpha                            # 激活斜率(LeakyReLU)的激活斜率
        self.concat = concat                          # 用来判断是不是最有一个attention

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))      # 建立一个w权重，用于对特征数F进行线性变化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)                          # 对权重矩阵进行初始化
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))              # 计算函数α，输入是上一层两个输出的拼接，输出的是eij，a的size为(2*F',1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)                          # 对a进行初始化
        self.leakyrelu = nn.LeakyReLU(self.alpha)                                 # 激活层

    def forward(self, h, adj):
        '''
        参数input：h 表示输入的各个节点的特征矩阵
        参数adj ：表示邻接矩阵
        '''
        # 线性变化特征的过程, h的size为(N,F')，N表示节点的数量，F‘表示输出的节点的特征的数量
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)           # [N,N]

        zero_vec = -9e15*torch.ones_like(e)                 # 生成一个矩阵，size为(N,N)
        attention = torch.where(adj > 0, e, zero_vec)       # 对于邻接矩阵中的元素，>0说明两种之间有变连接，就用e中的权值，否则表示没有变连接，就用一个默认值来表示
        attention = F.softmax(attention, dim=1)             # 做一个softmax，生成贡献度权重
        attention = F.dropout(attention, self.dropout, training=self.training)      # dropout操作
        h_prime = torch.matmul(attention, Wh)               # 根据权重计算最终的特征输出。

        if self.concat:
            return F.elu(h_prime)                           # 做一次激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])             # [N,out_feature] * [out_feature,1] --->[N,1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])             # [N,out_feature] * [out_feature,1] --->[N,1]
        # broadcast add
        e = Wh1 + Wh2.T                                                   # [N,N]
        return self.leakyrelu(e)

    def __repr__(self):
        # 打印输出类名称，输入特征数量，输出特征数量
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        """
            参数1 ：nfeat   输入层数量
            参数2： nhid    输出特征数量
            参数3： nclass  分类个数
            参数4： dropout dropout 斜率
            参数5： alpha  激活函数的斜率
            参数6： nheads 多头部分
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        # 根据多头部分给定的数量声明attention的数量
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # 将多头的各个attention作为子模块添加到当前模块中。
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 最后一个attention层，输出的是分类
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # 参数x：各个输入的节点得特征表示
        # 参数adj：邻接矩阵表示
        x = F.dropout(x, self.dropout, training=self.training)
        print("x1.size():",x.size())
        # 对每一个attention的输出做拼接
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print("s.size():",x.size())
        x = F.dropout(x, self.dropout, training=self.training)
        # 输出的是带有权重的分类特征
        x = F.elu(self.out_att(x, adj))
        # 各个分类的概率归一化
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class WebGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, style_feat, config=None):
        super(WebGAT,self).__init__()
        self.dropout = dropout
        self.style_feat = style_feat
        #style_embedding
        self.style_embedding_size = 50                            # 未用到
        self.embedding = nn.Linear(style_feat, style_feat)
        self.config = config
        self.device = config.device
        
        self.attentions1 = [GraphAttentionLayer(style_feat, nhid, dropout=dropout, alpha=alpha, concat=True).to(self.device) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention_layer1_{}'.format(i), attention)
        self.attentions2 = [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True).to(self.device) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention_layer2_{}'.format(i), attention)
            
        self.cls = nn.Linear(nhid * nheads, nclass)

    
    def forward(self, x, adj, node_length):
        '''
        x:  各个输入的节点得特征表示; [B, seq_len, dim]
        adj:  邻接矩阵表示; [B, seq_len, seq_len]
        node_length:
        '''
        style_embedding = x[:, :self.style_feat]
        text_embedding = x[:, self.style_feat:]
        style_embedding = self.embedding(style_embedding)
        x = style_embedding
        x = F.dropout(x, self.dropout, training=self.training)
        
        if self.config.gat_layers == 1:
          x = [att(x, adj) for att in self.attentions1]
        elif self.config.gat_layers == 2:
          x = [att(x, adj) for att in self.attentions1]
          x = [self.attentions2[idx](xi, adj) for idx, xi in enumerate(x)]         # 2layer
          
        x = torch.cat(x, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        node_0_embedding = x[node_length]     # 找到整个batch中每个html的first_node
        x = node_0_embedding                  # node_0_embedding: [b,1,dims_nhid_*_nheads]
        return x                              # [B, ]


if __name__ == "__main__":
  input = torch.randn([2,4])
  nfeat, nhid, nclass, dropout, alpha, nheads, style_feat = 4, 8, 2, 0.1, 0.1, 8, 2
  m = WebGAT(nfeat, nhid, nclass, dropout, alpha, nheads, style_feat)
  for k,v in m.state_dict().items():
    print(k,v.size())
  adj = torch.tensor([[1,0],[0,1]])
  output = m(input, adj)
  print(output)