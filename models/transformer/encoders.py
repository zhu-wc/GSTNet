from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.containers import Module
from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithSwinBox
import torch.nn.init as init
import numpy as np
device = torch.device('cuda:3')

def GridRelationalEmbedding(batch_size, grid_size=7, dim_g=64, wave_len=1000, trignometric_embedding=True):
    # make grid
    a = torch.arange(0, grid_size).float().to(device)
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(batch_size, -1, -1)/grid_size
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    temp_x = delta_x
    delta_x = torch.clamp(torch.abs(delta_x / 2), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    temp_y = delta_y
    delta_y = torch.clamp(torch.abs(delta_y / 2), min=1e-3)
    delta_y = torch.log(delta_y)

    sqrt = torch.sqrt(temp_x*temp_x + temp_y*temp_y)
    sqrt = torch.clamp(torch.abs(sqrt / 2), min=1e-3)
    sqrt = torch.log(sqrt)

    add = temp_x + temp_y
    add = torch.clamp(torch.abs(add / 2), min=1e-3)
    add = torch.log(add)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    sqrt = sqrt.view(batch_size, matrix_size[1], matrix_size[2], 1)
    add = add.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, add, sqrt), -1)  # bs * r * r *3

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).to(device)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        #bias = torch.ones(batch_size,grid_size*grid_size,grid_size*grid_size,16).to(device) # 50 49 49 16
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)

class ScaledDotProductAttentionRela(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttentionRela, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, relative,attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        #TODO: 相对位置编码
        att = att + relative

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        # if recorder.activate is True:
        #     recorder.record(att, comment=self.comment)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttentionDis(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttentionDis, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttentionRela(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, relative,attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values,relative ,attention_mask, attention_weights)
            out = self.dropout(out)
            # add connecttion_weight
            out = self.layer_norm(queries + out)
            # print("connection_weight from M-HEAD", connection_weight)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt_dis = MultiHeadAttentionDis(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs
                                        )
        self.mhatt_pos = MultiHeadAttentionWithSwinBox(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs
                                        )
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, dis_relative,pos_relative,attention_mask=None, attention_weights=None):
        if self.training:
            rho = 0.3
        else:
            rho = 0.0
        pro = torch.rand(2)
        pro = (pro>=rho).float()

        att_dis = (self.mhatt_dis(queries, keys, values, dis_relative, attention_mask, attention_weights)*pro[0]) / (1 - rho)
        att_pos = (self.mhatt_pos(queries, keys, values, pos_relative, attention_mask, attention_weights)*pro[1]) / (1 - rho)
        att = (att_dis + att_pos)*0.5

        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff



class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs
                                                  ) for _ in range(N)])
        self.padding_idx = padding_idx

        #构建siwn中所需的相对位置编码矩阵
        window_size = (7, 7)
        num_heads = 1
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)).to(device)  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).to(device)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # init by normal
        init.normal_(self.relative_position_bias_table, mean=0, std=0.02)

        #非线性增大所需要的可学习参数
        self.WGs_pos = nn.ModuleList([nn.Linear(32, 1, bias=True) for _ in range(h)])
        self.WGs_dis = nn.ModuleList([nn.Linear(96, 1, bias=True) for _ in range(h)])
        self.bias = torch.ones(50, 49, 49, 32).to(device)  # 50 49 49 16

    def forward(self, input, attention_weights=None):
        bs = input.shape[0]
        #swin的相对位置编码
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(7*7, 7*7, 1)  # shape = (49,49,1)
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # 49, 49

        swin_relative = relative_position_bias.unsqueeze(0).repeat(input.shape[0], 1, 1, 1)# bs 49 49 1
        #end

        # no-linear increase
        dim_g = 32
        feat_range = torch.arange(dim_g+1)[1:].to(device).view(1,1,1,-1) # 1 1 1 32
        dim_mat = 1.0/feat_range #转化为0-1之间的小数

        mul_mat = swin_relative*70*dim_mat # (bs 49 49 1)*(1,1,1,32) = bs 49 49 32
        sin_mat = torch.cos(mul_mat).view(-1,dim_g)

        relative_pos_head = [l(sin_mat).view(input.shape[0],1,49,49) for l in self.WGs_pos]
        relative_pos = torch.cat((relative_pos_head), 1) # bs 8 49 49

        # 相对距离编码
        relative_geometry_embeddings = GridRelationalEmbedding(bs)
        relative_geometry_embeddings = torch.cat([relative_geometry_embeddings,self.bias[:bs]],-1)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 96)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs_dis]
        relative_dis = torch.cat((relative_geometry_weights_per_head), 1)
        relative_dis = F.relu(relative_dis)

        attention_mask = (torch.sum(input == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        outs = []
        out = input

        for l in self.layers:
            out = l(out, out, out,relative_dis,relative_pos, attention_mask, attention_weights)

        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)  # add by luo
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)
