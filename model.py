import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from torch import Tensor
from torch_encodings import PositionalEncoding1D, Summer

################################################################################################
################################################################################################
################################################################################################

"""
Module: Time-encoder
"""
class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.as_tensor(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32), dtype=torch.float32)).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output



################################################################################################
################################################################################################
################################################################################################
"""
Module: DLP
"""

class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, dims, 
                 channel_expansion_factor=4, 
                 dropout=0.2,
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')

        self.dims = dims
        if 'token' in self.module_spec:
            self.transformer_encoder = _MultiheadAttention(d_model=dims, 
                                                           n_heads=2,
                                                           d_k=None,
                                                           d_v=None,
                                                           attn_dropout=dropout)
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        
    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.transformer_encoder.reset_parameters()
        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        x = self.transformer_encoder(x, x, x)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)
        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(attn_dropout))

    def reset_parameters(self):
        self.to_out[0].reset_parameters()
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


    
class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)

class Patch_Encoding(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers, dropout,
                 channel_expansion_factor,
                 window_size,
                 module_spec=None, 
                 use_single_layer=False
                ):
        super().__init__()
        self.per_graph_size = per_graph_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mixer_blocks.append(
                TransformerBlock(hidden_channels, 
                                 channel_expansion_factor, 
                                 dropout, 
                                 module_spec=None, 
                                 use_single_layer=use_single_layer)
            )
        # padding
        self.stride = window_size
        self.window_size = window_size
        self.pad_projector = nn.Linear(window_size*hidden_channels, hidden_channels)
        self.p_enc_1d_model_sum = Summer(PositionalEncoding1D(hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, batch_size, inds):
           

        # x : [ batch_size, graph_size, edge_dims+time_dims]
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        x = torch.zeros((batch_size * self.per_graph_size, 
                         edge_time_feats.size(1)), device=edge_feats.device)
        x[inds] = x[inds] + edge_time_feats         
        x = x. view(-1, self.per_graph_size//self.window_size, self.window_size*x.shape[-1])
        x = self.pad_projector(x)
        x = self.p_enc_1d_model_sum(x) 
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)    
        x = self.layernorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x
    



#############################################
# Dual edge predictor 
#############################################
class EdgePredictor_per_node(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int = 100):
        """
        dim_in: runtime embedding size D for each h_* vector (encoder out [+ node feats if used])
        We score MLP([h_src_combined || h_pos_dst1 || h_pos_dst2]) -> 1 logit
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in * 3, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, h, pos_edge_label, neg_samples=1, src_ids1=None, src_ids2=None):
        """
        Returns:
          pred_true  : logits for aligned pairs with label==1
          pred_false : logits for aligned pairs with label==0
          pred_neg   : logits for sampled negatives
          idx1, idx2 : alignment indices (for memory update)
        Layout of h (unchanged from your pipeline):
          [ src1 | pos1 | neg1 | src2 | pos2 | neg2 ]
        """
        D = h.shape[1]
        num_edge = h.shape[0] // (2 * neg_samples + 4)

        h_src1      = h[:num_edge]
        h_pos_dst1  = h[num_edge:2*num_edge]
        h_neg_dst1  = h[2*num_edge:2*num_edge + num_edge*neg_samples]

        base2       = 2*num_edge + num_edge*neg_samples
        h_src2      = h[base2 : base2 + num_edge]
        h_pos_dst2  = h[base2 + num_edge : base2 + 2*num_edge]
        h_neg_dst2  = h[base2 + 2*num_edge : ]

        # --- align src IDs across datasets ---
        id2_to_idx = {int(nid): j for j, nid in enumerate(src_ids2[:num_edge].tolist())}
        idx1_list, idx2_list = [], []
        for i, nid in enumerate(src_ids1[:num_edge].tolist()):
            j = id2_to_idx.get(int(nid))
            if j is not None:
                idx1_list.append(i)
                idx2_list.append(j)

        if len(idx1_list) == 0:
            dev = h.device
            empty = torch.empty(0, 1, device=dev)
            return empty, empty, empty, torch.empty(0, dtype=torch.long, device=dev), torch.empty(0, dtype=torch.long, device=dev)

        idx1 = torch.as_tensor(idx1_list, dtype=torch.long, device=h.device)
        idx2 = torch.as_tensor(idx2_list, dtype=torch.long, device=h.device)

        # --- positives (aligned) ---
        # h_src_combined = h_src1.index_select(0, idx1) + h_src2.index_select(0, idx2)
        h_src_combined = 0.5 * (h_src1.index_select(0, idx1) + h_src2.index_select(0, idx2))
        h_pos_dst1_a   = h_pos_dst1.index_select(0, idx1)
        h_pos_dst2_a   = h_pos_dst2.index_select(0, idx2)

        h_pos_edge = torch.cat([h_src_combined, h_pos_dst1_a, h_pos_dst2_a], dim=-1)
        pred_pos   = self.mlp(h_pos_edge)  # logits for all aligned pairs

        # split by ground-truth label
        y_pos = pos_edge_label.index_select(0, idx1).view(-1, 1)
        mask_true  = (y_pos == 1)
        mask_false = (y_pos == 0)
        pred_true  = pred_pos[mask_true].view(-1, 1)
        pred_false = pred_pos[mask_false].view(-1, 1)

        # --- negatives (3 cases) ---
        pred_neg = torch.empty(0, 1, device=h.device)
        if neg_samples > 0 and h_neg_dst1.numel() > 0 and h_neg_dst2.numel() > 0:
            rows1, rows2 = [], []
            for i in idx1.tolist():
                rows1.extend(range(i*neg_samples, (i+1)*neg_samples))
            for j in idx2.tolist():
                rows2.extend(range(j*neg_samples, (j+1)*neg_samples))
            rows1 = torch.as_tensor(rows1, dtype=torch.long, device=h.device)
            rows2 = torch.as_tensor(rows2, dtype=torch.long, device=h.device)

            h_neg_dst1_a = h_neg_dst1.index_select(0, rows1)  # [K*neg, D]
            h_neg_dst2_a = h_neg_dst2.index_select(0, rows2)  # [K*neg, D]

            K = idx1.size(0)
            h_src_rep  = h_src_combined.unsqueeze(1).expand(K, neg_samples, D).reshape(K*neg_samples, D)
            h_pos1_rep = h_pos_dst1_a.unsqueeze(1).expand(K, neg_samples, D).reshape(-1, D)
            h_pos2_rep = h_pos_dst2_a.unsqueeze(1).expand(K, neg_samples, D).reshape(-1, D)

            h_neg_edge1 = torch.cat([h_src_rep, h_neg_dst1_a, h_pos2_rep], dim=-1)  # [src, neg1, pos2]
            h_neg_edge2 = torch.cat([h_src_rep, h_pos1_rep, h_neg_dst2_a], dim=-1)  # [src, pos1, neg2]
            h_neg_edge3 = torch.cat([h_src_rep, h_neg_dst1_a, h_neg_dst2_a], dim=-1) # [src, neg1, neg2]

            pred_neg = torch.cat([self.mlp(h_neg_edge1), self.mlp(h_neg_edge2), self.mlp(h_neg_edge3)], dim=0)

        return pred_true, pred_false, pred_neg, idx1, idx2
    



############################################
# Persistent node memory (buffer + time-aware EMA)
############################################
class NodeMemory(nn.Module):
    def __init__(self, num_nodes, mem_dim):
        super().__init__()
        self.register_buffer("memory", torch.zeros(num_nodes, mem_dim))
        self.register_buffer("last_update_ts", torch.full((num_nodes,), -1.0))

    @torch.no_grad()
    def ema_update(self, node_ids, new_states, ts, half_life= 40.0):
        if node_ids.numel() == 0:
            return
        prev_ts = self.last_update_ts.index_select(0, node_ids)
        dt = torch.clamp(ts - prev_ts, min=0.0)
        # convert Δt to alpha via half-life: alpha = exp(-ln2*Δt/half_life)
        alpha = torch.exp(-0.69314718 * dt / half_life).unsqueeze(1)  # [B,1]
        old = self.memory.index_select(0, node_ids)
        new = alpha * old + (1 - alpha) * new_states.detach()
        self.memory.index_copy_(0, node_ids, new)
        self.last_update_ts.index_copy_(0, node_ids, ts)


    def get(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Return memory rows for node_ids"""
        if node_ids is None or node_ids.numel() == 0:
            return self.memory.new_zeros((0, self.memory.size(1)))
        if node_ids.dtype != torch.long:
            node_ids = node_ids.long()
        node_ids = node_ids.to(self.memory.device)
        return self.memory.index_select(0, node_ids)



############################################
# Gated memory fusion
############################################
class GatedMemoryFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d, d)

    def forward(self, x, m):
        # x, m: [N, d]
        g = self.gate(torch.cat([x, m], dim=-1))
        return x + g * self.proj(m)



############################################
# Light-weight dynamics
############################################
class NodePropensity(nn.Module):
    """
    Maintains a scalar b[u] ~ EMA of recent positive rate per src node.
    Read: get(node_ids) -> [N] scalars
    Write: ema_update(node_ids, targets in {0,1}, ts, half_life)
    """
    def __init__(self, num_nodes):
        super().__init__()
        self.register_buffer("bias", torch.zeros(num_nodes, 1))
        self.register_buffer("last_update_ts", torch.full((num_nodes,), -1.0))

    @torch.no_grad()
    def ema_update(self, node_ids, targets, ts, half_life=40.0):
        if node_ids.numel() == 0:
            return
        prev_ts = self.last_update_ts.index_select(0, node_ids)
        dt = torch.clamp(ts - prev_ts, min=0.0)
        alpha = torch.exp(-0.69314718 * dt / half_life).unsqueeze(1)  # [B,1]
        old = self.bias.index_select(0, node_ids)
        tgt = targets.view(-1, 1).float()
        new = alpha * old + (1.0 - alpha) * tgt
        self.bias.index_copy_(0, node_ids, new)
        self.last_update_ts.index_copy_(0, node_ids, ts)

    def get(self, node_ids):
        return self.bias.index_select(0, node_ids).view(-1)  # [N]


class PreferenceMemory(nn.Module):
    """
    Stores m[u] ~ EMA of dst embeddings for *positive* edges per src node.
    Score contribution: alpha * m[u]^T h_dst
    """
    def __init__(self, num_nodes, dim):
        super().__init__()
        self.register_buffer("pref", torch.zeros(num_nodes, dim))
        self.register_buffer("last_update_ts", torch.full((num_nodes,), -1.0))

    @torch.no_grad()
    def ema_update(self, node_ids, dst_states, ts, half_life=40.0):
        if node_ids.numel() == 0:
            return
        prev_ts = self.last_update_ts.index_select(0, node_ids)
        dt = torch.clamp(ts - prev_ts, min=0.0)
        alpha = torch.exp(-0.69314718 * dt / half_life).unsqueeze(1)
        old = self.pref.index_select(0, node_ids)
        new = alpha * old + (1.0 - alpha) * dst_states.detach()
        self.pref.index_copy_(0, node_ids, new)
        self.last_update_ts.index_copy_(0, node_ids, ts)

    def get(self, node_ids):
        return self.pref.index_select(0, node_ids)  # [N, D]


class RecencyTracker(nn.Module):
    """
    Tracks last-seen timestamp per src node.
    Recency feature: log(1 + Δt_since_last_src_event)
    """
    def __init__(self, num_nodes, default_dt=1.0):
        super().__init__()
        self.register_buffer("last_src_ts", torch.full((num_nodes,), -1.0))
        self.default_dt = float(default_dt)

    @torch.no_grad()
    def get_dt(self, node_ids, ts):
        last = self.last_src_ts.index_select(0, node_ids)
        dt = ts - last
        # for unseen nodes, use a default positive dt
        fill = torch.full_like(dt, self.default_dt)
        dt = torch.where(last >= 0, dt, fill)
        return torch.clamp(dt, min=0.0)

    @torch.no_grad()
    def update_src(self, node_ids, ts):
        if node_ids.numel() == 0:
            return
        self.last_src_ts.index_copy_(0, node_ids, ts)


############################################
# Dual Interface (Transformer + Memory)
############################################
# class Dual_Interface(nn.Module):
#     def __init__(self,
#                  mlp_mixer_configs,
#                  edge_predictor_configs,
#                  num_nodes,
#                  mem_dim,
#                  ema_alpha: float = 0.3,
#                  use_memory: bool = True,
#                  shared_memory: bool = False,
#                  consistency_coef: float = 0.05,
#                  half_life: float = 40.0):
#         super(Dual_Interface, self).__init__()

#         self.time_feats_dim = edge_predictor_configs['dim_in_time']
#         self.node_feats_dim = edge_predictor_configs['dim_in_node']
#         if self.time_feats_dim > 0:
#             self.base_model = Patch_Encoding(**mlp_mixer_configs)

#         self.enc_out = mlp_mixer_configs['out_channels']
#         self.eff_D   = self.enc_out + (self.node_feats_dim if self.node_feats_dim > 0 else 0)

#         self.edge_predictor = EdgePredictor_per_node(dim_in=self.eff_D,
#                                                      dim_hidden=max(100, self.enc_out))
#         self.criterion = nn.BCEWithLogitsLoss(reduction='none')

#         self.use_memory    = use_memory
#         self.shared_memory = shared_memory
#         self.ema_alpha     = ema_alpha
#         self.consistency_coef = consistency_coef
#         self.half_life = half_life

#         if self.use_memory:
#             if self.shared_memory:
#                 self.node_memory = NodeMemory(num_nodes, self.eff_D)
#             else:
#                 self.node_memory1 = NodeMemory(num_nodes, self.eff_D)
#                 self.node_memory2 = NodeMemory(num_nodes, self.eff_D)
#             self.mem_fuse = GatedMemoryFusion(self.eff_D)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.time_feats_dim > 0:
#             self.base_model.reset_parameters()
#         self.edge_predictor.reset_parameters()

#     def forward(self, model_inputs, neg_samples, node_feats):
#         # New trailing inputs: ..., pos_edge_label, src_ids1, src_ids2, src_ts1, src_ts2
#         pos_edge_label = model_inputs[-5].view(-1).float()
#         src_ids1, src_ids2 = model_inputs[-4], model_inputs[-3]
#         src_ts1,  src_ts2  = model_inputs[-2], model_inputs[-1]  # [B] float timestamps (UN-normalized)
#         base_inputs = model_inputs[:-5]

#         pred_true, pred_false, pred_neg, idx1, idx2, x_cache = self.predict(
#             base_inputs, pos_edge_label, neg_samples, node_feats, src_ids1, src_ids2
#         )

#         if pred_true.numel() == 0 and pred_false.numel() == 0:
#             device = pos_edge_label.device
#             loss = torch.tensor(0.0, device=device, requires_grad=True)
#             return loss, torch.empty(0, device=device), torch.empty(0, device=device)

#         # Negative undersampling (deterministic)
#         if pred_neg.numel() > 0:
#             target_neg = pred_true.size(0) + pred_false.size(0)
#             if pred_neg.size(0) > target_neg > 0:
#                 gen = torch.Generator(device=pred_neg.device).manual_seed(4242)
#                 idx = torch.randperm(pred_neg.size(0), generator=gen, device=pred_neg.device)[:target_neg]
#                 pred_neg = pred_neg.index_select(0, idx)

#         # Loss
#         loss_true  = self.criterion(pred_true,  torch.ones_like(pred_true)).mean() if pred_true.numel()  > 0 else 0.0
#         loss_false = self.criterion(pred_false, torch.zeros_like(pred_false)).mean() if pred_false.numel() > 0 else 0.0
#         loss_neg   = self.criterion(pred_neg,   torch.zeros_like(pred_neg)).mean()   if pred_neg.numel()   > 0 else 0.0
#         loss = loss_true + loss_false + loss_neg

#         all_pred  = torch.cat([pred_true, pred_false, pred_neg], dim=0)
#         all_label = torch.cat([torch.ones_like(pred_true),
#                                torch.zeros_like(pred_false),
#                                torch.zeros_like(pred_neg)], dim=0)

#         # Memory update 
#         if self.use_memory and pred_true.numel() > 0:
#             with torch.no_grad():
#                 mask_true = (pos_edge_label.index_select(0, idx1).view(-1, 1) == 1)
#                 up_idx1   = idx1[mask_true.view(-1)]
#                 up_idx2   = idx2[mask_true.view(-1)]

#                 num_edge = x_cache.shape[0] // (2 * neg_samples + 4)
#                 base2    = 2*num_edge + num_edge*neg_samples
#                 h_src1_now = x_cache[:num_edge]
#                 h_src2_now = x_cache[base2 : base2 + num_edge]

#                 # gather matching timestamps for those sources
#                 up_ts1 = src_ts1.index_select(0, up_idx1)  # [K]
#                 up_ts2 = src_ts2.index_select(0, up_idx2)  # [K]

#                 if self.shared_memory:
#                     self.node_memory.ema_update(
#                         src_ids1.index_select(0, up_idx1),
#                         h_src1_now.index_select(0, up_idx1),
#                         ts=up_ts1, half_life=self.half_life
#                     )
#                     self.node_memory.ema_update(
#                         src_ids2.index_select(0, up_idx2),
#                         h_src2_now.index_select(0, up_idx2),
#                         ts=up_ts2, half_life=self.half_life
#                     )
#                 else:
#                     self.node_memory1.ema_update(
#                         src_ids1.index_select(0, up_idx1),
#                         h_src1_now.index_select(0, up_idx1),
#                         ts=up_ts1, half_life=self.half_life
#                     )
#                     self.node_memory2.ema_update(
#                         src_ids2.index_select(0, up_idx2),
#                         h_src2_now.index_select(0, up_idx2),
#                         ts=up_ts2, half_life=self.half_life
#                     )


#     def predict(self, base_inputs, pos_edge_label, neg_samples, node_feats, src_ids1, src_ids2):
#         # encoder
#         if self.time_feats_dim > 0 and self.node_feats_dim == 0:
#             x = self.base_model(*base_inputs)
#         elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
#             x = self.base_model(*base_inputs)
#             x = torch.cat([x, node_feats], dim=1)
#         elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
#             x = node_feats
#         else:
#             raise ValueError('Either dim_in_time or dim_in_node must be > 0')

#         # memory fusion (gated) into source slices
#         num_edge = x.shape[0] // (2 * neg_samples + 4)
#         base2    = 2*num_edge + num_edge*neg_samples
#         if self.use_memory:
#             if self.shared_memory:
#                 m1 = self.node_memory.get(src_ids1[:num_edge])
#                 m2 = self.node_memory.get(src_ids2[:num_edge])
#                 x[:num_edge]             = self.mem_fuse(x[:num_edge], m1)
#                 x[base2:base2+num_edge]  = self.mem_fuse(x[base2:base2+num_edge], m2)
#             else:
#                 m1 = self.node_memory1.get(src_ids1[:num_edge])
#                 m2 = self.node_memory2.get(src_ids2[:num_edge])
#                 x[:num_edge]             = self.mem_fuse(x[:num_edge], m1)
#                 x[base2:base2+num_edge]  = self.mem_fuse(x[base2:base2+num_edge], m2)

#         pred_true, pred_false, pred_neg, idx1, idx2 = self.edge_predictor(
#             x, pos_edge_label, neg_samples=neg_samples, src_ids1=src_ids1, src_ids2=src_ids2
#         )
#         return pred_true, pred_false, pred_neg, idx1, idx2, x


class Dual_Interface(nn.Module):
    def __init__(self,
                 mlp_mixer_configs,
                 edge_predictor_configs,
                 num_nodes,
                 mem_dim,
                 ema_alpha: float = 0.3,
                 use_memory: bool = True,
                 shared_memory: bool = False,
                 consistency_coef: float = 0.05,
                 half_life: float = 40.0):
        super(Dual_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']
        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.enc_out = mlp_mixer_configs['out_channels']
        self.eff_D   = self.enc_out + (self.node_feats_dim if self.node_feats_dim > 0 else 0)

        # --- light-weight dynamics modules ---
        self.prop1 = NodePropensity(num_nodes)
        self.prop2 = NodePropensity(num_nodes)
        self.pref1 = PreferenceMemory(num_nodes, self.eff_D)
        self.pref2 = PreferenceMemory(num_nodes, self.eff_D)
        self.rec1  = RecencyTracker(num_nodes, default_dt=1.0)
        self.rec2  = RecencyTracker(num_nodes, default_dt=1.0)

        # trainable scalars to weight their contribution (start at 0 so it's safe)
        self.w_b        = nn.Parameter(torch.tensor(0.0))  # node propensity weight
        self.w_r        = nn.Parameter(torch.tensor(0.0))  # recency weight
        self.alpha_pref = nn.Parameter(torch.tensor(0.0))  # preference dot weight

        self.edge_predictor = EdgePredictor_per_node(dim_in=self.eff_D,
                                                     dim_hidden=max(100, self.enc_out))
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.use_memory    = use_memory
        self.shared_memory = shared_memory
        self.ema_alpha     = ema_alpha
        self.consistency_coef = consistency_coef
        self.half_life = half_life

        if self.use_memory:
            if self.shared_memory:
                self.node_memory = NodeMemory(num_nodes, self.eff_D)
            else:
                self.node_memory1 = NodeMemory(num_nodes, self.eff_D)
                self.node_memory2 = NodeMemory(num_nodes, self.eff_D)
            self.mem_fuse = GatedMemoryFusion(self.eff_D)

        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        # dynamics scalars already initialized to 0.0 (safe)

    def forward(self, model_inputs, neg_samples, node_feats):
        # New trailing inputs: ..., pos_edge_label, src_ids1, src_ids2, src_ts1, src_ts2
        pos_edge_label = model_inputs[-5].view(-1).float()
        src_ids1, src_ids2 = model_inputs[-4], model_inputs[-3]
        src_ts1,  src_ts2  = model_inputs[-2], model_inputs[-1]  # [B] float timestamps (UN-normalized)
        base_inputs = model_inputs[:-5]

        # predict (gets x_cache back for feature-based extras and updates)
        pred_true, pred_false, pred_neg, idx1, idx2, x_cache = self.predict(
            base_inputs, pos_edge_label, neg_samples, node_feats, src_ids1, src_ids2
        )

        # early exit on empty batch
        if pred_true.numel() == 0 and pred_false.numel() == 0:
            device = pos_edge_label.device
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return loss, torch.empty(0, device=device), torch.empty(0, device=device)

        # -------------------------------------------
        # (1)-(3) Add light-weight dynamics to logits
        # -------------------------------------------
        num_edge = x_cache.shape[0] // (2 * neg_samples + 4)
        base0     = 0
        base1     = num_edge                        # end of src1 block
        base_pos1 = base1 + num_edge                # end of pos dst1 block
        base_neg1 = base_pos1 + num_edge * neg_samples
        base2     = 2 * num_edge + num_edge * neg_samples
        base_pos2 = base2 + num_edge
        base_neg2 = base_pos2 + num_edge
        end_all   = base_neg2 + num_edge * neg_samples

        if idx1.numel() > 0:
            # (1) node propensity bias
            b1_all = self.prop1.get(src_ids1[:num_edge])  # [num_edge]
            b2_all = self.prop2.get(src_ids2[:num_edge])  # [num_edge]
            b_mean_aligned = 0.5 * (b1_all.index_select(0, idx1) + b2_all.index_select(0, idx2))  # [K]

            # (2) recency: log(1 + Δt_since_last_src_event)
            dt1_all = self.rec1.get_dt(src_ids1[:num_edge], src_ts1[:num_edge])  # [num_edge]
            dt2_all = self.rec2.get_dt(src_ids2[:num_edge], src_ts2[:num_edge])  # [num_edge]
            dt_mean_aligned = 0.5 * (dt1_all.index_select(0, idx1) + dt2_all.index_select(0, idx2))  # [K]
            rec_feat_aligned = torch.log1p(dt_mean_aligned)  # [K]

            # (3) preference vector dot against current positive dst embeddings
            m1_aligned = self.pref1.get(src_ids1[:num_edge]).index_select(0, idx1)  # [K, D]
            m2_aligned = self.pref2.get(src_ids2[:num_edge]).index_select(0, idx2)  # [K, D]
            h_pos_dst1_a = x_cache[base1:base_pos1].index_select(0, idx1)           # [K, D]
            h_pos_dst2_a = x_cache[base_pos2:base_pos2+num_edge].index_select(0, idx2)  # [K, D]
            pref_dot_aligned = 0.5 * (
                (m1_aligned * h_pos_dst1_a).sum(-1) + (m2_aligned * h_pos_dst2_a).sum(-1)
            )  # [K]

            extras_aligned = (self.w_b * b_mean_aligned
                              + self.w_r * rec_feat_aligned
                              + self.alpha_pref * pref_dot_aligned)  # [K]

            # split extras across pred_true / pred_false
            y_pos = pos_edge_label.index_select(0, idx1).view(-1, 1)              # [K,1]
            mask_true  = (y_pos == 1).view(-1)
            mask_false = (y_pos == 0).view(-1)

            if pred_true.numel() > 0:
                pred_true = pred_true + extras_aligned[mask_true].view(-1, 1)
            if pred_false.numel() > 0:
                pred_false = pred_false + extras_aligned[mask_false].view(-1, 1)

            # negatives: follow the same indexing layout (3 stacks)
            if pred_neg.numel() > 0 and neg_samples > 0:
                rows1, rows2 = [], []
                for i in idx1.tolist():
                    rows1.extend(range(i * neg_samples, (i + 1) * neg_samples))
                for j in idx2.tolist():
                    rows2.extend(range(j * neg_samples, (j + 1) * neg_samples))
                rows1 = torch.as_tensor(rows1, dtype=torch.long, device=x_cache.device)
                rows2 = torch.as_tensor(rows2, dtype=torch.long, device=x_cache.device)

                K = idx1.size(0)
                b_rep   = b_mean_aligned.unsqueeze(1).expand(K, neg_samples).reshape(-1)   # [K*neg]
                rec_rep = rec_feat_aligned.unsqueeze(1).expand(K, neg_samples).reshape(-1) # [K*neg]

                m1_rep = m1_aligned.unsqueeze(1).expand(K, neg_samples, m1_aligned.size(1)).reshape(-1, m1_aligned.size(1))
                m2_rep = m2_aligned.unsqueeze(1).expand(K, neg_samples, m2_aligned.size(1)).reshape(-1, m2_aligned.size(1))

                h_neg_dst1_a = x_cache[base_neg1:base_neg1 + num_edge * neg_samples].index_select(0, rows1)  # [K*neg, D]
                h_neg_dst2_a = x_cache[base_neg2:end_all].index_select(0, rows2)                               # [K*neg, D]

                pref_neg1 = (m1_rep * h_neg_dst1_a).sum(-1)      # [K*neg] case1: [src, neg1, pos2]
                pref_neg2 = (m2_rep * h_neg_dst2_a).sum(-1)      # [K*neg] case2: [src, pos1, neg2]
                pref_neg3 = 0.5 * ((m1_rep * h_neg_dst1_a).sum(-1) + (m2_rep * h_neg_dst2_a).sum(-1))  # case3

                extras_neg1 = (self.w_b * b_rep + self.w_r * rec_rep + self.alpha_pref * pref_neg1)
                extras_neg2 = (self.w_b * b_rep + self.w_r * rec_rep + self.alpha_pref * pref_neg2)
                extras_neg3 = (self.w_b * b_rep + self.w_r * rec_rep + self.alpha_pref * pref_neg3)

                extras_neg = torch.cat([extras_neg1, extras_neg2, extras_neg3], dim=0).view(-1, 1)  # [3*K*neg,1]
                pred_neg = pred_neg + extras_neg

        # --------------------
        # Negative undersample
        # --------------------
        if pred_neg.numel() > 0:
            target_neg = pred_true.size(0) + pred_false.size(0)
            if pred_neg.size(0) > target_neg > 0:
                gen = torch.Generator(device=pred_neg.device).manual_seed(4242)
                idx = torch.randperm(pred_neg.size(0), generator=gen, device=pred_neg.device)[:target_neg]
                pred_neg = pred_neg.index_select(0, idx)

        # ---- Loss ----
        loss_true  = self.criterion(pred_true,  torch.ones_like(pred_true)).mean() if pred_true.numel()  > 0 else 0.0
        loss_false = self.criterion(pred_false, torch.zeros_like(pred_false)).mean() if pred_false.numel() > 0 else 0.0
        loss_neg   = self.criterion(pred_neg,   torch.zeros_like(pred_neg)).mean()   if pred_neg.numel()   > 0 else 0.0
        loss = loss_true + loss_false + loss_neg

        all_pred  = torch.cat([pred_true, pred_false, pred_neg], dim=0)
        all_label = torch.cat([torch.ones_like(pred_true),
                               torch.zeros_like(pred_false),
                               torch.zeros_like(pred_neg)], dim=0)

        # -------------------------
        # Updates (memory + dynamics)
        # -------------------------
        if idx1.numel() > 0:
            with torch.no_grad():
                # --- original memory update on positives only (as you had) ---
                if self.use_memory and pred_true.numel() > 0:
                    mask_true = (pos_edge_label.index_select(0, idx1).view(-1, 1) == 1)
                    up_idx1   = idx1[mask_true.view(-1)]
                    up_idx2   = idx2[mask_true.view(-1)]

                    h_src1_now = x_cache[:num_edge]
                    h_src2_now = x_cache[base2 : base2 + num_edge]

                    up_ts1 = src_ts1.index_select(0, up_idx1)
                    up_ts2 = src_ts2.index_select(0, up_idx2)

                    if self.shared_memory:
                        self.node_memory.ema_update(
                            src_ids1.index_select(0, up_idx1),
                            h_src1_now.index_select(0, up_idx1),
                            ts=up_ts1, half_life=self.half_life
                        )
                        self.node_memory.ema_update(
                            src_ids2.index_select(0, up_idx2),
                            h_src2_now.index_select(0, up_idx2),
                            ts=up_ts2, half_life=self.half_life
                        )
                    else:
                        self.node_memory1.ema_update(
                            src_ids1.index_select(0, up_idx1),
                            h_src1_now.index_select(0, up_idx1),
                            ts=up_ts1, half_life=self.half_life
                        )
                        self.node_memory2.ema_update(
                            src_ids2.index_select(0, up_idx2),
                            h_src2_now.index_select(0, up_idx2),
                            ts=up_ts2, half_life=self.half_life
                        )

                # --- dynamics: propensity updates (all aligned), preference (positives), recency (all aligned) ---
                y_pos_aligned = pos_edge_label.index_select(0, idx1).float()  # [K]

                # per-node propensity uses y in {0,1}
                self.prop1.ema_update(src_ids1.index_select(0, idx1), y_pos_aligned,
                                      src_ts1.index_select(0, idx1), half_life=self.half_life)
                self.prop2.ema_update(src_ids2.index_select(0, idx2), y_pos_aligned,
                                      src_ts2.index_select(0, idx2), half_life=self.half_life)

                # preference memory only for positives
                mask_true_up = (y_pos_aligned == 1)
                if torch.any(mask_true_up):
                    up_idx1p = idx1[mask_true_up]
                    up_idx2p = idx2[mask_true_up]
                    dst1_pos_now = x_cache[base1:base_pos1].index_select(0, up_idx1p)                # [Kp, D]
                    dst2_pos_now = x_cache[base_pos2:base_pos2+num_edge].index_select(0, up_idx2p)   # [Kp, D]
                    self.pref1.ema_update(src_ids1.index_select(0, up_idx1p), dst1_pos_now,
                                          src_ts1.index_select(0, up_idx1p), half_life=self.half_life)
                    self.pref2.ema_update(src_ids2.index_select(0, up_idx2p), dst2_pos_now,
                                          src_ts2.index_select(0, up_idx2p), half_life=self.half_life)

                # recency tracker for all aligned sources
                self.rec1.update_src(src_ids1.index_select(0, idx1), src_ts1.index_select(0, idx1))
                self.rec2.update_src(src_ids2.index_select(0, idx2), src_ts2.index_select(0, idx2))

        # ---------------
        # Consistency loss
        # ---------------
        if self.use_memory and not self.shared_memory and idx1.numel() > 0:
            m1 = self.node_memory1.get(src_ids1.index_select(0, idx1))
            m2 = self.node_memory2.get(src_ids2.index_select(0, idx2))
            loss = loss + self.consistency_coef * torch.mean((m1 - m2) ** 2)

        return loss, all_pred, all_label

    def predict(self, base_inputs, pos_edge_label, neg_samples, node_feats, src_ids1, src_ids2):
        # encoder
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*base_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*base_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            raise ValueError('Either dim_in_time or dim_in_node must be > 0')

        # memory fusion (gated) into source slices
        num_edge = x.shape[0] // (2 * neg_samples + 4)
        base2    = 2 * num_edge + num_edge * neg_samples
        if self.use_memory:
            if self.shared_memory:
                m1 = self.node_memory.get(src_ids1[:num_edge])
                m2 = self.node_memory.get(src_ids2[:num_edge])
                x[:num_edge]            = self.mem_fuse(x[:num_edge],            m1)
                x[base2:base2+num_edge] = self.mem_fuse(x[base2:base2+num_edge], m2)
            else:
                m1 = self.node_memory1.get(src_ids1[:num_edge])
                m2 = self.node_memory2.get(src_ids2[:num_edge])
                x[:num_edge]            = self.mem_fuse(x[:num_edge],            m1)
                x[base2:base2+num_edge] = self.mem_fuse(x[base2:base2+num_edge], m2)

        pred_true, pred_false, pred_neg, idx1, idx2 = self.edge_predictor(
            x, pos_edge_label, neg_samples=neg_samples, src_ids1=src_ids1, src_ids2=src_ids2
        )
        return pred_true, pred_false, pred_neg, idx1, idx2, x