from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

N_MAX_POSITIONS = 4096 


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )

    def forward(self, input, mask=None, kv=None, use_cache=False):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
        Input is (bs, qlen, dim)
        Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """compute context"""
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if use_cache:
            if self.layer_id in self.cache:
                if kv is None:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)

        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)

        if mask is not None:
            mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
            mask = (
                (mask == 0).view(mask_reshape).expand_as(scores)
            )  # (bs, n_heads, qlen, klen)
            scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (bs, n_heads, qlen, klen)
        weights = F.dropout(
            weights, p=self.dropout, training=self.training
        )  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        if TransformerModel_f.STORE_OUTPUTS and not self.training:
            self.outputs = weights.detach().cpu()

        return self.out_lin(context)


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = nn.ModuleList()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = F.relu(x)
        for mlin in self.midlin:
            x = mlin(x)
            x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

### Modified: Bottleneck (From ReLSO paper)

class BaseBottleneck(nn.Module):
    """Basic fcn bottleneck

    Args:
        nn ([type]): [description]
    """
    def __init__(self, input_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()

        self.fc1 = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h):
        """
        b = batch size
        h = hidden dimension
        z = latent dim
        
        input_dim: b x s x h
        output_dim: b x z
        """

        z_rep = self.fc1(h)

        return z_rep

class TransformerModel_f(nn.Module):

    STORE_OUTPUTS = True

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
    ):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()
        assert is_encoder, "TransformerModel_f should be used for encoder only"

        # encoder / decoder, output layer
        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        self.apex = params.nvidia_apex

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]

        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words

        # model parameters
        self.dim = (
            params.enc_emb_dim if is_encoder else params.dec_emb_dim
        )  # 512 by default
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = (
            params.n_enc_heads if is_encoder else params.n_dec_heads
        )  # 8 by default
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"

        # embeddings

        if positional_embeddings is None or positional_embeddings == "alibi":
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
            )
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.embeddings = Embedding(
                self.n_words, self.dim, padding_idx=self.pad_index
            )
        else:
            self.embeddings = None
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

        # output layer
        if self.with_output:
            assert not self.use_prior_embeddings
            self.proj = nn.Linear(
                self.dim, self.n_words, bias=True
            )  ##added index for eos and tab
            if params.share_inout_emb:
                self.proj.weight = self.embeddings.weight
        
        ### Modified:
        self.glob_attn_module = nn.Sequential(
            nn.Linear(self.dim, 1), nn.Softmax(dim=1)
        )
        # for los and gradient checking
        self.z_rep = None
        # auxiliary network
        self.latent_dim = 32
        self.bottleneck_module = BaseBottleneck(self.dim, self.latent_dim)
        self.z_projection = nn.Linear(self.latent_dim, self.dim)
        

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        tensor,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
        """
        # # lengths = (x != self.pad_index).float().sum(dim=1)
        # # mask = x != self.pad_index

        # # check inputs
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        # x = x.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs
        # assert not (use_cache and self.cache is None)

        # # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = (
        #         torch.arange(src_len.max(), dtype=torch.long, device=lengths.device)
        #         < src_len[:, None]
        #     )

        # # positions
        # if positions is None:
        #     positions = x.new(slen).long()
        #     positions = torch.arange(slen, out=positions).unsqueeze(0)
        # else:
        #     assert positions.size() == (slen, bs)
        #     positions = positions.transpose(0, 1)

        # # do not recompute cached elements
        # if use_cache:
        #     _slen = slen - self.cache["slen"]
        #     x = x[:, -_slen:]
        #     positions = positions[:, -_slen:]
        #     mask = mask[:, -_slen:]
        #     attn_mask = attn_mask[:, -_slen:]

        # # all layer outputs
        # if TransformerModel_f.STORE_OUTPUTS and not self.training:
        #     self.outputs = []

        # # embeddings
        # if not self.use_prior_embeddings:
        #     tensor = self.embeddings(x)
        # else:
        #     tensor = x

        # if self.position_embeddings is not None:
        #     tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # tensor = self.layer_norm_emb(tensor)
        # tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        # tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        # if TransformerModel_f.STORE_OUTPUTS and not self.training:
        #     self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):

            # self attention
            self.attentions[i].cache = self.cache
            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     self.encoder_attn[i].cache = self.cache
            #     attn = self.encoder_attn[i](
            #         tensor, src_mask, kv=src_enc, use_cache=use_cache
            #     )
            #     attn = F.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel_f.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        # # update cache length
        # if use_cache:
        #     self.cache["slen"] += tensor.size(1)

        ## <<<<<<<<<<<<<<<<<< code for encoder_f <<<<<<<<<<<<<<<
        if self.is_encoder:
            glob_attn = self.glob_attn_module(tensor)  # output should be B x S x 1
            z_rep = torch.bmm(glob_attn.transpose(-1, 1), tensor).squeeze()

            # to regain the batch dimension
            if len(tensor) == 1:
                z_rep = z_rep.unsqueeze(0)

            z_rep = self.bottleneck_module(z_rep)
            return z_rep
        ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        if self.is_decoder:
            # move back sequence length to dimension 0
            tensor = tensor.transpose(0, 1)
            return tensor


    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)    # [all_token_nums, dim]
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def output_layer(self, tensor):
        return self.proj(tensor)    # [bs, seq_len, dim] -> [bs, seq_len, n_words]
    