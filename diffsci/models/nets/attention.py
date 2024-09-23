import math
import warnings

import torch
import einops


class NDimensionalAttention(torch.nn.Module):
    def __init__(self,
                 num_channels,
                 num_heads=1,
                 type="default",
                 attn_residual=False,
                 magnitude_preserving=False):
        """
        Parameters
        ----------
        num_channels : int
            The number of channels in the input
        num_heads : int
            The number of heads in the multihead attention
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.type = type
        self.attn_residual = attn_residual
        self.magnitude_preserving = magnitude_preserving
        if type == "default":
            if magnitude_preserving:
                warnings.warn("Magnitude preserving is not implemented for"
                              " in-built MultiheadAttention. Using in-house"
                              " implementation.")
                self.mhattn = MultiHeadAttention(
                    num_heads,
                    num_channels,
                    num_channels // num_heads,
                    num_channels // num_heads,
                    attn_type='dot',
                    magnitude_preserving=magnitude_preserving)
            else:
                self.mhattn = torch.nn.MultiheadAttention(num_channels,
                                                          num_heads=num_heads,
                                                          batch_first=True)
        elif type == "cosine":
            self.mhattn = MultiHeadAttention(
                num_heads,
                num_channels,
                num_channels // num_heads,
                num_channels // num_heads,
                attn_type='cosine',
                magnitude_preserving=magnitude_preserving)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, D, H, W)

        Returns
        -------
        torch.Tensor of shape (B, C, D, H, W)

        OBS: VARIABLES NAMES IN THIS FUNCTION ARE NOT GOOD!!! CHANGE IT IN THE
        FUTURE!!!
        """
        x_r, shape = self.rearrange_input(x)
        x_r, _ = self.mhattn(x_r, x_r, x_r, need_weights=False)
        x_r = self.unravel_output(x_r, shape)
        if self.attn_residual:
            x_r = x + x_r
        return x_r

    def rearrange_input(self, x):
        raise NotImplementedError

    def unravel_output(self, x, shape):
        raise NotImplementedError


class TwoDimensionalAttention(NDimensionalAttention):
    def rearrange_input(self, x):
        w, h = x.shape[-2], x.shape[-1]
        x_r = einops.rearrange(x, 'b c w h -> b (w h) c')
        return x_r, (w, h)

    def unravel_output(self, x, shape):
        w, h = shape
        x_r = einops.rearrange(x, 'b (w h) c -> b c w h', w=w, h=h)
        return x_r


class ThreeDimensionalAttention(NDimensionalAttention):
    def rearrange_input(self, x):
        d, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
        x_r = einops.rearrange(x, 'b c d h w -> b (d h w) c')
        return x_r, (d, h, w)

    def unravel_output(self, x, shape):
        d, h, w = shape
        x_r = einops.rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        return x_r


class MultiHeadAttention(torch.nn.Module):
    """
        Implements MultiHeadAttention in-house, so that we can implement
        cosine attention.
    """
    def __init__(self, nheads, dmodel, dk, dv, attn_type='dot',
                 magnitude_preserving=False):
        """
        Parameters
        ----------
        nheads : int
        dmodel : int
        dk : int
        dv : int
        """
        super().__init__()
        self.nheads = nheads
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.magnitude_preserving = magnitude_preserving
        self.epsilon = 1e-4

        if attn_type == 'dot':
            self.attn_fn = dot_product_attn
        elif attn_type == 'cosine':
            self.attn_fn = cosine_product_attn
        # (..., ntokens, dmodel),
        # (nheads, dmodel, dk) -> (..., nheads, ntokens, dv)
        self.q_proj_matrix = torch.nn.Parameter(
            torch.zeros([nheads, dmodel, dk]))
        self.k_proj_matrix = torch.nn.Parameter(
            torch.zeros([nheads, dmodel, dk]))
        self.v_proj_matrix = torch.nn.Parameter(
            torch.zeros([nheads, dmodel, dv]))
        self.o_proj_matrix = torch.nn.Parameter(
            torch.zeros([nheads, dmodel, dv]))
        self.initialize()

    def initialize(self):
        if self.magnitude_preserving:
            torch.nn.init.normal_(self.q_proj_matrix)
            torch.nn.init.normal_(self.k_proj_matrix)
            torch.nn.init.normal_(self.v_proj_matrix)
            torch.nn.init.normal_(self.o_proj_matrix)
        else:  # Uses Xavier initialization
            torch.nn.init.xavier_uniform_(self.q_proj_matrix)
            torch.nn.init.xavier_uniform_(self.k_proj_matrix)
            torch.nn.init.xavier_uniform_(self.v_proj_matrix)
            torch.nn.init.xavier_uniform_(self.o_proj_matrix)

    def forward(self, queries, keys, values,
                mask=None, need_weights=True):
        # Here, multihead attention is implemented
        # with the aid of einsum, therefore avoiding
        # clumsy concatenation and reshape operations,
        # as well as making the operation in general
        # clearer
        #
        # queries : (..., ntokens, dmodel)
        # keys : (..., ntokens, dmodel)
        # values : (..., ntokens, dmodel)
        # mask : None or (..., ntokens, ntokens)

        ws = []

        if self.training:
            with torch.no_grad():
                self.q_proj_matrix.copy_(self.normalize_weight(
                    self.q_proj_matrix, 'wq'))
                self.k_proj_matrix.copy_(self.normalize_weight(
                    self.k_proj_matrix, 'wk'))
                self.v_proj_matrix.copy_(self.normalize_weight(
                    self.v_proj_matrix, 'wv'))
                self.o_proj_matrix.copy_(self.normalize_weight(
                    self.o_proj_matrix, 'wo'))

        for kind, weight in zip(['wq', 'wk', 'wv', 'wo'],
                                [self.q_proj_matrix,
                                 self.k_proj_matrix,
                                 self.v_proj_matrix,
                                 self.o_proj_matrix]):
            w = self.normalize_weight(weight, kind)
            if kind in ['wq', 'wk', 'wv']:
                fan_in = w.shape[1]
            elif kind in ['wo']:
                fan_in = w.shape[0]*w.shape[2]
            else:
                raise ValueError(f"kind must be in ['wq', 'wk', 'wv', 'wo'],"
                                 f" got {kind}")
            w = w / math.sqrt(fan_in)
            ws.append(w)

        wq, wk, wv, wo = ws
        # (..., nheads, ntokens, dk)
        projected_queries = torch.einsum('...ij, kjm -> ...kim',
                                         queries,
                                         wq)
        # (..., nheads, ntokens, dk)
        projected_keys = torch.einsum('...ij, kjm -> ...kim',
                                      keys,
                                      wk)
        # (..., nheads, ntokens, dv)
        projected_values = torch.einsum('...ij, kjm -> ...kim',
                                        values,
                                        wv)
        # (..., nheads, ntokens, dv)
        new_projected_values, weights = self.attn_fn(
            projected_queries,
            projected_keys,
            projected_values,
            mask,
            need_weights
        )
        # (..., ntokens, dmodel)
        new_values = torch.einsum('...ijk, ilk -> ...jl',
                                  new_projected_values,
                                  wo)
        if need_weights:
            return new_values, weights
        else:
            return new_values, None

    def normalize_weight(self, weight, kind):
        if self.magnitude_preserving:
            # weight_shape = weight.shape
            if kind in ['wq', 'wk', 'wv']:
                norm = torch.linalg.vector_norm(weight, dim=1, keepdim=True)
            elif kind in ['wo']:
                norm = torch.linalg.vector_norm(weight, dim=[0, 2],
                                                keepdim=True)

            else:
                raise ValueError(f"kind must be in ['wq', 'wk', 'wv', 'wo'],"
                                 f" got {kind}")
            alpha = math.sqrt(norm.numel()/weight.numel())
            norm_weight = weight / (alpha*norm + self.epsilon)
            return norm_weight
        else:
            return weight


def dot_product_attn(queries, keys, values, mask=None, return_weights=False):
    """
        Implements Scaled Dot Product Attention,
        as in https://arxiv.org/abs/1706.03762

        Parameters
        ----------
        queries : torch.Tensor
            query vector of size (..., ntokens, dk)
        keys : torch.Tensor
            keys vector of size (..., ntokens, dk)
        values : torch.Tensor
            values vector of size (..., ntokens, dv)
        mask : Union[None, str, torch.Tensor]
            If None, no mask is applied. If mask in ['upper', 'causal'],
            apply causal mask. Else, apply
            custom mask of size (ntokens, ntokens).
            In the later case, if the mask is a bool or long,
            pairs valuing False (or 0) will be ignored,
            and pairs valuing True (or 1) will be considered.
            Else, the mask will be directly added.
        return_weights : bool
            Whether to return the weights matrix

        Returns
        -------
        weighted values of dimension (..., ntokens, dv), and
        weights of dimension (..., ntokens, ntokens) if return_weights is True

    """
    dk = queries.shape[-1]
    # (..., ntokens, ntokens)
    inner_product = torch.einsum('...ij, ...kj -> ...ik', queries, keys)
    inner_product /= math.sqrt(dk)
    if mask is not None:
        if isinstance(mask, str):
            ntokens = values.shape[-2]
            if mask == 'upper' or mask == 'causal':
                maskbool = torch.triu(torch.ones(ntokens, ntokens),
                                      diagonal=1)
                mask = torch.log(1 - maskbool)
            else:
                raise NotImplementedError
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool or mask.dtype == torch.long:
                mask = torch.log(mask.to(torch.float))
            inner_product += mask
    # (..., ntokens, ntokens)
    weights = torch.softmax(inner_product, dim=-1)
    # (..., ntokens, dv)
    wvalues = torch.einsum('...ij, ...jk -> ...ik', weights, values)
    if not return_weights:
        return wvalues, None
    else:
        return weights, wvalues


def cosine_product_attn(queries,
                        keys,
                        values,
                        mask=None,
                        return_weights=False):
    """
        Implements Cosine Attention,
        as in https://arxiv.org/pdf/2211.06828

        Parameters
        ----------
        queries : torch.Tensor
            query vector of size (..., ntokens, dk)
        keys : torch.Tensor
            keys vector of size (..., ntokens, dk)
        values : torch.Tensor
            values vector of size (..., ntokens, dv)
        mask : Union[None, str, torch.Tensor]
            If None, no mask is applied. If mask in ['upper', 'causal'],
            apply causal mask. Else, apply
            custom mask of size (ntokens, ntokens).
            In the later case, if the mask is a bool or long,
            pairs valuing False (or 0) will be ignored,
            and pairs valuing True (or 1) will be considered.
            Else, the mask will be directly added.
        return_weights : bool
            Whether to return the weights matrix

        Returns
        -------
        weighted values of dimension (..., ntokens, dv), and
        weights of dimension (..., ntokens, ntokens) if return_weights is True

    """

    inner_product = cosine_similarity(queries, keys)  # (..., ntokens, ntokens)
    if mask is not None:
        if isinstance(mask, str):
            ntokens = values.shape[-2]
            if mask == 'upper' or mask == 'causal':
                maskbool = torch.triu(torch.ones(ntokens, ntokens),
                                      diagonal=1)
                mask = torch.log(1 - maskbool)
            else:
                raise NotImplementedError
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool or mask.dtype == torch.long:
                mask = torch.log(mask.to(torch.float))
            inner_product += mask
    # (..., ntokens, ntokens)
    weights = torch.softmax(inner_product, dim=-1)
    # (..., ntokens, dv)
    wvalues = torch.einsum('...ij, ...jk -> ...ik', weights, values)
    if not return_weights:
        return wvalues, None
    else:
        return weights, wvalues


def cosine_similarity(a, b, eps=1e-8):
    # a : shape [..., n, d]
    # b : shape [..., m, d]
    # returns : shape [..., n, m]

    # [..., n, d]
    # [..., m, d]
    a = a / (torch.linalg.vector_norm(a, dim=-1, keepdim=True) + eps)
    b = b / (torch.linalg.vector_norm(b, dim=-1, keepdim=True) + eps)
    return torch.einsum('...nd,...md->...nm', a, b)
