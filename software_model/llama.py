from software_model.operators import (
    Operator,
    Reshape,
    Transpose,
)
from software_model.matmul import Matmul
from software_model.softmax import Softmax
from software_model.gelu import GeLU

from software_model.utils import Tensor, data_type_dict
from software_model.graph import DependencyGraph
from software_model.utils import SymbolTable

from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 1
    n_heads: int = 32
    n_kv_heads: Optional[int] = 32
    vocab_size: int = 32000 # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 1
    max_seq_len: int = 2048

class RMSNorm(Operator):
    __count = 0
    def __init__(self, dim: int, data_type, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """

        self.name = f"{self.__class__.__name__}_{RMSNorm.__count}"
        self.eps = eps
        self.weight = Tensor([dim])
        self.data_type = data_type

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x

    def __call__(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        # output = self._norm(x.float()).type_as(x)
        output = x
        DependencyGraph.add_node_to_graph(output, [x], self.__class__.__name__, self.name)
        return output

class ColumnParallelLinear(Operator):
    def __init__(self, in_feat: int, out_feat:int, bias:bool, data_type):
        self.in_features = in_feat
        self.out_features = out_feat
        if bias:
            self.weight_bias = Tensor([self.in_features+1, self.out_features], data_type)
        else:
            self.weight_bias = Tensor([self.in_features, self.out_features], data_type)
        self.data_type = data_type
    
    def __call__ (self, input: Tensor) -> Tensor:
        output = Matmul(data_type=self.data_type)(input, self.weight_bias)
        return output

class RowParallelLinear(Operator):
    def __init__(self, in_feat: int, out_feat:int, bias:bool, data_type):
        self.in_features = in_feat
        self.out_features = out_feat
        if bias:
            self.weight_bias = Tensor([self.in_features+1, self.out_features], data_type)
        else:
            self.weight_bias = Tensor([self.in_features, self.out_features], data_type)
        self.data_type = data_type
    
    def __call__ (self, input: Tensor) -> Tensor:
        output = Matmul(data_type=self.data_type)(input, self.weight_bias)
        return output

class Attention(Operator):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs, data_type):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.data_type = data_type

        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, False, data_type=data_type)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, data_type=data_type)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, data_type=data_type)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, data_type=data_type)

        self.q_transpose = Transpose(data_type=data_type)
        self.keys_transpose = Transpose(data_type=data_type)
        self.values_transpose = Transpose(data_type=data_type)

        self.cache_k = Tensor([args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim])
        self.cache_v = Tensor([args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim])
        self.softmax = Softmax(data_type=data_type)

    def __call__(
        self,
        x: Tensor,
        start_pos: int
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = Reshape(data_type=self.data_type)(xq, [bsz, seqlen, self.n_local_heads, self.head_dim])
        xk = Reshape(data_type=self.data_type)(xk, [bsz, seqlen, self.n_local_kv_heads, self.head_dim])
        xv = Reshape(data_type=self.data_type)(xv, [bsz, seqlen, self.n_local_kv_heads, self.head_dim])

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = xq
        self.cache_v = xq

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = self.q_transpose(xq, [0, 2, 1, 3])  # (bs, n_local_heads, seqlen, head_dim)
        keys = self.keys_transpose(keys, [0, 2, 1, 3]) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = self.values_transpose(values, [0, 2, 1, 3]) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = Matmul(data_type=self.data_type)(xq, self.keys_transpose(keys, [0, 1, 3, 2]))
        # if mask is not None:
        #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = self.softmax(scores)
        output = Matmul(data_type=self.data_type)(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = Transpose(data_type=self.data_type)(output, [0, 2, 1, 3])
        output = Reshape(data_type=self.data_type)(output, [bsz, seqlen, self.n_local_heads*self.head_dim])
        output = self.wo(output)
        return output

class FeedForward(Operator):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        data_type
    ):
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, data_type=data_type
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, data_type=data_type
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, data_type=data_type
        )
        self.data_type = data_type

    def __call__(self, x):
        w1 = self.w1(x)
        a1 = GeLU(self.data_type)(w1)
        w3 = self.w3(x)
        w2 = self.w2(w3)
        return w2
    
class TransformerBlock:
    def __init__(self, layer_id: int, args: ModelArgs, data_type):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, data_type)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            data_type=data_type
        )

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, data_type=data_type, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, data_type=data_type, eps=args.norm_eps)
    
    def __call__(
        self,
        x: Tensor,
        start_pos: int
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (Tensor): Precomputed cosine and sine frequencies.
            mask (Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = self.attention(self.attention_norm(x), start_pos)
        out = self.feed_forward(self.ffn_norm(h))
        return out
        

class LLAMA(Operator):
    def __init__(self, params: ModelArgs, data_type):
        super().__init__(0, 0, 0, 0, data_type)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = Tensor([self.vocab_size, params.dim])

        self.layers = []
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, data_type))
        
        self.norm = RMSNorm(params.dim, eps=params.norm_eps, data_type=data_type)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, data_type=data_type
        )


    def __call__(self, x: Tensor, start_pos: int) -> Tensor:
        _bsz, seqlen = x.shape
        h = Tensor([_bsz, seqlen, self.params.dim], data_type=x.data_type)

        for layer in self.layers:
            h = layer(h, start_pos)
        h = self.norm(h)
        output = self.output(h)
        return output

if __name__ == "__main__":
    from pathlib import Path
    params = ModelArgs()
    model = LLAMA(params=params, data_type=data_type_dict["int8"])
    x = Tensor([1, 100], data_type=data_type_dict["int8"])
    out = model(x, 0)

    symbol_table_path = Path("symbol_table_llama_one_layer.json")
    dep_graph_path = Path("dep_graph_llama_one_layer.json")
    SymbolTable.dump_symbol_table_to_json(symbol_table_path)
    DependencyGraph.dump_graph_to_json(dep_graph_path)
    # total_params = Matmul(data_type_dict["int8"]).get_learnable_parameters()
    total_params = DependencyGraph.get_learnable_parameters()

    print("symbol table dumped to: ", symbol_table_path)
    print("dep graph dumped to: ", dep_graph_path)
    print(f"Matmul Learnable Parameter Count: {total_params}")
