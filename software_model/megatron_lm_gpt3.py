from software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
    ElementWiseAddition,
    BarrierSync
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU

from software_model.utils import Tensor, DataType, data_type_dict
from software_model.communication_primitives import AllReduceMultiPCB
from software_model.graph import DependencyGraph
from software_model.utils import SymbolTable
from math import ceil
from typing import List, Tuple
from hardware_model.system import System


# class TransformerBlockInitComputationTP(Operator):
#     def __init__(self, d_model, n_heads, d_head, device_count, data_type: DataType):
#         super().__init__(0, 0, 0, 0, data_type)
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.device_count = device_count
#         self.d_head = d_head
#         # parameters per device
#         d = d_model
#         self.Wq = Tensor([d, d // device_count], data_type)
#         self.Wk = Tensor([d, d // device_count], data_type)
#         self.Wv = Tensor([d, d // device_count], data_type)
#         self.W0 = Tensor([d // device_count, d], data_type)
#         self.W1 = Tensor([d, 4 * d // device_count], data_type)
#         self.W2 = Tensor([4 * d // device_count, d], data_type)
#         # operators per device
#         # # multi-head attention
#         self.Q_proj = Matmul(data_type)
#         self.Q_proj.set_core_device(0)

#         self.K_proj = Matmul(data_type)
#         self.K_proj.set_core_device(1)

#         self.V_proj = Matmul(data_type)
#         self.V_proj.set_core_device(2)

#         self.Q_reshape = Reshape(data_type)
#         self.Q_reshape.set_core_device(0)

#         self.K_reshape = Reshape(data_type)
#         self.K_reshape.set_core_device(1)

#         self.V_reshape = Reshape(data_type)
#         self.V_reshape.set_core_device(2)

#         self.Q_transpose = Transpose(data_type)
#         self.Q_transpose.set_core_device(0)

#         self.K_transpose = Transpose(data_type)
#         self.K_transpose.set_core_device(1)

#         self.V_transpose = Transpose(data_type)
#         self.V_transpose.set_core_device(2)

#         self.Q_mul_K = BatchedMatmul(data_type)
#         self.Q_mul_K.set_core_device(0)

#         self.A_softmax = Softmax(data_type)
#         self.A_softmax.set_core_device(0)

#         self.A_mul_V = BatchedMatmul(data_type)
#         self.A_mul_V.set_core_device(2)

#         self.H_transpose = Transpose(data_type)
#         self.H_transpose.set_core_device(2)

#         self.H_reshape = Reshape(data_type)
#         self.H_reshape.set_core_device(2)

#         self.H_matmul0 = Matmul(data_type)
#         self.H_matmul0.set_core_device(3)

#         self.layer_norm0 = LayerNorm(data_type)
#         self.layer_norm0.set_core_device(3)

#         self.allreduce_mha = AllReduceMultiPCB(data_type)
#         # # feed-forward network
#         self.H_matmul1 = Matmul(data_type)
#         self.H_matmul1.set_core_device(3)
        
#         self.H_gelu = GeLU(data_type)
#         self.H_gelu.set_core_device(3)

#         self.H_matmul2 = Matmul(data_type)
#         self.H_matmul2.set_core_device(3)

#         self.layer_norm1 = LayerNorm(data_type)
#         self.layer_norm1.set_core_device(3)

#         self.allreduce_ffn = AllReduceMultiPCB(data_type)

#     def __call__(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
#         # b: batch size
#         # s: sequence length
#         # d: hidden dimension
#         # d_h: dimension per head
#         b, s, d = X.shape
#         assert d == self.d_model
#         h = self.n_heads
#         dev_cnt = self.device_count
#         d_h = self.d_head

#         # multi-head attention
#         Q = self.Q_proj(X, self.Wq)  # [b, s, d / dev_cnt]
#         assert Q.shape == [b, s, d // dev_cnt]
#         K = self.K_proj(X, self.Wk)  # [b, s, d / dev_cnt]
#         V = self.V_proj(X, self.Wv)  # [b, s, d / dev_cnt]
#         Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
#         K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
#         V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
#         Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
#         assert Q_T.shape == [b, h // dev_cnt, s, d_h]
#         K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, s]
#         K_T.set_desc("K_cache")
#         assert K_T.shape == [b, h // dev_cnt, d_h, s]
#         V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
#         V_T.set_desc("V_cache")
#         assert V_T.shape == [b, h // dev_cnt, s, d_h]
#         A = self.Q_mul_K(Q_T, K_T)  # [b, h / dev_cnt, s, s]
#         A.set_desc("A")
#         assert A.shape == [b, h // dev_cnt, s, s]
#         A_prob = self.A_softmax(A)
#         H = self.A_mul_V(A_prob, V_T)  #  [b, h / dev_cnt, s, d_h]
#         assert H.shape == [b, h // dev_cnt, s, d_h]
#         H = self.H_transpose(H, [0, 2, 1, 3])  #  [b, s, h / dev_cnt, d_h]
#         assert H.shape == [b, s, h // dev_cnt, d_h]
#         H = self.H_reshape(H, [b, s, d // dev_cnt])
#         assert H.shape == [b, s, d // dev_cnt]
#         H0 = self.H_matmul0(H, self.W0)  #  [b, s, d]
#         assert H0.shape == [b, s, d]
#         H0 = self.layer_norm0(H0)
#         assert H0.shape == [b, s, d]
#         if dev_cnt > 1:
#             H0 = self.allreduce_mha(H0)

#         # feed-forward network
#         H1 = self.H_matmul1(H0, self.W1)  # [b, s, 4 * d / dev_cnt]
#         assert H1.shape == [b, s, 4 * d // dev_cnt]
#         H1 = self.H_gelu(H1)
#         H2 = self.H_matmul2(H1, self.W2)  #  [b, s, d]
#         assert H2.shape == [b, s, d]
#         H2 = self.layer_norm1(H2)
#         if dev_cnt > 1:
#             H2 = self.allreduce_ffn(H2)

#         assert H2.shape == [b, s, d]
#         return H2, K_T, V_T


class TransformerBlockAutoRegressionTP(Operator):
    def __init__(self, d_model, n_heads, d_head, device_count, data_type: DataType, device_count_sz: int = 8):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.device_count = device_count # 4 device of 8-core each
        self.device_count_sz = device_count_sz # 8 cores
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d], data_type)
        self.Wq.set_desc("Wq")

        self.Wk = Tensor([d, d], data_type)
        self.Wk.set_desc("Wk")

        self.Wv = Tensor([d, d], data_type)
        self.Wv.set_desc("Wv")
        
        self.W0 = Tensor([d, d], data_type)
        self.W1 = Tensor([d, 4 * d], data_type)
        self.W2 = Tensor([4 * d, d], data_type)
        # operators per device
        # # multi-head attention

        self.Q_proj: List[Matmul] = []
        self.K_proj: List[Matmul] = []
        self.V_proj: List[Matmul] = []
        self.Q_reshape = []
        self.K_reshape = []
        self.V_reshape = []
        self.Q_transpose = []
        self.K_transpose = []
        self.V_transpose = []
        self.K_concat = []
        self.V_concat = []

        self.H_transpose = []
        self.H_reshape = []
        self.H_matmul0: List[Matmul] = []
        self.layer_norm0 = []

        self.H_matmul1: List[Matmul] = []
        self.H_gelu = []
        self.H_matmul2: List[Matmul] = []
        self.layer_norm1 = []

        for device_id in range(device_count):
            base_device_id = device_id * device_count_sz

            Q_proj_i = Matmul(data_type)
            Q_proj_i.set_desc("Q_proj")
            Q_proj_i.set_core_device(base_device_id)
            self.Q_proj.append(Q_proj_i)

            Q_reshape_i = Reshape(data_type)
            Q_reshape_i.set_desc("Q_reshape")
            Q_reshape_i.set_core_device(base_device_id)
            self.Q_reshape.append(Q_reshape_i)

            Q_transpose_i = Transpose(data_type)
            Q_transpose_i.set_desc("Q_transpose")
            Q_transpose_i.set_core_device(base_device_id)
            self.Q_transpose.append(Q_transpose_i)

            K_proj_i = Matmul(data_type)
            K_proj_i.set_desc("K_proj")
            K_proj_i.set_core_device(base_device_id + 1)
            self.K_proj.append(K_proj_i)

            K_reshape_i = Reshape(data_type)
            K_reshape_i.set_desc("K_reshape")
            K_reshape_i.set_core_device(base_device_id + 1)
            self.K_reshape.append(K_reshape_i)

            K_transpose_i = Transpose(data_type)
            K_transpose_i.set_desc("K_transpose")
            K_transpose_i.set_core_device(base_device_id + 1)
            self.K_transpose.append(K_transpose_i)

            K_concat_i = Concat(data_type)
            K_concat_i.set_desc("K_concat")
            K_concat_i.set_core_device(base_device_id + 1)
            self.K_concat.append(K_concat_i)

            V_proj_i = Matmul(data_type)
            V_proj_i.set_desc("V_proj")
            V_proj_i.set_core_device(base_device_id + 2)
            self.V_proj.append(V_proj_i)

            V_reshape_i = Reshape(data_type)
            V_reshape_i.set_desc("V_reshape")
            V_reshape_i.set_core_device(base_device_id + 2)
            self.V_reshape.append(V_reshape_i)
            
            V_transpose_i = Transpose(data_type)
            V_transpose_i.set_desc("V_transpose")
            V_transpose_i.set_core_device(base_device_id + 2)
            self.V_transpose.append(V_transpose_i)

            V_concat_i = Concat(data_type)
            V_concat_i.set_desc("V_concat")
            V_concat_i.set_core_device(base_device_id + 2)
            self.V_concat.append(V_concat_i)

            H_transpose_i = Transpose(data_type)
            H_transpose_i.set_desc("H_transpose")
            H_transpose_i.set_core_device(((device_id + 1)*device_count_sz) - 1)
            self.H_transpose.append(H_transpose_i)

            H_reshape_i = Reshape(data_type)
            H_reshape_i.set_desc("H_reshape")
            H_reshape_i.set_core_device(H_transpose_i.core_device)
            self.H_reshape.append(H_reshape_i)

            H_matmul0_i = Matmul(data_type)
            H_matmul0_i.set_desc("H_matmul0")
            H_matmul0_i.set_core_device(H_reshape_i.core_device)
            self.H_matmul0.append(H_matmul0_i)

            layer_norm0_i = LayerNorm(data_type)
            layer_norm0_i.set_core_device(H_matmul0_i.core_device)
            self.layer_norm0.append(layer_norm0_i)

            # # feed-forward network
            H_matmul1_i = Matmul(data_type)
            H_matmul1_i.set_core_device(H_matmul0_i.core_device)
            self.H_matmul1.append(H_matmul1_i)

            H_gelu_i = GeLU(data_type)
            H_gelu_i.set_core_device(H_matmul1_i.core_device)
            self.H_gelu.append(H_gelu_i)

            H_matmul2_i = Matmul(data_type)
            H_matmul2_i.set_core_device(H_gelu_i.core_device)
            self.H_matmul2.append(H_matmul2_i)

            layer_norm1_i = LayerNorm(data_type)
            layer_norm1_i.set_core_device(H_matmul2_i.core_device)
            self.layer_norm1.append(layer_norm1_i)

        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.allreduce_mha.set_core_device(self.H_matmul0[0].core_device)

        self.allreduce_ffn = AllReduceMultiPCB(data_type)
        self.allreduce_ffn.set_core_device((device_count * device_count_sz) - 1)

    def __call__(self, x: Tensor, seq_len: int, K_cache:Tensor, V_cache:Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = self.d_head

        # KV cache
        K_cache = Tensor([b, h, d_h, s], self.data_type)
        K_cache.set_desc("K_cache")

        V_cache = Tensor([b, h, s, d_h], self.data_type)
        V_cache.set_desc("V_cache")

        # multi-head attention
        self.q = Tensor([b, 1, d])
        self.k = Tensor([b, 1, d])
        self.v = Tensor([b, 1, d])

        self.K_T = Tensor([b, h, d_h, s + 1])
        self.K_T.set_desc("K_cache")

        self.V_T = Tensor([b, h, s + 1, d_h])
        self.V_T.set_desc("V_cache")

        a_out = Tensor([b, h, 1, s + 1])
        h0_out = Tensor([b, h, 1, d_h])

        h0_matmul_out = [Tensor([b, 1, d]) for _ in range(dev_cnt)]
        all_h0_matmul_out: List[Tensor] = []

        for device_id in range(device_count):

            # Shards the Q, K, V calculation on different devices
            start_idx = device_id * (self.d_model // dev_cnt)
            end_idx = (device_id + 1) * (self.d_model // dev_cnt)

            Wq_i = self.Wq[:, start_idx:end_idx]
            Wk_i = self.Wk[:, start_idx:end_idx]
            Wv_i = self.Wv[:, start_idx:end_idx]

            q_i = self.q[:, :, start_idx:end_idx]
            k_i = self.k[:, :, start_idx:end_idx]
            v_i = self.v[:, :, start_idx:end_idx]

            start_head_idx = device_id * (self.n_heads // dev_cnt)
            end_head_idx = (device_id + 1) * (self.n_heads // dev_cnt)

            branch_K_T_i = self.K_T[:, start_head_idx:end_head_idx, :, :]
            branch_V_T_i = self.V_T[:, start_head_idx:end_head_idx, :, :]

            Wq_offset = SymbolTable.get_base_address(Wq_i) - SymbolTable.get_base_address(self.Wq)
            Wk_offset = SymbolTable.get_base_address(Wk_i) - SymbolTable.get_base_address(self.Wk)
            Wv_offset = SymbolTable.get_base_address(Wv_i) - SymbolTable.get_base_address(self.Wv)

            q_offset = SymbolTable.get_base_address(q_i) - SymbolTable.get_base_address(self.q)
            k_offset = SymbolTable.get_base_address(k_i) - SymbolTable.get_base_address(self.k)
            v_offset = SymbolTable.get_base_address(v_i) - SymbolTable.get_base_address(self.v)

            assert Wq_offset >= 0
            assert Wk_offset >= 0
            assert Wv_offset >= 0

            assert q_offset >= 0
            assert k_offset >= 0
            assert v_offset >= 0

            self.Q_proj[device_id].sharded_matmul_details = {
                "desc" : "Q_proj",
                "device_id" : device_id,
                x.name : {
                    "base_addr" : SymbolTable.get_base_address(x),
                    "offset": 0,
                },
                Wq_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.Wq),
                    "offset": Wq_offset,
                },
                q_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.q),
                    "offset": q_offset,
                }
            }

            self.K_proj[device_id].sharded_matmul_details = {
                "desc" : "K_proj",
                "device_id" : device_id,
                x.name : {
                    "base_addr" : SymbolTable.get_base_address(x),
                    "offset": 0,
                },
                Wk_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.Wk),
                    "offset": Wk_offset,
                },
                k_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.k),
                    "offset": k_offset,
                }
            }

            self.V_proj[device_id].sharded_matmul_details = {
                "desc" : "V_proj",
                "device_id" : device_id,
                x.name : {
                    "base_addr" : SymbolTable.get_base_address(x),
                    "offset": 0,
                },
                Wv_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.Wv),
                    "offset": Wv_offset,
                },
                v_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.v),
                    "offset": v_offset,
                }
            }

            new_q = self.Q_proj[device_id](x, Wq_i, q_i)
            assert new_q.shape == [b, 1, d // dev_cnt]

            new_k = self.K_proj[device_id](x, Wk_i, k_i)
            new_v = self.V_proj[device_id](x, Wv_i, v_i)

            new_q = self.Q_reshape[device_id](new_q, [b, 1, h // dev_cnt, d_h])
            new_k = self.K_reshape[device_id](new_k, [b, 1, h // dev_cnt, d_h])
            new_v = self.V_reshape[device_id](new_v, [b, 1, h // dev_cnt, d_h])

            q_T = self.Q_transpose[device_id](new_q, [0, 2, 1, 3])
            assert q_T.shape == [b, h // dev_cnt, 1, d_h]

            k_T = self.K_transpose[device_id](new_k, [0, 2, 3, 1])
            assert k_T.shape == [b, h // dev_cnt, d_h, 1]

            v_T = self.V_transpose[device_id](new_v, [0, 2, 1, 3])
            assert v_T.shape == [b, h // dev_cnt, 1, d_h]

            K_T = self.K_concat[device_id](K_cache[:, start_head_idx:end_head_idx, :, :], k_T, 3, branch_K_T_i)
            assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]

            V_T = self.V_concat[device_id](V_cache[:, start_head_idx:end_head_idx, :, :], v_T, 2, branch_V_T_i)
            assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]


            # Attention Head Calculation
            attn_head_core_ids = []
            all_h0: List[Tensor] = []
            max_used_core_id = self.V_concat[device_id].core_device + 1       

            heads_on_device = h // dev_cnt
            start_head = device_id * heads_on_device
            end_head = (device_id + 1) * heads_on_device
            
            # maps a core to the most recent tensor it produces
            most_recent_tensor = {}
            all_out_tensors = []
            for i in range(start_head, end_head):
                target_core_id = min(self.V_concat[device_id].core_device + 1 + (i%4), (device_id + 1) * self.device_count_sz - 1)
                
                q_T_i = q_T[:, i, :, :]
                K_T_i = K_T[:, i, :, :]
                a_out_i = a_out[:, i, :, :]
                
                q_T_offset = SymbolTable.get_base_address(q_T_i) - SymbolTable.get_base_address(q_T)
                K_T_offset = SymbolTable.get_base_address(K_T_i) - SymbolTable.get_base_address(K_T)
                a_out_offset = SymbolTable.get_base_address(a_out_i) - SymbolTable.get_base_address(a_out)
                assert q_T_offset >= 0
                assert K_T_offset >= 0
                assert a_out_offset >= 0

                q_mul_k_single_matmul = Matmul(self.V_concat[device_id].data_type)
                q_mul_k_single_matmul.set_core_device(target_core_id)
                q_mul_k_single_matmul.batched_matmul_details = {
                    "device_id" : device_id,
                    "inner_batch_id" : i,
                    q_T.name : {
                        "base_addr" : SymbolTable.get_base_address(q_T),
                        "offset": q_T_offset,
                    },
                    K_T.name : {
                        "base_addr" : SymbolTable.get_base_address(K_T),
                        "offset": K_T_offset,
                    },
                    a_out.name : {
                        "base_addr" : SymbolTable.get_base_address(a_out),
                        "offset" : a_out_offset
                    }
                }
                
                new_a_out = q_mul_k_single_matmul(q_T_i, K_T_i, a_out_i)
                assert new_a_out.shape == [b, 1, 1, s + 1]

                softmax_obj = Softmax(new_a_out.data_type)
                softmax_obj.set_core_device(target_core_id)
                a_prob = softmax_obj(new_a_out)

                V_T_i = V_T[:, i, :, :]
                h0_out_i = h0_out[:, i, :, :]

                V_T_offset = SymbolTable.get_base_address(V_T_i) - SymbolTable.get_base_address(V_T)
                h0_out_offset = SymbolTable.get_base_address(h0_out_i) - SymbolTable.get_base_address(h0_out)
                assert V_T_offset >= 0
                assert h0_out_offset >= 0

                a_mul_v_single_matmul = Matmul(self.data_type)
                a_mul_v_single_matmul.set_core_device(target_core_id)
                a_mul_v_single_matmul.batched_matmul_details = {
                    "device_id" : device_id,
                    "inner_batch_id" : i,
                    a_prob.name : {
                        "base_addr" : SymbolTable.get_base_address(a_prob),
                        "offset": a_out_offset,
                    },
                    V_T.name : {
                        "base_addr" : SymbolTable.get_base_address(V_T),
                        "offset": V_T_offset,
                    },
                    h0_out.name : {
                        "base_addr" : SymbolTable.get_base_address(h0_out),
                        "offset": h0_out_offset,
                    }
                }
                new_h0_out = a_mul_v_single_matmul(a_prob, V_T_i, h0_out_i)
                assert new_h0_out.shape == [b, 1, 1, d_h]

                most_recent_tensor[a_mul_v_single_matmul.core_device] = new_h0_out
                all_out_tensors.append(new_h0_out)

                attn_head_core_ids.append(target_core_id)
                max_used_core_id = max(max_used_core_id, target_core_id)

            # all-reduce only gathers data from the cores to which attention head was
            # mapped to
            # for _, tensor in most_recent_tensor.items():
            #     all_h0.append(tensor)
            for tensor in all_out_tensors:
                all_h0.append(tensor)

            all_reduce_obj = AllReduceMultiPCB(self.V_concat[device_id].data_type)
            all_reduce_obj.set_core_device(min(max_used_core_id + 1, (device_id + 1) * self.device_count_sz - 1))
            h0 = all_reduce_obj(all_h0)
        
            # h0 = self.A_mul_V(a_prob, V_T)  #  [b, h / dev_cnt, 1, d_h]
            assert h0.shape == [b, h // dev_cnt, 1, d_h]
            self.H_transpose[device_id].set_core_device(all_reduce_obj.core_device)
            h0 = self.H_transpose[device_id](h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
            assert h0.shape == [b, 1, h // dev_cnt, d_h]

            self.H_reshape[device_id].set_core_device(all_reduce_obj.core_device)
            h0 = self.H_reshape[device_id](h0, [b, 1, d // dev_cnt])
            assert h0.shape == [b, 1, d // dev_cnt]

            # divides self.W0 row-parallel 
            W0_i = self.W0[start_idx:end_idx, :]
            h0_matmul_out_i = h0_matmul_out[device_id]

            W0_offset = SymbolTable.get_base_address(W0_i) - SymbolTable.get_base_address(self.W0)
            h0_offset = SymbolTable.get_base_address(h0) - SymbolTable.get_base_address(h0_out)
            h0_matmul_out_i_offset = SymbolTable.get_base_address(h0_matmul_out_i) - SymbolTable.get_base_address(h0_matmul_out_i)

            assert W0_offset >= 0
            assert h0_offset >= 0
            assert h0_matmul_out_i_offset >= 0

            self.H_matmul0[device_id].sharded_matmul_details = {
                "desc" : "H_matmul0",
                "device_id" : device_id,
                h0.name : {
                    "base_addr" : SymbolTable.get_base_address(h0_out),
                    "offset": h0_offset, 
                },
                W0_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.W0),
                    "offset": W0_offset,
                },
                h0_matmul_out_i.name : {
                    "base_addr" : SymbolTable.get_base_address(h0_matmul_out_i),
                    "offset": h0_matmul_out_i_offset,
                }
            }

            self.H_matmul0[device_id].set_core_device(all_reduce_obj.core_device)
            out = self.H_matmul0[device_id](h0, self.W0, h0_matmul_out_i)  #  [b, 1, d]
            assert out.shape == [b, 1, d]

            self.layer_norm0[device_id].set_core_device(all_reduce_obj.core_device)
            out = self.layer_norm0[device_id](out)
            assert out.shape == [b, 1, d]

            all_h0_matmul_out.append(out)

        # all-reduce all [b, 1, d] tensors from other devices
        h0 = all_h0_matmul_out[0]
        for idx, tensor in enumerate(all_h0_matmul_out):
            obj = ElementWiseAddition(self.data_type)
            if idx != 0:
                obj.set_core_device(self.device_count_sz-1)
                out = obj(h0, tensor, h0)
                h0 = out

        # synchronizes all cores/devices at this point
        _ = BarrierSync(self.data_type)()
        
        h1_matmul_out = Tensor([b, 1, 4 * d])
        h2_matmul_out = [Tensor([b, 1, d]) for _ in range(dev_cnt)]

        all_h2_matmul_out = []
        for device_id in range(dev_cnt):
            start_idx = device_id * ((4 * self.d_model) // dev_cnt)
            end_idx = (device_id + 1) * ((4 * self.d_model) // dev_cnt)

            # feed-forward network
            W1_i = self.W1[:, start_idx:end_idx]
            h0_i = h0
            h1_matmul_out_i = h1_matmul_out[:, :, start_idx:end_idx]

            W1_offset = SymbolTable.get_base_address(W1_i) - SymbolTable.get_base_address(self.W1)
            h0_offset = SymbolTable.get_base_address(h0_i) - SymbolTable.get_base_address(h0)
            h1_matmul_out_i_offset = SymbolTable.get_base_address(h1_matmul_out_i) - SymbolTable.get_base_address(h1_matmul_out)

            assert W1_offset >= 0
            assert h0_offset >= 0
            assert h1_matmul_out_i_offset >= 0

            self.H_matmul1[device_id].sharded_matmul_details = {
                "desc" : "H_matmul1",
                "device_id" : device_id,
                h0_i.name : {
                    "base_addr" : SymbolTable.get_base_address(h0),
                    "offset": h0_offset,
                },
                W1_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.W1),
                    "offset": W1_offset,
                },
                h1_matmul_out_i.name : {
                    "base_addr" : SymbolTable.get_base_address(h1_matmul_out),
                    "offset": h1_matmul_out_i_offset,
                }
            }

            h1 = self.H_matmul1[device_id](h0_i, W1_i, h1_matmul_out_i)  # [b, 1, 4 * d / dev_cnt]
            assert h1.shape == [b, 1, 4 * d // dev_cnt]

            h1 = self.H_gelu[device_id](h1)

            W2_i = self.W2[start_idx:end_idx, :]
            h2_matmul_out_i = h2_matmul_out[device_id]

            W2_offset = SymbolTable.get_base_address(W2_i) - SymbolTable.get_base_address(self.W2)
            h2_matmul_out_i_offset = SymbolTable.get_base_address(h2_matmul_out_i) - SymbolTable.get_base_address(h2_matmul_out_i)

            assert W2_offset >= 0
            assert h2_matmul_out_i_offset >= 0

            self.H_matmul2[device_id].sharded_matmul_details = {
                "desc" : "H_matmul2",
                "device_id" : device_id,
                h1.name : {
                    "base_addr" : SymbolTable.get_base_address(h1_matmul_out_i),
                    "offset": h1_matmul_out_i_offset,
                },
                W2_i.name : {
                    "base_addr" : SymbolTable.get_base_address(self.W2),
                    "offset": W2_offset,
                },
                h2_matmul_out_i.name : {
                    "base_addr" : SymbolTable.get_base_address(h2_matmul_out_i),
                    "offset": h2_matmul_out_i_offset,
                }
            }

            h2 = self.H_matmul2[device_id](h1, W2_i, h2_matmul_out_i)  #  [b, 1, d]
            assert h2.shape == [b, 1, d]
            
            h2 = self.layer_norm1[device_id](h2)
            # if dev_cnt > 1:
            #     h2 = self.allreduce_ffn(h2)
            all_h2_matmul_out.append(h2)

            assert h2.shape == [b, 1, d]
        
        _ = BarrierSync(self.data_type)()
        # all-reduce all [b, 1, d] tensors from other devices
        h2 = all_h2_matmul_out[0]
        for idx, tensor in enumerate(all_h2_matmul_out):
            obj = ElementWiseAddition(self.data_type)
            if idx != 0:
                obj.set_core_device(self.device_count_sz-1)
                out = obj(h2, tensor, h2)
                h2 = out

        return out

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        # print(f'memory requirement: {self.memory_requirement/1e9*96}GB')
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        pcb = system.device
        interconnect = system.interconnect

        # matmul
        # print("simulating qkv")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )
        # gelu_latency = max(gelu_latency, 1e-7)

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class GPTModel:
    def __init__(self, d_model, n_heads, d_head, n_layers, device_count, data_type):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.device_count = device_count
        self.data_type = data_type

        # self.blocks_init = [
        #     TransformerBlockInitComputationTP(d_model, n_heads, d_head, device_count, data_type)
        #     for _ in range(n_layers)
        # ]
        self.blocks_decode = [
            TransformerBlockAutoRegressionTP(d_model, n_heads, d_head, device_count, data_type)
            for _ in range(n_layers)
        ]
        self.K_cache = [None] * n_layers
        self.V_cache = [None] * n_layers

        self.vocab_size = 50000
        self.token_embedding = Tensor([self.vocab_size, d_model], data_type)

        self.lm_head_weight = self.token_embedding 

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        h = x
        for i, block in enumerate(self.blocks_decode):
            h = block(h, seq_len, self.K_cache[i], self.V_cache[i])

        # self.lm_head_weight = Transpose(data_type=self.data_type)(self.lm_head_weight, [1,0])
        # logits = Matmul(self.data_type)(h, self.lm_head_weight)

        return h

    def prefill(self, prompt: Tensor):
        h = prompt
        for i, block in enumerate(self.blocks_init):
            h, K_T, V_T = block(h)
            self.K_cache[i] = K_T
            self.V_cache[i] = V_T
        return h


if __name__ == "__main__":
    from pathlib import Path

    d_model = 2048
    n_heads = 32
    d_head = d_model//n_heads
    n_layers = 1
    device_count = 4
    batch_size = 1
    seq_len = 2048

    model = GPTModel(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        n_layers=n_layers,
        device_count=device_count,
        data_type=data_type_dict["fp16"]
    )

    
    # prompt = Tensor([batch_size, seq_len, d_model], data_type_dict["fp16"])
    # model.prefill(prompt=prompt)
    # dep_graph_path = Path("dep_graph_gpt3_small_prefill_one_block.json")
    # total_prefill_params = DependencyGraph.get_learnable_parameters()
    # DependencyGraph.reset_and_dump_graph(dep_graph_path)

    x = Tensor([batch_size, 1, d_model], data_type_dict["fp16"])
    logits = model.forward(x, seq_len=seq_len)

    symbol_table_path = Path("symbol_table.json")
    dep_graph_path = Path("dep_graph_gpt3_xl_decode_one_block.json")
    # dep_graph_path = Path("tiny_decode.json")
    SymbolTable.dump_symbol_table_to_json(symbol_table_path)
    DependencyGraph.dump_graph_to_json(dep_graph_path)
    total_decode_params = DependencyGraph.get_learnable_parameters()

    print("symbol table dumped to: ", symbol_table_path)
    print("dep graph dumped to: ", dep_graph_path)
    # print(f"Matmul (Prefill) Learnable Parameter Count: {total_prefill_params}")
    print(f"Matmul (Decode) Learnable Parameter Count: {total_decode_params}")
