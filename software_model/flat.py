from jax import custom_vjp
import jax
import jax.numpy as jnp
import numpy as np
import gc
import time
import pandas as pd

@jax.jit
def matmul(A, B):
  return jnp.matmul(A, B)


@jax.jit
def matmul3(A, B, C):
  O = jnp.matmul(A, B)
  O = jax.nn.softmax(O, axis=-1)
  return jnp.matmul(O, C)


def einsum_layer(query, key, output, B=1, H=1, R=1):
  assert len(key.shape) == len(query.shape) == 4
  batch, head, hidden,seq_k  = key.shape
  batch, head, seq_q, hidden = query.shape
  for bt in range(0, batch, B):
    for ht in range(0, head, H):
      for mt in range(0, seq_q, R):
        out = matmul(jnp.array(query[bt:bt+B, ht:ht+H, mt:mt+R, :]), jnp.array(key[bt:bt+B, ht:ht+H, :, :]))
        output[bt:bt+B, ht:ht+H, mt:mt+R, :] = out
  return output


def L_softmax_A(query, key, value, output, B=1, H=1, R=1):
  '''
  Baseline: Un-fused Logit-Softmax-Attend
  
  B: Batch graularity
  H: Head granularity
  R: Row granularity
  '''
  assert len(key.shape) == len(query.shape) == len(query.shape) == 4
  batch, head, hidden, seq_k  = key.shape
  batch, head, seq_q, hidden = query.shape
  batch, head, seq_v, hidden = value.shape
  intermediate = np.zeros((batch, head, seq_q, seq_k))
  assert seq_k == seq_v
  intermediate = einsum_layer(query, key, intermediate, B=B, H=H, R=R)
  intermediate = jax.nn.softmax(jnp.array(intermediate), axis=-1)
  output = einsum_layer(np.array(intermediate), value, output, B=B, H=H, R=R)
  return output


def fused_L_softmax_A(query, key, value, output, B=1, H=1, R=1):
  '''
  FLAT: fused Logit-Softmax-Attend

  B: Batch graularity
  H: Head granularity
  R: Row granularity
  '''
  assert len(key.shape) == len(query.shape) == len(query.shape) == 4
  batch, head, hidden,seq_k  = key.shape
  batch, head, seq_q, hidden = query.shape
  batch, head, seq_v, hidden = value.shape
  assert seq_k == seq_v
  for bt in range(0, batch, B):
    for ht in range(0, head, H):
      for mt in range(0, seq_q, R):
        out = matmul3(jnp.array(query[bt:bt+B, ht:ht+H, mt:mt+R, :]), jnp.array(key[bt:bt+B, ht:ht+H, :, :]), jnp.array(value[bt:bt+B, ht:ht+H, :, :]))
        output[bt:bt+B, ht:ht+H, mt:mt+R, :] = out
  return output

# sweep batch size

def baseline():
    #Baseline
    head = 12
    seq_k = seq_v = seq_q = seq =256
    hidden = 768
    print('=========Baseline============')
    for batcht in range(8):
        batch = 2** batcht
        key = np.ones((batch, head, hidden, seq_k))
        query = np.ones((batch, head, seq_q, hidden))
        value = np.ones((batch, head, seq_v, hidden))
        output = np.zeros((batch, head, seq_q, hidden))
        B = batch
        H = head
        R = seq_q
        print(f'Running Model Batch-{batch}, Head-{head}, Seq-{seq_q}, Hidden-{hidden}, with granularity B-{B}, H-{H}, R-{R}')
        L_softmax_A(query, key, value, output, B, H, R)
        timing = %timeit -o L_softmax_A(query, key, value, output, B, H, R)
        del key, query, value, output
        gc.collect()


def flat():
    head = 12
    seq_k = seq_v = seq_q = seq =256
    hidden = 768

    # ===hyperparameter of FLAT===
    batch_tile = 64   # 1<=batch_tile<=batch
    head_tile = head  # 1<=head_tile<=head
    seq_tile = seq  # 1<=seq_tile<=seq
    #=============================

    print('=========FLAT============')

    for batcht in range(8):
        batch = 2** batcht
        key = np.ones((batch, head, hidden, seq_k))
        query = np.ones((batch, head, seq_q, hidden))
        value = np.ones((batch, head, seq_v, hidden))
        output = np.zeros((batch, head, seq_q, hidden))
        B = min(batch_tile, batch)
        H = min(head_tile, head)
        R = min(seq_tile, seq_q)
        print(f'Running Model Batch-{batch}, Head-{head}, Seq-{seq_q}, Hidden-{hidden}, with granularity B-{B}, H-{H}, R-{R}')
        fused_L_softmax_A(query, key, value, output, B, H, R)
        timing = %timeit -o fused_L_softmax_A(query, key, value, output, B, H, R)
        gc.collect()

# sweep sequence length

def baseline2():
    batch = 1
    head = 12
    hidden = 768

    print('=========Baseline============')

    for seqt in range(8):
        seq = 2** seqt
        seq_k = seq_v = seq_q =seq
        key = np.ones((batch, head, hidden, seq_k))
        query = np.ones((batch, head, seq_q, hidden))
        value = np.ones((batch, head, seq_v, hidden))
        output = np.zeros((batch, head, seq_q, hidden))
        B = min(batch_tile, batch)
        H = min(head_tile, head)
        R = min(seq_tile, seq_q)
        print(f'Running Model Batch-{batch}, Head-{head}, Seq-{seq_q}, Hidden-{hidden}, with granularity B-{B}, H-{H}, R-{R}')
        L_softmax_A(query, key, value, output, B, H, R)
        timing = %timeit -o L_softmax_A(query, key, value, output, B, H, R)
        del key, query, value, output
        gc.collect()

def flat2():
    batch = 1
    head = 12
    hidden = 768

    # ===hyperparameter of FLAT===
    batch_tile = 64   # 1<=batch_tile<=batch
    head_tile = head  # 1<=head_tile<=head
    seq_tile = float('Inf')  # 1<=seq_tile<=seq
    #=============================

    print('=========FLAT============')

    for seqt in range(8):
        seq = 2** seqt
        seq_k = seq_v = seq_q =seq
        key = np.ones((batch, head, hidden, seq_k))
        query = np.ones((batch, head, seq_q, hidden))
        value = np.ones((batch, head, seq_v, hidden))
        output = np.zeros((batch, head, seq_q, hidden))
        B = min(batch_tile, batch)
        H = min(head_tile, head)
        R = min(seq_tile, seq_q)
        print(f'Running Model Batch-{batch}, Head-{head}, Seq-{seq_q}, Hidden-{hidden}, with granularity B-{B}, H-{H}, R-{R}')
        fused_L_softmax_A(query, key, value, output, B, H, R)
        timing = %timeit -o fused_L_softmax_A(query, key, value, output, B, H, R)
        del key, query, value, output
        gc.collect()