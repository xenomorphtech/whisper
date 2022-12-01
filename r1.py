#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from collections import OrderedDict

import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum 

def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        assert name in pt_params
        mapped_pt_params[ait_name] = pt_params[name]
    return mapped_pt_params

#class Conv1d(nn.Conv1d):
#    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
#        return super()._conv_forward(
#            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
#        )


class AITMultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
    ):
        q = self.query(x)

        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        #wv = self.qkv_attention(q, k, v, mask)

        return self.out(q)

class AITMultiHeadAttention1(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
    ):
        q = self.query(x)
        print("q shape", q.shape())
        print("xa", xa)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        wv = self.qkv_attention(q, k, v)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor):
        n_batch, n_ctx, n_state = (1, 384, 384) 
        seqlen = 384
        seqlen_kv = 384
        head_dim = n_state // self.n_head
        scale = (n_state // self.n_head) ** -0.25
        print("pre", q)
        q= ops.permute()(
            ops.reshape()(q, [seqlen, self.n_head, head_dim]), [1, 0, 2]
        )
        k= ops.permute()(
            ops.reshape()(k, [seqlen_kv, self.n_head, head_dim]), [1, 2, 0]
        )
        v= ops.permute()(
            ops.reshape()(v, [seqlen_kv, self.n_head, head_dim]),
            [1, 0, 2],
        )

        #q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        print("post", q)
        #k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        #v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        #qk = q @ k
        #if mask is not None:
        #    qk = qk + mask[:n_ctx, :n_ctx]


        qk = ops.bmm_rrr()(q, k)
        score = ops.elementwise(FuncEnum.MUL)(qk, scale)
        score = ops.softmax()(score, -1)
        #out = ops.bmm_rcr()(score, v)
        out = ops.reshape()(v, [384, 384])
        
        #w = F.softmax(qk.float(), dim=-1)
        #return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return out 

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        # print("ResidualAttentionBlock", n_state, n_head)
        self.attn = MultiHeadAttention(n_state, n_head) #will use mask
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) # will use xa
        self.cross_attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp, specialization="fast_gelu"), nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
    ):
        # print("residual input", x, xa, mask)
        # dump({"x": x.cpu().numpy().tolist(), "xa": xa.cpu().numpy().tolist(), "mask": mask.cpu().numpy().tolist() }, "input.json")
        o = self.attn_ln(x)
        x = x + self.attn(o, None, mask=mask)
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, None)
        x = x + self.mlp(self.mlp_ln(x))
        # print("residual output", x)
        # raise("Error")
        return x

class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: dict = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        start = time.time()
        print("TextDecoder | forward ", start)
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        print("TextDecoder | has cache ", offset)
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        print(x.cpu().numpy())
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        print("TextDecoder | forward out ", time.time() - start)

        return logits



import r2 
#model_tiny = whisper.load_model("tiny", device="cuda")
#print(model_tiny.dims)
#print("loaded")
#one_attention = model_tiny.decoder.blocks[0].attn
#print(one_attention)
#verify_simple_model(one_attention)

def verify_multihead_attention():
    n_state, n_head = 384, 6
    model = r2.MultiHeadAttention(n_state, n_head).cuda().half()
    # create pt input
    x = torch.randn([1, n_state, n_state]).cuda().half()
    z = torch.randn([1, n_state, n_state]).cuda().half()
    print("> before eval", x)
    m_y = model(x, z, None)
    print("> after eval", m_y)

    # create ait model
    ait_model = AITMultiHeadAttention(n_state, n_head)

    X = Tensor(
        shape=[1, 384, 384],
        name="X",
        dtype="float16",
        is_input=True,
    )
    XA = Tensor(
        shape=[1, n_state, n_state],
        name="XA",
        dtype="float16",
        is_input=True,
    )
    Y = ait_model(X, XA)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"

    # map pt weights to ait
    weights = map_pt_params(ait_model, model)

    # code gen
    target = detect_target()
    with compile_model(
        Y, target, "./tmp", "simple_model_demo", constants=weights
    ) as module:
        # create storage for output tensor
        y = torch.empty([n_state]).cuda().half()

        # inputs and outputs dict
        inputs = {"X": x, "XA": z}
        outputs = {"Y": y}

        # run
        module.run_with_tensors(inputs, outputs, graph_mode=True)

        # verify output is correct
        print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

        # benchmark ait and pt
        #count = 1000
        #ait_t, _, _ = module.benchmark_with_tensors(
        #    inputs, outputs, graph_mode=True, count=count
        #)
        #print(f"AITemplate time: {ait_t} ms/iter")

        #pt_t = benchmark_torch_function(count, pt_model.forward, x)
        #print(f"PyTorch eager time: {pt_t} ms/iter")

        ## check out the fused graph
        ## there are only fused ops in the final graph
        ## gemm_rcr_bias_fast_gelu, gemm_rcr_bias_add, and layernorm
        #graph = module.debug_sorted_graph
        #print("Final graph:")
        #print(sorted_graph_pseudo_code(graph))


verify_multihead_attention()

def verify_simple_model(random_head):
    # create pt input
    x = torch.randn([1, 384, 384]).cuda().half()
    z = torch.randn([384, 384]).cuda().half()
    print("before eval", x)
    m_y = random_head(x, None, None)
    print("after eval", m_y)
    random_head.forward(x)

    # create ait model
    n_state, n_head = 384, 6
    ait_model = ResidualAttentionBlock(n_state, n_head)
    X = Tensor(
        shape=[1, 384, 384],
        name="X",
        dtype="float16",
        is_input=True,
    )
    XA = Tensor(
        shape=[n_state, n_head],
        name="XA",
        dtype="float16",
        is_input=True,
    )
    Y = ait_model(X, XA, None)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"

    # map pt weights to ait
    weights = map_pt_params(ait_model, random_head)

    # code gen
    target = detect_target()
    with compile_model(
        Y, target, "./tmp", "simple_model_demo", constants=weights
    ) as module:
        # create storage for output tensor
        y = torch.empty([batch_size, hidden]).cuda().half()

        # inputs and outputs dict
        inputs = {"X": x}
        outputs = {"Y": y}

        # run
        module.run_with_tensors(inputs, outputs, graph_mode=True)

        # verify output is correct
        print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

        # benchmark ait and pt
        #count = 1000
        #ait_t, _, _ = module.benchmark_with_tensors(
        #    inputs, outputs, graph_mode=True, count=count
        #)
        #print(f"AITemplate time: {ait_t} ms/iter")

        #pt_t = benchmark_torch_function(count, pt_model.forward, x)
        #print(f"PyTorch eager time: {pt_t} ms/iter")

        ## check out the fused graph
        ## there are only fused ops in the final graph
        ## gemm_rcr_bias_fast_gelu, gemm_rcr_bias_add, and layernorm
        #graph = module.debug_sorted_graph
        #print("Final graph:")
        #print(sorted_graph_pseudo_code(graph))


