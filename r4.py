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
from aitemplate.compiler import Model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum 

import numpy as np

import time

def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        gname = name
        if name == "mlp.1.weight":
            gname = "mlp.2.weight"
        if name == "mlp.1.bias":
            gname = "mlp.2.bias"
        if not(gname in pt_params):
            raise Exception("name not in pt_params", gname)
        mapped_pt_params[ait_name] = pt_params[gname]
    return mapped_pt_params

def map_pt_params_head(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        gname = name.replace("mlp.1.weight","mlp.2.weight").replace("mlp.1.bias", "mlp.2.bias")
        if not(gname in pt_params):
            raise Exception("name not in pt_params", gname)
        mapped_pt_params[ait_name] = pt_params[gname]
    return mapped_pt_params

 

class AITMultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        mask: Tensor,
    ):
        print("x before query ", self.get_shape(x))

        q = self.query(x)

        print(q)
        print("q after query ", self.get_shape(q))

        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        _, n_ctx, n_state = self.get_shape(q)

        total_in_q = 1
        for a in self.get_shape(q):
            total_in_q *= a

        scale = (self.n_state // self.n_head) ** -0.25

        head_dim = (self.n_state // self.n_head)
        assert head_dim > 1
       
        print("total_in_q // self.n_state", self.n_state, total_in_q // self.n_state, head_dim)
        print([total_in_q // self.n_state, self.n_head, head_dim])
        q= ops.permute()(
            ops.reshape()(q, [total_in_q // self.n_state, self.n_head, head_dim]), [1, 0, 2]
        ) * scale


        [a,b,c] = self.get_shape(k)

        chunk_size = a * b * c // self.n_state

        print(self.get_shape(k), chunk_size)

        k= ops.permute()(
            ops.reshape()(k, [-1, self.n_head, head_dim]), [1, 2, 0]
        ) * scale

        v = ops.permute()(
            ops.reshape()(v, [-1, self.n_head, head_dim]), [1, 0, 2]
        )

        qk = ops.bmm_rrr()(q, k)


        print("qkv ", self.get_shape(q), self.get_shape(k), self.get_shape(v))
        print("qk ", self.get_shape(qk), qk)

        if mask is not None:
          print("mask", self.get_shape(mask))
          print(x)
          print(qk)
          qk = qk + mask

        #score = ops.elementwise(FuncEnum.MUL)(qk, scale)
        w = ops.softmax()(qk, -1)

        wv = ops.bmm_rrr()(w, v) 
        
        out = ops.flatten(start_dim=1)(ops.permute()(wv, [1,0,2]))
        out = self.out(out)

        return out 


    def get_shape(self, x):
        shape = [it.value() for it in x._attrs["shape"]]
        return shape

class AITResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        # print("ResidualAttentionBlock", n_state, n_head)
        self.attn = AITMultiHeadAttention(n_state, n_head) #will use mask
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = AITMultiHeadAttention(n_state, n_head) # will use xa
        self.cross_attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp, specialization="fast_gelu"), nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        mask: Tensor,
     ):
        # print("residual input", x, xa, mask)
        # dump({"x": x.cpu().numpy().tolist(), "xa": xa.cpu().numpy().tolist(), "mask": mask.cpu().numpy().tolist() }, "input.json")
        x = x + self.attn(self.attn_ln(x), xa=None, mask=mask) #should use mask
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, None)
        x = x + self.mlp(self.mlp_ln(x))
        # print("residual output", x)
        # raise("Error")
        return x

class AITHeads(nn.Module):
    def __init__(self, n_state : int, n_head : int, n_ctx : int):
        super().__init__()

        self.blocks = nn.ModuleList([AITResidualAttentionBlock(n_state, n_head) for i in range(4)])

    def forward(self, x, xa, mask):
        for i, l in enumerate(self.blocks):
            x = l(x, xa, mask)
        return x


class AITResidualAttentionBlockPartial(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.cross_attn = AITMultiHeadAttention(n_state, n_head) # will use xa
        self.cross_attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp, specialization="fast_gelu"), nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        attention: Tensor,
     ):
        # print("residual input", x, xa, mask)
        # dump({"x": x.cpu().numpy().tolist(), "xa": xa.cpu().numpy().tolist(), "mask": mask.cpu().numpy().tolist() }, "input.json")
        x = x + attention 
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, None)
        x = x + self.mlp(self.mlp_ln(x))
        # print("residual output", x)
        # raise("Error")
        return x



#import r2 

print("loading whisper")
s = time.time()
import whisper

tiny = whisper.load_model("medium", device="cuda")
print("loaded ", time.time() - s )

n_text_ctx = tiny.dims.n_text_ctx


# tiny model dimentions
#r2.ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4, n_vocab=51865, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4)

print(tiny.decoder)
pt_params = dict(tiny.decoder.named_parameters())
print(list(pt_params.keys()))

def compile_partial_block(blocknum, input_size= 1, n_state= 384, n_head= 6, weights={}):

    model = tiny.decoder.cuda().half()
    
    # create ait model
    ait_model = AITResidualAttentionBlockPartial(n_state, n_head)
    
    ait_model.name_parameter_tensor()
    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        if not(ait_name in weights):
            raise Exception(ait_name + " not in weights")

 
    X = Tensor(
        shape=[1, input_size, n_state],
        name="X",
        dtype="float16",
        is_input=True,
    )
    XA = Tensor(
        shape=[1, 1500, n_state],
        name="XA",
        dtype="float16",
        is_input=True,
    )
    ATTENTION = Tensor(
        shape=[1, input_size, n_state],
        name="ATTENTION",
        dtype="float16",
        is_input=True,
    )


    Y = ait_model(X, XA, ATTENTION)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    print(Y)
    
    # code gen
    target = detect_target()
    return compile_model(
        Y, target, "./tmp", "block_%d_%d" % (blocknum, input_size), constants=weights
    )

def compile_block(input_size, block_index):
          blockn = "blocks.%d." % (block_index)
          #'attn.query.weight'
          #'attn.query.bias'
          #'attn.key.weight'
          #'attn.value.weight'
          #'attn.value.bias'
          #'attn.out.weight'
          #'attn.out.bias'
          #'attn_ln.weight'
          #'attn_ln.bias'
          vals = [
          'cross_attn.query.weight',
          'cross_attn.query.bias',
          'cross_attn.key.weight',
          'cross_attn.value.weight',
          'cross_attn.value.bias',
          'cross_attn.out.weight',
          'cross_attn.out.bias',
          'cross_attn_ln.weight',
          'cross_attn_ln.bias',
          'mlp.0.weight',
          'mlp.0.bias',
          'mlp.2.weight',
          'mlp.2.bias',
          'mlp_ln.weight',
          'mlp_ln.bias',
          ]
           
          mapped_pt_params = OrderedDict()
          for name in vals: 
                full = blockn + name
                ait_name = name.replace("mlp.2.", "mlp.1.").replace(".", "_")
                if not(full in pt_params):
                    raise Exception("name not in pt_params", full)
                mapped_pt_params[ait_name] = pt_params[full]
    
          print(mapped_pt_params.keys())
          compile_partial_block(block_index, input_size, 
                n_state = tiny.dims.n_text_state, 
                n_head = tiny.dims.n_text_head, 
                weights = mapped_pt_params)

def compile_modules():
    for input_size in [1,3]: 
       for block_index in range(len(tiny.decoder.blocks)):
           compile_block(input_size, block_index) 
    exit()

#compile_block(3, 0) 
#compile_modules()
tiny.float()

def compile_w_model(input_size, n_state = 384, n_head = 6):

    model = tiny.decoder.cuda().half()
    
    # create ait model
    ait_model = AITHeads(n_state, n_head, n_text_ctx)
    
    X = Tensor(
        shape=[1, input_size, n_state],
        name="X",
        dtype="float16",
        is_input=True,
    )
    XA = Tensor(
        shape=[1, 1500, n_state],
        name="XA",
        dtype="float16",
        is_input=True,
    )
    MASK = Tensor(
        shape=[input_size, input_size],
        name="MASK",
        dtype="float16",
        is_input=True,
    )
    Y = ait_model(X, XA, MASK)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    print(Y)
    
    # map pt weights to ait
    weights = map_pt_params_head(ait_model, model)
    
    # code gen
    target = detect_target()
    return compile_model(
        Y, target, "./tmp", "heads%d" % (input_size), constants=weights
    )

for block_index in range(len(tiny.decoder.blocks)): 
  file3 = Model("./tmp/block_%d_3/test.so" % (block_index)) 
  file1 = Model("./tmp/block_%d_1/test.so" % (block_index)) 
  tiny.decoder.blocks[block_index].hooks = {1: file1, 3: file3}

start = time.time()
print("api entry", start)
result = tiny.transcribe(
"../s.mp3",
language="ru",
task=None,
fp16=torch.cuda.is_available(),
#**transcribe_options
)
end = time.time()
print(result)
print({"result": result['text'], "took": end-start, "lang": lang, "qual": qual})
print({"took": end-start})
   
d = json.loads(open("input.json","r").read())

#dump({"x": initial.cpu().numpy().tolist(), "xa": xa.cpu().numpy().tolist(), "out": x.cpu().numpy().tolist() }, "input.json")
 




