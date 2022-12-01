#HTTP Server stuff
from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import json
import base64
from io import BytesIO
import math

#ML stuff
import torch
import os, glob, random

gpu_model = None
gpu_pipe = {}


import whisper
import os
import numpy as np
import torch

if not(torch.cuda.is_available()): 
    throw("should be")

DEVICE = "cuda" 

whisper.something()
print(DEVICE)

qual="tiny"
model_tiny = whisper.load_model(qual, device=DEVICE)
print(model_tiny.dims)
print("load done")
#model_med = whisper.load_model("medium", device=DEVICE)
#print(
#    f"Model is {'multilingual' if model_base.is_multilingual else 'English-only'} "
#    f"and has {sum(np.prod(p.shape) for p in model_base.parameters()):,} parameters."
#)

import tempfile
import time

lang="ru"
start = time.time()
print("api entry", start)
model = model_tiny
result = model.transcribe(
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

