import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import sys
import time
import numpy as np
import glob
import os

@dataclass()
class Model_Config():
    block_size:int  = 1024
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int =12
    n_embd:int =768

class mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.SCALE = 1
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (config.n_embd % config.n_head) == 0

        self.c_att = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.SCALE = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_att(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        q = q.view(B,T,self.n_head, C //self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        '''attn = (q@k.transpose(-2,-1)) *(1.0/math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        attn = F.softmax(attn,dim=-1)
        y = attn @ v
        '''

        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.att = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = mlp(config)

    def forward(self,x):
        x = x+self.att(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x

def load_token(file):
    tokens = torch.tensor(np.load(file).astype('float32'), dtype=torch.long)
    return tokens

class Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            l_f = nn.LayerNorm(config.n_embd),
        ))
        self.l_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

        self.transformer.wte.weight = self.l_head.weight
        self.apply(self.init_weights)

    def init_weights(self,module):
        std = 0.02
        if hasattr(module,'SCALE'):
            std *= (2*self.config.n_layer)**-0.25
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,i,target=None):
        B,T = i.size()
        assert (T<=self.config.block_size) ,f"cannot forward seq len{T} less then block size"

        pos = torch.arange(0,T,dtype=torch.long,device=i.device)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(i)
        x = tok_embd+pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.l_f(x)
        logits = self.l_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        return logits,loss

    def config_optimizer(self,wd,lr,device):
        params = {pn:p for pn,p in self.named_parameters()}
        params = {pn:p for pn,p in params.items() if p.requires_grad}
        total_params = sum(p.numel() for p in params.values())
        print("total params:",total_params)
        decay_params = [p for n,p in params.items() if p.dim()>=2]
        nodecay_params = [p for n, p in params.items() if p.dim() < 2]
        d_params = sum(p.numel() for p in decay_params)
        nd_params = sum(p.numel() for p in nodecay_params)
        print("total decay params:",d_params)
        print("total no deacy params:",nd_params)
        optim_groups =[
            {'params':decay_params,'weight_decay':wd},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        f = fused_available and 'cuda' == device.type
        print("using fused AdamW")
        optimizer = torch.optim.AdamW(optim_groups,lr=lr,betas=(0.9,0.95),eps=1e-8,fused=f)
        return optimizer

    def train_(self,epochs):
        for i in range(epochs):
            t1 = time.time()
            optimizer.zero_grad()
            l = 0.0
            for step in range(grad_acc_stps):
                x, y = train_loader.nxt_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_acc_stps
                l += loss.detach()
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            torch.cuda.synchronize()
            t2 = time.time()
            dt = (t2 - t1)
            tks = (train_loader.B * train_loader.T * grad_acc_stps) / (t2 - t1)
            print(
                f"step: {i:4d} | loss: {l.item():.8f} | lr: {lr:.4e} | time: {dt:02.2f}s | norm: {norm:02.4f} | tkn/s: {tks:.2f}")

            if i%50==0:
                self.save(i)

    def save(self,i=None):
        chk_point = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        if i is not None:
            torch.save(chk_point, f"model_{i}.pt")
            print("model saved at ", i)
        else:
            torch.save(chk_point, "model.pt")
            print("model saved")

    def load(self,file):
        chekpoint = torch.load(file)
        model.load_state_dict(chekpoint["model"])
        optimizer.load_state_dict(chekpoint["optimizer"])

class DataLoader():
    def __init__(self,B,T):
        self.B = B
        self.T = T
        #en = tiktoken.get_encoding('gpt2')
        #with open('Shakespeare.txt', 'r') as f:
        #    txt = f.read()
        #assert split in {'train','val'}
        root = f"/home/pss/PycharmProjects/ML/edu_fineweb10B/"
        shards = os.listdir(root)
        shards = [s for s in shards if s.endswith(".npy")]
        shards = sorted (shards)
        shards = [os.path.join(root,s) for s in shards]
        #shard = sorted(glob.glob(f"/home/pss/PycharmProjects/ML/edu_fineweb10B/edufineweb_train_000001.npy"))
        #assert len(shard)>0
        self.shards = shards
        self.shard = 0
        tokens = load_token(shards[0])
        self.len_tokens = len(tokens)
        self.tokens = tokens
        print(f"{len(self.tokens)} tokens loaded")
        print(f"1 epoch = {len(self.tokens)//(B*T)}")
        self.i =0


    def nxt_batch(self):
        B,T = self.B,self.T
        if self.i+B*T+1>self.len_tokens:
            self.i=0
            self.shard = (self.shard+1)%len(self.shards)
            self.tokens = load_token(self.shards[self.shard])
            self.len_tokens = len(self.tokens)

        buf = torch.tensor(self.tokens[self.i:self.i+B*T+1])
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.i +=B*T
        return x,y

def model_test(model,input):
    model = model.eval()
    en = tiktoken.get_encoding('gpt2')
    tokens = en.encode(input)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
    x = tokens.to('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while (x.size(1) < max_len):
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            p = F.softmax(logits, dim=-1)
            tok_p, tok_i = torch.topk(p, 50, dim=-1)

            ix = torch.multinomial(tok_p, 1)
            xcol = torch.gather(tok_i, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_seq):
        tokens = x[i, :max_len].tolist()
        decoded = en.decode(tokens)
        print("->", decoded)


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda:0')
torch.manual_seed(420)
num_return_seq = 5
max_len = 30


total_batch = 524288
B=8
T=1024
assert total_batch%(B*T)==0
grad_acc_stps = total_batch//(B*T)
print("total batch:",total_batch)
print("grad acc steps:",grad_acc_stps)
train_loader = DataLoader(8,1024)


warmup_stps = 190
max_lr = 3e-4
min_lr = max_lr*0.1
max_stps = 50
def get_lr(i):
    if i<warmup_stps:
        return max_lr*(i+1)/warmup_stps
    if i>max_stps:
        return min_lr
    decay_r = (i-warmup_stps)/(max_stps-warmup_stps)
    assert 0<= decay_r <= 1
    c = 0.5* (1.0+math.cos(math.pi*decay_r))
    return min_lr+c*(max_lr-min_lr)

model = Model(Model_Config(vocab_size=50304))
model = model.to(device)
model = torch.compile(model)
optimizer = model.config_optimizer(wd=0.1,lr=6e-4,device=device)

model.load("model.pt")

total_tokens = 500* 1e6
steps = int(total_tokens/(5*1e5))
print("total steps:",steps)

model.train_(200)

model.save()
sys.exit(0)


model_test(model=model,input="hi what are you doing")
