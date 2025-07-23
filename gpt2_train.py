from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import os
import numpy as np
import tiktoken #importing tiktoken library from openAI
enc = tiktoken.get_encoding('gpt2') #getting the gpt2 encoding, tokenizer for gpt2

#---------------------------------------------------------------------4
#Attention Mechanism: Multi Head SA
class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        #output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT =1 #setting some kind of flag for this module

        #regurlarization 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a 'bias', more of a mask, but following the OPENAI/HF naming through
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    

    def forward(self, x):
        B, T, C = x.size() #batch_size, sequence length, embedding dimensionality
        #calculate the query, key, values for all heads in batch and move head forward to be the batch
        #nh is "number of heads", hs is "head size" and C (number of channels) = nh * hs
        #in GPT 2(124M), n_heads = 12, hs = 64, so nh * hs = C = 768 channels in the transformer

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #same
        v = v.view(B, T, self.n_head, C // self.n_head). transpose(1, 2) #same
        #attention- materializes the large (T, T) matrix for all the queries and keys
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        #applying casual mask to block future tokens
        #att = F.softmax(att, dim = -1)
        #y = att @ v #(B, nh, T, T) * (B, nh, T, T) -> (B, nh, T, hs)

        #implementing above 4 line with flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True) #scaled_dot_product_attention - it will calls the flash attention  
        y = y.transpose(1, 2).contiguous().view(B,T,C) #re-assemble all head output side by side

        #output projection
        y = self.c_proj(y) #merging the head back
        return y


#----------------------------------------------------------3
class MLP(nn.Module):
    #position wise FFL in transformers

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


#---------------------------------------------------------------------------------2
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x  
    

    #x → LayerNorm → CausalSelfAttention → +Residual → LayerNorm → MLP (GELU + Linear) → +Residual

#-----------------------------------------------------------------1
@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 #number of tokens: 50, 000 BPE merges + 256 bytes tokens + 1<|endoftext\> token
    n_layer: int = 12 # number of layers 
    n_head: int = 12 #number of heads
    n_embd: int = 768 #embedding dimension

#--------------------------------------------------5

class GPT(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(  #this is basically a module that allows to index into submodule using keys like a dictionary
            wte = nn.Embedding(config.vocab_size, config.n_embd), #nn.Embedding is just like a wrapper around the tensor that allows to access its elements by indexing into the rows.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #ModuleList cause we are indexing it using the integers
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight #768 * 50257 ~= 40M, so 40 / 124 ~= 30 % parameters are being saved using this weight time scheme

#--------------------------------------------------------------------------------------------------9
        #init params
        self.apply(self._init_weights) #we are calling the apply() of NN modules and that iterates all the submodules of this module and apply _init_weights() on them

    def _init_weights(self, module):

        #here we are iterating all the module here 
        if isinstance(module, nn.Linear): #if they are nn.linear module then we're going to make sure to initialize the weight 
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** 0.05 #number of residual layer is 2 * self.config.n_layer
                #the 2 times number of layer comes from the fact that every single one of our layers actually has two blocks that adds to the residual pathway. one is attention and another is MLP, so 2 times
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std) #using normal with the std of 0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) #if there is bias in this layer, we make sure to initialize that to zero

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
#------------------------------------------------------------------------------------------------------------------------6
#6. to generate from this model, forward passing it
    def forward(self, idx, targets = None): #input to the model= is token indices 
        #idx is of shape (B, T) - batch dimension, time_dimension

        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size" #T can be more than the block size(max seq length)

        #forward the token and positional embedding
        pos = torch.arange(0, T, dtype= torch.long, device = idx.device)#creating position sort of indices and shape(T)
        pos_emb = self.transformer.wpe(pos) #position embedding of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embedding of shape (B, T, n_emb
        x = tok_emb + pos_emb # broadcasting is hidden in +, where we have to create an additional dimension here and
        #then these two add up, cause same position embedding apply at every single row of our example stacked up in a batch

        #forward the blocks of the transformers
        for block in self.transformer.h:
            x = block(x)

        #forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size - max possible tokens), and if input was B by T(B,T) indices then at every 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #flattening out the 3D tensors to 2D of logits : i.e first one : B * T and second one : vocab_size
             #single B by T we will calculate logits for what tokens comes next in the sequence
        #and the tensore are what we are going to obtain and logits are softmax away from becoming probabilities
        return logits, loss
#-----------------------------------------------------------------------------------------------------------------------5
    @classmethod
    def from_pretrained(cls, model_type):
        #from_pretrained is the constructor or class object in python that  returns the GPT object(above) if we just give 
        # it the model type which in our case is GPT2  
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" %model_type)

        #n_layer, n_head and n_embd are determined from model_type

        config_args = {
            'gpt2':         dict(n_layer = 12, n_head = 12, n_embd = 768), #124M params
            'gpt-medium':   dict(n_layer = 24, n_head = 16, n_embd = 1024), #358M params
            'gpt2-large':         dict(n_layer = 36, n_head =20, n_embd = 1280),#774M params
            'gpt2-xl':         dict(n_layer = 48, n_head = 25, n_embd = 1600), #155M params

        }[model_type]

        config_args['vocab_size' ] = 50257 #always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 #always 1024 for GPT model checkpoints 
        #create a from-scratch initialized miniGPT model
        config = GPTConfig(**config_args) #creating the config object
        model = GPT(config) #and creating our own model
        #creating the state dict both to our model and for the hugging face model
        sd = model.state_dict() 
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask / buffer -attn.bias just used for the auto regressive mask

        #init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while ensuring all of the parameters are aligned and match in names and shapes

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #ignore these masked bias
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #same just the mask is not
        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight', 
            'mlp.c_fc.weight', 'mlp.c_proj.weight'
            ]
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these masked bias
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]         # same just the mask is not
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Add this debug block here
        sd_keys_hf_set = set(sd_keys_hf)
        sd_keys_set = set(sd_keys)

        extra_in_hf = sd_keys_hf_set - sd_keys_set
        missing_in_hf = sd_keys_set - sd_keys_hf_set

        print("Extra keys in HuggingFace:", extra_in_hf)
        print("Missing keys in HuggingFace:", missing_in_hf)

        #  Assertion that fails
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        #print("Keys in HuggingFace model:", len(sd_keys_hf))
        #print("Keys in your model:", len(sd_keys))
        #for k1, k2 in zip(sd_keys_hf, sd_keys):
        #    print(f"{k1}  <--->  {k2}")
        
        #basically the openai checkpoints use a "Conv1D" module. but we only want to use a vanilla linear
        #this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            
            if any(k.endswith(w) for w in transposed):
                #special treatment for the a Conv1D weigths we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                #vanilla copy over the parameters 
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():

                    sd[k].copy_(sd_hf[k])
                
        return model
    
    #the below configure_optimizers returns the optimizer object


    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        #optim_groups that goes into the AdamW
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters #inspect.signature - checks if the fused quar is present inside adamw
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW : {use_fused}")
        
        #fused makes sure all the kernel are fused into a single kernel, get rid of overhead, instead of iterating over all the tensors
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


#-------------------------------------------------------------------------------------------------------------------------------------------9
import tiktoken

import os
import numpy as np
import torch

def load_tokens(filename):
    # Load shard directly from disk when needed
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)  # No .pin_memory(), just standard tensor
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, num_processes, split):
        self.B = B
        self.T = T
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Get shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        master_process = 1
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset() #reset the dataloader

    def reset(self):
        # State, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        needed = B * T + 1

    # If not enough tokens left, load the next shard
        while self.current_position + needed > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0  # reset position in new shard
        # Check if we have enough tokens left in the current shard
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)#inputs
        y = (buf[:-1]).view(B, T) #targets

        #advance the position in the tensor
        self.current_position +=B*T*self.num_processes
        #if we run out of tokens, load the next shard
        if self.current_position+(B * T * self.num_processes+1)> len(self.tokens):
            self.current_shard = (self.current_shard + 1) %len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            #self.current_position = B * T * self.process_rank
        return x,y
       
#------------------------------------------------------------------------------------------8
#attempt to autodetect the device
import time

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"using device : {device}")

#get a data batch

#import tiktoken
#enc = tiktoken.get_encoding('gpt2')
#with open('input.txt', 'r') as f:
#   text = f.read()
#text = text[:1000]
#tokens = enc.encode(text)
#B, T = 4, 32
#buf = torch.tensor(tokens[:B*T + 1])
#x = buf[:-1].view(B, T).to(device)
#y = buf[1:].view(B,T).to(device)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#gradient accumulation steps - 2**19 / 8 * 512 = 128, so forward and backward steps should be repeated 128 times in training loop 
total_batch_size = 524288 #2 **19, ~0.5M , in number of tokens
B = 8 #micro batch size
T = 512 #sequence length
assert total_batch_size % (B * T) == 0 ,"make sure total_batch_size is divisible by B * T"
grad_accum_steps = 32 # we manually set to 32 but formula is , total_batch_size // (B * T)
print(f"total desired batch size : {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B = 8, T = 512, num_processes=1, split = 'train')
val_loader = DataLoaderLite(B = 8, T = 512 , num_processes=1, split = "val")
#enabling TF32 precision : because in every nn.Linear inside there is matrix multiplication, so expecting that matrix mul to be running on tensor course utilizing the TF32 precision (%).
torch.set_float32_matmul_precision('high') #scale our throughput like 8x times, 3x times

#create model
model = GPT(GPTConfig(vocab_size=50304)) #we override the vocab_size, cause 
model.to(device)
model = model.to(torch.float16)   # convert weights to float16
model = torch.compile(model, backend="eager")

#compile: to make the code faster, since trintion is not available in wnd, we try to skip it.
#model = torch.compile(model, backend="aot_eager")  # still compiles, safer on Windows this is something like compiler for the neural network. it compile the model and return it, it cost comilation time but makes faster

#cosine decay - from gpt3 paper implementation, see section B
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 #gpt 3 paper says they warmup the learning rate over 375 million tokens so 37fe6/2**19 = 715 steps  
max_steps = 5000 #10B /2**19 == around 19,073 steps
def get_lr(it):
    #i. linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr*(it+1) / warmup_steps
    #ii. if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    #iii. in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
 

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, 'w') as f:
    pass
#optimize
#optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-5, betas = (0.9, 0.95), eps = 1e-8)
import time
import warnings

from torch.amp import autocast, GradScaler

# === FORCE model to float32 ===
model = model.to(dtype=torch.float16)  #  This is CRUCIAL to fix the bfloat16 crash

# === OPTIMIZER ===
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# === SCALER ===
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    scaler = GradScaler(device='cuda')

 #LOAD CHECKPOINT IF EXISTS ===
resume_path = os.path.join(log_dir, 'model_04700.pt')  # Or the latest checkpoint
if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])  # If you're using scheduler and saving it
    start_step = checkpoint['step'] + 1
    print(f"Resumed training from step {start_step}")
else:
    print("Starting training from scratch")

start_step = checkpoint['step'] + 1 if 'step' in checkpoint else 0

# === TRAINING LOOP ===
for step in range(start_step, max_steps):
    t0 = time.time()

    #once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type = device, dtype = torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
                
            master_process = 1
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} val{val_loss_accum.item():.4f}\n")

                if step > 0 and step % 100 == 0:
        
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {'model' : model.state_dict(),
                                  'config':model.config,
                                  'step': step,
                                  'val_loss' : val_loss_accum.item(),
                                  }
                    torch.save(checkpoint, checkpoint_path)  

        model.eval()
        num_return_sequences = 4
        max_length= 32
        tokens = enc.encode("Hello, I'm a language model,") #encode this string and get a list of integers which are the tokensf
        tokens = torch.tensor(tokens, dtype = torch.long) #(8, ) so we gonna replicate these 8 tokens for 5 times to get 5 rows of 8 tokens
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5, 8)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            #forward the model to get the logits
            with torch.no_grad():
                logits, loss = model(xgen) #B, T, vocal size
                #take the logits at the last position
                logits = logits[:, -1, :] #(B, vocab_size) #-1 logits only the last location, we throw away all the other logits, as we only care about last column logits

                #get the probabilities
                probs = F.softmax(logits, dim = -1)

                #do top-k sampling of 50 - huggingface pipeline default
                #topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)#we take our probabilities  and only keep the top 50 prob and 
                #anything lower than 50th probability, we just clamp to zero and renoramalize - this way we are never sampling very rare tokens
                #meaning the token we are sampling in the top50 of the most likely tokens which helps model to put on the track

                #select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator= sample_rng) #(B, 1)
                #gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) #(B, 1)

                #append to the sequence
                xgen = torch.cat((xgen, xcol), dim = 1)          
    
        #print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded= enc.decode(tokens)
            print(f"sample {i}: {decoded}")

    #training loop
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type='cuda',dtype=torch.float16):
            logits, loss = model(x, y)
        #we have to scale the loss to accout for gradient accumulation,
        #because the gradient just add on each successive backward()
        #addition of gradient corresponds to a SUM in the objective, but 
        #instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        scaler.scale(loss).backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
    master_process = 1
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.6f} | lr : {lr:.4e} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


    del x, y, logits, loss
    torch.cuda.empty_cache()


#print(logits.shape)
#print(loss) #and our loss comes like 10.9687
#and the thing is : cross entropy loss = -log(1/50257) ~= 10.82 so at initialization(we expect this) so not bad. Its the probability we expect, now performing the optimization

#import sys; sys.exit(0)
#-------------------------------------------------------------------------
#model = GPT.from_pretrained('gpt2')
#print("nice!")
        
#-------------------------------------------------------------------------------------------------------------------7

#num_return_sequences = 5
#max_length = 30

#creating the prefix tokens - the starting some words or sentence before generating the tokens




