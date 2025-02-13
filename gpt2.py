import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

import requests

'''
params
'''
batch_size = 16
block_size = 32
learning_rate = 1e-3
max_iters = 5000
eval_iters = 200
eval_interval = 100
n_embd = 64
n_heads = 4
n_layers = 4
dropout = 0.3



def encode(s):
    return [stoi[c] for c in s]
def decode(s):
    return ''.join([itos[i] for i in s])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return xb, yb

@torch.no_grad
def estimate_loss():
    out = {}
    m.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb,yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    ''' single attention head '''

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-1,-2) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    ''' this implemets several attention heads '''
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out

class FeedForward(nn.Module):
    ''' simple feed forward nn '''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(*[
                                  nn.Linear(n_embd, 4 * n_embd),
                                  nn.ReLU(),
                                  nn.Linear(4 * n_embd, n_embd),
                                  nn.Dropout(dropout)
                                ])
        
    def forward(self, x):
        x = self.net(x)
        return x
    


class Block(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_heads)
        self.ffwd  = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class LanguageModel(nn.Module):

    def __init__(self, n_layers, n_embd, n_heads):
        super().__init__()
        self.token_embs = nn.Embedding(vocab_size, n_embd)
        self.pos_embs = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        token_embs = self.token_embs(x)
        pos_embs = self.pos_embs(torch.arange(T))
        x = token_embs + pos_embs
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x)
        loss = None

        if targets != None:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_tokens):

        context = idx[:,-block_size:]
        logits, _ = self.forward(context)

        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        new_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat([context, new_token], dim=-1)

        return context


# URL of the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Download the dataset
response = requests.get(url)
with open("input.txt", "w", encoding="utf-8") as file:
    file.write(response.text)

with open('input.txt', 'r') as f:
    data = f.read()

vocab = sorted(list(set(data)))
vocab_size = len(vocab)

stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}

data = torch.tensor(encode(data))
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

m = LanguageModel(n_layers, n_embd, n_heads)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    x, y = get_batch('train')
    logits, loss = m(x, y)

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f'at {iter} iteration -- train loss: {losses["train"]}, val loss: {losses["val"]}')

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.tensor([[20]])
for _ in range(int(1e3)):
    context = context[:,-block_size:]
    new_token = m.generate(context, max_tokens=1)
    print(decode(new_token.tolist()[0][-1:]))
    context = torch.cat([context,new_token], dim=-1)



    
