#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import string
import random
import time
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
f = open("dino.txt",'r')
dataset = []
vocab = string.ascii_lowercase
n_letters = len(vocab) + 1
for line in f:
    dataset.append(line.lower().rstrip())
data_size = len(dataset)
vocab_size = len(vocab)
print(list(range(n_letters)))


# In[2]:


def letterToIndex(letter):
    return vocab.find(letter)
def letterToTensor(letter):
    tensor = torch.zeros(1,vocab_size)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
def wordToTensor(word):
    tensor = torch.zeros(len(word),1,n_letters)
    letter_indexes = [vocab.find(word[li]) for li in range(1, len(word))]
    letter_indexes.append(n_letters - 1) # EOS
    target_tensor = torch.zeros(len(word),1,n_letters)
    for i,j in enumerate(word):
        tensor[i][0][letterToIndex(j)] = 1
    for i,j in enumerate(letter_indexes):
        target_tensor[i][0][j] = 1
    return tensor,target_tensor


# In[3]:


a,b = wordToTensor('az')
print(a)
print(b)


# In[4]:


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.inp_to_hidden = torch.nn.Linear(155,128)
        self.hidden_to_output = torch.nn.Linear(128,27)
        self.softmax = torch.nn.Softmax(dim = 1)
    def forward(self,input, hidden):
        inp = torch.cat((input,hidden),dim=1)
        a = torch.tanh(self.inp_to_hidden(inp))
        b = self.hidden_to_output(a)
        y = self.softmax(b)
        return a,y
    def initHidden(self,hidden=128):
        return torch.zeros(1,hidden)


# In[5]:


tens = torch.Tensor([[-0.0643, -0.0104,  0.0593, -0.0488, -0.0704,  0.0590,  0.0488,
          -0.0294, -0.0270,  0.0717,  0.0898, -0.0386,  0.0457,  0.0257,
           0.1794,  0.0761, -0.0853,  0.0087,  0.1660,  0.0853,  0.0206,
           0.0641, -0.0271,  0.0417, -0.0240,  0.0837]])
b = torch.nn.Softmax(dim=1)
d = b(tens)
print(d)


# In[6]:


criterion = torch.nn.BCELoss()
rnn = RNN()
x_1,t_1 = wordToTensor('hello')
print(x_1)
def returnTop(y):
    top_v, top_index = y.topk(1)
    letters = [vocab[i] for i in top_index]
    return letters

def randomChoice(data):
    return data[random.randint(0,len(data)-1)]

def randomWord():
    word = randomChoice(dataset)
    tensor,target = wordToTensor(word)
    return tensor,target


# In[7]:


def sample(rnn):
    x = torch.zeros(1,27)
    a = torch.zeros(1,128)
    y = torch.zeros(1,27)
    counter= 0
    qq = 26
    idx = -1
    temp = []
    name = ""
    while (counter!=50 and qq != idx ):
        a,y = rnn(x,a)
        idx = np.random.choice(list(range(n_letters)),p = y[0].detach().numpy())
        x = torch.zeros(1,27)
        temp.append(idx)
        x[0][idx] = 1
        counter+=1
    for i,j in enumerate(temp):
        if(j!=26):
            name += vocab[j]
    return name


# In[8]:


print(sample(rnn))


# In[9]:



learning_rate = 0.005

def train(x,t):
    a =torch.zeros(1, 128)
    rnn.zero_grad()
    loss = 0
    for i in range(x.shape[0]):
        a,y = rnn(x[i],a)
        l = criterion(y,t[i])
        loss+= l
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return y, loss.item()/x.shape[0]


# In[10]:


train(x_1,t_1)


# In[11]:


n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

for iter in tqdm(range(1, n_iters + 1)):
    x,t = randomWord()
    output, loss = train(x,t)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
        print(sample(rnn))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


# In[14]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()


# In[20]:


print(sample(rnn))


# In[ ]:




