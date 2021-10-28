

import numpy as np
import torch

def train_barrier(L0,L1, num_itr ,lr, landa):
  t = torch.tensor(0.5,requires_grad=False).float()
  x = torch.tensor(0.0,requires_grad=True).float()
  y = torch.tensor(0.0,requires_grad=True).float()
  z = torch.tensor(0.0,requires_grad=True).float()
  optimizer = torch.optim.SGD([x,y,z], lr=lr)
  for i in range(num_itr+1):
    if i%250==0:
      t += 0.1
    loss =L1(x,y,z)+ t*torch.nn.functional.relu((L0(x,y,z)-landa))**2
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  return (x,y,z)

def Algorithm1(L0,L1,opt0,opt1,epsilon,gamma):

  count=0
  mu_start = L0(opt0[0],opt0[1],opt0[2])
  mu_end = L0(opt1[0],opt1[1],opt1[2])

  while True:
    count += 1
    mu_mid = (mu_end+mu_start)/2
    opt = train_barrier(L0,L1,10,0.0001,mu_mid)
    l1 = L1(opt[0],opt[1],opt[2])+gamma    
    if l1>=mu_mid:
      mu_start = mu_mid
    else:
      mu_end = mu_mid
    if abs(mu_start-mu_end)<=epsilon:
      break
  return opt,L0(opt[0],opt[1],opt[2]),L1(opt[0],opt[1],opt[2])

def Algorithm2(L0,L1,opt0,opt1,epsilon,gamma):
  model1, model1_L0, model1_L1 = Algorithm1(L0,L1,opt0,opt1,epsilon,gamma)
  model2, model2_L0, model2_L1 = Algorithm1(L0,L1,opt0,opt1,epsilon,-gamma)
  if model1_L0+model1_L1<model2_L0+ model2_L1:
    return model1,model1_L0,model1_L1,model1_L0+model1_L1
  else:
    return model2,model2_L0,model2_L1,model2_L0+model2_L1

def Algorithm3(L0,L1,opt0,opt1,epsilon,gamma):
  x = torch.tensor(3.6,requires_grad=True).float()
  y = torch.tensor(3.6,requires_grad=True).float()
  z = torch.tensor(3.6,requires_grad=True).float()

  optimizer = torch.optim.SGD([x,y,z], lr=0.0001)
  for i in range(10):
    loss =L1(x,y,z)+ L0(x,y,z)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

  l0 = L0(x,y,z)
  l1 = L1(x,y,z)
  if l0<l1:
    opt = opt1
  else:
    opt = opt0
  beta0 = 0.0
  beta1 = 1.0
  while beta1-beta0>epsilon:
    beta = (beta1+beta0)/2
    w0 = (1-beta)*x + beta*opt[0]
    w1 = (1-beta)*y + beta*opt[1]
    w2 = (1-beta)*z + beta*opt[2]
    if l1<l0:
      loss = L0(w0,w1,w2) - L1(w0,w1,w2)-gamma
    else:
      loss = L1(w0,w1,w2) -L0(w0,w1,w2) -gamma
    if loss>0:
      beta0 = beta
    else:
      beta1 = beta
 
  return w0,w1,w2, L0(w0,w1,w2)+L1(w0,w1,w2),L0(w0,w1,w2), L1(w0,w1,w2)

#running algorithm 2

L0 = lambda x0,y0,z0: (x0+5)**2+(y0+2)**2+(z0+1)**2+4*x0*z0
opt0 = [1,-2,-3]
L1= lambda x1,y1,z1: (x1-9)**2+(y1-9)**2+ x1*y1+1 + (z1-9)**2 + z1*x1 + z1*y1
opt1 = [4.5,4.5,4.5]
gamma = 0
loss = []
model = []
Gamma = np.linspace(0.5,50,10)
for gamma in Gamma:
  w,_,_,l = Algorithm2(L0,L1,opt0,opt1,0.01,gamma)
  model.append(w)
  loss.append(l)

#running algorithm 3

model_apx = []
loss_apx = []
for gamma in Gamma:
  w0,w1,w2, l,l0,l1 = Algorithm3(L0,L1,opt0,opt1,0.01,gamma)
  model_apx.append((w0,w1,w2))
  loss_apx.append(l)

def baseline(L0,L1, num_itr ,lr, gamma):
  t = torch.tensor(0.5,requires_grad=False).float()
  x = torch.tensor(0.0,requires_grad=True).float()
  y = torch.tensor(0.0,requires_grad=True).float()
  z = torch.tensor(0.0,requires_grad=True).float()
  optimizer = torch.optim.SGD([x,y,z], lr=lr)
  for i in range(num_itr+1):
    if i%250==0:
      t = t+0.1
    loss =L1(x,y,z)+L0(x,y,z)+ t*torch.nn.functional.relu(((L0(x,y,z)-L1(x,y,z)) -gamma))**2 + t*torch.nn.functional.relu(((L1(x,y,z)-L0(x,y,z)) -gamma))**2 
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  return x,y,z, L1(x,y,z)+L0(x,y,z),L0(x,y,z),L1(x,y,z)

model_baseline = []
loss_baseline = []
for gamma in Gamma:
  w0,w1,w2, l,l0,l1 = baseline(L0,L1,10,0.0001,gamma)
  model_baseline.append((w0,w1,w2))
  loss_baseline.append(l)

font = {'family' : 'normal',
        'size'   : 14}
import matplotlib
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,4.3))
plt.plot(Gamma,loss,'-o',linewidth=2)
plt.plot(Gamma,loss_apx,':x',linewidth=2)
plt.plot(Gamma,loss_baseline,':>',linewidth=2)
plt.ylabel('$L({w})$')
plt.xlabel('$\gamma$')
plt.legend(['Algorithm2','Algorithm3','baseline'])
plt.grid()
plt.show()
fig.savefig('acc.eps', format='eps',bbox_inches='tight')

distanceAlg3 = []
distanceBaseLine = []
for i in range(len(Gamma)):
  x,y,z = model[i]
  xx,yy,zz = model_apx[i]
  distanceAlg3.append(((x-xx)**2+(y-yy)**2+(z-zz)**2)**0.5)
  xx,yy,zz = model_baseline[i]
  distanceBaseLine.append(((x-xx)**2+(y-yy)**2+(z-zz)**2)**0.5)
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=False)

fig = plt.figure(figsize=(5,4.3))
plt.plot(Gamma,distanceAlg3,':x',linewidth=2)
plt.plot(Gamma,distanceBaseLine,':>',linewidth=2)
plt.ylabel('$||{w} - {w}^*||$')
plt.xlabel('$\gamma$')
plt.legend(['Algorithm3 vs Algorithm2','baseline vs Algorithm2'])
plt.grid()
plt.show()
fig.savefig('w.eps', format='eps',bbox_inches='tight')