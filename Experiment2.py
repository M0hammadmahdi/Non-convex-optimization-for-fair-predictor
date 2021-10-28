
#loading the data

import numpy as np
from Adult_data import Adult_dataset
import torch 
from algorithms import Algorithm1, Algorithm2, Algorithm3, LR
X_0, y_0, X_1, y_1, X_test_0, y_test_0, X_test_1, y_test_1 = Adult_dataset()
X_0 = np.vstack([X_0,X_test_0])
y_0 = np.vstack([y_0.reshape(-1,1),y_test_0.reshape(-1,1)]).ravel()
X_0 = torch.tensor(X_0,requires_grad=False).float()
n0 = X_0.shape[0]
X_1 = np.vstack([X_1,X_test_1])
y_1 = np.vstack([y_1.reshape(-1,1),y_test_1.reshape(-1,1)]).ravel()
X_1 = torch.tensor(X_1,requires_grad=False).float()
n1 = X_1.shape[0]
y_0 = torch.tensor(y_0,requires_grad=False).float()
y_1 = torch.tensor(y_1,requires_grad=False).float()
X_train = np.vstack([X_0,X_1])
s = X_train.std(axis=0)+10**(-6)
m = X_train.mean(axis=0)
X_0 = (X_0-m)/s # normalize the data
X_1 = (X_1-m)/s # normalize the data

model_apx = []
loss_apx = []
import numpy as np
Gamma = np.linspace(0,0.3,6)

for gamma in Gamma:
  w, l0,l1,l = Algorithm3('LR', X_0, y_0, X_1, y_1,10000,0.001,0.01,gamma)
  model_apx.append(w)
  loss_apx.append(l)

import numpy as np
gamma = 0
loss = []
model = []
for gamma in Gamma:
  w,_,_,l = Algorithm2('LR', X_0, y_0, X_1, y_1, 10000 ,0.001,0.01,gamma)
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)])
  X = torch.vstack([X_0,X_1])
  X = torch.vstack([X_0,X_1])
  model.append(w)
  loss.append(l)

def baseline(method, X_0, y_0, X_1, y_1,num_itr ,lr,gamma):
  X_0 = torch.tensor(X_0,requires_grad=False).float()
  X_1 = torch.tensor(X_1,requires_grad=False).float()
  y_0 = torch.tensor(y_0,requires_grad=False).float()
  y_1 = torch.tensor(y_1,requires_grad=False).float()
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)])
  X = torch.vstack([X_0,X_1])
  if method == 'LR':
    model = LR(X.shape[1])
    LL = torch.nn.BCELoss()
  t = torch.tensor(0.5,requires_grad=False).float()
  for i in range(num_itr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    LL = torch.nn.BCELoss()
    if i%250==0:
      t+=0.1
    loss = LL(model.forward(X),y.reshape(-1,1))+ t*torch.nn.functional.relu(LL(model.forward(X_0),y_0.unsqueeze(1))-LL(model.forward(X_1),y_1.unsqueeze(1))-gamma)**2 + t*torch.nn.functional.relu(LL(model.forward(X_1),y_1.unsqueeze(1))-LL(model.forward(X_0),y_0.unsqueeze(1))-gamma)**2 
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  return model,LL(model.forward(X_0),y_0.unsqueeze(1)),LL(model.forward(X_1),y_1.unsqueeze(1)), LL(model.forward(X),y.reshape(-1,1))

model_baseline = []
loss_baseline = []
for gamma in Gamma:
  w, l0,l1,l = baseline('LR', X_0, y_0, X_1, y_1,10000 ,0.001,gamma)
  model_baseline.append(w)
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
fig.savefig('accexp2.eps', format='eps',bbox_inches='tight')

distanceAlg3 = []
distanceBaseLine = []
for i in range(len(Gamma)):
  v1 = torch.cat([param.view(-1) for param in model[i].parameters()])
  v2 = torch.cat([param.view(-1) for param in model_apx[i].parameters()])
  distanceAlg3.append(torch.norm(v1-v2) )
  v2 = torch.cat([param.view(-1) for param in model_baseline[i].parameters()])
  distanceBaseLine.append(torch.norm(v1-v2) )
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=False)

fig=plt.figure(figsize=(5,4.3))
plt.plot(Gamma,distanceAlg3,':x',linewidth=2)
plt.plot(Gamma,distanceBaseLine,':>',linewidth=2)
plt.ylabel('$||{w} - {w}^*||$')
plt.xlabel('$\gamma$')
plt.legend(['Algorithm3 vs Algorithm2','baseline vs Algorithm2'])
plt.grid()
plt.show()
fig.savefig('wexp2.eps', format='eps',bbox_inches='tight')