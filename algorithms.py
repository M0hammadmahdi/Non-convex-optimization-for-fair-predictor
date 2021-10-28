import torch
from torch import nn
class LR(torch.nn.Module):
    def __init__(self,d_in):
        super(LR, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d_in, 1))
    def forward(self, x):
        logits = torch.sigmoid(self.linear_relu_stack(x))
        return logits
class LinR(torch.nn.Module):
    def __init__(self,d_in):
        super(LinR, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d_in, 1))
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train_barrier(method, X_0, y_0, X_1, y_1, num_itr ,lr, landa):
  if method == 'LR':
    model = LR(X_0.shape[1])
    LL = nn.BCELoss()
  if method == 'LinR':
    model = LinR(X_0.shape[1])
    LL = nn.MSELoss()

  t = torch.tensor(0.1,requires_grad=False).float()
  count = 0
  while count < num_itr:# or torch.nn.functional.relu((LL(model.forward(X_0),y_0.unsqueeze(1))-landa))>0.01:
    count +=1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if count%250 == 0:
      t = t+1
    loss = LL(model.forward(X_1),y_1.unsqueeze(1))+  t*torch.nn.functional.relu((LL(model.forward(X_0),y_0.unsqueeze(1))-landa))**2 
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  return model

def Algorithm1(method, X_0, y_0, X_1, y_1, num_itr ,lr,epsilon,gamma):
  y = torch.cat((y_0, y_1), 0)
  X = torch.vstack([X_0,X_1])
  if method == 'LR':
    model0 = LR(X_0.shape[1])
    model1 = LR(X_0.shape[1])
    LL = nn.BCELoss()
  elif method == 'LinR':
    model0 = LinR(X_0.shape[1])
    model1 = LinR(X_0.shape[1])
    LL = nn.MSELoss()
  optimizer = torch.optim.SGD(model0.parameters(), lr=lr)
  for i in range(num_itr):
    loss = LL(model0.forward(X_0),y_0.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    #Obtaining W_G2
  mu_start =loss-10**-3
  optimizer = torch.optim.SGD(model1.parameters(), lr=lr)
  for i in range(num_itr):
    loss = LL(model1.forward(X_1),y_1.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  mu_end = LL(model1.forward(X_0),y_0.unsqueeze(1))+10**-3
  count = 1
  while True:
    count += 1
    mu_mid = (mu_end+mu_start)/2
    model = train_barrier(method,X_0,y_0,X_1,y_1,num_itr,lr,mu_mid)
    L1 = LL(model.forward(X_1),y_1.unsqueeze(1))+gamma
    L0 = LL(model.forward(X_0),y_0.unsqueeze(1))
    if L1>=mu_mid:
      mu_start = mu_mid
    else:
      mu_end = mu_mid
    if abs(mu_start-mu_end)<=epsilon:
      break
  return model, L0,L1

def Algorithm2(method, X_0, y_0, X_1, y_1, num_itr ,lr,epsilon,gamma):
  model1, model1_L0, model1_L1 = Algorithm1(method, X_0, y_0, X_1, y_1, num_itr ,lr,epsilon,gamma)
  model2, model2_L0, model2_L1 = Algorithm1(method, X_0, y_0, X_1, y_1, num_itr ,lr,epsilon,-gamma)
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)])
  X = torch.vstack([X_0,X_1])
  if method=='LR':
    LL = nn.BCELoss()
    Loss1 = LL(model1.forward(X),y.reshape(-1,1))
    Loss2 = LL(model2.forward(X),y.reshape(-1,1))
  if method=='LinR':
    LL = nn.MSELoss()
    Loss1 = LL(model1.forward(X),y.reshape(-1,1))
    Loss2 = LL(model2.forward(X),y.reshape(-1,1))
  if Loss1<Loss2:
    return model1,model1_L0,model1_L1, Loss1
  else:
    return model2,model2_L0,model2_L1, Loss2

def Algorithm3(method, X_0, y_0, X_1, y_1, num_itr ,lr,epsilon,gamma):
  y = torch.vstack([y_0.unsqueeze(1),y_1.unsqueeze(1)])
  X = torch.vstack([X_0,X_1])
  if method == 'LR':
    #Obtaining W_O
    model_O = LR(X.shape[1])
    model1 = LR(X.shape[1])
    model = LR(X.shape[1])
    LL = nn.BCELoss()
  elif method=='LinR':
    model_O = LinR(X.shape[1])
    model1 = LinR(X.shape[1])
    model = LinR(X.shape[1])
    LL = nn.MSELoss()  
  optimizer = torch.optim.SGD(model_O.parameters(), lr=lr)
  for i in range(num_itr):
    loss = LL(model.forward(X),y.reshape(-1,1))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  #Obtaining W_G1
  optimizer = torch.optim.SGD(model1.parameters(), lr=lr)
  L0 = LL(model_O.forward(X_0),y_0.unsqueeze(1))
  L1 = LL(model_O.forward(X_1),y_1.unsqueeze(1))
  for i in range(num_itr):
    if L1>L0:
      if method=='SVM':
        output = model1(X_1).squeeze()
        weight = model1.weight.squeeze()
        loss = torch.mean(torch.clamp(1 - y_1 * output, min=0))
        loss += c * (weight.t() @ weight) / 2.0
      else:
        loss = LL(model1.forward(X_1),y_1.unsqueeze(1))
    else:
      if method=='SVM':
        output = model_1(X_0).squeeze()
        weight = model_1.weight.squeeze()
        loss = torch.mean(torch.clamp(1 - y_0 * output, min=0))
        loss += c * (weight.t() @ weight) / 2.0
      else:
        loss = LL(model1.forward(X_0),y_0.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  params_O = {}
  params_1 = {}
  for name, params in model_O.named_parameters():
    params_O[name] = params.clone()

  for name, params in model1.named_parameters():
    params_1[name] = params.clone()
  beta0 = 0
  beta1 = 1
  while beta1-beta0>epsilon:
    beta = (beta1+beta0)/2
    for name, params in model.named_parameters():
      params.data.copy_((1-beta)*params_O[name]+beta*params_1[name])
    loss0 = LL(model.forward(X_0),y_0.unsqueeze(1))
    loss1 = LL(model.forward(X_1),y_1.unsqueeze(1))
    if L1>L0:
      loss = loss1 - loss0-gamma
    else:
      loss = loss0 - loss1 -gamma
    if loss>0:
      beta0 = beta
    else:
      beta1 = beta
  loss_final = LL(model.forward(X),y.reshape(-1,1))
  return model, loss0,loss1, loss_final