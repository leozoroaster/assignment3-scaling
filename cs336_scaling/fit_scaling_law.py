import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "isoflops_curves.json"

with open(DATA_PATH, 'r') as file:
    data = json.load(file)

num_data=len(data)

N = np.array([dp["parameters"] for dp in data], dtype=np.float64)
C = np.array([dp["compute_budget"] for dp in data], dtype=np.float64)
D = C / (6.0 * N)
L = np.array([dp["final_loss"] for dp in data], dtype=np.float64)

#predict L=E+A/N^{alpha}+B/D^{beta}
#transform x1=1/N, x2=1/D
x = torch.tensor(np.stack([1.0 / N, 1.0 / D], axis=1), dtype=torch.float32)
y = torch.tensor(L.reshape(-1, 1), dtype=torch.float32)

#initialize nn and optim
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Parameter(torch.tensor([0.1]))
        self.A = nn.Parameter(torch.tensor([1.0]))
        self.a = nn.Parameter(torch.tensor([0.1]))
        self.B = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([0.1]))

    def forward(self,x):
        x1 = x[:,0:1]
        x2 = x[:, 1:2]

        new_x1=self.A * (x1 ** self.a)
        new_x2=self.B * (x2 ** self.b)

        y_hat = self.E + new_x1+new_x2

        return y_hat

model=MyModel()
optimizer = optim.LBFGS(
    model.parameters(),
    lr=0.1,
    max_iter=20,
    line_search_fn="strong_wolfe",
)

loss_fn = nn.HuberLoss(delta=0.001)

def pen_fn(x):
    return torch.relu(-x)

def closure():
    optimizer.zero_grad()
    y_hat = model(x)
    loss = loss_fn(torch.log(y_hat), torch.log(y)) + 0.1*pen_fn(model.E)
    loss.backward()
    return loss

T=100
for t in range(T):
    loss = optimizer.step(closure)

#check predicted loss
print("E = ",model.E)
print("A = ",model.A)
print("B = ",model.B)
print("a = ",model.a)
print("b = ",model.b)

y_hat=model(x)

for i in range(10):
    print(y_hat[i], y[i])

G=(model.a*model.A/(model.b*model.B))**(1/(model.a+model.b))

def predict(compute):
    N_opt=G* (compute/6)**(model.b/(model.a+model.b))
    D_opt = (1/G) * (compute / 6) ** (model.a / (model.a + model.b))
    return N_opt, D_opt

print(predict(1e23))

print(predict(1e24))

