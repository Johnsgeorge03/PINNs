#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:40:00 2026

@author: john
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Physical parameters
m = 1.0
c = 0.4
k = 4.0

x0 = 1.0
v0 = 0.0

t_min = 0.0
t_max = 5.0


class PINN(nn.Module):
    def __init__(self, hidden_dim=32, num_hidden_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, t):
        return self.net(t)
    
## Compute derivatives
def compute_derivatives(model, t):
    x = model(t)
    
    dx_dt = torch.autograd.grad(x, t,
                            grad_outputs=torch.ones_like(x), 
                            create_graph=True)[0]
    
    d2x_dt2 = torch.autograd.grad(dx_dt, t,
                                  grad_outputs=torch.ones_like(x), 
                                  create_graph=True)[0]
    
    return x, dx_dt, d2x_dt2

## Compute losses
def pinn_loss(model, t_collocation):
    # physics loss
    x, dx_dt, d2x_dt2 = compute_derivatives(model, t_collocation)
    residual = m * d2x_dt2 + c * dx_dt + k * x
    physics_loss = torch.mean(residual ** 2)
    
    # initial condition loss
    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, 
                      requires_grad=True)
    
    x_t0, dx_dt_t0, _ = compute_derivatives(model, t0)
    ic_loss = torch.mean((x_t0 - x0) ** 2) + torch.mean((dx_dt_t0 - v0) ** 2)
    total_loss = physics_loss + ic_loss
    return total_loss, physics_loss , ic_loss


## Create model
model = PINN().to(device)
n_collocation = 200
t_collocation = torch.linspace(t_min, t_max, n_collocation).view(-1, 1).to(device)
t_collocation.requires_grad_(True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## Train model
epochs = 20000
loss_history = []
physics_history = []
ic_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss, physics_loss, ic_loss = pinn_loss(model, t_collocation)
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    physics_history.append(physics_loss.item())
    ic_history.append(ic_loss.item())
    
    if epoch % 500 == 0:
        print(epoch, loss.item(), physics_loss.item(), ic_loss.item())


## Evaluate
t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
x_t0, dx_dt_t0, _ = compute_derivatives(model, t0)

x_c, dx_c, d2x_c = compute_derivatives(model, t_collocation)
residual = m * d2x_c + c * dx_c + k * x_c

print("Mean residual^2:", torch.mean(residual**2).item())
print("Max |residual|:", torch.max(torch.abs(residual)).item())

print("x(0) predicted:", x_t0.item())
print("x'(0) predicted:", dx_dt_t0.item())

t_eval = torch.linspace(t_min, t_max, 400).view(-1, 1).to(device)

with torch.no_grad():
    x_pred = model(t_eval)


## Visualize
t_plot = t_eval.cpu().numpy()
x_plot = x_pred.cpu().numpy()


plt.figure(figsize=(8, 4))
plt.plot(t_plot, x_plot, label="PINN prediction")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Damped Harmonic Oscillator")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.semilogy(loss_history, label="Total loss")
plt.semilogy(physics_history, label="Physics loss")
plt.semilogy(ic_history, label="IC loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training history")
plt.grid(True)
plt.legend()
plt.show()



    
    





