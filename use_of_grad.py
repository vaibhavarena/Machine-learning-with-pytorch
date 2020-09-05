import torch

datatype = torch.float
device = torch.device("cuda:0")

N, D_in, D_out, H = 64, 1000, 10, 100

x = torch.randn(N, D_in, device=device, dtype=datatype)
y = torch.randn(N, D_out, device=device, dtype=datatype)

w1 = torch.randn(D_in, H, device=device, dtype=datatype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=datatype, requires_grad=True)

learning_rate = 1e-6

for i in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()

    if(i%100==99):
        print(i, loss.item())
    
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

    w1.grad.zero_()
    w2.grad.zero_()