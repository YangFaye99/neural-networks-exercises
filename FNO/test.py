import torch

def test():
    x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    y = 2.0 * x;
    z = torch.autograd.grad(outputs=y[0], inputs=x, retain_graph=True)[0]
    z.requires_grad = True
    print(y.grad())
    f = torch.norm(y)
    g = torch.autograd.grad(outputs=f, inputs=y, grad_outputs=torch.ones_like(f), retain_graph=True)[0]
    print(g.data)
    g = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=torch.ones_like(f), retain_graph=True)[0]
    print(g.data)
    f.backward()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()