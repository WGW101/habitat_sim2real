import torch
import sys

has_cuda = torch.cuda.is_available()
if not has_cuda:
	print("Error! torch did not find CUDA...")
	sys.exit(1)
print("torch found cuda!")

dev = torch.device("cuda" if has_cuda else "cpu")

a = torch.randn((8, 10, 2))
b = torch.randn((8, 2, 2))

res_cpu = torch.bmm(a, b)

a = a.to(dev)
b = b.to(dev)
res = torch.bmm(a, b)

all_close = torch.allclose(res_cpu, res.cpu())
if all_close:
	print("bmm successfully returned same result on gpu and cpu!")


class TestModule(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = torch.nn.Linear(3, 512)
		self.fc2 = torch.nn.Linear(512, 1)
	
	def forward(self, x):
		return self.fc2(torch.nn.functional.relu(self.fc1(x)))


mod = TestModule()
mod.to(dev)

x = torch.randn((8, 3))
x = x.to(dev)
y = mod(x)
print(y.cpu())
