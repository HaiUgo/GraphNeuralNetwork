import torch

input_channels, output_channels = 5, 10
batch_size = 1
kernel_size = 3
width, height = 100, 100
input = torch.randn(batch_size, input_channels, width, height)

conv_layer = torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)   # torch.Size([1, 5, 100, 100])
print(output.shape)   # torch.Size([1, 10, 98, 98])
print(conv_layer.weight.shape)   # torch.Size([10, 5, 3, 3])


import torch
input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]
input = torch.Tensor(input).view(1, 1, 5, 5)
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data
output = conv_layer(input)
print(output)

