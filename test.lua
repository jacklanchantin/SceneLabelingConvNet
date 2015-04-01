require 'nn'
require 'utility'

h = torch.randn(2,3,3)
mod = nn.Upscale(2)
out = mod:forward(h)
print (out)
print (mod:updateGradInput(h, out))
