require '../utility.lua'
require 'image'

local MODEL = "../Models/nhu=25,50,pools=8,2,conv_kernels=6,3,7,droput=0,indropout=0,num_images=-1,shifted_inputs=false.net"

local infile = arg[1]
local outfile = arg[2]
print(infile)
print(outfile)

local input = image.load(infile)

-- Placeholder for utility.test_model
function test_file(input, outfile, filename)
    input = image.scale(input, 300)
    local res = test_model(MODEL, input, nil, filename)
    local out = torch.zeros(input:size())
    image.scale(out, res)
    local blended = out * 0.5 + input * 0.5

    local img = torch.zeros(3, input:size(2), 3*input:size(3))
    img[{{},{},{1, input:size(3)}}] = input
    img[{{},{},{input:size(3)+1, 2*input:size(3)}}] = blended
    img[{{},{},{2*input:size(3)+1, 3*input:size(3)}}] = out
    image.save(outfile, img)
end

test_file(input, outfile, infile)
