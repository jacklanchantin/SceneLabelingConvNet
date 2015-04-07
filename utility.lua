require 'nn'
require 'image'
local M = require 'moses'

function patch_size_finder(fs, pools, iter)
    local start
    if iter == 1 then 
	start = 1
    else
	start = patch_size_finder(fs, pools, iter-1)
    end
    for i=table.getn(fs),1,-1 do
	if pools[i] ~= nil then
	    start = start * pools[i]
	end
	start = start + fs[i] - 1
    end
    return start
end


-- Flattens a 2-D image to be used with ClassNLLCriterion
local Flatten, parent = torch.class('nn.Flatten', 'nn.Module')

function Flatten:__init()
    parent.__init(self)
end

function Flatten:updateOutput(input)
    input = input:contiguous()
    self.output:set(input):resize(input:size(1), input:size(2)*input:size(3))
    return self.output
end

function Flatten:updateGradInput(input, gradOutput)
    gradOutput = gradOutput:contiguous()
    self.gradInput:set(gradOutput):resizeAs(input)
    return self.gradInput
end

-- Upscales feature plane
local Upscale, parent = torch.class('nn.Upscale', 'nn.Module')

function Upscale:__init(scale)
    parent.__init(self)
    self.scale = scale
end

function Upscale:updateOutput(input)
    local s = self.scale
    local h = input:size(2)*s
    local w = input:size(3)*s
    self.output:resize(input:size(1), h, w)
    input.image.scaleBilinear(input, self.output)
    return self.output
end

function Upscale:updateGradInput(input, gradOutput)
    local s = self.scale
    self.gradInput:resizeAs(input)
    gradOutput.image.scaleBilinear(gradOutput, self.gradInput)
    return self.gradInput
end

-- Given a torch r x c tensor of integer values from 1 to up to 24, convert it to a
-- image and write it out. Optionally saves it to filename
LABELS_TO_IMAGE_CONTRASTING_COLORS = {{255,0,0}, {228,228,0}, {0,255,0}, {0,255,255}, {176,176,255}, {255,0,255}, {228,228,228}, {176,0,0}, {186,186,0}, {0,176,0}, {0,176,176}, {132,132,255}, {176,0,176}, {186,186,186}, {135,0,0}, {135,135,0}, {0,135,0}, {0,135,135}, {73,73,255}, {135,0,135}, {135,135,135}, {85,0,0}, {84,84,0}, {0,85,0}, {0,85,85}, {0,0,255}, {85,0,85}, {84,84,84}}
LABELS_TO_IMAGE_CONTRASTING_COLORS = M.map(LABELS_TO_IMAGE_CONTRASTING_COLORS, function (k,v) return torch.Tensor(v)/255 end)
function label2img(labels, filename)
    if labels:nDimension() ~= 2 then return nil end
    local img = torch.zeros(3, labels:size(1), labels:size(2))
    
    for x=1,labels:size(1) do
        for y=1,labels:size(2) do
            img[{{}, x, y}] = LABELS_TO_IMAGE_CONTRASTING_COLORS[labels[x][y]]
        end
    end

    if type(filename) == type("") then image.save(filename, img) end
    return img
end

-- Takes a model, runs our covnet on it, scales it up by some amount, and
-- writes it out to outfilename
function test_model(modelname, img, outfilename, scale)
    scale = scale or 1
    local model = torch.load(modelname)
    local out = model:forward(img)
    out = image.scale(out, out:size(3)*scale, out:size(2)*scale)
    local _,labels = out:max(1)
    labels = labels[1]
    local im = label2img(labels, outfilename)
    return im
end
