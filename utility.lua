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
LABELS_TO_IMAGE_CONTRASTING_COLORS = {{2, 63, 165}, {74, 111, 227}, {17, 198, 56}, {15, 207, 192}, {125, 135, 185}, {133, 149, 225}, {141, 213, 147}, {156, 222, 214}, {190, 193, 212}, {181, 187, 227}, {198, 222, 199}, {213, 234, 231}, {214, 188, 192}, {230, 175, 185}, {234, 211, 198}, {243, 225, 235}, {187, 119, 132}, {224, 123, 145}, {240, 185, 141}, {246, 196, 225}, {255, 255, 255}, {211, 63, 106}, {239, 151, 8}, {247, 156, 212}}
LABELS_TO_IMAGE_CONTRASTING_COLORS = M.map(LABELS_TO_IMAGE_CONTRASTING_COLORS, function (k,v) return torch.Tensor(v)/255 end)
function label2img(labels, filename)

    if x:nDimension() ~= 2 then return nil end
    local img = torch.zeros(3, labels:size(1), labels:size(2))
    
    for x=1,labels:size(1) do
        for y=1,labels:size(2) do
            img[{{}, x, y}] = LABELS_TO_IMAGE_CONTRASTING_COLORS[labels[x][y]]
        end
    end

    if type(filename) == type("") then image.save(filename, img) end
    return img
end
