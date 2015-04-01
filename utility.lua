require 'nn'
require 'image'

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
    input.image.scaleSimple(input, self.output)
    --[[
    for p=1,input:size(1) do
	for x=0,input:size(2)-1 do
	    for y=0,input:size(3)-1 do
		for i=1,s do
		    for j=1,s do
			self.output[p][x*s+i][y*s+j] = input[p][x+1][y+1]
		    end
		end
	    end
	end
    end
    ]]
    return self.output
end

function Upscale:updateGradInput(input, gradOutput)
    --Averages the gradient from the surrounding locations
    local s = self.scale
    self.gradInput:resizeAs(input)
    gradOutput.image.scaleBilinear(gradOutput, self.gradInput)
    --[[
    for p=1,input:size(1) do
	for x=0,input:size(2)-1 do
	    for y=0,input:size(3)-1 do
		total = 0
		for i=1,s do
		    for j=1,s do
			total = total + gradOutput[p][x*s+i][y*s+j]
		    end
		end
		self.gradInput[p][x+1][y+1] = total / 4;
	    end
	end
    end
    ]]
    return self.gradInput
end
