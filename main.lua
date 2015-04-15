require 'image'
require 'io'
require 'nn'
require 'math'
require 'utility'
require 'ModifiedSGD'
require 'xlua'
local M = require('moses')


if not opt then
    print '==> processing options'
    cmd = torch.CmdLine()
    cmd:text()
    cmd:option('-layers', '25,50' , 'Hidden Units Per Layer')
    cmd:option('-pools', '8,2', 'Pooling Layer Sizes')
    cmd:option('-kernels', '6,3,7', 'Kernel Sizes') -- should be size of #layers + 1
    cmd:option('-relu', false, 'use ReLU nonlinearity layers?')
    cmd:option('-dropout', 0, 'dropout rate (0-1)')
    cmd:option('-indropout', 0, 'dropout rate for input (0-1)')
    cmd:text()
    opt = cmd:parse(arg or {})
end


layers = {}
kernels = {}
pools = {}

for i in string.gmatch(opt.layers, "[^,]+") do
    table.insert(layers, tonumber(i))
end

for i in string.gmatch(opt.kernels, "[^,]+") do
    table.insert(kernels, tonumber(i))
end

for i in string.gmatch(opt.pools, "[^,]+") do
    table.insert(pools, tonumber(i))
end



-- NN Statistics
nInput = 3; 		-- RGB
nClasses = 9;		-- The 8 classes in the Stanford Set+1 unknown

patch_size = patch_size_finder(kernels, pools, 1)
step_pixel = M.reduce(pools, function(acc,v) return acc*v end, 1)
print("Patch size of " .. patch_size)
print("Step pixel is " .. step_pixel)


---------------------------------------------------------------------------
----------------------- Building the training set -------------------------
---------------------------------------------------------------------------
-- 1 represents unknown pixel
images = {}
answers = {}
size = 0;
--table.insert(images, image.load('iccv09Data/images/0000047.jpg'))
--for filename in io.popen('find iccv09Data/images/*.jpg | sort -r -R | head -n  1000'):lines() do
for filename in io.popen('find iccv09Data/images/*.jpg | sort -r -R | head -n  1000'):lines() do
    -- Sort -R will randomize the files
    -- head -n x will get the first x training sets.
    local im = image.load(filename)
    -- Open the corresponding region files

    local region_file = filename:gsub("images", "labels"):gsub(".jpg", ".regions.txt")

    local file = io.open(region_file)

    
    local answer = {}
    -- The classes are as below:
    -- labels:	1	    2		    3		    4	    5
    -- 		    unknown	sky 		tree 		road	grass 
    -- 		    6	    7		    8		    9
    -- 		    water	building	mountain	foreground obj
    for i=1,im:size(2) do
    	answer[i] = {}
    	for j=1,im:size(3) do
    	    answer[i][j] = file:read("*number")+2
    	end
    end
    size = size+1
    answers[size] =  answer
    images[size] = im
end

print("Doing downscaling of training examples")
training_size = 0
training = {} -- train/test dataset
--eg of 1st element of dataset
--  1 : 
--     {
--       1 : DoubleTensor - size: 3x240x320 --> image
--       2 : DoubleTensor - size: 300 --> labels (downscaled)
--     }

start_pixel = (patch_size+1)/2
for ind=1,size do
    local ans = {}
    local k = 0
    -- Set up the related answer set, since downscaling occurs
    for i=1,images[ind]:size(2),step_pixel do
	for j=1,images[ind]:size(3),step_pixel do
	    k = k+1
	    ans[k] = answers[ind][i][j]
	end
    end
    ans.size = function () return k end
    training_size = training_size + 1
    training[training_size] =  { images[ind], torch.Tensor(ans) }
    answers[ind] = nil
    images[ind] = nil
    collectgarbage()
end
training.size = function () return math.floor(training_size*9/10) end
training.testSize = function () return training_size end
print("training size: "..tostring(training.size()))
print("testing size: "..tostring(training.testSize() - training.size()))


---------------------------------------------------------------------------
---------------------------- Creating Model -------------------------------
---------------------------------------------------------------------------
--specify nonlinearity
nonlinearity = nn.Tanh
if opt.relu then
    nonlinearity = nn.ReLU
end


cnn = nn.Sequential();

-- Record dropout layers
opt.dropout_layers = {}

-- input dropout
if opt.indropout > 0 then
    local drop = nn.Dropout(opt.indropout)
    table.insert(opt.dropout_layers, drop)
    cnn:add(drop)
end

-- pad for convolution
cnn:add(nn.SpatialZeroPadding(start_pixel-1,start_pixel-1,start_pixel-1,start_pixel-1))


layers[0] = nInput
for L=1, (#layers) do
    print('creating layer: '..L)
    cnn:add(nn.SpatialConvolution(layers[L-1], layers[L], kernels[L], kernels[L]))
    cnn:add(nn.SpatialMaxPooling(pools[L], pools[L]))
    cnn:add(nonlinearity())
    -- output dropout
    if opt.dropout > 0 then
        local drop = nn.Dropout(opt.dropout)
        table.insert(opt.dropout_layers, drop)
        cnn:add(drop)
    end
end

cnn:add(nn.SpatialConvolution(layers[#layers], nClasses, kernels[#kernels], kernels[#kernels]))


-- Run through CNN and stich together for full output.
-- run single time using the outputs
-- propagate erros back using BPTT
print(cnn:forward(training[1][1]):size())


model = nn.Sequential()
-- Reorganizes to make suitable for criterion
model:add(cnn)
model:add(nn.Flatten())
model:add(nn.Transpose({1,2}))
model:add(nn.LogSoftMax())

print(model:forward(training[1][1]):size())


---------------------------------------------------------------------------
----------------------------- Train/Testing -------------------------------
---------------------------------------------------------------------------

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(model, criterion)
trainer.maxIterations = 50
trainer.learningRate = 0.01
curitr = 1

--hookExample called  during training after each example forwarded and backwarded through the network.
trainer.hookExample = function(self, iteration) xlua.progress(curitr, training.size()); curitr = curitr + 1 end --
--hookIteration calledduring training after a complete pass over the dataset.
trainer.hookIteration =
    function(self, iteration)  
        print("Doing iteration " .. iteration .. "...)");
        curitr = 1
        correct = 0
        total = 0
        --run model on test set and compare output to groud truth
        for i=training.size()+1, training.testSize() do
            local ans = model:forward(training[i][1]):apply(math.exp)
            for k=1,ans:size(1) do
                if ans[k]:max() == ans[k][training[i][2][k]] then correct = correct+1 end
                total = total+1
            end
        end
        print("we got "..tostring(correct/total*100).."% correct!")
        print(correct)
        print(total)
        local filename = 'model.net'
        print('==> saving model to '..filename)
        torch.save(filename, cnn)
    end


trainer:train(training)

--print ("Testing on the first image with classes:")
--print(training[training:size()+1][2])
--print ("Result class probabilities are given: ")
--print (model:forward(training[training:size()+1][1]):apply(math.exp))


