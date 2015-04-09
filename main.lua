require 'image'
require 'io'
require 'nn'
require 'math'
require 'utility'
require 'ModifiedSGD'
require 'xlua'
local M = require('moses')

-- NN Statistics
nInput = 3; 		-- RGB
nClasses = 9;		-- The 8 classes in the Stanford Set+1 unknown
nHU1 = 25; nHU2 = 50;  	-- Hidden Units per layer
fs = {6, 3, 7};		-- Filter Sizes
pools = {8, 2};		-- Pooling layer sizes

patch_size = patch_size_finder(fs, pools, 1)
step_pixel = M.reduce(pools, function(acc,v) return acc*v end, 1)
print("Patch size of " .. patch_size)
print("Step pixel is " .. step_pixel)

-- Building the training set
-- 1 represents unknown pixel
images = {}
answers = {}
size = 0;
--table.insert(images, image.load('iccv09Data/images/0000047.jpg'))
for filename in io.popen('find iccv09Data/images/*.jpg | sort -r -R | head -n 1000'):lines() do
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

cnn = nn.Sequential();

cnn:add(nn.SpatialZeroPadding(start_pixel-1,start_pixel-1,start_pixel-1,start_pixel-1))
cnn:add(nn.SpatialConvolution(nInput, nHU1, fs[1], fs[1]))
cnn:add(nn.SpatialMaxPooling(pools[1], pools[1]))
cnn:add(nn.Tanh())

cnn:add(nn.SpatialConvolution(nHU1, nHU2, fs[2], fs[2]))
cnn:add(nn.SpatialMaxPooling(pools[2], pools[2]))
cnn:add(nn.Tanh())

cnn:add(nn.SpatialConvolution(nHU2, nClasses, fs[3], fs[3]))

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

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.maxIterations = 50
trainer.learningRate = 0.01
curitr = 1
trainer.hookExample = function(self, iteration) xlua.progress(curitr, training.size()); curitr = curitr + 1 end
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


