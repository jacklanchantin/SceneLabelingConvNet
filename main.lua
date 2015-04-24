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
    cmd:option('-nhu', '25,50' , 'Hidden Units Per Layer')
    cmd:option('-pools', '8,2', 'Pooling Layer Sizes')
    cmd:option('-conv_kernels', '6,3,7', 'Kernel Sizes') -- should be size of #layers + 1
    cmd:option('-relu', true, 'use ReLU nonlinearity layers?')
    cmd:option('-dropout', 0.5, 'dropout rate (0-1)')
    cmd:option('-indropout', 0.2, 'dropout rate for input (0-1)')
    cmd:text()
    opt = cmd:parse(arg or {})
end


nhu = {}
conv_kernels = {}
pools = {}

for i in string.gmatch(opt.nhu, "[^,]+") do
    table.insert(nhu, tonumber(i))
end
for i in string.gmatch(opt.conv_kernels, "[^,]+") do
    table.insert(conv_kernels, tonumber(i))
end
for i in string.gmatch(opt.pools, "[^,]+") do
    table.insert(pools, tonumber(i))
end


-- NN Statistics
nInput = 3; 		-- RGB
nClasses = 9;		-- The 8 classes in the Stanford Set+1 unknown


--finds the input patch size based on size of convolutional and pooling conv_kernels
patch_size = patch_size_finder(conv_kernels, pools, 1)
step_pixel = M.reduce(pools, function(acc,v) return acc*v end, 1) --product of pooling kernel sizes
start_pixel = (patch_size+1)/2
print("Patch size of " .. patch_size)
print("Step pixel is " .. step_pixel)
print("Start pixel is " .. start_pixel)


---------------------------------------------------------------------------
-------------------------- Building the datasets --------------------------
---------------------------------------------------------------------------

--[[
 Function: Retrieves images and corresponding labels from test or train dataset 
 Inputs:
 	setType: 'train' or 'test'
 	numSamples: number of images to retrieve
 Outputs:
 	images: table of images (3xHeightxWidth)
 	answers: table of labels corresponding to a particular image (HeightxWidth)
 	size: number of images retrieved
]]	
function get_files(setType, numSamples)
    local images = {}
    local answers = {}
    local size = 0;
  	-- if numsamples == -1, then retrieve all images in set, otherwise retrieve numSamples
    if numSamples == -1 then
        fileString = 'find '..setType..'/*.jpg | sort -r -R'
    else
        fileString = 'find '..setType..'/*.jpg | sort -r -R | head -n '..numSamples
    end

    for filename in io.popen(fileString):lines() do
        -- Sort -R will randomize the files
        -- head -n x will get the first x training sets.
        local im = image.load(filename)

        -- Open the corresponding region files
        local region_file = filename:gsub("images", "labels"):gsub(".jpg", ".regions.txt")

        local file = io.open(region_file)
        
        local answer = {}
        -- The classes are as below:
        -- labels:  1       2           3           4       5
        --          unknown sky         tree        road    grass 
        --          6       7           8           9
        --          water   building    mountain    foreground obj
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
    return images,answers,size
end

--[[
 Function: Creates either train or test dataset, which is composed of "setSize" images
 Inputs:
	setType: 'test' or 'train'
    setSize: number of images to use for this dataset 
 Outputs:
 	set: test or train dataset (table consisting of image and corresponding labels)
 	datasetInd: number of samples in dataset
]]	
function create_dataset(setType,setSize)
    images,labels,size = get_files(setType,setSize)
    print('\n==> Creating '..setType..' dataset ('..setSize..' images)')

    set = {}
    datasetInd = 0
    for ind=1,size do
    	print(setType..' image '..ind)
        -- for xMap=0,0 do
        --     for yMap=0,0 do
        for xMap=0,step_pixel-1 do
            for yMap=0,step_pixel-1 do

            	-- Pad Image
				paddedImg = nn.SpatialZeroPadding(start_pixel-xMap-1,start_pixel-1,start_pixel-yMap-1,start_pixel-1):forward(images[ind])

                -- Set up the related answer set, since downscaling occurs
                local ans = {}
                local k = 0
                for i=1,images[ind]:size(2)-yMap,step_pixel do
                    for j=1,images[ind]:size(3)-xMap,step_pixel do
                        k = k + 1
                        ans[k] = labels[ind][i+yMap][j+xMap]
                    end
                end

                -- Add image and labels to dataset
                ans.size = function () return k end
                datasetInd = datasetInd + 1 
                set[datasetInd] = {paddedImg, torch.Tensor(ans)}
                ans = nil
                paddedImg = nil
                collectgarbage()
            end
        end
        labels[ind] = nil
        images[ind] = nil
        collectgarbage()
    end
    return set,datasetInd
end


totalSamples = 200
-- training,trainSz = create_dataset('test',totalSamples*0.9)
-- testing,testSz = create_dataset('train',totalSamples*0.1)
testing,testSz = create_dataset('train',20)
training,trainSz = create_dataset('test',20)

training.size = function () return trainSz end
testing.size = function () return testSz end

print("training size: "..tostring(training.size()))
print("testing size: "..tostring(testing.size()))


---------------------------------------------------------------------------
---------------------------- Creating Model -------------------------------
---------------------------------------------------------------------------
print('\n==> Creating network')

-- Create Convolutional Network
cnn = nn.Sequential();

-- Pad for convolution (EACH feature map of given input is padded with specified number of zeros)
-- *Note*: we do padding manually during the creation of the dataset for the different image maps, so we don't need to here
-- cnn:add(nn.SpatialZeroPadding(start_pixel-1,start_pixel-1,start_pixel-1,start_pixel-1))

-- Specify nonlinearity type
nonlinearity = nn.Tanh
if opt.relu then
    nonlinearity = nn.ReLU
end

-- Table to record dropout layers
opt.dropout_layers = {}

-- Add input dropout
if opt.indropout > 0 then
    local drop = nn.Dropout(opt.indropout)
    table.insert(opt.dropout_layers, drop)
    cnn:add(drop)
end

-- Create network layers
nhu[0] = nInput
for L=1, (#nhu) do
    print('creating layer '..L)

    cnn:add(nn.SpatialConvolution(nhu[L-1], nhu[L], conv_kernels[L], conv_kernels[L]))
    cnn:add(nn.SpatialMaxPooling(pools[L], pools[L]))
    cnn:add(nonlinearity())

    -- Add output dropout
    if opt.dropout > 0 then
        local drop = nn.Dropout(opt.dropout)
        table.insert(opt.dropout_layers, drop)
        cnn:add(drop)
    end
end

-- Applies a 2D convolution over input image composed of 3 (RGB) input planes
cnn:add(nn.SpatialConvolution(nhu[#nhu], nClasses, conv_kernels[#conv_kernels], conv_kernels[#conv_kernels]))

-- Run through CNN and stich together for full output.
print(cnn:forward(training[1][1]):size())

-- Reorganizes to make suitable for criterion
model = nn.Sequential()
model:add(cnn)
model:add(nn.Flatten())
model:add(nn.Transpose({1,2}))
model:add(nn.LogSoftMax())

print(model:forward(training[1][1]):size())

-- Specify Loss Criterion: we use negative log likelihood for the 9-way classification
criterion = nn.ClassNLLCriterion()


---------------------------------------------------------------------------
----------------------------- Train/Testing -------------------------------
---------------------------------------------------------------------------
print('\n==> Training network')

trainer = nn.StochasticGradient(model, criterion)
trainer.maxIterations = 50
trainer.learningRate = 0.01
curitr = 1

--hookExample called  during training after each example forwarded and backwarded through the network.
trainer.hookExample = function(self, iteration) xlua.progress(curitr, training.size()); curitr = curitr + 1 end --

--hookIteration called during training after a complete pass over the dataset.
trainer.hookIteration =
    function(self, iteration)  
        print("--> Doing iteration " .. iteration .. "...");
        curitr = 1
        correct = 0
        total = 0
        --run model on test set and compare output to groud truth
        for i=1, testing.size() do
            local ans = model:forward(testing[i][1]):apply(math.exp)
            for k=1,ans:size(1) do
                if ans[k]:max() == ans[k][testing[i][2][k]] then correct = correct+1 end
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

 --Run Training
trainer:train(training)


--print ("Testing on the first image with classes:")
--print(training[training:size()+1][2])
--print ("Result class probabilities are given: ")
--print (model:forward(training[training:size()+1][1]):apply(math.exp))