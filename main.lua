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
    cmd:option('-relu', false, 'use ReLU nonlinearity layers?')
    cmd:option('-dropout', 0, 'dropout rate (0-1)')
    cmd:option('-indropout', 0, 'dropout rate for input (0-1)')
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
nInput = 3;         -- RGB
nClasses = 9;       -- The 8 classes in the Stanford Set+1 unknown


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
function get_files(setType, numSamples)
    local images = {}
    local answers = {}
    local size = 0;
    if numSamples == -1 then
        fileString = 'find '..setType..'/*.jpg | sort -r -R'
    else
        fileString = 'find '..setType..'/*.jpg | sort -r | head -n '..numSamples
    end

    for filename in io.popen(fileString):lines() do
        -- Sort -R will randomize the files
        -- head -n x will get the first x training sets.
        print(filename)
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

function create_dataset(setType,setSize,downsample)
    images,labels,numImages = get_files(setType,setSize)
    print("creating "..setType.." dataset")

    if not downsample then
        step_pixel = 1
    end

    set = {}
    datasetInd = 0
    for ind=1,numImages do
        print("image #: "..ind)
        -- for xMap=0,0 do
        --     for yMap=0,0 do
        for xMap=0,step_pixel-1 do
            for yMap=0,step_pixel-1 do

  --[[              -- Create padded image (includes padding for convolution)
                paddedImg = torch.zeros(3, images[ind]:size(2)+((start_pixel-1)*2)-(yMap), images[ind]:size(3)+((start_pixel-1)*2)-(xMap))
                for c=1,3 do
                    yS = (start_pixel)-yMap
                    yF = images[ind]:size(2)+(start_pixel)-(yMap)-1
                    xS = (start_pixel)-xMap
                    xF = images[ind]:size(3)+(start_pixel)-(xMap)-1
                    paddedImg[c][{{yS,yF},{xS,xF}}] = images[ind][c]
                end
]]
				paddedImg = nn.SpatialZeroPadding(start_pixel-xMap-1,start_pixel-1,start_pixel-yMap-1,start_pixel-1):forward(images[ind])

                -- Set up the related answer set, since downscaling occurs
                local ans = {}
                local k = 0
                for i=1,images[ind]:size(2)-xMap,step_pixel do
                    for j=1,images[ind]:size(3)-yMap,step_pixel do
                        k = k + 1
                        ans[k] = labels[ind][i+xMap][j+yMap]
                    end
                end

                -- Add image and lables to dataset
                ans.size = function () return k end
                datasetInd = datasetInd + 1 
                set[datasetInd] =  {paddedImg, torch.Tensor(ans)}
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




totalSamples = 10
-- training,trainSz = create_dataset('test',totalSamples*0.9,true)
-- testing,testSz = create_dataset('train',totalSamples*0.1,true)

training,trainSz = create_dataset('train',2,true)
testing,testSz = create_dataset('test',1,true)


print(training)

training.size = function () return trainSz end
testing.size = function () return testSz end
print("training size: "..tostring(training.size()))
print("testing size: "..tostring(testing.size()))



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


--TODO:  Pad image right and bottom by p, pad image left and top by (p-x, p-y)
for x=1,step_pixel do
    for y=1,step_pixel do
        padLeft = start_pixel - x
        padTop = start_pixel - y
    end
end

-- pad for convolution (EACH feature map of given input is padded with specified number of zeros)
--cnn:add(nn.SpatialZeroPadding(start_pixel-(step_pixel)-1))


nhu[0] = nInput
for L=1, (#nhu) do
    print('creating layer: '..L)
    cnn:add(nn.SpatialConvolution(nhu[L-1], nhu[L], conv_kernels[L], conv_kernels[L]))
    cnn:add(nn.SpatialMaxPooling(pools[L], pools[L]))
    cnn:add(nonlinearity())
    -- output dropout
    if opt.dropout > 0 then
        local drop = nn.Dropout(opt.dropout)
        table.insert(opt.dropout_layers, drop)
        cnn:add(drop)
    end
end

cnn:add(nn.SpatialConvolution(nhu[#nhu], nClasses, conv_kernels[#conv_kernels], conv_kernels[#conv_kernels]))


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

---------------------------------------------------------------------------
----------------------------- Train/Testing -------------------------------
---------------------------------------------------------------------------

trainer = nn.StochasticGradient(model, criterion)
trainer.maxIterations = 50
trainer.learningRate = 0.01
trainer.shuffleIndices = false
curitr = 1

--hookExample called  during training after each example forwarded and backwarded through the network.
trainer.hookExample = 
	function(self, iteration) 
		xlua.progress(curitr, training.size());
		curitr = curitr + 1; 
		if curitr < training.size() then
			print(curitr); 
			print(training[curitr])
			local out = model:forward(training[curitr][1])
			local output = 0
			local input = out
			local target = training[curitr][2]
				for i=1,target:size(1) do
					output = output - input[i][target[i]]
				end
			print(out:size())
			print(criterion:forward(out, training[curitr][2]))
		end

	end

--hookIteration called during training after a complete pass over the dataset.
trainer.hookIteration =
    function(self, iteration)  
        print("==> Doing iteration " .. iteration .. "...");
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


trainer:train(training)


--print ("Testing on the first image with classes:")
--print(training[training:size()+1][2])
--print ("Result class probabilities are given: ")
--print (model:forward(training[training:size()+1][1]):apply(math.exp))
