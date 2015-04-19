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
function get_files(setType, numSamples)
    local images = {}
    local answers = {}
    local size = 0;
    if numSamples == -1 then
        fileString = 'find '..setType..'/*.jpg | gsort -r -R'
    else
        fileString = 'find '..setType..'/*.jpg | gsort -r -R | head -n '..numSamples
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

function create_dataset(setType,setSize,downsample)
    images,labels,size = get_files(setType,setSize)

    if not downsample then
        step_pixel = 1
    end

    set = {}
    for ind=1,size do
        local ans = {}
        local k = 0
        -- Set up the related answer set, since downscaling occurs
        for i=1,images[ind]:size(2),step_pixel do
            for j=1,images[ind]:size(3),step_pixel do
                k = k+1
                ans[k] =labels[ind][i][j]
            end
        end
        ans.size = function () return k end

        set[ind] =  { images[ind], torch.Tensor(ans) }
        labels[ind] = nil
        images[ind] = nil
        collectgarbage()
    end
    return set,size
end


totalSamples = 200
training,trainSz = create_dataset('test',totalSamples*0.9,true)
testing,testSz = create_dataset('train',totalSamples*0.1,true)



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
cnn:add(nn.SpatialZeroPadding(start_pixel-1,start_pixel-1,start_pixel-1,start_pixel-1))


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
curitr = 1

--hookExample called  during training after each example forwarded and backwarded through the network.
trainer.hookExample = function(self, iteration) xlua.progress(curitr, training.size()); curitr = curitr + 1 end --

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