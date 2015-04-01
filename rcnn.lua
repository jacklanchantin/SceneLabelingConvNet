require 'image'
require 'io'
require 'nn'
require 'math'
require 'utility'
require 'ModifiedSGD'
 
-- NN Statistics
nClasses = 9;		-- The 8 classes in the Stanford Set+1 unknown
nInput = 3 + nClasses; 	-- RGB, and feature space
nHU1 = 25; nHU2 = 50;  	-- Hidden Units per layer
fs = {8, 8, 1};		-- Filter Sizes
pools = {2, 2};		-- Pooling layer sizes

patch_size = patch_size_finder(fs, pools, 1)
step_pixel = pools[1]*pools[2] -- Hacky solution, assumes 2 pooling layers

-- Building the training set
-- 1 represents unknown pixel
images = {}
answers = {}
size = 0;
--table.insert(images, image.load('iccv09Data/images/0000047.jpg'))
for filename in io.popen('find stanford/*.jpg'):lines() do
    -- Sort -R will randomize the files
    -- head -n x will get the first x training sets.
    local im = image.load(filename)
    -- Open the corresponding region files
    local file = io.open(filename:sub(0,-4).."regions.txt")
    local answer = {}
    -- The classes are as below:
    -- labels:	1	2		3		4	5
    -- 		unknown	sky 		tree 		road	grass 
    -- 		6	7		8		9
    -- 		water	building	mountain	foreground obj
    for i=1,im:size(2) do
	answer[i] = {}
	for j=1,im:size(3) do
	    answer[i][j] = file:read("*n")+2
	end
    end
    size = size+1
    answers[size] =  answer
    images[size] = im
end

training_size = 0
training = {}
start_pixel = (patch_size+1)/2
for ind=1,size do
    local ans = {}
    local k = 0
    local h = images[ind]:size(2)%step_pixel
    local w = images[ind]:size(3)%step_pixel
    local im = nn.SpatialZeroPadding(0,-w,0,-h):forward(images[ind])
    local feat = torch.zeros(9, im:size(2), im:size(3))
    -- Set up the related answer set, since downscaling occurs
    for i=1,im:size(2)-h,step_pixel do
	for j=1,im:size(3)-w,step_pixel do
	    k=k+1
	    ans[k] = answers[ind][i][j]
	end
    end
    ans.size = function () return k end
    training_size = training_size + 1
    training[training_size] =  { im, ans }
end
training.size = function () return math.floor(training_size*9/10) end
training.testSize = function () return training_size end
print("training size: "..tostring(training.size()))
print("testing size: "..tostring(training.testSize() - training.size()))

cnn1 = nn.Sequential();

cnn1:add(nn.SpatialZeroPadding(start_pixel-1,start_pixel-1,start_pixel-1,start_pixel-1))

conv_net11 = nn.SpatialConvolution(nInput, nHU1, fs[1], fs[1])
cnn1:add(conv_net11)
cnn1:add(nn.SpatialMaxPooling(pools[1], pools[1]))
cnn1:add(nn.Tanh())

conv_net12 = nn.SpatialConvolution(nHU1, nHU2, fs[2], fs[2])
cnn1:add(conv_net12)
cnn1:add(nn.SpatialMaxPooling(pools[2], pools[2]))
cnn1:add(nn.Tanh())

conv_net13 = nn.SpatialConvolution(nHU2, nClasses, fs[3], fs[3])
cnn1:add(conv_net13)

cnn2 = nn.Sequential();

cnn2:add(nn.SpatialZeroPadding(start_pixel-1,start_pixel-1,start_pixel-1,start_pixel-1))

conv_net21 = nn.SpatialConvolution(nInput, nHU1, fs[1], fs[1])
cnn2:add(conv_net11)
cnn2:add(nn.SpatialMaxPooling(pools[1], pools[1]))
cnn2:add(nn.Tanh())

conv_net22 = nn.SpatialConvolution(nHU1, nHU2, fs[2], fs[2])
cnn2:add(conv_net12)
cnn2:add(nn.SpatialMaxPooling(pools[2], pools[2]))
cnn2:add(nn.Tanh())

conv_net23 = nn.SpatialConvolution(nHU2, nClasses, fs[3], fs[3])
cnn2:add(conv_net13)

averager = { { conv_net11, conv_net21},
             { conv_net12, conv_net22},
             { conv_net13, conv_net23} }
averager.size = function () return 3 end

	    
initial_feats = torch.zeros(9, training[1][1]:size(2), training[1][1]:size(3))
joiner = nn.JoinTable(1)
example = joiner:forward{training[1][1], initial_feats}
--print(example:size())


cnn = nn.Sequential();
    -- Recurrent architecture, splits input into input and input with
    -- just the image, runs it trhough, and upscales output label plane
    rec1 = nn.Concat(1)
    rec1:add(nn.Sequential():add(cnn1):add(nn.Upscale(step_pixel)))
    rec1:add(nn.Narrow(1,1,3))
cnn:add(rec1)
cnn:add(cnn2)

--print(cnn:forward(example):size())
--print(training[1][2]:size())

-- Reorganizes to make suitable for criterion
cnn2:add(nn.Flatten())
cnn2:add(nn.Transpose({1,2}))
cnn2:add(nn.LogSoftMax())

--print(cnn:forward(training[1][1]):size())
--print(training[1][2]:size())

criterion = nn.ClassNLLCriterion()
trainer = nn.ModifiedSGD(cnn2, cnn, criterion, averager)
trainer.maxIteration = 25
trainer.learningRate = 0.01
trainer:train(training)

correct = 0
total = 0
for i=training.size()+1, training.testSize() do
    local initial_feats = torch.zeros(9, training[i][1]:size(2), training[i][1]:size(3))
    local example = joiner:forward{training[i][1], initial_feats}
    local ans = cnn:forward(example):apply(math.exp)
    for k=1,ans:size(1) do
	if ans[k]:max() == ans[k][training[i][2][k]] then correct = correct+1 end
	total = total+1
    end
end
print("we got "..tostring(correct/total*100).."% correct!")
print(correct)
print(total)
