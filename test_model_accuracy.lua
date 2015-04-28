require 'image'
require 'utility'
require 'paths'
require 'io'
require 'nn'
require 'math'
require 'ModifiedSGD'
require 'xlua'
require 'lfs'
local M = require('moses')


create_shifted_inputs = true



function get_files(setType, numSamples)
    local images = {}
    local answers = {}
    local size = 0;
    -- if numsamples == -1, then retrieve all images in set, otherwise retrieve numSamples
    if numSamples == -1 then
        fileString = 'find ./iccv09Data/images/*.jpg | sort -r -R'
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
function get_files(setType, numSamples)
    local images = {}
    local answers = {}
    local size = 0;
    -- if numsamples == -1, then retrieve all images in set, otherwise retrieve numSamples
    if numSamples == -1 then
        fileString = 'find ./iccv09Data/images/*.jpg | sort -r -R'
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
        print('loading '..setType..' image '..ind)
        if create_shifted_inputs then
            num_shifts = step_pixel-1
        else 
            num_shifts = 0
        end
        local xMap = torch.floor(torch.uniform(0, num_shifts-0.000000001))
        local yMap = torch.floor(torch.uniform(0, num_shifts-0.000000001))

        -- Pad Image
        paddedImg = nn.SpatialZeroPadding(start_pixel-xMap-1,start_pixel-1,start_pixel-yMap-1,start_pixel-1):forward(images[ind])
        -- Set up the related answer set, since downscaling occurs

        y = math.ceil((#labels[ind]-yMap)/step_pixel)
        x = math.ceil((#labels[ind][#labels[ind]]-xMap)/step_pixel)
        ans = torch.Tensor(y,x)
        for i=1,#labels[ind]-yMap,step_pixel do
            for j=1,#labels[ind][#labels[ind]]-xMap,step_pixel do
                nI = math.ceil(i/step_pixel)
                nJ = math.ceil(j/step_pixel)
                L = labels[ind][i+yMap][j+xMap]
                ans[nI][nJ] = L
            end

        end
        

        -- Add image and labels to dataset
        -- ans.size = function () return k end
        datasetInd = datasetInd + 1 
        set[datasetInd] = {paddedImg, ans}
        ans = nil
        paddedImg = nil
        collectgarbage()

        labels[ind] = nil
        images[ind] = nil
        collectgarbage()
    end
    return set,datasetInd
end


dir = '/if19/jjl5sw/GitHub/Scene-Labeling-Conv-Net/Models/'

for modelname in io.popen('ls "'..dir..'"'):lines() do
    if modelname ~= 'model_accuracy.txt' then
        print(modelname)
        --modelname = "nhu=32,64,pools=8,2,conv_kernels=6,3,7,droput=0.5,indropout=0.2,num_images=500,shifted_inputs=false.net"
        local model = torch.load('./Models/'..tostring(modelname))
        patch_size = model.patch_size
        step_pixel = model.step_pixel
        start_pixel = model.start_pixel
        print("Patch size of " .. patch_size)
        print("Step pixel is " .. step_pixel)
        print("Start pixel is " .. start_pixel)



        model2 = nn.Sequential()
        model2:add(model)
        model2:add(nn.Flatten())
        model2:add(nn.Transpose({1,2}))
        -- model2:add(nn.LogSoftMax())
        -- criterion = nn.ClassNLLCriterion()

        model2:evaluate()


        testing,testSz = create_dataset('test',5)
        testing.size = function () return testSz end

        curitr = 1
        correct = 0
        total = 0

        --run model on test set and compare output to groud truth
        for k=1, testing.size() do
            local out = model:forward(testing[k][1])
            _,labels = out:max(1)
            for i=1,labels[1]:size(1) do
                for j=1,labels[1]:size(2) do
                    l = labels[1][i][j]
                    t = testing[k][2][i][j]
                    if l == t then correct = correct+1 end
                    total = total+1
                end
            end
            acc = tostring(correct/total*100)
            print("we got "..acc.."% correct!")
            print(correct)
            print(total)
        end
        os.execute('echo '..modelname:gsub(",", "-")..','..acc..' >> '..dir..'model_accuracy.txt')
    end
end


