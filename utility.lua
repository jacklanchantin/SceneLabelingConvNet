require 'nn'
require 'image'
local M = require 'moses'

function patch_size_finder(fs, pools, iter)
    local start
    if iter == 1 then start = 1 else
        start = patch_size_finder(fs, pools, iter-1)
    end
    for i=#fs,1,-1 do
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
function test_model(modelname, img, outfilename, filename)
    print("==> Testing Image "..filename)
    local model = torch.load(modelname)

    local patch_size = model.patch_size
    local step_pixel = model.step_pixel
    local start_pixel = model.start_pixel

    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model:evaluate()

    merged = torch.ones(img:size(2), img:size(3)) --img:size(2) = 240 = y and img:size(3) = 320 = x1

    labelMap = {}
    -- for each feature map
    for xMap=0,step_pixel-1 do
        labelMap[xMap+1] = {}
        for yMap=0,step_pixel-1 do

            --print(model)
            paddedImg = nn.SpatialZeroPadding(start_pixel-xMap-1,start_pixel-1,start_pixel-yMap-1,start_pixel-1):forward(img)

            local out = model:forward(paddedImg)
            _,labels = out:max(1)
            labelMap[xMap+1][yMap+1] = labels[1]

            -- for each pixel inside the feature map
            for x=0,((labelMap[xMap+1][yMap+1]:size(2)-1)) do
                for y=0,((labelMap[xMap+1][yMap+1]:size(1)-1)) do
                    local mergeX = (step_pixel*(x)) + xMap
                    local mergeY = (step_pixel*(y)) + yMap
                    --if (mergeX <= merged:size(2)) and (mergeY <= merged:size(1)) then
                    local currMap = labelMap[xMap+1][yMap+1]
                    local val = currMap[y+1][x+1]
                    merged[mergeY+1][mergeX+1] = val
                    --end
                end
            end

        end
    end

    local im = label2img(merged, outfilename)
    return im
end



function create_folds(numFolds)
    os.execute("mkdir -p cv_folds")
    for i=1,numFolds do
        os.execute("mkdir -p ./cv_folds/fold"..i)
    end

    k = 0
    for img_file_name in io.popen('find iccv09Data/images/*.jpg | gsort -r -R'):lines() do
        local region_file_name = img_file_name:gsub("images", "labels"):gsub(".jpg", ".regions.txt")
        os.execute("cp "..img_file_name.." ./cv_folds/fold"..((k%5) + 1))
        os.execute("cp "..region_file_name.." ./cv_folds/fold"..((k%5) + 1))
        k = k + 1
    end
end


function create_sets(numFolds,testFold)
    os.execute("mkdir -p train")
    os.execute("mkdir -p test")
    for i=1,numFolds do
        if i == testFold then
            os.execute("cp ./cv_folds/fold"..i.."/* test/")
        else
            os.execute("cp ./cv_folds/fold"..i.."/* train/")
        end
    end
end



