require 'image'
require 'utility'
require 'paths'

-- Makes a striped image of the colors we assign to each class
function make_comp()
    x = torch.zeros(500, 500)
    for i=1,10 do
        x[{{}, {(i-1)*50+1, i*50}}] = i
    end
    label2img(x, "comp.png")
end
make_comp()

-- Example of how to test our model
for filename in io.popen('find iccv09Data/images/*.jpg | sort -r -R | head -n 20'):lines() do
    local bn = paths.basename(filename)
    local input = image.load(filename)
    local out = test_model("model.net", input, nil, 16)
    local img = torch.zeros(3, out:size(2), out:size(3)+input:size(3))
    print(input:size())
    print(out:size())
    print(img:size())
    img[{{},{},{input:size(3)+1, out:size(3)+input:size(3)}}] = out
    img[{{},{1,input:size(2)},{1, input:size(3)}}] = input
    image.save("sampleimg/"..bn, img)
end