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
for filename in io.popen('find test/*.jpg | sort -r -R | head -n 10'):lines() do
    --filename = './test/9004581.jpg'
    local bn = paths.basename(filename)
    local input = image.load(filename)
    local out = test_model("nhu=6,32,64,32,16,pools=2,2,2,2,2,conv_kernels=4,3,5,3,5,5,droput=0,indropout=0,num_images=500,shifted_inputs=false.net", input, nil, filename)
    local img = torch.zeros(3, out:size(2), out:size(3)+input:size(3))
    img[{{},{},{input:size(3)+1, out:size(3)+input:size(3)}}] = out
    img[{{},{1,input:size(2)},{1, input:size(3)}}] = input
    image.save("sampleimg/"..bn, img)
end

for word in string.gmatch(s,"pools+") do; print(word); end