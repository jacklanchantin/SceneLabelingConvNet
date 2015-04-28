require 'image'
require 'utility'
require 'paths'
require 'io'

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
    local res = test_model("nhu=25,50,pools=8,2,conv_kernels=6,3,7,droput=0,indropout=0,num_images=-1,shifted_inputs=false.net", input, nil, filename)
    local out = torch.zeros(input:size())
    image.scale(out, res)
    local blended = out * 0.5 + input * 0.5

    local img = torch.zeros(3, input:size(2), 3*input:size(3))
    img[{{},{},{1, input:size(3)}}] = input
    img[{{},{},{input:size(3)+1, 2*input:size(3)}}] = blended
    img[{{},{},{2*input:size(3)+1, 3*input:size(3)}}] = out
    image.save("sampleimg/"..bn, img)

    local img = torch.zeros(3, input:size(2), 3*input:size(3))

    local ansfile = filename:sub(1,-4) ..  "regions.txt"
    local file = io.open(ansfile)

    local answer = {}
    for i=1,input:size(2) do
        answer[i] = {}
        for j=1,input:size(3) do
            answer[i][j] = file:read("*number")+2
        end
    end
    answer = torch.Tensor(answer)
    answer = label2img(answer)

    local blended = answer*0.5 + input*0.5
    local img = torch.zeros(3, input:size(2), 3*input:size(3))
    img[{{},{},{1, input:size(3)}}] = input
    img[{{},{},{input:size(3)+1, 2*input:size(3)}}] = blended
    img[{{},{},{2*input:size(3)+1, 3*input:size(3)}}] = answer
    image.save("sampleimg/"..bn:sub(1,-4).."true.jpg", img)
end

--for word in string.gmatch(s,"pools+") do; print(word); end
