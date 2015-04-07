require 'image'
require 'utility'

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
test_model("test.png", "model.net", image.load("iccv09Data/images/6000020.jpg"), 16)
