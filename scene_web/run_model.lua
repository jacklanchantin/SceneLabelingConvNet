require '../utility.lua'
require 'image'

local infile = arg[1]
local outfile = arg[2]
print(infile)
print(outfile)

local input = image.load(infile)

-- Placeholder for utility.test_model
function test_file(model, img, outfile, filename)
    local k = -img + img:max()
    image.save(outfile, k)
end

test_file(nil, input, outfile, infile)
