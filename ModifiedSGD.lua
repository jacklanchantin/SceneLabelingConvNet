require 'nn'

local ModifiedSGD = torch.class('nn.ModifiedSGD')

function ModifiedSGD:__init(rec, module, criterion, rc)
    self.learningRate = 0.01
    self.learningRateDecay = 0
    self.maxIteration = 25
    self.shuffleIndices = true
    self.module = module
    self.rec = rec
    self.criterion = criterion
    -- recurrent modules
    self.rc = rc
end

function ModifiedSGD:train(dataset)
    local iteration = 1
    local currentLearningRate = self.learningRate
    local module = self.module
    local rec = self.rec
    local criterion = self.criterion

    local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
    if not self.shuffleIndices then
	for t = 1,dataset:size() do
	    shuffledIndices[t] = t 
	end 
    end 
    -- Make the initial weight an biases the same
    for i=1,self.rc:size() do
	local w = (self.rc[i][1].weight+self.rc[i][2].weight)/2
	local b = (self.rc[i][1].bias+self.rc[i][2].bias)/2
	self.rc[i][1].weight = w:clone()
	self.rc[i][2].weight = w:clone()
	self.rc[i][1].bias = b:clone()
	self.rc[i][2].bias = b:clone()
    end
    print("# StochasticGradient: training")
    local joiner = nn.JoinTable(1)
    while true do
	local currentError = 0 
	for t = 1,dataset:size() do
	    print("Training element "..t.." with error "..(currentError/t))
	    local example = dataset[shuffledIndices[t]]
	    local initial_feats= torch.zeros(9, example[1]:size(2), example[1]:size(3))
	    local input = joiner:forward{example[1], initial_feats}
	    local target = example[2]

	    -- Randomly train of f or f^2
	    if torch.randn(1)[1] > 0 then

		currentError = currentError + criterion:forward(module:forward(input), target)

		module:updateGradInput(input, criterion:updateGradInput(module.output, target))
		module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

		-- Average the instances together
		for i=1,self.rc:size() do
		    local w = (self.rc[i][1].weight+self.rc[i][2].weight)/2
		    local b = (self.rc[i][1].bias+self.rc[i][2].bias)/2
		    self.rc[i][1].weight = w:clone()
		    self.rc[i][2].weight = w
		    self.rc[i][1].bias = b:clone()
		    self.rc[i][2].bias = b
		end
	    else
		currentError = currentError + criterion:forward(rec:forward(input), target)

		rec:updateGradInput(input, criterion:updateGradInput(rec.output, target))
		rec:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

		-- Set the first instance to be the second
		for i=1,self.rc:size() do
		    self.rc[i][1].weight = self.rc[i][2].weight:clone()
		    self.rc[i][1].bias = self.rc[i][2].bias:clone()
		end
	    end
	    
	end
	collectgarbage()

	currentError = currentError / dataset:size()
	print("# current error = " .. currentError .. " at iteration " .. iteration)
	iteration = iteration + 1
	currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
	if self.maxIteration > 0 and iteration > self.maxIteration then
	    print("# StochasticGradient: you have reached the maximum number of iterations")
	    break
	end
    end
end

