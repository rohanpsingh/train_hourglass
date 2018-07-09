-- Model parameters
param, gradparam = model:getParameters()

-- Optimization function
local optfn = optim['rmsprop']


-- Main processing step
function train(input,label)

    local output, err, idx
    local function evalFn(x) return criterion.output, gradparam end

    model:training()

    if useGPU_ ~= -1 then
        -- Convert to CUDA
        input = applyFn(function (x) return x:cuda() end, input)
        label = applyFn(function (x) return x:cuda() end, label)
    end

    -- Do a forward pass and calculate loss
    local output = model:forward(input)
    local err = criterion:forward(output, label)


    -- Training: Do backpropagation and optimization
    model:zeroGradParameters()
    model:backward(input, criterion:backward(output, label))
    optfn(evalFn, param, optimState)

    -- Synchronize with GPU
    if useGPU_ ~= -1 then cutorch.synchronize() end

    -- Calculate accuracy
    local acc = accuracy(output, label)
    local loss = err

    print("Accuracy: ", acc, "   Loss: ", loss)


    if snapWeight_ ~= 0 and iternum % snapWeight_ == 0 then
        model:clearState()
        torch.save(paths.concat(savePath_, 'model_' .. iternum .. '.t7'), model)
        torch.save(paths.concat(savePath_, 'optimState.t7'), optimState)
    end

end
