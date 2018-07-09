-------------------------------------------------------------------------------
-- Load necessary libraries and files
-------------------------------------------------------------------------------

require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'string'
require 'image'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')

torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if useGPU_ == -1 then
    nnlib = nn
else
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    cutorch.setDevice(useGPU_)
end


torch.seed()

nParts = optNumParts_
dataDim = {3, optInputRes_, optInputRes_}
labelDim = {nParts, optOutputRes_, optOutputRes_}

paths.dofile('util/' .. optTask_ .. '.lua')             -- dofile pose-int.lua

function applyFn(fn, t, t2)
    -- Helper function for applying an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

