--- Load up network model or initialize from scratch
paths.dofile('models/' .. optNetType_ .. '.lua')
print('==> Creating model from file: models/' .. optNetType_ .. '.lua')
model = createModel()

-- Criterion (can be set in the opt.task file as well)
if not criterion then
    criterion = nn[optCrit_ .. 'Criterion']()
end

if useGPU_ ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
    
    cudnn.fastest = true
    cudnn.benchmark = true
end
