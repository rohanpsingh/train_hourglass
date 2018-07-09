-- Get prediction coordinates
predDim = {nParts,2}

criterion = nn.ParallelCriterion()
for i = 1,nStack_ do criterion:add(nn.MSECriterion()) end

-- Code to generate training samples from raw images.
function generateSample(input_image, in_pts, in_c, in_s)

    local img = input_image
    local pts = in_pts
    local c = in_c
    local s = in_s

    local inp = crop(img, c, s, 0, optInputRes_)
    local out = torch.zeros(nParts, optOutputRes_, optOutputRes_)
    for i = 1,nParts do
        if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, optOutputRes_), 1)
        end
    end

    return inp,out
end

function preprocess(input, label)
    newLabel = {}
    for i = 1,nStack_ do newLabel[i] = label end
    return input, newLabel
end

function accuracy(output,label)
    local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8}}
    return heatmapAccuracy(output[#output],label[#output],nil,jntIdxs[optDataset_])
    -- return basicAccuracy(output,label)
end
