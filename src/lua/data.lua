inputDim = dataDim
outputDim = {}
outputDim[1] = labelDim
outputDim[2] = labelDim

local function print_dims(prefix,d)
    local s = ""
    if #d == 0 then s = "single value"
    elseif #d == 1 then s = string.format("vector of length: %d", d[1])
    else
        s = string.format("tensor with dimensions: %d", d[1])
        for i = 2,table.getn(d) do s = s .. string.format(" x %d", d[i]) end
    end
    print(prefix .. s)
end

function loadAndAugmentData(input_image, in_pts, in_c, in_s)
    -- Load in a mini-batch of data
    local input,label

    input = torch.Tensor(optBatchSize_, unpack(dataDim))
    label = torch.Tensor(optBatchSize_, unpack(labelDim))
    for i = 1, optBatchSize_ do    
        input[i],label[i] = generateSample(input_image[i], in_pts[i], in_c[i], in_s[i][1])
    end	

    if input:max() > 2 then
       input:div(255)
    end

    local s = torch.randn(optBatchSize_):mul(optScaleFactor_):add(1):clamp(1-optScaleFactor_,1+optScaleFactor_)
    local r = torch.randn(optBatchSize_):mul(optRotFactor_):clamp(-2*optRotFactor_,2*optRotFactor_)

    for i = 1, optBatchSize_ do
        -- Color
    	local lo_lim = 1 - optColorVar_;
	local up_lim = 1 + optColorVar_;
	input[{i, 1, {}, {}}]:mul(torch.uniform(lo_lim, up_lim)):clamp(0, 1)
    	input[{i, 2, {}, {}}]:mul(torch.uniform(lo_lim, up_lim)):clamp(0, 1)
    	input[{i, 3, {}, {}}]:mul(torch.uniform(lo_lim, up_lim)):clamp(0, 1)

    	-- Scale/rotation
    	if torch.uniform() <= .6 then r[1] = 0 end
    	local inp,out = optInputRes_, optOutputRes_
    	input[i] = crop(input[i], {(inp+1)/2,(inp+1)/2}, inp*s[i]/200, r[i], inp)
    	label[i] = crop(label[i], {(out+1)/2,(out+1)/2}, out*s[i]/200, r[i], out)
    end

--[=====[ -- avoid flipping due to labelling ambiguity
    -- Flip
    if torch.uniform() <= .5 then
        input = flip(input)
        label = flip(label)
    end
--]=====]

    if preprocess then input,label = preprocess(input,label) end

    return input, label
end


function preprocessData(input_image, in_pts, in_c, in_s)
    local temp_input,temp_label
    if preprocess then
        --print_dims("Original input is a ", dataDim)
        --print_dims("Original output is a ", labelDim)
        --print("After preprocessing ---")
        temp_input,temp_label = loadAndAugmentData(input_image, in_pts, in_c, in_s)

        -- Input
        if type(temp_input) == "table" then
            inputDim = {}
            print("Input is a table of %d values" % table.getn(temp_input))
            for i = 1,#temp_input do
                inputDim[i] = torch.totable(temp_input[i][1]:size())
                print_dims("Input %d is a "%i, inputDim[i])
            end
        else
            inputDim = torch.totable(temp_input[1]:size())
            --print_dims("Input is a ", inputDim)
        end

        -- Output
        if type(temp_label) == "table" then
            outputDim = {}
            outputDim[1] = torch.totable(temp_label[1][1]:size())
            outputDim[2] = torch.totable(temp_label[2][1]:size())
        else
            outputDim = torch.totable(temp_label[1]:size())
        end
    else
        inputDim = dataDim
        outputDim = labelDim
        print_dims("Input is a ", inputDim)
        print_dims("Output is a ", outputDim)
    end
    return temp_input, temp_label
end
