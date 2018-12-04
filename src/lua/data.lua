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

function loadData(input_image, in_pts, in_c, in_s)
    -- Load in a mini-batch of data
    local input,label

    input = torch.Tensor(1, unpack(dataDim))
    label = torch.Tensor(1, unpack(labelDim))

    input[1],label[1] = generateSample(input_image, in_pts, in_c, in_s)

    if input:max() > 2 then
       input:div(255)
    end

    local s = torch.randn(1):mul(optScaleFactor_):add(1):clamp(1-optScaleFactor_,1+optScaleFactor_)
    local r = torch.randn(1):mul(optRotFactor_):clamp(-2*optRotFactor_,2*optRotFactor_)

    -- Color
    local lo_lim = 1 - optColorVar_;
    local up_lim = 1 + optColorVar_;
    input[{1, 1, {}, {}}]:mul(torch.uniform(lo_lim, up_lim)):clamp(0, 1)
    input[{1, 2, {}, {}}]:mul(torch.uniform(lo_lim, up_lim)):clamp(0, 1)
    input[{1, 3, {}, {}}]:mul(torch.uniform(lo_lim, up_lim)):clamp(0, 1)

    -- Scale/rotation
    if torch.uniform() <= .6 then r[1] = 0 end
    local inp,out = optInputRes_, optOutputRes_
    input[1] = crop(input[1], {(inp+1)/2,(inp+1)/2}, inp*s[1]/200, r[1], inp)
    label[1] = crop(label[1], {(out+1)/2,(out+1)/2}, out*s[1]/200, r[1], out)

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
        temp_input,temp_label = loadData(input_image, in_pts, in_c, in_s)

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
