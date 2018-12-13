paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('data.lua')    -- Set up data processing
paths.dofile('model.lua')   -- Read in network model
paths.dofile('train.lua')   -- Load up training/testing functions

--[=====[
function loadPseudoData()
    input_image = image.lena()
    input_scale = 1.444
    input_parts = torch.DoubleTensor({{100,1},{2,3},{4,5},{6,7},{8,9}})
    input_center_x = 128
    input_center_y = 128

    annot.part = input_parts
    annot.scale = input_scale
    annot.center = torch.DoubleTensor({input_center_x, input_center_y})
end
--]=====]

function save_sample_image(inimage, filename)
    local rgb = inimage
    local img = image.minmax{tensor = rgb}
    local out = image.toDisplayTensor{img}
    local img_save_path = paths.concat(savePath_, filename)
    image.save(img_save_path, out)
end

function loadInputData()
    os.execute('mkdir -p ' .. savePath_)
    local timage = torch.cat(inImage_c3, inImage_c2, 3):cat(inImage_c1, 3):permute(3,1,2)
    local input_image = unsqueeze:forward(timage:double())
    input_image_batch = torch.cat(input_image_batch, input_image, 1)
    save_sample_image(input_image[1], 'sample_image.jpg')
    annot.part = input_parts
    annot.scale = input_scale
    annot.center = torch.DoubleTensor({input_center_x, input_center_y})
end

function trainOnThis()
    -- Main training loop
    local input = torch.DoubleTensor()
    local label = {}
    input, label = preprocessData(input_image_batch, annot.part, annot.center, annot.scale)
    save_sample_image(input[1], 'processed.jpg')
    train(input, label)
    collectgarbage()
    iternum = iternum + 1
end
