paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('data.lua')    -- Set up data processing
paths.dofile('model.lua')   -- Read in network model
paths.dofile('train.lua')   -- Load up training/testing functions


function save_sample_image(inimage, filename)
    local rgb = inimage
    local img = image.minmax{tensor = rgb}
    local out = image.toDisplayTensor{img}
    local img_save_path = paths.concat(savePath_, filename)
    image.save(img_save_path, out)
end

function loadInputData()
    os.execute('mkdir -p ' .. savePath_)
    local input_image = torch.cat(inImage_c3, inImage_c2, 3):cat(inImage_c1, 3):permute(3,1,2):double()
    input_image_batch = torch.cat(input_image_batch, input_image, 4)
    input_keypt_batch = torch.cat(input_keypt_batch,input_keypt,3)
    input_scale_batch = torch.cat(input_scale_batch,torch.DoubleTensor({input_scale}),2)
    input_center_batch = torch.cat(input_center_batch,torch.DoubleTensor({input_center_x, input_center_y}),2)
    save_sample_image(input_image, 'sample_image.jpg')
end

function trainOnThis()
    -- Main training loop
    local input = torch.DoubleTensor()
    local label = {}
    input, label = preprocessData(input_image_batch:permute(4,1,2,3), input_keypt_batch:permute(3,1,2), input_center_batch:permute(2,1), input_scale_batch:permute(2,1))
    save_sample_image(input[1], 'processed.jpg')
    train(input, label)
    -- clean batches
    input_image_batch = torch.DoubleTensor()
    input_keypt_batch = torch.DoubleTensor()
    input_scale_batch = torch.DoubleTensor()
    input_center_batch = torch.DoubleTensor()

    collectgarbage()
    iternum = iternum + 1
end
