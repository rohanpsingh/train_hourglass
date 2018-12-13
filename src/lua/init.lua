require 'paths'
require 'torch'
require 'nn'

optNumParts_ = 20
optTask_ = 'pose-int'
optDataset_ = 'tool'
optInputRes_ = 256
optOutputRes_ = 64
optScaleFactor_ = .25
optRotFactor_ = 30
optNetType_ = 'hg-stacked'
optContinue_ = false
optBranch_ = 'none'
optCrit_ = 'MSE'
optLR_ = 2.5e-4
optLRdecay_ = 0
optMomentum_ = 0
optDamp_ = 0
optWeightDecay_ = 0
nStack_ = 2
useGPU_ = 1
snapWeight_ = 1000
savePath_ = paths.concat(os.getenv('HOME'),'tmp/train_weights')
optColorVar_ = 0.2
optBatchSize_ = 1

inImage_c1 = torch.DoubleTensor()
inImage_c2 = torch.DoubleTensor()
inImage_c3 = torch.DoubleTensor()
input_image_batch = torch.DoubleTensor()

input_keypt = torch.DoubleTensor()
input_scale = 0
input_center_x = 0
input_center_y = 0
input_scale_batch = torch.DoubleTensor()
input_keypt_batch = torch.DoubleTensor()
input_center_batch = torch.DoubleTensor()


optimState = {}

iternum = 1

optimState = {
    learningRate = optLR_,
    learningRateDecay = optLRdecay_,
    momentum = optMomentum_,
    dampening = optDamp_,
    weightDecay = optWeightDecay_
}
