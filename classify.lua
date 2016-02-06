require './init.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate classified data')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-evaluation', 'evaluation file from where extract classes')
-- optional parameters
cmd:option('-seed', 123, 'random number generator\'s seed')
cmd:option('-output_number', 1, 'output number per char to use for extract classes')
cmd:option('-sample', false, 'true to sample at each timestep, otherwise use argmax')
cmd:option('-temperature', 1, 'temperature of sampling, used only if -sample is true')
-- GPU/CPU
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-opencl', false, 'use OpenCL (instead of CUDA)')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

if not lfs.attributes(opt.evaluation, 'mode') then
  print('Error the file ' .. opt.evaluation .. ' does not exists, \n specify a right evaluation file')
end

-- looking for a suitable gpu
if opt.gpuid >= 0 then
  io.write("Checking GPU...")
  if not opt.opencl then -- with CUDA
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if ok and ok2 then
      print('using CUDA on GPU' .. opt.gpuid)
      cutorch.setDevice(opt.gpuid + 1)
      cutorch.manualSeed(opt.seed)
    else
      opt.opencl = true -- try with OpenCL
    end
  end

  if opt.opencl then -- with OpenCL
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if ok and ok2 then
      print('using OpenCL on GPU ' .. opt.gpuid)
      cltorch.setDevice(opt.gpuid + 1)
      torch.manualSeed(opt.seed)
    else
      print('no suitable GPU, falling back on CPU mode')
      print('if cutorch and cunn or cltorch and clnn are installed,')
      print('your CUDA toolkit or your OpenCL driver may be improperly configured.')
      print(' - check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print(' - check your OpenCL driver installation, check output of clinfo command, and try again.')
      opt.gpuid = -1 -- overwrite user setting
    end
  end
end -- end looking for a suitable gpu

local evaluation = torch.load(opt.evaluation)
local outputs = evaluation.outputs
merge(opt, evaluation.opt)

local data = loadstring('return datasets.'..opt.dataset..'('..tostring(opt.use_space)..')')()
local test_x, test_y = data:testCharTensors()
local test_computed_y = torch.IntTensor(test_y:size(1))
local test_size = test_x:size(1)

local inverted_vocabulary = {}
for c,i in pairs(data.per_char_tag_vocabulary) do inverted_vocabulary[i] = c end
local conf = optim.ConfusionMatrix(inverted_vocabulary)


for i=1, test_size do
  local char_max_outputs
  if opt.sample then -- use sampling
    outputs[i]:div(opt.temperature) -- scale by temperature
    local probs = torch.exp(outputs[i])
    probs:div(torch.sum(probs)) -- renormalize so probs sums to one
    char_max_outputs =
      torch.multinomial(probs:float(), opt.output_number):resize(opt.output_number):float()
  else
    char_max_outputs = torch.FloatTensor(opt.output_number)
    local min = outputs[i]:min()

    for j=1, opt.output_number do
      local _, max_index = outputs[i]:max(2)
      char_max_outputs[j] = max_index:resize(1)
      outputs[i][1][char_max_outputs[j]] = min
    end
  end

  test_computed_y[i] = torch.mode(char_max_outputs):resize(1)
  conf:add(test_computed_y[i], test_y[i])
end

local to_save = {}
to_save.confusion_matrix = conf
to_save.classes = test_computed_y
to_save.opt = opt

local savefile = opt.evaluation:sub(1, opt.evaluation:len() - ( 12 + paths.extname(opt.evaluation):len())) .. '_classification_using_'..opt.output_number..'_outputs.'..paths.extname(opt.evaluation)
print(opt.model)
torch.save(savefile, to_save)
print('Classes saved to ' .. savefile)
