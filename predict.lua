require './init.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict from a model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model', 'model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed', 123, 'random number generator\'s seed')
cmd:option('-sample', 1, '0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-temperature', 1, 'temperature of sampling')
cmd:option('-verbose', 1, 'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

if not lfs.attributes(opt.model, 'mode') then
  print('Error the file ' .. opt.model .. ' does not exists, \n specify a right model file')
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

local checkpoint = torch.load(opt.model)
merge(opt, checkpoint.opt)

local protos = checkpoint.protos
protos.rnn:evaluate()

-- init the translator
local translator = _G[opt.loader..'Translator'](opt.dataDir)

-- init the rnn state to all zeros
print('Creating an LSTM')
local current_state = {}

for layer = 1, opt.layers_number do
  local initial_h = torch.zeros(1, opt.layer_size):double()
  if opt.gpuid >= 0 then
    if opt.opencl then initial_h = initial_h:cl()
    else initial_h = initial_h:cuda() end
  end
  table.insert(current_state, initial_h:clone())
  table.insert(current_state, initial_h:clone())
end
local state_size = #current_state

local data = loadstring('return datasets.'..opt.dataset..'('..tostring(opt.use_space)..')')()
local test_x, _ = data:testCharTensors()

if opt.gpuid >= 0 then
  if opt.opencl then test_x = test_x:cl()
  else test_x = test_x:cuda() end
end

for i = 1, #test_x do

end


local seed_text = opt.prime_text
if string.len(seed_text) > 0 then
  print('Seeding with text ' .. seed_text)
  print('--------')
  for c in seed_text:gmatch('.') do
    prevChar = torch.Tensor{ translator.translate(c) }
    io.write(translator.reversedTranslate(prevChar[1]))
    if opt.gpuid >= 0 and opt.opencl == 0 then prevChar = prevChar:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then prevChar = prevChar:cl() end
    local lst = protos.rnn:forward{prevChar, unpack(current_state) }
    -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
    current_state = {}
    for i = 1,state_size do table.insert(current_state, lst[i]) end
    prediction = lst[#lst]
  end
else
  -- fill with uniform probabilities over characters
  print('missing seed text, using uniform probability over first character')
  print('--------------------------')
  prediction = torch.Tensor(1, translator.size):fill(1)/(translator.size)
  if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
  if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
end

for i = 1, opt.length do
  --log probabilities from the previous timestep
  if opt.sample == 0 then
    -- user argmax
    local _, _prevChar = prediction:max(2)
    prevChar = _prevChar:resize(1)
  else
    -- se sampling
    prediction:div(opt.temperature)
    local probs = torch.exp(prediction):squeeze()
    probs:div(torch.sum(probs)) -- renormalize so probs sum to one
    prevChar = torch.multinomial(probs:float(), 1):resize(1):float()
  end

  -- forward the rnn for the next character
  local lst = protos.rnn:forward{prevChar, unpack(current_state) }
  current_state = {}
  for i = 1, state_size do table.insert(current_state, lst[i]) end
  prediction = lst[#lst]

  io.write(translator.reversedTranslate(prevChar[1]))
end
io.write('\n')
io.flush()
