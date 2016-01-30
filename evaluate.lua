require './init.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify data using a model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model', 'model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed', 123, 'random number generator\'s seed')
-- GPU/CPU
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-opencl', false, 'use OpenCL (instead of CUDA)')
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
local test_size = test_x:size(1)

local outputs = {}

test_x:resize(test_size, 1)

if opt.gpuid >= 0 then
  if opt.opencl then test_x = test_x:cl()
  else test_x = test_x:cuda() end
end

io.write('Classifying data:  ')
io.flush()

local prev_percentage = -1

local function test_percentage(i)
  return math.floor(i * 100 / test_size);
end

for i = 1, test_size do
  -- forward the rnn for next character
  local lst = protos.rnn:forward{test_x[i], unpack(current_state)}

  current_state = {}
  for j=1, state_size do
    table.insert(current_state, lst[j])
  end
  table.insert(outputs, lst[#lst]:clone():double())

  local percentage = test_percentage(i)
  if(percentage ~= prev_percentage) then
    prev_percentage = percentage
    if(percentage % 10 == 0) then
      io.write(percentage..'%')
    elseif(percentage % 2 == 0) then
      io.write('.')
    end
    io.flush()
  end
end

io.write('\n')
io.flush()

local to_save = {}
to_save.outputs = outputs
to_save.opt = opt

local savefile = opt.model:sub(1, opt.model:len() - ( 1 + paths.extname(opt.model):len())) .. '_evaluation.'..paths.extname(opt.model)
torch.save(savefile, to_save)
print('Evaluation saved to ' .. savefile)
