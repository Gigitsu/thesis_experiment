
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--
require './init.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level part-of-speech tagger')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-dataset', 'Evalita', 'The dataset used for training.')
cmd:option('-use_space', false, 'Use a space to separate words during training')
-- model params
cmd:option('-layer_size', 128, 'size of LSTM internal state')
cmd:option('-layers_number', 2, 'number of layers in the LSTM')
-- optimization
cmd:option('-learning_rate', 2e-3, 'learning rate')
cmd:option('-learning_rate_decay', 0.97, 'learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('-dropout', 0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length', 50, 'number of timesteps to unroll for')
cmd:option('-batch_size', 50, 'number of sequences to train on in parallel')
cmd:option('-max_epochs', 50, 'number of full passes through the training data')
cmd:option('-grad_clip', 5, 'clip gradients at this value')
-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 1000, 'every how many iterations should we evaluate on validation data?')
cmd:option('-accurate_gpu_timing', 0, 'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
cmd:option('-checkpoint_dir', paths.concat(g2.SAVE_DIR, 'cp'), 'output directory where checkpoints get written')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-savefile', 'lstm', 'filename to autosave the checkpoint to. Will be inside checkpoint_dir/')
cmd:option('-no_resume', false, 'whether resume or not from last checkpoint')
-- GPU/CPU
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-opencl', false, 'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

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

-- create the data loader class
local data = loadstring('return datasets.'..opt.dataset..'('..tostring(opt.use_space)..')')()
local loader = CoNLLoader(data, opt.batch_size, opt.seq_length)
--print('vocab size: ' .. vocab_size)

local protos = {}

-- create directory for the check points if does not exists
if not opt.checkpoint_dir:sub(1, 1) then
  opt.checkpoint_dir = paths.concat(g2.SAVE_DIR, opt.dataset, opt.checkpoint_dir)
end
if not path.exists(opt.checkpoint_dir) then
  lfs.mkdir(opt.checkpoint_dir)
end

-- looking for candidate checkpoint file
if string.len(opt.init_from) == 0 then
  local mrc = g2.mostRecentFile(opt.checkpoint_dir)

  if mrc ~= nil then opt.init_from = path.join(opt.checkpoint_dir, mrc)
  else opt.no_resume = true end
else
  opt.init_from = path.join(opt.checkpoint_dir, opt.init_from)
end

if not opt.no_resume then -- try to restore the model from a previous checkpoint
  io.write('Trying to resume model from a checkpoint in ' .. opt.init_from .. '...')

  local checkpoint = torch.load(opt.init_from)
  if checkpoint.opt.dataset == opt.dataset then
    protos = checkpoint.protos

    if opt.layer_size ~= checkpoint.opt.layer_size then
      print('WARNING: overwriting layer_size value with ' .. checkpoint.opt.layer_size .. ' found in checkpoint')
    end
    if opt.layers_number ~= checkpoint.opt.layers_number then
      print('WARNING: overwriting layers_number value with ' .. checkpoint.opt.layers_number .. ' found in checkpoint')
    end

    opt.layer_size = checkpoint.opt.layer_size
    opt.layers_number = checkpoint.opt.layers_number

    print('done')
  else
    opt.no_resume = true
  end
end

if opt.no_resume then -- define the model: prototypes for one timestep, then clone them in time
  io.write('Creating an LSTM with ' .. opt.layers_number .. ' layers...')

  local tablex = require('pl.tablex')
  local input_size = tablex.size(data.char_vocabulary)
  local output_size = tablex.size(data.per_char_tag_vocabulary)

  protos.rnn = LSTM(
    input_size,
    output_size,
    OneHot,
    opt.layer_size,
    opt.layers_number,
    opt.dropout
  )

  protos.criterion = nn.ClassNLLCriterion()

  print('done')
end

-- the initial state of the cell/hidden states
init_state = {}
for layer_idx= 1, opt.layers_number do
    local h_init = torch.zeros(opt.batch_size, opt.layer_size)
    if opt.gpuid >= 0 then
      if opt.opencl then h_init = h_init:cl()
      else h_init = h_init:cuda() end
    end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
  if opt.opencl then
    for _, v in pairs(protos) do v:cl() end
  else
    for _, v in pairs(protos) do v:cuda() end
  end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if not opt.no_resume then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

-- initialize the LSTM forget gates with slightly higher biases
-- to encourage remembering in the beginning
for layer_idx = 1, opt.layers_number do
  for _,node in ipairs(protos.rnn.forwardnodes) do
    if node.data.annotations.name == "i2h_" .. layer_idx then
      print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
      -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
      node.data.module.bias[{ { opt.layer_size + 1, opt.layer_size * 2 } }]:fill(1.0)
    end
  end
end

print('number of parameters in the model: ' .. params:nElement())

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name, proto in pairs(protos) do
  print('cloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x,y)
  x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
  y = y:transpose(1,2):contiguous()

  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    if opt.opencl then
      x = x:cl()
      y = y:cl()
    else
      -- have to convert to float because integers can't be cuda()'d
      x = x:float():cuda()
      y = y:float():cuda()
    end
  end

  return x, y
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        x,y = prepro(x,y)
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst]
            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then params:copy(x) end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    x,y = prepro(x,y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t= 1, opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.train_losses = train_losses
        checkpoint.val_losses = val_losses
        checkpoint.val_loss = val_loss
        checkpoint.protos = protos
        checkpoint.epoch = epoch
        checkpoint.opt = opt
        checkpoint.i = i
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end
