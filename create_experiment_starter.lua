cmd = torch.CmdLine()
cmd:text()
cmd:text('Creates a shell script to start the experiment')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-use_space', false, 'True to train with space too.')

cmd:option('-min_nodes',  128, 'The min number of node to use.')
cmd:option('-max_nodes', 1024, 'The max number of node to use.')
cmd:option('-max_layers', 5, 'The max number of layers to use')

cmd:option('-min_seq_lengths', 60, 'The min value of sequence length to use')
cmd:option('-max_seq_lengths', 100, 'The max value of sequence length to use')
cmd:option('-seq_length_step', 20, 'The step value to use for increment sequence length')
cmd:option('-max_epochs', 50, 'The max number of full passes through the training data')
cmd:text()

opt = cmd:parse(arg)

local node_iterator = function(max)
  local node = opt.min_nodes/2
  return function()
    node = node * 2
    if(node <= max) then return node end
  end
end

local get_spaces = function(actual, max)
  if (#tostring(actual) < #tostring(max)) then
    return ' '
  else
    return ''
  end
end

local datasets    = {'English2000', 'Evalita'}

local commands = ''

for _, d in pairs(datasets) do
  commands = commands .. '#Train '..d..' dataset with different nodes and layers\n'
  for l = 2, opt.max_layers do
    for n in node_iterator(opt.max_nodes) do
      for s = opt.min_seq_lengths, opt.max_seq_lengths, opt.seq_length_step do
        commands = commands .. 'th train.lua'..
          ' -dataset '..d..
          ' -max_epochs '..opt.max_epochs..
          ' -layers_number '..get_spaces(l, opt.max_layers)..l..
          ' -layer_size '..get_spaces(n, opt.max_nodes)..n..
          ' -seq_length '..get_spaces(s, opt.max_seq_lengths)..s..
          ' -checkpoint_dir cp_'..n..'h_'..l..'l_'..s..'s\n'
        if(opt.use_space) then
            commands = commands .. 'th train.lua'..
              ' -dataset '..d..
              ' -max_epochs '..opt.max_epochs..
              ' -layers_number '..get_spaces(l, opt.max_layers)..l..
              ' -layer_size '..get_spaces(n, opt.max_nodes)..n..
              ' -seq_length '..get_spaces(s, opt.max_seq_lengths)..s..
              ' -use_space -checkpoint_dir cp_'..n..'h_'..l..'l_'..s..'s_ws\n'
        end
      end
    end
  end
  commands = commands .. '\n'
end

local filename = './experiment_starter.sh'
local file = require("pl.file")
file.write(filename, commands)

require('fs')
fs.chmod(filename, 755)

print('Experiment starter script created successfully')
