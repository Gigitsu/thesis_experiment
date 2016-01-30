cmd = torch.CmdLine()
cmd:text()
cmd:text('Creates a shell script to start the experiment')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-use_space', false, 'True to train with space too.')

cmd:option('-max_nodes', 1024, 'The max number of node to use.')
cmd:option('-max_layers', 5, 'The max number of layers to use')

cmd:option('-min_seq_lengths', 60, 'The min value of sequence length to use')
cmd:option('-max_seq_lengths', 100, 'The max value of sequence length to use')
cmd:option('-seq_length_step', 20, 'The step value to use for increment sequence length')


opt = cmd:parse(arg)

local node_iterator = function(max)
  local node = 64
  return function()
    node = node * 2
    if(node <= max) then return node end
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
          ' -layers_number '..l..
          ' -layer_size '..n..
          ' -seq_length '..s..
          ' -checkpoint_dir cp_'..n..'h_'..l..'l_'..s..'s'..
          ' -dataset '..d..'\n'
        if(opt.use_space) then
          commands = commands ..
            'th train.lua -use_space -seq_length '..s..' -layer_size '..n..' -layers_number '..l..' -checkpoint_dir cp_'..n..'h_'..l..'l_'..s..'s_ws -dataset '..d..'\n'
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
