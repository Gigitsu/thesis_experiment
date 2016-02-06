require('./init.lua')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Creates a shell script to start the evaluations')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-output_number', 1, 'output number per char to use for extract classes')
cmd:text()

opt = cmd:parse(arg)

local models = {}
local evaluations = {}

local stringx = require('pl.stringx')

for ds in g2.dirs(g2.SAVE_DIR) do
  for cp in g2.dirs(ds) do
    local min_loss, min_path  = 100, ''
    local max_epoch, max_path =   0, ''
    for t7 in g2.files(cp, '.t7') do
      local lidx = stringx.lfind(t7, 'loss') + 4
      local eidx = stringx.lfind(t7, 'epoch') + 5
      local loss = tonumber(t7:sub(lidx, lidx+5))
      local epoch = tonumber(t7:sub(eidx, lidx-6))
      if(loss and loss < min_loss) then
        min_loss = loss
        min_path = path.join(cp, t7)
      end
      if(epoch and epoch > max_epoch) then
        max_epoch = epoch
        max_path = path.join(cp, t7)
      end
    end
    table.insert(models, min_path)
    table.insert(models, max_path)
  end
end

local commands = ''

for _, m in pairs(models) do
  local b = m:sub(1, m:len() -  3)
  local e = b..'_evaluation.t7'
  local c = b..'_classification_using_'..opt.output_number..'_outputs.t7'
  if(not paths.filep(e)) then
    commands = commands..
      'th evaluate.lua '..m..'\n'
  end
  if(not paths.filep(c)) then
    commands = commands..
      'th classify.lua '..e..' -output_number '..opt.output_number..'\n'
  end
end

local filename = './evaluation_starter.sh'
local file = require("pl.file")
file.write(filename, commands)

require('fs')
fs.chmod(filename, 755)

print('Evaluation starter script created successfully')
