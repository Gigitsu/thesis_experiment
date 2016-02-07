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
      print('your CUDA toolkit or your OpenCL driver may be improperly char_configured.')
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
local test_char_computed_y = torch.IntTensor(test_y:size(1)):zero()
local test_size = test_x:size(1)

local inverted_char_vocabulary, inverted_word_vocabulary = {}, {}
for c,i in pairs(data.per_char_tag_vocabulary) do inverted_char_vocabulary[i] = c end
for c,i in pairs(data.tag_vocabulary) do inverted_word_vocabulary[i] = c end

local char_conf, word_conf =
  optim.ConfusionMatrix(inverted_char_vocabulary),
  optim.ConfusionMatrix(inverted_word_vocabulary)

local tablex = require('pl.tablex')
local i = 0

tablex.foreach(data.test_data, function(sentence) -- start iterate through sentences
  tablex.foreach(sentence, function(word_parts) -- start iterate through words
    local word_max_outputs

    if opt.use_space then
      word_max_outputs = torch.FloatTensor(#word_parts[CoNLL.c_chars] - 1, opt.output_number)
    else
      word_max_outputs = torch.FloatTensor(#word_parts[CoNLL.c_chars], opt.output_number)
    end

    for k = 1, #word_parts[CoNLL.c_chars] do -- start prediction through chars
      i = i + 1

      --char prediction
      local _, char_tag = outputs[i]:max(2)
      test_char_computed_y[i] = char_tag:resize(1)
      char_conf:add(test_char_computed_y[i], test_y[i])

      --word prediction
      if not opt.use_space or ( opt.use_space and k < #word_parts[CoNLL.c_chars] ) then
        local min = outputs[i]:min()
        for j=1, opt.output_number do
          local _, max_index = outputs[i]:max(2)
          word_max_outputs[k][j] = max_index:resize(1)
          outputs[i][1][word_max_outputs[k][j]] = min
        end
      end

    end -- end prediction through chars

    local word_tag = torch.mode(word_max_outputs:view(word_max_outputs:nElement()))

    word_tag = inverted_char_vocabulary[word_tag[1]]
    word_tag = word_tag:sub(1, #word_tag - 2)
    word_tag = data.tag_vocabulary[word_tag]

    word_conf:add(word_tag, word_parts[CoNLL.c_word_tag])

  end) -- end iterate through words
end) -- end iterate through sentences

local to_save = {}
to_save.char_confusion_matrix = char_conf
to_save.word_confusion_matrix = word_conf
to_save.char_classes = test_char_computed_y
to_save.opt = opt

local baseSavefile = opt.evaluation:sub(1, opt.evaluation:len() - ( 12 + paths.extname(opt.evaluation):len())) .. '_classification_using_'..opt.output_number..'_outputs.'

local savefile = baseSavefile..paths.extname(opt.evaluation)
torch.save(savefile, to_save)
print('Classes saved to ' .. savefile)

local stringx = require('pl.stringx')

function log10(n)
   if math.log10 then
      return math.log10(n)
   else
      return math.log(n) / math.log(10)
   end
end

local function ConfusionMatrixToCsv(conf_matrix)
  conf_matrix:updateValids()
  local str = {}
  local nclasses = conf_matrix.nclasses
  local maxCnt = conf_matrix.mat:max()
  local nDigits = math.max(8, 1 + math.ceil(log10(maxCnt)))
  for t = 1,nclasses do
    local pclass = conf_matrix.valids[t] * 100
    pclass = string.format('%2.3f', pclass)
    pclass = stringx.replace(pclass,'.',',')
    for p = 1,nclasses do
      table.insert(str, conf_matrix.mat[t][p]..',')
    end
    if conf_matrix.classes and conf_matrix.classes[1] then
      local class = conf_matrix.classes[t] or ''
      stringx.replace(class, '"', '""')
      table.insert(str, '"' .. pclass .. '%","' .. (conf_matrix.classes[t] or '') .. '"\n')
    else
      table.insert(str, '"' .. pclass .. '%" \n')
    end
  end
  table.insert(str, ' + average row correct: ' .. (conf_matrix.averageValid*100) .. '% \n')
  table.insert(str, ' + average rowUcol correct (VOC measure): ' .. (conf_matrix.averageUnionValid*100) .. '% \n')
  table.insert(str, ' + global correct: ' .. (conf_matrix.totalValid*100) .. '%')
  return table.concat(str)
end

local file = require('pl.file')

file.write(baseSavefile..'_char_matrix.csv', ConfusionMatrixToCsv(char_conf))
file.write(baseSavefile..'_word_matrix.csv', ConfusionMatrixToCsv(word_conf))

print('Confusion matrices exported')
