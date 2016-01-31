require 'os'
require 'fs'
require 'nn'
require 'lfs'
require 'optim'
require 'torch'
require 'nngraph'
utf8 = require 'lua-utf8'
-- luarocks luautf8

g2 = {}
-- torch.include('g2', 'utils/utils.lua')
require 'utils/utils.lua'

g2.TORCH_DIR = os.getenv('TORCH_DATA_PATH')
  or './torch_data'
g2.mkdir(g2.TORCH_DIR)

--[[ directory structure ]]--
g2.DATA_DIR = os.getenv('DEEP_DATA_PATH')
   or paths.concat(g2.TORCH_DIR, 'data')
g2.mkdir(g2.DATA_DIR)

g2.SAVE_DIR = os.getenv('DEEP_SAVE_PATH')
   or paths.concat(g2.TORCH_DIR, 'save')
g2.mkdir(g2.SAVE_DIR)

g2.LOG_DIR = os.getenv('DEEP_LOG_PATH')
   or paths.concat(g2.TORCH_DIR, 'log')
g2.mkdir(g2.LOG_DIR)

g2.UNIT_DIR = os.getenv('DEEP_UNIT_PATH')
   or paths.concat(g2.TORCH_DIR, 'unit')
g2.mkdir(g2.UNIT_DIR)

-- Modules
require './modules/LSTM.lua'
require './modules/OneHot.lua'

-- CoNLL data sets
require './data/datasets.lua'
require './data/loader.lua'
require './data/CoNLL.lua'
