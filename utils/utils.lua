function string.tomodule(modulename,splitter)
  splitter = splitter or '[.]'
  assert(type(modulename) == 'string')
  local modula = _G
  for i, name in ipairs(_.split(modulename,splitter)) do
    modula = modula[name] or require(modula)
  end
  return modula
end

--[[ From https://github.com/rosejn/lua-util: ]]--
-- Boolean predicate to determine if a path points to a valid file or directory.
function g2.isFile(path)
  return paths.filep(path) or paths.dirp(path)
end

-- Check that a data directory exists, and create it if not.
function g2.mkdir(dir)
  if not paths.dirp(dir) then
    paths.mkdir(dir)
  end
end

-- Download the file at location url.
function g2.download(url)
  local protocol, scpurl, filename = url:match('(.-)://(.*)/(.-)$')
  if protocol == 'scp' then
    os.execute(string.format('%s %s %s', 'scp', scpurl .. '/' .. filename, filename))
  else
    os.execute('wget ' .. url)
  end
end

-- Temporarily changes the current working directory to call fn,
-- returning its result.
function g2.doWithCwd(path, fn)
  local cur_dir = fs.cwd()
  fs.chdir(path)
  local res = fn()
  fs.chdir(cur_dir)
  return res
end


-- If file doesn't exists at path, downloads it from url into path
local function check(dirPath, url)
  g2.mkdir(dirPath)
  local path = paths.concat(dirPath, paths.basename(url))
  local exists = paths.filep(path)

  return exists, path
end

function g2.checkAndDownload(dirPath, url)
  local exists, path = check(dirPath, url)
  if not exists then
    g2.doWithCwd(
      dirPath,
      function() g2.download(url) end
    )
  end
  return path
end

function g2.checkDownloadAndDecompress(dirPath, url)
  local exists, path = check(dirPath, url)
  local data = g2.checkAndDownload(dirPath, url)
  if not exists then
    g2.decompress(data, dirPath)
  end

  return path
end

-- Decompress a .tar, .tgz or .tar.gz file.
function g2.untar(srcPath, dstPath)
  local dstPath = dstPath or '.'
  paths.mkdir(dstPath)
  if srcPath:match("%.tar$") then
    os.execute('tar -xvf ' .. srcPath .. ' -C ' .. dstPath)
  else
    os.execute('tar -xvzf ' .. srcPath .. ' -C ' .. dstPath)
  end
end

-- Decompress a .zip file
function g2.unzip(srcPath, dstPath)
  local dstPath = dstPath or '.'
  paths.mkdir(dstPath)
  os.execute('unzip ' .. srcPath .. ' -d ' .. dstPath)
end

-- Decompress a .gz file
function g2.gunzip(srcPath, dstPath)
  --assert(not dstPath, "destination path not supported with gunzip")
  os.execute('gunzip -c ' .. srcPath .. ' > ' .. paths.concat(dstPath, paths.basename(srcPath, 'gz')) )
end

-- Decompress a file
function g2.decompress(srcPath, dstPath)
  if string.find(srcPath, ".zip") then
    g2.unzip(srcPath, dstPath)
  elseif string.find(srcPath, ".tar") or string.find(srcPath, ".tgz") then
    g2.untar(srcPath, dstPath)
  elseif string.find(srcPath, ".gz") or string.find(srcPath, ".gzip") then
    g2.gunzip(srcPath, dstPath)
  else
    print("Don't know how to decompress file: ", srcPath)
  end
end
--[[ End From ]]--

function g2.isString(args)
  return torch.type(args) == 'string'
end

function g2.isTable(args)
  return torch.type(args) == 'table'
end

function g2.getExt(file)
  local rev = string.reverse(file)
  local len = rev:find("%.")
  return string.reverse(rev:sub(1,len))
end

function g2.files(dir, ext)
  local iter, dirObj = lfs.dir(dir)
  return function() -- iterator function
    local item = iter(dirObj)
    while item do
      if lfs.attributes(path.join(dir, item)).mode == "file" then -- is file
        if ext == nil or ext == lfs.getExt(item) then
          return item
        end
      end
      item = iter(dirObj)
    end
    return nil -- no more items
  end -- end of iterator
end

function g2.mostRecentFile(dir)
  local mrFile, mrfMod = nil, 0
  for f in g2.files(dir) do
    local file = path.join(dir,f)
    local mod = lfs.attributes(file).modification
    if mod > mrfMod then
      mrfMod = mod
      mrFile = f
    end
  end
  return mrFile
end

-------------------
-- from karpathy --
-------------------

-- takes a list of tensors and returns a list of cloned tensors
function clone_list(tensor_list, zero_too)
  local out = {}
  for k,v in pairs(tensor_list) do
    out[k] = v:clone()
    if zero_too then out[k]:zero() end
  end
  return out
end

model_utils = {}

function model_utils.batch_iterator(x, y, batch_size, seq_length)
  local x = x
  local y = y

  -- cut off the end so that it divides evenly
  local len = x:size(1)
  if len % (batch_size * seq_length) ~= 0 then
      print('cutting off end of data so that the batches/sequences divide evenly')
      local s_e = batch_size * seq_length * math.floor(len / (batch_size * seq_length))
      x = x:sub(1, s_e)
      y = y:sub(1, s_e)
  end

  local x_b = x:view(batch_size, -1):split(seq_length, 2)
  local y_b = y:view(batch_size, -1):split(seq_length, 2)

  local i = 0
  local n = #x_b

  return function()
    i = i + 1
    if i <= n then return x_b[1], y_b[1] end
  end
end

function model_utils.combine_all_parameters(...)
  --[[ like module:getParameters, but operates on many modules ]]--

  -- get parameters
  local networks = {...}
  local parameters = {}
  local gradParameters = {}
  for i = 1, #networks do
    local net_params, net_grads = networks[i]:parameters()

    if net_params then
      for _, p in pairs(net_params) do
        parameters[#parameters + 1] = p
      end
      for _, g in pairs(net_grads) do
        gradParameters[#gradParameters + 1] = g
      end
    end
  end

  local function storageInSet(set, storage)
    local storageAndOffset = set[torch.pointer(storage)]
    if storageAndOffset == nil then
      return nil
    end
    local _, offset = unpack(storageAndOffset)
    return offset
  end

  -- this function flattens arbitrary lists of parameters,
  -- even complex shared ones
  local function flatten(parameters)
    if not parameters or #parameters == 0 then
      return torch.Tensor()
    end
    local Tensor = parameters[1].new

    local storages = {}
    local nParameters = 0
    for k = 1,#parameters do
      local storage = parameters[k]:storage()
      if not storageInSet(storages, storage) then
        storages[torch.pointer(storage)] = {storage, nParameters}
        nParameters = nParameters + storage:size()
      end
    end

    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()

    for k = 1,#parameters do
      local storageOffset = storageInSet(storages, parameters[k]:storage())
      parameters[k]:set(flatStorage,
        storageOffset + parameters[k]:storageOffset(),
        parameters[k]:size(),
        parameters[k]:stride())
      parameters[k]:zero()
    end

    local maskParameters=  flatParameters:float():clone()
    local cumSumOfHoles = flatParameters:float():cumsum(1)
    local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
    local flatUsedParameters = Tensor(nUsedParameters)
    local flatUsedStorage = flatUsedParameters:storage()

    for k = 1,#parameters do
      local offset = cumSumOfHoles[parameters[k]:storageOffset()]
      parameters[k]:set(flatUsedStorage,
        parameters[k]:storageOffset() - offset,
        parameters[k]:size(),
        parameters[k]:stride())
    end

    for _, storageAndOffset in pairs(storages) do
      local k, v = unpack(storageAndOffset)
      flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
    end

    if cumSumOfHoles:sum() == 0 then
      flatUsedParameters:copy(flatParameters)
    else
      local counter = 0
      for k = 1,flatParameters:nElement() do
        if maskParameters[k] == 0 then
          counter = counter + 1
          flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
        end
      end
      assert (counter == nUsedParameters)
    end
    return flatUsedParameters
  end

  -- flatten parameters and gradients
  local flatParameters = flatten(parameters)
  local flatGradParameters = flatten(gradParameters)

  -- return new flat vector that contains all discrete parameters
  return flatParameters, flatGradParameters
end

function model_utils.clone_many_times(net, T)
  local clones = {}

  local params, gradParams
  if net.parameters then
    params, gradParams = net:parameters()
    if params == nil then
      params = {}
    end
  end

  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
        cloneParamsNoGrad = clone:parametersNoGrad()
        for i =1,#paramsNoGrad do
          cloneParamsNoGrad[i]:set(paramsNoGrad[i])
        end
      end
    end

    clones[t] = clone
    collectgarbage()
  end

  mem:close()
  return clones
end

function merge(t1, t2)
  for k,v in pairs(t2) do t1[k] = v end
end
