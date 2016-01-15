local CoNLLoader = torch.class('CoNLLoader')

local function make_batches(x, y, batch_size, seq_length)
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

  return x_b, y_b
end

function CoNLLoader:__init(data, batch_size, seq_length)
  local train_x, train_y = data:trainCharTensors()
  local test_x, test_y = data:testCharTensors()
  local dev_x, dev_y = data:devCharTensors()

  train_x, train_y = make_batches(train_x, train_y, batch_size, seq_length)
  test_x, test_y = make_batches(test_x, test_y, batch_size, seq_length)
  dev_x, dev_y = make_batches(dev_x, dev_y, batch_size, seq_length)

  self.ntrain = #train_x
  self.ntest = #test_x
  self.nval = #dev_x

  self.split_sizes = {self.ntrain, self.nval, self.ntest}
  self.batch_ix = {0,0,0}

  self.train_x, self.train_y = train_x, train_y
  self.test_x, self.test_y = test_x, test_y
  self.val_x, self.val_y = dev_x, dev_y

end

function CoNLLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function CoNLLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end

    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]

    if split_index == 2 then return self.val_x[ix], self.val_y[ix] end
    if split_index == 3 then return self.test_x[ix], self.test_y[ix] end

    return self.train_x[ix], self.train_y[ix]
end
