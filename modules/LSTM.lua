require 'nn'
require 'nngraph'

-- see https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf

-- nn.Identity()() creates a module that returns whatever is input to it as output without transformation
-- nn.Linear(input_size,outputSize)(whatTranform) applies a linear transformation to the incoming data, i.e. y = Ax + b

local LSTM, parent = torch.class('LSTM','nn.gModule')

function LSTM:__init(input_size, output_size, input_module, layer_size, layers_number, dropout)
  -- During training, dropout masks parts of the input using binary samples
  -- from a bernoulli distribution. Each input element has a probability of p
  -- of being dropped, i.e having its commensurate output element be zero.
  dropout = dropout or 0

  -- there will be 2*layers_number+1 inputs
  local inputs, outputs = {}, {}

  table.insert(inputs, nn.Identity()()) -- x, the input layer
  for layer = 1,layers_number do -- for every hidden layers
    -- I have the two inputs
    table.insert(inputs, nn.Identity()()) -- prev_c[layer]
    table.insert(inputs, nn.Identity()()) -- prev_h[layer]
  end

  local x, layer_input_size
  for layer = 1,layers_number do
    -- c,h from previos timesteps
    local prev_c = inputs[layer*2] -- the cells states
    local prev_h = inputs[layer*2+1] -- the hidden nodes states

    -- the input to this layer
    if layer == 1 then
      x = input_module(input_size)(inputs[1])
      layer_input_size = input_size
    else
      x = outputs[(layer-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      layer_input_size = layer_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(layer_input_size, 4 * layer_size)(x):annotate{name='i2h_'..layer}
    local h2h = nn.Linear(layer_size, 4 * layer_size)(prev_h):annotate{name='i2h_'..layer}

    local reshaped = nn.Reshape(4, layer_size)(nn.CAddTable()({i2h, h2h}))
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

    -- decode the gates
    local input_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local output_gate = nn.Sigmoid()(n3)

    -- decode the write inputs
    local in_transform  = nn.Tanh()(n4)

    -- perform the LSTM update
    local next_c = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({input_gate, in_transform})
    })

    -- gated cells form the output
    local next_h = nn.CMulTable()({output_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(layer_size, output_size)(top_h)
  table.insert(outputs, nn.LogSoftMax()(proj))

  parent.__init(self, inputs, outputs)
end
