-- column names
local c_word, c_word_tag, c_chars, c_chars_tags =
  'word', 'word_tag', 'chars', 'chars_tags'

local c_dataset, c_tag_vocabulary, c_word_vocabulary, c_char_vocabulary, c_per_char_tag_vocabulary =
  'dataset', 'tag_vocabulary', 'word_vocabulary', 'char_vocabulary', 'per_char_tag_vocabulary'
-- end column names

local CoNLL = torch.class('CoNLL')

function CoNLL:__init(data, splits, word_column, tag_column, column_separator, with_space)
  local tablex = require('pl.tablex')

  local with_space = with_space or false
  local splits = splits or {0.93, 0.5, 0.2}
  local column_separator = column_separator or '\t'


  if not g2.isTable then data = {data} end

  print('Parsing data files...')
  local ds = CoNLL.encode(CoNLL.readFiles(data, column_separator), word_column, tag_column, with_space)

  local split_sizes = {#ds.dataset, 0, 0}

  for i=1,#splits do
    split_sizes[i] = torch.round(#ds.dataset * splits[i])
  end

  self.word_vocabulary = ds[c_word_vocabulary]
  self.tag_vocabulary = ds[c_tag_vocabulary]

  self.char_vocabulary = ds[c_char_vocabulary]
  self.per_char_tag_vocabulary = ds[c_per_char_tag_vocabulary]

  self.train_data = tablex.sub(ds[c_dataset], 0, split_sizes[1])
  self.test_data = tablex.sub(ds[c_dataset], split_sizes[1] + 1, split_sizes[1] + split_sizes[2])
  self.dev_data = tablex.sub(ds[c_dataset], split_sizes[1] + split_sizes[2] + 1, split_sizes[1] + split_sizes[2] + split_sizes[3])
end

local function charIterator(dataset)
  local sentence_index = 1
  local word_index = 1
  local char_index = 1

  return function()
    while sentence_index <= #dataset do
      sentence = dataset[sentence_index]
      while word_index <= #sentence do
        word = sentence[word_index]
        while char_index <= #word.chars do
          x = word[c_chars][char_index]
          y = word[c_chars_tags][char_index]
          char_index = char_index + 1
          return x, y
        end -- if char in word
        char_index = 1
        word_index = word_index + 1
      end -- if word in sentence
      word_index = 1
      sentence_index = word_index + 1
    end -- if sentence in dataset
  end
end

local function charToTensor(dataset)
  local tablex = require('pl.tablex')
  local input, target = {}, {}

  tablex.foreach(dataset, function(s)
      tablex.foreach(s, function(w)
        for i = 1,#w[c_chars] do
          table.insert(input, w[c_chars][i])
          table.insert(target, w[c_chars_tags][i])
        end
      end)
  end)

  return torch.IntTensor(input), torch.IntTensor(target)
end

function CoNLL:trainCharIterator()
  return charIterator(self.train_data)
end

function CoNLL:testCharIterator()
  return charIterator(self.test_data)
end

function CoNLL:devCharIterator()
  return charIterator(self.dev_data)
end

function CoNLL:trainCharTensors()
  return charToTensor(self.train_data)
end

function CoNLL:testCharTensors()
  return charToTensor(self.test_data)
end

function CoNLL:devCharTensors()
  return charToTensor(self.dev_data)
end

-- Parser section

-- Read a dataset from a file in CoNLL format.
function CoNLL.readFiles(paths, column_separator)
  local file = require('pl.file')

  local str = ''

  for i=1,#paths do
    str = str .. file.read(paths[i])
  end

  return CoNLL.makeTable(str, column_separator)
end

-- Given a data file path, returns a table with the loaded dataset
function CoNLL.makeTable(str, column_separator)
  local tablex, stringx = require('pl.tablex'), require('pl.stringx')

  local makeSentence = function(s)
    return tablex.map(
      function(w) return w:split(column_separator) end,
      s:split('\n')
    )
  end

  local data = str:split('\n\n')

  return tablex.map(
    function(s) return makeSentence(s) end,
    data
  )
end

-- Given a dataset in table format, returns a string in the original CoNLL format.
-- Usefull to compare original with rnn output.
function CoNLL.makeString(tbl)
  local tablex = require('pl.tablex')

  local makeWord = function(p1, p2) return p1 .. '\t' .. p2 end

  local makeSentence = function(w1, w2)
    local w1 = g2.isString(w1) and w1 or tablex.reduce(makeWord, w1)
    local w2 = g2.isString(w2) and w2 or tablex.reduce(makeWord, w2)
    return w1 .. '\n' .. w2
  end

  local makeCorpus = function(s1, s2)
    local s1 = g2.isString(s1) and s1 or tablex.reduce(makeSentence, s1)
    local s2 = g2.isString(s2) and s2 or tablex.reduce(makeSentence, s2)
    local s1 = g2.isString(s1) and s1 or tablex.reduce(makeWord, s1)
    local s2 = g2.isString(s2) and s2 or tablex.reduce(makeWord, s2)
    return s1 .. '\n\n' .. s2
  end

  return tablex.reduce(makeCorpus, tbl)
end

-- Creates a dataset, along with a vocabulary.
-- This function takes 2 argumenst:
--   @word_column: the column index where the word is stored
--   @tag_column: the column index where the tag is stored
function CoNLL.encode(dataset, word_column, tag_column, with_space)
  local tablex = require('pl.tablex')

  -- start generating vocabulary

  local wordVocabulary, tagVocabulary = {}, {} -- for word based encoding
  local charVocabulary, perCharTagVocabulary = {}, {} -- for char based encoding

  -- add space in char vocabulary
  if with_space then
    charVocabulary[' '] = true
    perCharTagVocabulary['S'] = true
  end

  tablex.foreach(dataset, function(sentence) -- start iterate through sentences
    tablex.foreach(sentence, function(wordParts) -- start iterate through words
      local word, tag = wordParts[word_column], wordParts[tag_column]

      if tag == nil then print(wordParts) end

      -- insert word
      wordVocabulary[word] = true
      tagVocabulary[tag] = true

      --insert chars
      for c in utf8.gmatch(word, '.') do
        charVocabulary[c] = true
      end
      perCharTagVocabulary[tag .. '-S'] = true
      if utf8.len(word) > 1 then perCharTagVocabulary[tag..'-I'] = true end
      --if utf8.len(word) > 2  then perCharTagVocabulary[tag..'-I'] = true end

    end) -- end iterate through words
  end) -- end iterate through sentences

  local function sortVocabulary(vocabulary)
    local tmp = {}

    for k in pairs(vocabulary) do table.insert(tmp, k) end
    table.sort(tmp)
    for k, v in ipairs(tmp) do vocabulary[v] = k end
  end -- end of sort function

  sortVocabulary(wordVocabulary)
  sortVocabulary(tagVocabulary)

  sortVocabulary(charVocabulary)
  sortVocabulary(perCharTagVocabulary)

  -- end generating vocabulary

  -- start encoding dataset

  tablex.foreach(dataset, function(sentence) -- start iterate through sentences
    for k, v in pairs(sentence) do -- start iterate through words

      local word, tag = v[word_column], v[tag_column]
      local encodedChars, encodedCharsTags = {}, {}

      -- per word encoding
      local encodedWord, encodedWordTag = wordVocabulary[word], tagVocabulary[tag]

      -- per char encoding
      for c in utf8.gmatch(word, '.') do
        table.insert(encodedChars, charVocabulary[c])
        table.insert(encodedCharsTags, perCharTagVocabulary[tag..'-I'])
      end
      encodedCharsTags[1] = perCharTagVocabulary[tag..'-S']
      --if utf8.len(word) > 1 then encodedCharsTags[utf8.len(word)] = perCharTagVocabulary[tag..'-E'] end
      if with_space then
        table.insert(encodedChars, charVocabulary[' '])
        table.insert(encodedCharsTags, perCharTagVocabulary['S'])
      end

      -- replace original sentence word with an encoded one
      sentence[k] = {
        [c_word] = encodedWord, [c_word_tag] = encodedWordTag,
        [c_chars] = encodedChars, [c_chars_tags] = encodedCharsTags
      }
    end -- end iterate through words
  end) -- end iterate through sentences

  -- end encoding dataset

  return {
    [c_dataset] = dataset,
    [c_tag_vocabulary] = tagVocabulary,
    [c_word_vocabulary] = wordVocabulary,
    [c_char_vocabulary] = charVocabulary,
    [c_per_char_tag_vocabulary] = perCharTagVocabulary
  }
end

function CoNLL.decode(dataset, wordVocabulary, tagVocabulary)
  -- body...
end

-- End of parser section
