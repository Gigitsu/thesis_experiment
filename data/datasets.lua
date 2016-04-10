datasets = datasets or {}

-- Download and decompress CoNLL data files if does not exists.
local conllPath = paths.concat(g2.DATA_DIR, 'CoNLL')
local conllWordColumn, conllTagColumn = 2, 6

local downloadCoNLLDataset = function (url, path, basename)
  g2.checkDownloadAndDecompress(path, url)

  local trainPath = basename .. 'train.txt'
  local trialPath = basename .. 'trial.txt'
  local develPath = basename .. 'development.txt'

  if(not paths.filep(trainPath) or not paths.filep(trialPath) or not paths.filep(develPath)) then
    print("something went wrong, the data files does not exists")
  end

  return trainPath, trialPath, develPath
end

-- All the following methods returns, in this order, train, trial and develop file paths

-- English dataset
function datasets.English2000(with_space)
  local path = paths.concat(conllPath, '2000', 'English')
  local trainUrl = "http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz"
  local testUrl = "http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz"

  g2.checkDownloadAndDecompress(path, trainUrl)
  g2.checkDownloadAndDecompress(path, testUrl)

  return CoNLL(
    {paths.concat(path, 'train.txt'), paths.concat(path, 'test.txt')},
    {0.9, 0.1},
    1, 2,
    ' ',
    with_space
  )
end

-- German dataset
function datasets.German2009(with_space)
  local path = paths.concat(conllPath, '2009', 'German')
  local url = 'https://ufal.mff.cuni.cz/conll2009-st/data/CoNLL2009-ST-German-traindevB.zip'
  local basename = paths.concat(path, 'CoNLL2009-ST-German-traindev', 'CoNLL2009-ST-German-')

  return CoNLL(
    {downloadCoNLLDataset(url, path, basename)},
    {0.93754, 0.01041, 0.05205},
    conllWordColumn, conllTagColumn,
    '\t',
    with_space
  )
end

-- Spanish dataset
function datasets.Spanish2009(with_space)
  local path = paths.concat(conllPath, '2009', 'Spanish')
  local url = 'https://ufal.mff.cuni.cz/conll2009-st/data/CoNLL2009-ST-Spanish-traindevB.zip'
  local basename = paths.concat(path, 'CoNLL2009-ST-Spanish-traindev', 'datasets', 'CoNLL2009-ST-Spanish-')

  return CoNLL(
    {downloadCoNLLDataset(url, path, basename)},
    {0.89366, 0.00311, 0.10321},
    conllWordColumn, conllTagColumn,
    '\t',
    with_space
  )
end

-- Catalan dataset
function datasets.Catalan2009(with_space)
  local path = paths.concat(conllPath, '2009', 'Catalan')
  local url = 'https://ufal.mff.cuni.cz/conll2009-st/data/CoNLL2009-ST-Catalan-traindevC.zip'
  local basename = paths.concat(path, 'CoNLL2009-ST-Catalan-traindev', 'datasets', 'CoNLL2009-ST-Catalan-')

  return CoNLL(
    {downloadCoNLLDataset(url, path, basename)},
    {0.88152, 0.00333, 0.11513},
    conllWordColumn, conllTagColumn,
    '\t',
    with_space
  )
end

function datasets.Evalita(with_space)
  local base = paths.concat(g2.DATA_DIR, 'CoNLL', 'Evalita')

  return CoNLL(
    {paths.concat(base, 'train'), paths.concat(base, 'devel')},
    {0.9, 0.1},
    1, 2,
    '\t',
    with_space
  )
end

function datasets.TurinUniversityTreebank(with_space)
  local path = paths.concat(g2.DATA_DIR, 'CoNLL', 'Tut')
  local urls = {
    "http://www.di.unito.it/~tutreeb/corpora/tutINconll/NEWS-22nov2010.conl.zip",
    "http://www.di.unito.it/~tutreeb/corpora/tutINconll/VEDCH-22nov2010.conl.zip",
    "http://www.di.unito.it/~tutreeb/corpora/tutINconll/CODICECIVILE-22nov2010.conl.zip",
    "http://www.di.unito.it/~tutreeb/corpora/tutINconll/EUDIR-22nov2010.conl.zip",
    "http://www.di.unito.it/~tutreeb/corpora/tutINconll/WIKI-22nov2010.conl.zip",
  }

  for i, u in pairs(urls) do
    g2.checkDownloadAndDecompress(path, u)
  end

  return CoNLL(
    {
      paths.concat(path, 'CODICECIVILE-22nov2010.conl'),
      paths.concat(path, 'EUDIR-22nov2010.conl'),
      paths.concat(path, 'VEDCH-22nov2010.conl'),
      paths.concat(path, 'NEWS-22nov2010.conl'),
      paths.concat(path, 'WIKI-22nov2010.conl'),
    },
    {0.93, 0.05, 0.02},
    conllWordColumn, conllTagColumn,
    '\t',
    with_space
  )
end
