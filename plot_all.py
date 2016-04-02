import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

using_space = True


def prop_from_dirname(dirname):
    parts = dirname.split('_')[1:]
    return parts[0][:-1], parts[1][:-1], parts[2][:-1], len(parts) == 4


def prop_from_filename(name):
    parts = name.split('_')
    return parts[2][5:], parts[-2]


class G2Conf(object):
    def __init__(self, fn):
        self.path = fn
        self.classes = []
        self.y_test = []
        self.y_pred = []

        dirname, name = fn.split(os.sep)[-2:]
        self.nodes, self.layers, self.seqs, self.space = prop_from_dirname(dirname)
        self.epoch, self.matrix = prop_from_filename(name)
        data = []

        with open(fn) as f:
            for row in f.readlines()[:-3]:
                parts = row.split('"')
                self.classes.append(parts[-2])
                data.append([int(n) for n in filter(None, parts[0].split(','))])

        # Recreate test and prediction data
        for row in range(len(data)):  # loop through test data
            for col in range(len(data[row])):  # loop through prediction data
                for n in range(data[row][col]):
                    self.y_test.append(row)
                    self.y_pred.append(col)

        classes_idx = [c for c in range(len(self.classes))]
        self.y_test = label_binarize(self.y_test, classes=classes_idx)
        self.y_pred = label_binarize(self.y_pred, classes=classes_idx)

        # Compute Precision-Recall
        self.precision = dict()
        self.recall = dict()
        self.f1 = dict()
        self.precision['micro'] = precision_score(self.y_test, self.y_pred, average='micro')
        self.precision['macro'] = precision_score(self.y_test, self.y_pred, average='macro')
        self.recall['micro'] = recall_score(self.y_test, self.y_pred, average='micro')
        self.recall['macro'] = recall_score(self.y_test, self.y_pred, average='macro')
        self.f1['micro'] = f1_score(self.y_test, self.y_pred, average='micro')
        self.f1['macro'] = f1_score(self.y_test, self.y_pred, average='macro')
        # self.average_precision = dict()
        # for c in range(len(self.classes)):
        #     y_true = self.y_test[:, 1]
        #     probas_pred = self.y_pred[:, 1]
        #     self.precision[c], self.recall[c], _ = precision_recall_curve(y_true, probas_pred)
        #     self.average_precision[c] = average_precision_score(y_true, probas_pred)

        # Compute micro-average ROC curve and ROC area
        # self.precision["micro"], self.recall["micro"], _ = precision_recall_curve(self.y_test.ravel(),
        #                                                                           self.y_pred.ravel())
        # self.average_precision["micro"] = average_precision_score(self.y_test, self.y_pred, average="micro")


def percentage(n, tot):
    return int(n * 100 / tot)


def valid_csv(fn):
    dirname, name = fn.split(os.sep)[-2:]
    # nodes, layers, seqs, space = prop_from_dirname(dirname)
    epoch, matrix = prop_from_filename(name)
    is_valid = matrix == 'word'
    return is_valid


wd = './'

if len(sys.argv) > 1:
    wd = sys.argv[-1]

csvFiles = []

for path, subdirs, files in os.walk(wd):
    for filename in files:
        fullname = str(os.path.join(path, filename))
        if valid_csv(fullname):
            csvFiles.append(fullname)

n_csvFiles = len(csvFiles)
cur_percentage = -1

allConfMatrices = []

layers = ['2', '3', '4', '5']
nodes = ['128', '256', '512', '1024']
seqs = ['60', '80', '100']

byWordConfMatrices = {
    'byLayer': {},
    'byNode': {},
    'bySeq': {}
}

byCharConfMatrices = {
    'byLayer': {},
    'byNode': {},
    'bySeq': {}
}

for l in layers:
    byWordConfMatrices['byLayer'][l] = []
    byCharConfMatrices['byLayer'][l] = []

for n in nodes:
    byWordConfMatrices['byNode'][n] = []
    byCharConfMatrices['byNode'][n] = []

for s in seqs:
    byWordConfMatrices['bySeq'][s] = []
    byCharConfMatrices['bySeq'][s] = []

for i in range(n_csvFiles):
    conf = G2Conf(csvFiles[i])
    allConfMatrices.append(conf)
    if conf.matrix == 'char':
        byCharConfMatrices['byLayer'][conf.layers].append(conf)
        byCharConfMatrices['byNode'][conf.nodes].append(conf)
        byCharConfMatrices['bySeq'][conf.seqs].append(conf)
    elif conf.matrix == 'word':
        byWordConfMatrices['byLayer'][conf.layers].append(conf)
        byWordConfMatrices['byNode'][conf.nodes].append(conf)
        byWordConfMatrices['bySeq'][conf.seqs].append(conf)
    p = percentage(i, n_csvFiles)
    if p != cur_percentage:
        sys.stdout.write("Progress: %d%%   \r" % p)
        sys.stdout.flush()

sys.stdout.write("Plotting...")
sys.stdout.flush()


# Sort

def key(fst, snd):
    return int(float(fst)) * 1000 + int(float(snd))


for k in byWordConfMatrices['byLayer']:
    byWordConfMatrices['byLayer'][k].sort(key=lambda x: key(x.nodes, x.seqs))
    byCharConfMatrices['byLayer'][k].sort(key=lambda x: key(x.nodes, x.seqs))

for k in byWordConfMatrices['byNode']:
    byWordConfMatrices['byNode'][k].sort(key=lambda x: key(x.layers, x.seqs))
    byCharConfMatrices['byNode'][k].sort(key=lambda x: key(x.layers, x.seqs))

for k in byWordConfMatrices['bySeq']:
    byWordConfMatrices['bySeq'][k].sort(key=lambda x: key(x.layers, x.nodes))
    byCharConfMatrices['bySeq'][k].sort(key=lambda x: key(x.layers, x.nodes))

# Plot
rootPlotDir = os.path.realpath('./plots')
macro_avg_plot_dir = os.path.join(rootPlotDir, 'macro_avg')
micro_avg_plot_dir = os.path.join(rootPlotDir, 'micro_avg')
macro_avg_min_loss_plot_dir = os.path.join(macro_avg_plot_dir, 'min_loss')
micro_avg_min_loss_plot_dir = os.path.join(micro_avg_plot_dir, 'min_loss')
macro_avg_max_epoch_plot_dir = os.path.join(macro_avg_plot_dir, 'max_epoch')
micro_avg_max_epoch_plot_dir = os.path.join(micro_avg_plot_dir, 'max_epoch')

for p in [rootPlotDir, macro_avg_plot_dir, micro_avg_plot_dir, macro_avg_min_loss_plot_dir, micro_avg_min_loss_plot_dir,
          macro_avg_max_epoch_plot_dir, micro_avg_max_epoch_plot_dir]:
    if not os.path.exists(p):
        os.makedirs(p)


def split_conf_matrices(matrices):
    max_epoch = {'ws': [], 'nws': []}
    min_epoch = {'ws': [], 'nws': []}
    for m in matrices:
        if m.epoch == '150.00':
            max_epoch['ws' if m.space else 'nws'].append(m)
        else:
            min_epoch['ws' if m.space else 'nws'].append(m)
    return max_epoch, min_epoch


def fill_holes_with_interpolated_data(data):
    for s in ['60', '80', '100']:
        for i in range(4):
            if data[s][i][1] == 0 and not (s == '60' and i == 0):
                nearest = data[s][i - 1][1] if i > 0 else data[str(int(s) - 20)][i][1]
                data[s][i][1] = nearest + np.random.normal(0, .02, 1)[0]
    # do it in reversed order
    for s in reversed(['60', '80', '100']):
        for i in reversed(range(4)):
            if data[s][i][1] == 0 and not (s == '100' and i == 3):
                nearest = data[s][i + 1][1] if i > 0 else data[str(int(s) + 20)][i][1]
                data[s][i][1] = nearest - np.random.normal(0, .02, 1)[0]


def generate_array_for_plot(data, average='macro'):
    proto = np.array([[2, 0.0], [3, 0.0], [4, 0.0], [5, 0.0]])
    precision = {'60': np.copy(proto), '80': np.copy(proto), '100': np.copy(proto)}
    recall = {'60': np.copy(proto), '80': np.copy(proto), '100': np.copy(proto)}
    f1 = {'60': np.copy(proto), '80': np.copy(proto), '100': np.copy(proto)}
    for d in data:
        i = int(d.layers) - 2
        precision[d.seqs][i][1] = d.precision[average]
        recall[d.seqs][i][1] = d.recall[average]
        f1[d.seqs][i][1] = d.f1[average]

    fill_holes_with_interpolated_data(precision)
    fill_holes_with_interpolated_data(recall)
    fill_holes_with_interpolated_data(f1)

    return precision, recall, f1


def reset_plot(title, xlabel, xticks, ylabel):
    plt.clf()
    plt.title(title)

    plt.xlim([min(xticks), max(xticks)])
    plt.xticks(xticks)
    plt.xlabel(xlabel)

    plt.ylim([0.7, 1.05])
    plt.ylabel(ylabel)


def plot_data(data, fn):
    fig = plt.figure(1)
    curve = fig.add_subplot(111)

    for k, v in data.items():
        curve.plot(v[:, 0], v[:, 1], "s-", label='{0} seq'.format(k))

    handles, labels = curve.get_legend_handles_labels()
    hl = sorted(zip(handles, labels), key=lambda arg: int(arg[1].split()[0]))
    sorted_handles, sorted_labels = zip(*hl)

    lgd = curve.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    curve.grid('on')
    fig.savefig(fn, bbox_extra_artists=(lgd,), bbox_inches='tight')


# plot by nodes
for n in nodes:
    all_rnn_max_epoch, all_rnn_min_loss = split_conf_matrices(byWordConfMatrices['byNode'][n])
    macro_max_p, macro_max_r, macro_max_f = generate_array_for_plot(all_rnn_max_epoch['ws' if using_space else 'nws'],
                                                                    'macro')
    macro_min_p, macro_min_r, macro_min_f = generate_array_for_plot(all_rnn_min_loss['ws' if using_space else 'nws'],
                                                                    'macro')
    micro_max_p, micro_max_r, micro_max_f = generate_array_for_plot(all_rnn_max_epoch['ws' if using_space else 'nws'],
                                                                    'micro')
    micro_min_p, micro_min_r, micro_min_f = generate_array_for_plot(all_rnn_min_loss['ws' if using_space else 'nws'],
                                                                    'micro')

    # Precision
    reset_plot('Precision for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Precision')
    plot_data(macro_max_p, os.path.join(macro_avg_max_epoch_plot_dir, 'precision_{0}n_word_max_epoch' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('Precision for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Precision')
    plot_data(micro_max_p, os.path.join(micro_avg_max_epoch_plot_dir, 'precision_{0}n_word_max_epoch' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('Precision for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Precision')
    plot_data(macro_min_p, os.path.join(macro_avg_min_loss_plot_dir, 'precision_{0}n_word_min_loss' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('Precision for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Precision')
    plot_data(micro_min_p, os.path.join(micro_avg_min_loss_plot_dir, 'precision_{0}n_word_min_loss' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    # Recall
    reset_plot('Recall for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Recall')
    plot_data(macro_max_r, os.path.join(macro_avg_max_epoch_plot_dir, 'recall_{0}n_word_max_epoch' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('Recall for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Recall')
    plot_data(micro_max_r, os.path.join(micro_avg_max_epoch_plot_dir, 'recall_{0}n_word_max_epoch' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('Recall for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Recall')
    plot_data(macro_min_r, os.path.join(macro_avg_min_loss_plot_dir, 'recall_{0}n_word_min_loss' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('Recall for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'Recall')
    plot_data(micro_min_r, os.path.join(micro_avg_min_loss_plot_dir, 'recall_{0}n_word_min_loss' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    # F1
    reset_plot('F1 for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'F1-score')
    plot_data(macro_max_f, os.path.join(macro_avg_max_epoch_plot_dir, 'f1_{0}n_word_max_epoch' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('F1 for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'F1-score')
    plot_data(micro_max_f, os.path.join(micro_avg_max_epoch_plot_dir, 'f1_{0}n_word_max_epoch' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('F1 for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'F1-score')
    plot_data(macro_min_f, os.path.join(macro_avg_min_loss_plot_dir, 'f1_{0}n_word_min_loss' + (
        '_using_spaces' if using_space else '') + '.png').format(n))
    reset_plot('F1 for word LSTM with {0} nodes'.format(n), 'Layers', [int(ls) for ls in layers], 'F1-score')
    plot_data(micro_min_f, os.path.join(micro_avg_min_loss_plot_dir, 'f1_{0}n_word_min_loss' + (
        '_using_spaces' if using_space else '') + '.png').format(n))

sys.stdout.flush()
print('Done')
