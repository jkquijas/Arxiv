from sklearn.manifold import TSNE
import numpy as np
import time
from numpy import random
import matplotlib

start = time.time()
data_path = 'Data/Training/training_data.txt'
labels_path = 'Data/Training/training_labels.txt'

X = np.loadtxt(data_path, delimiter=',')
with open(labels_path) as f:
    y = f.read().splitlines()

# Create subset and reduce to first 50 dimensions
indices = range(X.shape[0])
random.shuffle(indices)
n_train_samples = 5000

# Plotting function
matplotlib.rc('font', **{'family' : 'sans-serif',
                         'weight' : 'bold',
                         'size'   : 18})
matplotlib.rc('text', **{'usetex' : True})

def plot_mnist(X, y, X_embedded, name, min_dist=10.0):
    fig = figure(figsize=(10, 10))
    ax = axes(frameon=False)
    title("\\textbf{Keyword dataset} -- Two-dimensional "
          "bigram plot with %s" % name)
    setp(ax, xticks=(), yticks=())
    subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=y, marker="x")

    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = np.array([[15., 15.]])
        indices = range(X_embedded.shape[0])
        random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            """imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap=cm.gray_r), X_embedded[i])
            ax.add_artist(imagebox)"""

X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X[indices[:n_train_samples]])
plot_mnist(X[indices[:n_train_samples]], y, X_embedded,
           "t-SNE", min_dist=20.0)
