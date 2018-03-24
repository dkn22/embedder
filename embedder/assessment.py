import matplotlib.pyplot as plt
from sklearn import manifold

def visualize(emb, labels=None):
    '''
    Performs t-SNE dimensionality reduction on the embedding
    matrix and plots the data in 2D.

    :param emb: embedding matrix for one variable
    :param labels: labels for embeddings
    :return: Axes object
    '''

    tsne = manifold.TSNE(init='pca', random_state=0)
    Y = tsne.fit_transform(emb)
    ax = plt.scatter(-Y[:, 0], -Y[:, 1])

    if labels is not None:
        assert len(labels) == Y.shape[0]

        for i, txt in enumerate(labels):
            plt.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext=(-20, 8),
                        textcoords='offset points')

    return ax