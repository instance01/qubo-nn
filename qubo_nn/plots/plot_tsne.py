import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


problems = ["NP", "MC", "MVC", "SP", "M2SAT", "SPP", "GC", "QA", "QK", "M3SAT", "TSP"]  # noqa


def plot(small_size=False):
    for i in [10, 20, 30, 50, 70, 100, 200, 500, 1000]:
        with open('tsne_data%d.pickle' % i, 'rb') as f:
            (Y, y) = pickle.load(f)

        vis_x = Y[:, 0]
        vis_y = Y[:, 1]
        plt.scatter(
            vis_x,
            vis_y,
            c=y,
            # cmap=plt.cm.get_cmap("jet", 11),
            cmap=plt.cm.get_cmap("Spectral", 11),
            marker='.',
            s=6 if small_size else 9
        )
        plt.clim(-0.5, 10.5)
        # plt.title("Perplexity %d" % i)
        cbar = plt.colorbar(ticks=list(range(11)))
        cbar.ax.set_yticklabels(problems)
        plt.tight_layout()
        fname = "tsne%d" % i
        if small_size:
            fname += '_small'
        plt.savefig(fname + ".png")
        plt.savefig(fname + ".pdf")
        plt.show()


# plot(True)
plot(False)
