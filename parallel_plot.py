import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np



# Code from: https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
# See https://www.data-to-viz.com/graph/parallel.html for additional configuration options and explanation
def parallel_plot(input_data, plot_title, reverse_axes = [], shown_feat = [], plot_file_name = "", show_plot=True, visualization_variance=0.0):
    '''
    Create a parallel plot from a set of feature vectors with a target classification of clustering

    Parameters
    ----------
    input_data: Dictionary-like object, with the following attributes.
        data: ndarray of shape (num_vectors, vector_len): The data matrix. 
        target: ndarray of shape (num_vectors,)         : The classification target (Assignation of each row to classes or clusters)
        feature_names: list                             : The names of the dataset columns (features).
        target_names: list                              : The names of target classes or clusters.
    plot_title: str
        The title on the plot
    reverse_axes: list of feature indices (Default [])
        A list of axes that will be reversed (this may help to avoid some crossing lines)
    shown_feat: list of feature indices (default [])
        A list of all feature indices that will appear in the plot. If empty list, all the features will appear.
    plot_file_name: str (default "")
        A file name where to save the plot. If "", the plot is not saved to disk.
    show_plot: bool (default True)
        Whether to show the plot or not.
    visualization_variange: float
        Variance of noise added to the measures. This is useful when measures are integers with low dynamic range
    '''
    
    ynames       = input_data['feature_names']      # Names of the variables
    ys           = input_data['data'].astype(float) # Numeric data
    target_names = input_data['target_names']       # Names of the target classes or clusters
    target       = input_data['target']             # Assignation of each row to classes or clusters


    if len(shown_feat) != 0:
        ynames = [ynames[ii] for ii in shown_feat]
    
    ymins  = ys.min(axis=0)
    ymaxs  = ys.max(axis=0)
    dys    = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05

    if len(reverse_axes) != 0:
        for ax_ind in reverse_axes:
            ymaxs[ax_ind], ymins[ax_ind] = ymins[ax_ind], ymaxs[ax_ind]  # reverse axis to have less crossings
        dys = ymaxs - ymins


    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    zs = zs + np.random.normal(0.0, visualization_variance, zs.shape)

    
    fig, host = plt.subplots(figsize=(2*len(ynames), 10))   # ??????? 4 or len()?????


    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title(plot_title, fontsize=18, pad=12)

    colors = plt.cm.Set2.colors
    legend_handles = [None for _ in target_names]
    for jj in range(ys.shape[0]):
        # If this feature is not on the list of features to show, skip it
        if len(shown_feat) != 0 and jj not in shown_feat:
            continue
        
        # to just draw straight lines between the axes:
        # host.plot(range(ys.shape[1]), zs[jj,:], c=colors[(target[jj] - 1) % len(colors) ])

        # create bezier curves
        # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
        # at one third towards the next axis; the first and last axis have one less control vertex
        # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
        # y-coordinate: repeat every point three times, except the first and last only twice

        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)], np.repeat(zs[jj, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.7, edgecolor=colors[target[jj]])
        legend_handles[target[jj]] = patch
        host.add_patch(patch)

    host.legend(legend_handles, target_names, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=len(target_names), fancybox=True, shadow=True)
    plt.tight_layout()
    if plot_file_name != "":
        plt.savefig(plot_file_name)    
    plt.show()


