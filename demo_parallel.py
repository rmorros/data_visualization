import pandas as pd
import numpy as np
from parallel_plot import parallel_plot
from sklearn import datasets

iris = datasets.load_iris()
parallel_plot(iris, 'Parallel Coordinates Plot — Iris', reverse_axes = [], shown_feat = [], plot_file_name = "", show_plot=True)
