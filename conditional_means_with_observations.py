import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# https://seaborn.pydata.org/examples/jitter_stripplot.html
def cond_means(input_dataset, class_column, xaxis_title="value", yaxis_title="measurement", visualization_variance=0.0):
    '''
    input_dataset: pandas
    class_column: str
        Name of the column with the classification (or clustering) labels
    xaxis_title: str
        Title for the horizontal axis
    yaxis_title: str
        Title for the vertical axis
    visualization_variange: float
        Variance of noise added to the measures. This is useful when measures are integers with low dynamic range
    '''
    sns.set_theme(style="whitegrid")

    # Generate noise
    noise = np.random.normal(0.0, visualization_variance, (input_dataset.shape[0], input_dataset.shape[1]-1))
    # Avoid adding noise to class column
    classes = list(input_dataset.columns)
    classes.remove(class_column)
    input_dataset[classes] = input_dataset[classes] + noise
    
    # "Melt" the dataset to "long-form" or "tidy" representation
    input_dataset = pd.melt(input_dataset, class_column, var_name="measurement")  # ???????

    # Initialize the figure
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    # Show each observation with a scatterplot
    sns.stripplot(x=xaxis_title, y=yaxis_title, hue=class_column, data=input_dataset, dodge=True, alpha=.25, zorder=1)

    # Show the conditional means
    sns.pointplot(x=xaxis_title, y=yaxis_title, hue=class_column,
                  data=input_dataset, dodge=.532, join=False, palette="dark",
                  markers="d", scale=.85, ci=None)

    # Improve the legend 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:], labels[3:], title=class_column,
              handletextpad=0, columnspacing=1,
              loc="lower right", ncol=3, frameon=True)
    plt.show()

