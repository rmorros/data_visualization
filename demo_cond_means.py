import pandas as pd
import numpy as np
from conditional_means_with_observations import cond_means
import seaborn as sns


# pandas.DataFrame
iris = sns.load_dataset("iris")
cond_means(iris, "species", xaxis_title="value", yaxis_title="measurement")

