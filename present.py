#seaborn
import seaborn as sns

#Loading iris data with seaborn

iris = sns.load_dataset('iris')

#%matplotlib inline

import seaborn as sns;sns.set()

sns.pairplot(iris, hue='species',height=3)