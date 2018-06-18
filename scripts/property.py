import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
x = np.random.randint(1, 50, 100)
y = np.random.randint(1, 50, 100)
coords = np.column_stack([x, y])
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(coords)
plt.subplot(1,1,1)
plt.scatter(x, y, c=y_pred)
plt.show()
