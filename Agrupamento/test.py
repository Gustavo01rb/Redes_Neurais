import matplotlib.pyplot as plt
from C_means import C_Means

x, y = C_Means.generate_data(n_samples=1000, centers=4, cluster_std=1.3)
model = C_Means()
model.init_random_center(n_centers= 4, x1_min= x[:,0].min(), x1_max= x[:,0].max(), x2_min= x[:,1].min(), x2_max= x[:,1].max())



model.fit(x)
model.info(x=x)
