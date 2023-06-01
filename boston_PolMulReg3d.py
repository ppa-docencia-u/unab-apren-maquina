# Polinomial Multiple Regression (3D)
# Cap. 6 pag. 146

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Boston Housing Dataset
# https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
# Polynomial Multiple Regression
#  
# # Solo para pruebas
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Boston Housing DF
df = pd.DataFrame(
    data, 
    columns= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE', 
              'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
)
df['MEDV'] = target

#X = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
X = df[['LSTAT', 'RM']]
y = df['MEDV']

# Plotting the 3D Hyperplane
# Cap. 6 pag. 146
fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['LSTAT'],
           X['RM'],
           y, 
           c='b'
    )
ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")

#---create a meshgrid of all the values for LSTAT and RM---
x_surf = np.arange(0, 40, 1)   #---for LSTAT---
y_surf = np.arange(0, 10, 1)   #---for RM---
x_surf, y_surf = np.meshgrid(x_surf, y_surf)


# Función polinomial de grado 2 (cuadrática)
degree = 2
polynomial_features = PolynomialFeatures(degree = degree)

X_poly = polynomial_features.fit_transform(X)
print(polynomial_features.get_feature_names_out())

# Aplica Regresión Lineal
model = LinearRegression()
model.fit(X_poly, y)

#---calculate z(MEDC) based on the model---
z = lambda x,y: (model.intercept_ +
                  (model.coef_[1] * x) +
                  (model.coef_[2] * y) +
                  (model.coef_[3] * x**2) +
                  (model.coef_[4] * x*y) +
                  (model.coef_[5] * y**2)
                )

ax.plot_surface(x_surf, y_surf, z(x_surf,y_surf),
                rstride=1,
                cstride=1,
                color='None',
                alpha = 0.4
                )
plt.show()