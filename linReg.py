# A basic but realistic insight about global warming.
# Pedro Santos - https://github.com/PPSantos

"""
In this notebook you can find multiple regressions
plotted on NASA datasets (Global climate change).
Scores and coefficients are shown below.
(resources here: https://climate.nasa.gov/).

    1) 2D regressions.
        1.1) Global earth temperature.
        1.2) Carbon dioxide.
        1.3) Land ice (GreenLand Mass Variation).
        1.4) Sea level.
    2) 3D regresions.
        2.1) Greenland mass loss and CO2 concentration.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

"""
1.1) Global earth temperature anomaly through years (1880-2016).
     Data Source: NASA's Goddard Institute for Space Studies (GISS).
"""

# Reads data.
df_1 = pd.read_fwf('data/647_Global_Temperature_Data_File.txt')

# DATA: Years (X), temperature variation (Y).
x_values = df_1[['Year']]
y_values = df_1[['Temp1']]

# Concatenate data.
df = pd.concat([x_values, y_values], axis=1)
X = df.as_matrix(['Year'])
Y = df.as_matrix(['Temp1'])

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X, Y, test_size=0.1))

# Linear model (Y = AX + B).
reg = LinearRegression()
reg.fit(X_train, Y_train)
p = reg.predict(X_test).T

print '1.1) Temperature anomaly through years (1880-2016).'
print 'Linear model (Y = AX + B)'
print 'Coeficients: A)', reg.coef_[0][0]
print 'Intercept: B)', reg.intercept_[0]
print 'Score:', reg.score(X_test, Y_test)

# Polynomial model (Y = A(X + B)^2 + C).

# Dimensionality reduction.
xdata = X_train.flatten()
ydata = Y_train.flatten()

def f(t, a, b, c):
    return a*np.power(t + b, 2) + c

par, cov = curve_fit(f, xdata, ydata, p0=(0.0001, 1880, -0.3))

# Compute score (Sum of squares, Chi-Square).
residuals = ydata - f(xdata, *par)
chi = sum(residuals**2)

print 'Polynomial model (Y = A(X + B)^2 + C)'
print 'Coeficients: A)', par[0], ', B)', par[1], ', C)', par[2]
print 'Chi-Square:', chi, '\n'

# Graphic visualization.
plt.scatter(df['Year'], df['Temp1'])
plt.plot(X_test, p[0], color='green')
xline = np.linspace(min(xdata), max(xdata), 500)
plt.plot(xline, f(xline, *par), color='red')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (C)')
plt.title('Global average temperature anomaly (1880-2016).')
plt.show()

"""
1.2) CO2 concentration(ppm) through years (1958-2017).
     Data Source: NOAA ESRL DATA.
"""

# Reads data.
df_2 = pd.read_fwf('data/co2_mm_mlo_original.txt')

# DATA: Years (X), CO2 concentration(ppm)(Y).
x_values = df_2[['decimal']]
y_values = df_2[['average']]

# Concatenate data.
df = pd.concat([x_values, y_values], axis=1)
X = df.as_matrix(['decimal'])
Y = df.as_matrix(['average'])

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X, Y, test_size=0.1))

# Linear model (Y = AX + B).
reg = LinearRegression()
reg.fit(X_train, Y_train)
p = reg.predict(X_test).T

print '1.2) CO2 concentration through years (1958-2017).'
print 'Linear model (Y = AX + B)'
print 'Coeficients: A)', reg.coef_[0][0]
print 'Intercept: B)', reg.intercept_[0]
print 'Score:', reg.score(X_test, Y_test)

# Polynomial model (Y = A(X + B)^2 + C).

# Dimensionality reduction.
xdata = X_train.flatten()
ydata = Y_train.flatten()

def f(t, a, b, c):
    return a*np.power(t + b, 2) + c

par, cov = curve_fit(f, xdata, ydata, p0=(0.3, 1965, 315))

# Compute score (Sum of squares, Chi-Square).
residuals = ydata - f(xdata, *par)
chi = sum(residuals**2)

print 'Polynomial model (Y = A(X + B)^2 + C)'
print 'Coeficients: A)', par[0], ', B)', par[1], ', C)', par[2]
print 'Chi-Square:', chi, '\n'

# Graphic visualization.
plt.scatter(df['decimal'], df['average'])
plt.plot(X_test, p[0], color='green')
xline = np.linspace(min(xdata), max(xdata), 500)
plt.plot(xline, f(xline, *par), color='red')
plt.xlabel('Year')
plt.ylabel('CO2 concentration (ppm)')
plt.title('CO2 concentration (1958-2017).')
plt.show()

"""
1.3) GreenLand Mass Variation through years (2002-2017).
     Data Source: Wiese, D. N., D.-N. Yuan, C. Boening, F. W. Landerer and
     M. M. Watkins (2016) JPL GRACE Mascon Ocean, Ice, and Hydrology Equivalent
     HDR Water Height RL05M.1 CRI Filtered Version 2., Ver. 2., PO.DAAC, CA, USA.
     Dataset accessed [2017-08-07] at http://dx.doi.org/10.5067/TEMSC-2LCR5.
"""
# Reads data.
df_3 = pd.read_fwf('data/greenland_mass_200204_201701.txt')

# DATA: Years (X), Mass Variation(Gt)(Y).
x_values = df_3[['Year']]
y_values = df_3[['Mass']]

# Concatenate data.
df = pd.concat([x_values, y_values], axis=1)
X = df.as_matrix(['Year'])
Y = df.as_matrix(['Mass'])

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X, Y, test_size=0.1))

# Linear model (Y = AX + B).
reg = LinearRegression()
reg.fit(X_train, Y_train)
p = reg.predict(X_test).T

print '1.3) GreenLand Mass Variation (2002-2017).'
print 'Linear model (Y = AX + B)'
print 'Coeficients: A)', reg.coef_[0][0]
print 'Intercept: B)', reg.intercept_[0]
print 'Score:', reg.score(X_test, Y_test), '\n'

# Graphic visualization.
plt.scatter(df['Year'], df['Mass'])
plt.plot(X_test, p[0], color='green')
plt.xlabel('Year')
plt.ylabel('GreenLand Mass Variation (Gt)')
plt.title('Greenland Mass Variation (2002-2017).')
plt.show()

"""
1.4) Global Mean Sea Level variation(mm) through years(1993-2017).
     Data Source: GSFC. 2017. Global Mean Sea Level Trend from
     Integrated Multi-Mission Ocean Altimeters TOPEX/Poseidon,
     Jason-1, OSTM/Jason-2 Version 4. Ver. 4. PO.DAAC, CA, USA.
     Dataset accessed [2017-08-07] at 10.5067/GMSLM-TJ124.
"""

# Reads data.
df_4 = pd.read_fwf('data/GMSL_TPJAOS_V4_199209_201704.txt')

# DATA: Years (X), Sea level variation(mm)(Y).
x_values = df_4[['Year']]
y_values = df_4[['GMSL']]

# Concatenate data.
df = pd.concat([x_values, y_values], axis=1)
X = df.as_matrix(['Year'])
Y = df.as_matrix(['GMSL'])

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X, Y, test_size=0.1))

# Linear model (Y = AX + B).
reg = LinearRegression()
reg.fit(X_train, Y_train)
p = reg.predict(X_test).T

print '1.4) Sea height variation (1993-2017).'
print 'Linear model (Y = AX + B)'
print 'Coeficients: A)', reg.coef_[0][0]
print 'Intercept: B)', reg.intercept_[0]
print 'Score:', reg.score(X_test, Y_test), '\n'

# Graphic visualization.
plt.scatter(df['Year'], df['GMSL'])
plt.plot(X_test, p[0], color='green')
plt.xlabel('Year')
plt.ylabel('Sea height variation (mm)')
plt.title('Sea height variation (1993-2017).')
plt.show()

"""
2.1) Greenland mass loss and CO2 concentration through years (2003-2016).
     Data source: (shown above).
"""

# Reads data.
df_5 = pd.read_fwf('data/greenland_mass_200204_201608.txt')
df_6 = pd.read_fwf('data/co2_mm_mlo.txt')

# DATA: Years (X), CO2 concentration (Y), Greenland mass loss (Z).
x_values = df_6[['date']]
y_values = df_6[['average']]
z_values = df_5[['massloss']]

# Concatenate data.
df = pd.concat([x_values, y_values, z_values], axis=1)

X = df.as_matrix(['date'])
Y = df.as_matrix(['average', 'massloss']).astype('float32')

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X, Y, test_size=0.1))

# Linear model.
reg = LinearRegression()
reg.fit(X_train, Y_train)

print '2.1) Greenland mass loss and CO2 concentration through years (2003-2016).'
print 'Coeficients:', reg.coef_[0][0], ',', reg.coef_[1][0]
print 'Intercept:', reg.intercept_[0]
print 'Score:', reg.score(X_test, Y_test)

p = reg.predict(X_test).T

# Graphic visualization.
fig = plt.figure()
fig.set_size_inches(12.5, 7.5)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Year'); ax.set_ylabel('CO2 concentration (ppm)'); ax.set_zlabel('GreenLand Mass loss (Gt)')
ax.scatter(xs=df['date'], ys=df['average'], zs=df['massloss'])
ax.plot(xs=X_test, ys=p[0], zs=p[1], color='green')
ax.view_init(10, -45)
plt.title('Greenland mass loss and CO2 concentration (2003-2016).')
plt.show()
