import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
#regressione lineare e polinomiale e calcolo metriche con numpy

#carico il database
url="https://www.statlearning.com/s/Advertising.csv"
df = pd.read_csv(url)


#regressione lineare
df["mesi"] = range(len(df))
x=df["mesi"].values
y=df["sales"].values

print(df.head())

coeffs = np.polyfit(x, y, deg=1)
a, b = coeffs  # a = coefficiente angolare, b = intercetta

# Predizioni
y_pred_np = a * x + b
#metrice
mae= np.mean(np.abs(y-y_pred_np)) 
rmse=np.sqrt(np.mean((y-y_pred_np)**2))
#regressione polinomiale

fit=np.polyfit(x,y,deg=2)
poly=np.poly1d(fit)

y_pred_npp=poly(x)
#metrice

maep=np.mean(np.abs(y-y_pred_npp))
rmsep=np.sqrt(np.mean((y-y_pred_npp)**2))

plt.figure(figsize=(12, 6))
plt.scatter(x, y, marker='o', linestyle='-', label='dati reali')
plt.scatter(x, y_pred_npp, color="red",label=f'polinomiale, mae:{maep:2f}, rmse:{rmsep:2f}')
plt.plot(x,y_pred_np, label=f"Regressione lineare:, mae:{mae:2f}, rmse:{rmse:2f}", color="green")
plt.title('Vendite vs spesa tv')
plt.xlabel('spesa tv')
plt.ylabel('vendite')
plt.legend()
plt.grid(True)
plt.show()

#scipy

#regressione lineare

def lineare(x,c,d):
    return c*x+d

coeff,_=sp.optimize.curve_fit(lineare,x,y)

c,d=coeff

y_pred_scl=lineare(x,c,d)


maesc=np.mean(np.abs(y-y_pred_scl))
rmsesc=np.sqrt(np.mean((y-y_pred_scl)**2))

#polinomiale

def modello_polinomiale(x,c_p,d_p,e_p):
    return c_p*x**2 + d_p*x+e_p

coeffp,_=sp.optimize.curve_fit(modello_polinomiale,x,y)

c_p,d_p,e_p=coeffp

y_pred_scp=modello_polinomiale(x,c_p,d_p,e_p)

maescp=np.mean(np.abs(y-y_pred_scp))
rmsescp=np.sqrt(np.mean((y-y_pred_scp)**2))

plt.figure()
plt.scatter(x,y, label="Valori reali")
plt.plot(x,y_pred_scl, label=f"Regressione lineare, mae:{maesc:2f}, rmse:{rmsesc:2f}", color="green")
plt.scatter(x,y_pred_scp, label=f"Regressione polinomiale, mae:{maescp:2f}, rmse:{rmsescp:2f}", color="red")
plt.xlabel("Spesa tv")
plt.ylabel("Vendite")
plt.title("Regressione lineare vs polinomiale SCIPY")
plt.legend()
plt.grid(True)
plt.show()



#regressione con sckity-learn

from sklearn.model_selection import train_test_split

#splittiamo in dati di test e train

x_train, X_test, y_train, y_test=train_test_split(x,y,test_size=24, shuffle=False)

plt.figure()
plt.plot(np.sort(x_train),np.sort(y_train), marker='o',linestyle="-",label="Dati di allenamento ", color="red")
plt.plot(np.sort(X_test),np.sort(y_test),  marker='o',linestyle="-",label="Dati di test ", color="green")
plt.grid(True)
plt.show()

#regressione lineare

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
modello_linearesk=LinearRegression()

modello_linearesk.fit(x_train.reshape(-1,1),y_train)
y_pred_linearesk=modello_linearesk.predict(X_test.reshape(-1,1))
mae_skl=mean_absolute_error(y_test,y_pred_linearesk)
rmse_skl=root_mean_squared_error(y_test,y_pred_linearesk)

#regressione polinomiale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

modello_polinomialesk=make_pipeline(PolynomialFeatures(3),LinearRegression())
modello_polinomialesk.fit(x_train.reshape(-1,1),y_train)
y_pred_polinomialesk=modello_polinomialesk.predict(X_test.reshape(-1,1))

mae_skp=mean_absolute_error(y_test,y_pred_polinomialesk)
rmse_skp=root_mean_squared_error(y_test,y_pred_polinomialesk)

plt.figure()
plt.scatter(x,y, marker='o',linestyle="-",label="Dati reali ", color="red")
plt.plot(X_test, y_pred_linearesk,  marker='o',linestyle="-",label=f"regressione linare, mae={mae_skl:2f}, rmse={rmse_skl:2f}", color="green")
plt.plot(X_test,y_pred_polinomialesk, marker='o',linestyle="-",label=f"regressione polinomiale, mae={mae_skp:2f}, rmse={rmse_skp:2f}", color="red")
plt.legend()
plt.grid(True)
plt.show()



from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plotly  # NB: attenzione: qui si sta rinominando pyplot come 'plotly'
from matplotlib import font_manager as font_manager
from datetime import datetime, timedelta
import numpy as np


import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

fig=go.Figure()
fig.add_trace(go.Scatter(x=x,y=y, name="Dati reali", mode="markers", line=dict(color="yellow", width=4)))
fig.add_trace(go.Scatter(x=X_test,y=y_pred_linearesk, name="Lineare", mode="lines", line=dict(color="blue", width=4)))
fig.add_trace(go.Scatter(x=X_test,y=y_pred_polinomialesk, name="Polinomiale", mode="lines", line=dict(color="red", width=4)))
fig.update_layout(
    
    title="Lineare vs Polinomiale ",
    xaxis_title="Mesi",
    yaxis_title="Vendite"
)

fig.show()



