import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plotly  # NB: attenzione: qui si sta rinominando pyplot come 'plotly'
from matplotlib import font_manager as font_manager
from datetime import datetime, timedelta
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

# STEP 1: Carica o genera dati

url="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df=pd.read_csv(url)

print(df.head())

x=df["rm"].values
y=df["medv"].values

# STEP 2: NumPy fit + predizione + errori

#lineare

coeff_npl=np.polyfit(x,y,deg=1)
a,b = coeff_npl
y_pred_npl= a*x+b

mae_npl=np.mean(np.abs(y-y_pred_npl))
rnse_npl= np.sqrt(np.mean((y-y_pred_npl)**2))

#polinomiale

fitp=np.polyfit(x,y,deg=2)
poly=np.poly1d(fitp)

y_pred_npp=poly(x)

mae_npp=np.mean(np.abs(y-y_pred_npp))
rnse_npp=np.sqrt(np.mean((y-y_pred_npp)**2))

plt.figure()

plt.scatter(x,y, label="Dati reali", color="yellow")
plt.plot(x,y_pred_npl, label=f"Regressione lineare, mae:{mae_npl:2f}, rnse:{rnse_npl:2f}", color="green")
plt.scatter(x,y_pred_npp, label=f"Regressione polinomiale, mae:{mae_npp:2f}, rnse:{rnse_npp:2f}", color="red")
plt.title("NUMPY: LINEARE VS POLINOMIALE")
plt.xlabel("Numero di stanze")
plt.ylabel("Costo case")
plt.legend()
plt.grid(True)
plt.show()


# STEP 3: SciPy fit + predizione + errori

#lineare

def lineare(x,a_l,b_l):
    return a_l*x+b_l

coeff_scl,_=sp.optimize.curve_fit(lineare,x,y)
a_l,b_l=coeff_scl

y_pred_scl=lineare(x,a_l,b_l)

mae_scl=np.mean(np.abs(y-y_pred_scl))
rnse_scl=np.sqrt(np.mean((y-y_pred_scl)**2))

#polinomiale

def polinomiale(x,a_p,b_p,c_p):
    return a_p*x**2+b_p*x +c_p

coeff_scp,_=sp.optimize.curve_fit(polinomiale,x,y)
a_p,b_p,c_p=coeff_scp

y_pred_scp=polinomiale(x,a_p,b_p,c_p)

mae_scp=np.mean(np.abs(y-y_pred_scp))
rnse_scp=np.sqrt(np.mean((y-y_pred_scp)**2))

plt.figure()

plt.scatter(x,y, label="Dati reali", color="yellow")
plt.plot(x,y_pred_scl, label=f"Regressione lineare, mae:{mae_scl:2f}, rnse:{rnse_scl:2f}", color="green")
plt.scatter(x,y_pred_scp, label=f"Regressione polinomiale, mae:{mae_scp:2f}, rnse:{rnse_scp:2f}", color="red")
plt.title("SCIPY: LINEARE VS POLINOMIALE")
plt.xlabel("Numero di stanze")
plt.ylabel("Costo case")
plt.legend()
plt.grid(True)
plt.show()


# STEP 4: sklearn fit + predizione + errori

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#lineare

model_linear=LinearRegression()
model_linear.fit(x_train.reshape(-1,1), y_train)
y_pred_skl=model_linear.predict(x_test.reshape(-1,1))

mae_skl=mean_absolute_error(y_test,y_pred_skl)
rmse_skl=root_mean_squared_error(y_test,y_pred_skl)

#polinomiale

model_poly=make_pipeline(PolynomialFeatures(2),LinearRegression())
model_poly.fit(x_train.reshape(-1,1),y_train)
y_pred_skp=model_poly.predict(x_test.reshape(-1,1))

mae_skp=mean_absolute_error(y_test,y_pred_skp)
rnse_skp=root_mean_squared_error(y_test,y_pred_skp)

plt.figure()
plt.scatter(x,y, label="dati reali", color="yellow")
plt.plot(x_test,y_pred_skl,label=f"Regressione lineare, mae:{mae_skl:2f}, rmse:{rmse_skl:2f}", color="green")
plt.scatter(x_test,y_pred_skp,label=f"Regressione polinomiale, mae:{mae_skp:2f}, rmse:{rnse_skp:2f}", color="red")
plt.title("SKLEARN: LINEARE VS POLINOMIALE")
plt.xlabel("Numero di stanze")
plt.ylabel("Costo case")
plt.legend()
plt.grid(True)
plt.show()
# STEP 5: Plot finale e confronto

fig=go.Figure()

fig.add_trace(go.Scatter(x=x,y=y, name="dati reali", mode="markers",line=dict(color="yellow", width=4)))
fig.add_trace(go.Scatter(x=x,y= y_pred_npl, name="numpy lineare", mode="lines", line=dict(color="red")))
fig.add_trace(go.Scatter(x=x,y= y_pred_scl, name="scipy lineare", mode="lines", line=dict(color="green")))
fig.add_trace(go.Scatter(x=x_test,y= y_pred_skl, name="sklearn lineare", mode="lines", line=dict(color="orange")))

fig.update_layout(
    
    title="Confronto lineare",
    xaxis_title="N.stanze",
    yaxis_title="Costo case"
)

fig.show()

fig=go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, name="Dati reali", mode="markers", line=dict(color="yellow")))
fig.add_trace(go.Scatter(x=x, y=y_pred_npp, name= "NUMPY POLINOMIALE", mode="markers", line=dict(color="red")))
fig.add_trace(go.Scatter(x=x,y=y_pred_scp, name="SCIPY POLINOMALE", mode="markers", line=dict(color="green")))
fig.add_trace(go.Scatter(x=x_test, y=y_pred_skp, name="SKLEARN POLINOMIALE", mode="markers", line=dict(color="orange")))

fig.update_layout(
    
    title="confronto polinomiale",
    xaxis_title="N.stanze",
    yaxis_title="Costo case"
)

fig.show()