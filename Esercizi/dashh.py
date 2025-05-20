import dash                      
from dash import dcc, html      
from dash.dependencies import Input, Output
import plotly.express as px     
import pandas as pd  


import plotly.express as px
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plotly  # NB: attenzione: qui si sta rinominando pyplot come 'plotly'
from matplotlib import font_manager as font_manager
from datetime import datetime, timedelta
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = dash.Dash(__name__)   # __name__ serve per sapere se il file è eseguito direttamente

# Layout dell'applicazione: ciò che l'utente vedrà nel browser
app.layout = html.Div([
    html.H3("Seleziona un numero"),  # Titolo

    # Slider da 1 a 10 con step 1 e valore iniziale 4
    dcc.Slider(
        id='slider-numero',        # ID del componente (serve per il callback)
        min=1,
        max=10,
        step=1,
        value=4,                   # Valore iniziale
        marks={i: str(i) for i in range(1, 11)}  # Etichette visibili sullo slider
    ),
    # Div dove verrà mostrato il risultato (es: "4² = 16")
    dcc.Graph(id='output-quadrato')
])


@app.callback(
    Output('output-quadrato','figure'),
    Input('slider-numero','value')
)

def regressione_polinomiale(valore):
    df=pd.read_csv('esamee.csv')
    x=df["ore_studio"].values
    y=df["punteggio_esame"].values
    modello_lineare=LinearRegression()
    modello_lineare.fit(x.reshape(-1,1),y)
    y_pred_skl=modello_lineare.predict(x.reshape(-1,1))
    modello_polinomiale=make_pipeline(PolynomialFeatures(valore),LinearRegression())
    modello_polinomiale.fit(x.reshape(-1,1),y)
    y_pred_skp=modello_polinomiale.predict(x.reshape(-1,1))
    mae_l=mean_absolute_error(y,y_pred_skl)
    rmse_l=root_mean_squared_error(y,y_pred_skl)
    mae_p=mean_absolute_error(y,y_pred_skp)
    rmse_p=root_mean_squared_error(y,y_pred_skp)
    
    fig=go.Figure()
    
    fig.add_trace(go.Scatter(x=x,y=y, name="Dati reali", mode="markers", line=dict(color="Yellow")))
    fig.add_trace(go.Scatter(x=x,y=y_pred_skp, name=f"Regressione polinomiale, rmse:{rmse_p:2f}, mae:{mae_p:2f}", mode="lines", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=x,y=y_pred_skl, name=f"Regressione lineare, rmse:{rmse_l:2f}, mae:{mae_l:2f} ", mode="markers", line=dict(color="Red")))
    fig.update_layout(
        
        title="Regressione polinomiale vs lineare",
        xaxis_title="Ore di studio",
        yaxis_title="Punteggio esame"
        
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True,port=8051)


