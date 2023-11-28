import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html,  dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from boot import *
from scipy.stats import norm
from scipy import stats
from scipy.stats import kstest, normaltest, anderson, shapiro
import shutil
from sklearn.ensemble import IsolationForest
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
import numpy as np



df = pd.read_csv("futbol.csv")

df2 =   df.copy()
df2['date'] = pd.to_datetime(df2['date'])
df2 = df2[df2['date'].dt.year == 2023]


atipicos = pd.DataFrame()
variable = {}
for i in ["home_goal", "away_goal", "home_corner", "away_corner", "home_attack", "away_attack", "home_shots", "away_shots", "ht_diff", "at_diff", "total_corners"]:
        data = df2[i]
        median = np.median(data)
        abs_deviation = np.abs(data - median)
        mad = np.median(abs_deviation)
        threshold = 1.4826 * mad * 3
        outliers = data[abs_deviation > threshold]
        if len(outliers) > 0:
            variable[i] = len(outliers)
        
atipicos["Variable"] = variable.keys()
atipicos["Datos atípicos"] = variable.values()
atipicos = atipicos.sort_values("Datos atípicos", ascending = False)
atipicos = atipicos.set_index('Variable')['Datos atípicos'].to_dict()


def calculate_entropy(labels):
    """
    Calculate the entropy of a label distribution.

    :param labels: array-like, list of class labels (categorical data)
    :return: float, entropy value
    """
    # Count the occurrences of each label
    label_counts = np.unique(labels, return_counts=True)[1]

    # Calculate the probabilities for each label
    probabilities = label_counts / label_counts.sum()

    # Calculate the entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def barh(diccionario, titulo, titulox):
    fig = px.bar(y = diccionario.keys(), x=diccionario.values(), orientation="h", text_auto=True, color_discrete_sequence=['#186b2e'], template = "simple_white", title =  titulo )
    fig.update_coloraxes(showscale = False)
    fig.update_xaxes(ticksuffix = "%", showgrid = True)
    fig.update_traces(textfont_size = 10, textangle = 0, textposition = ('inside'))
    fig.update_xaxes(ticksuffix="%", showgrid=True, title="")
    fig.update_yaxes(title="")
    fig.update_layout(paper_bgcolor = "white", plot_bgcolor = "rgba(0,0,0,0)",title_font_color = "black", title_y = 0.95,margin=dict(l=10, b=0,r=0, t = 80), barmode = 'stack', yaxis = {'categoryorder': 'total ascending'})
    fig.update_layout(title_y=0.95)
    
    return fig.update_layout(margin = dict(t=50, l=50, r=50, b=0)).update_xaxes(ticksuffix = "", title = titulox)

def line_plot(diccionario, titulo, titulox):
    fig = px.line(x=diccionario.keys(), y=diccionario.values(), title=titulo)

    fig.update_traces(line=dict(color='#186b2e'))

    fig.update_layout(template="simple_white")
    fig.update_layout(title=titulo, paper_bgcolor="white", plot_bgcolor="rgba(0,0,0,0)",
                      title_font_color="black", title_y=0.95, margin=dict(l=10, b=0, r=0, t=80))
    fig.update_xaxes(showgrid=True, title=titulox)
    fig.update_yaxes(title="")
    fig.update_layout(margin=dict(t=50, l=50, r=50, b=0))

    return fig

fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=["texto", " "], values=[0.29, 1-0.29], hole=0.5, name="Donut 1", textinfo='percent', marker_colors=["#186b2e", "lightgray"]), 1, 1)
fig.add_trace(go.Pie(labels=["texto", " "], values=[0.58, 1-0.58], hole=0.5, name="Donut 1", textinfo=' percent', marker_colors=["#186b2e", "lightgray"]), 1, 2)
fig.add_trace(go.Pie(labels=["texto", " "], values=[0.12, 1-0.11], hole=0.5, name="Donut 1", textinfo='percent', marker_colors=["#186b2e", "lightgray"]), 1, 3)
fig.update_layout(
    margin = dict(t=50, l=50, r=50, b=10),
    showlegend = False,
    title_text="Tipos de datos en el Data Frame",
    annotations=[
        dict(text='Variables Categóricas', x=0.10, y=1.1, font_size=12, showarrow=False),
        dict(text='Variables numéricas', x=0.50, y=1.1, font_size=12, showarrow=False),
        dict(text='Otros tipos de variables', x=0.90, y=1.1, font_size=12, showarrow=False)
    ]
)




def inicio():
    children = []
    children += [html.H5("Características de la Base de datos")]
    children += [dcc.Graph(figure=fig, style={"height": "30vh"})]
    children += [dcc.Graph(figure=barh({key: value for key, value in df2.isna().sum().to_dict().items() if value > 0}, "Datos Nulos", "Datos nulos").update_xaxes(ticksuffix=""), style={"height": "25vh", "width": "50%", "float": "left"})]
    children += [html.Div(
        "Variable Objetivo: Home Corners",
        style={
            'border': '10px solid #186b2e',
            'border-radius': '15px', 
            'background-color': 'white',
            'color': 'black',
            'text-align': 'center',  
            'padding': '20px',     
            'width': '200px',      
            'margin': '20px auto',
            'margin-left': '50%',
            "width":"50%" , # Ajusta el margen para centrar
            'display': 'block',
            "margin-top":"5%",
            "margin-bottom":"10%"      # Asegura que el div se comporte como un bloque
        })]

    children += [html.Div([html.H5("Características de la Base de Datos", style={"display":"block"}), botones], style={"display":"flex"})]
    children += [html.Div(id="equipos")]
    return children

def equiposP(v):
    equiposs = {1: "Premier League", 2: "England Championship", 3:"England League 1", 4:"England League 2", 5:"England National League"}
    seleccionado = equiposs[v]
    tabla = df2.copy()
    tabla[["ht_result", "at_result"]] = tabla[["ht_result", "at_result"]].replace({"WON":3, "DRAW":1, "LOST":0}).copy()
    tabla = tabla[tabla["tournament"] == seleccionado]
    tabla['date'] = pd.to_datetime(tabla['date'])
    tabla2 = tabla[tabla['date'].dt.year == 2023]

    

    children = [html.Img(src = "assets/" + seleccionado + ".png", style={"width":"15%", "height":"10%", "float":"left", "margin-left":"10%"})]

    texto = [html.H5(seleccionado, style={"font-weight":"bold"})]
    texto += [html.H5("Partidos Jugados: " + str(len(tabla)))]
    texto += [html.H5("Partidos Jugados en 2023 : " + str(len(tabla2)))]
    children += [dcc.Graph(figure = barh(tabla[["home", "away", "ht_result"]].groupby(by="home").sum()["ht_result"].sort_values(ascending=False)[0:5].to_dict(), " Mejores equipos 2023 ", "Puntos"), style={"float":"right", "width":"40%", "height":"30vh"} )]

    children += [html.Div(texto, style={"width":"25%", "height":"10%", "float":"left", "margin-left":"5%", "align":"center", "margin-top":"8vh"})]
    return children


def univariado():
    children = []
    children += [html.H5("Análisis Univariado")]
    children += [html.H6("Columnas disponibles: ")]

    izquierda = html.Div([desplegable, html.Div(id="univar", style={"margin-top":"5%"})], style={"float":"left", "width":"40%", "margin-bottom":"7%"})
    derecha = html.Div([html.Div(id="graphuni")], style={"float":"right","width":"60%"})
    children += [izquierda, derecha]

    entropia = {}
    for i in ["tournament", "home", "away", "ht_result", "at_result"]:
        entropia[i] = calculate_entropy(df[i])
    children += [html.H6("Datos atípicos", style={"margin-top":"6%"})]
    children += [dcc.Graph(figure=barh(atipicos, "Datos atípicos", "Datos atípicos"), style={"width":"50%", "float":"left", "height":"35vh"}), dcc.Graph(figure=line_plot(entropia, "Entropía de las variables categóricas", "Variables"), style={"width":"50%", "float":"right", "height":"35vh"})]



    children += []



    return children

def medidas(valor):
    print("hola")
    children = []
    tabla = df.copy()
    tabla['date'] = pd.to_datetime(tabla['date'])
    tabla = tabla[tabla['date'].dt.year == 2023]
    tabla = tabla[valor].dropna()

    tournament_counts = tabla.value_counts(normalize=True) * 100
    df_tournament = pd.DataFrame({valor: tournament_counts.index, 'Percentage': tournament_counts.values}).sort_values("Percentage", ascending = False).head(5)

    if valor in ["tournament", "home", "away", "ht_result", "at_result"]:
        children += [html.H6("La moda es: " + str(tabla.mode()[0]))]
        children += [html.H6("No hay una clase con frecuencia dominante.")]

        children += [dash_table.DataTable(
        df_tournament.to_dict('records'),
        [{"name": i, "id": i} for i in df_tournament.columns],
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
        )]


    elif valor in ["home_goal", "away_goal", "home_corner", "away_corner", "home_attack", "away_attack", "home_shots", "away_shots", "ht_diff", "at_diff", "total_corners"]:
        children += [html.H6("El promedio es: " + str(round(tabla.mean(), 2)))]
        children += [html.H6("La varianza es: " + str(round(tabla.var(), 2)))]


        print(stats.kstest(tabla, 'norm'))
        children += [html.H5("Prubas de normalidad")]
        children += [html.H6("prueba de Kolmogorov: "+ str(round(stats.kstest(tabla, 'norm')[1],3)))]
        children += [html.H6("prueba de Shapiro: "+ str(round(stats.shapiro(tabla)[1],3)     ))]
        children += [html.H6("prueba de D'agostino: "+ str(round(stats.normaltest(tabla)[1],3)))]

        if (round(stats.kstest(tabla, 'norm')[1],3) < 0.06) & (round(stats.shapiro(tabla)[1],3)  < 0.06):
            children += [html.H6("Se rechaza la hipótesis nula, los datos no siguen una distribución normal")]
        else: 
            children += [html.H6("No hay suficiente evidencia estadística para rechazar la hipótesis nula")]
        
    return children

def graficas(valor):
    children = []
    tabla = df.copy()
    tabla['date'] = pd.to_datetime(tabla['date'])
    tabla = tabla[tabla['date'].dt.year == 2023]
    tabla = tabla[valor]
    if valor in ["tournament", "home", "away", "ht_result", "at_result"]:
        children += [dcc.Graph(figure = px.histogram(tabla, color_discrete_sequence= ["#186b2e" for i in tabla]).update_layout(showlegend = False, title = "Histograma " + valor))]
       
    elif valor in ["home_goal", "away_goal", "home_corner", "away_corner", "home_attack", "away_attack", "home_shots", "away_shots", "ht_diff", "at_diff", "total_corners"]:
        children += [dcc.Graph(figure = px.box(tabla, color_discrete_sequence= ["#186b2e" for i in tabla], orientation='h').update_layout(showlegend = False, title = "Boxplot " + valor))]
        
    return children



    
