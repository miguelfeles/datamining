import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
botones = html.Div(
    [
        dbc.RadioItems(
            id="radios",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Premier League", "value": 1},
                {"label": "England Championship", "value": 2},
                {"label": "England League 1", "value": 3},
                {"label": "England League 2", "value": 4},
                {"label": "England National League", "value": 5},
            ],
            value=1,
        ),
    ],
    className="radio-group",


)


columnas = ['tournament',
 'home',
 'home_goal',
 'away_goal',
 'away',
 'home_corner',
 'away_corner',
 'home_attack',
 'away_attack',
 'home_shots',
 'away_shots',
 'ht_diff',
 'at_diff',
 'ht_result',
 'at_result',
 'total_corners']
desplegable = dcc.Dropdown(columnas, columnas[0], id='demo-dropdown')
