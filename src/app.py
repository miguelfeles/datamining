import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from paginas import *
from scipy.stats import norm
from scipy import stats
from scipy.stats import kstest, normaltest, anderson, shapiro
import shutil
from sklearn.ensemble import IsolationForest
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "11rem",
    "padding": "2rem 1rem",
    "background-color": "rgba(24, 107, 46, 0.5)",
    "border": "3px solid black"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Fútbol"),
        html.Hr(),
        html.P(
            "Análisis partidos del fútbol de inglaterra", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Contexto", href="/", active="exact", style={"color": "White"}),
                dbc.NavLink("Univariado", href="/page-1", active="exact", style={"color": "White"}),
                dbc.NavLink("Multivariado", href="/page-2", active="exact", style={"color": "White"}),
            ],
            vertical=True,
            pills=True,
        ),

         html.P(
            "Dennis Arango Laura Paiba Miguel Feles", className="lead", style={"margin-top":"180%"}
        ),
        html.Img(src = "assets/balón.png", style={"width":"150%", "margin-top":"20%"})
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return inicio()
    elif pathname == "/page-1":
        return univariado()
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(Output("equipos", "children"), [Input("radios", "value")])
def display_value(value):
    return equiposP(value)

@app.callback(Output('univar', 'children'),Input('demo-dropdown', 'value'), suppress_callback_exceptions=True)
def update_output(value):
    return medidas(value)

@app.callback(Output('graphuni', 'children'),Input('demo-dropdown', 'value'), suppress_callback_exceptions=True)
def update_output(value):
    return graficas(value)

if __name__ == "__main__":
    app.run_server(port=8888, debug = True)