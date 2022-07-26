# %%
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.compose import (
    make_column_selector,
    make_column_transformer,
)
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# %%
car_price_df = pd.read_csv(
    "./dataset/data_pricing_challenge.csv", parse_dates=["registration_date", "sold_at"]
)

# %%
features = [f"feature_{i}" for i in range(1, 9)]
cat_var = ["model_key", "fuel", "paint_color", "car_type", *features]
num_var = ["mileage", "engine_power", "registration_date", "sold_at"]
target = ["price"]

# transform column to sold month (model seasonality)
car_price_df.loc[:, "sold_at"] = car_price_df.sold_at.dt.month
# transform column to registrated year (simplify car age)
car_price_df.loc[:, "registration_date"] = car_price_df.registration_date.dt.year

# %%
# drop the two cases above
car_price_df.drop(car_price_df[car_price_df.engine_power == 0].index, inplace=True)
car_price_df.drop(car_price_df[car_price_df.mileage < 0].index, inplace=True)

# %%
# Decisions to make on Categorical Variables
# Merge minority colors
minority_colors = ["red", "beige", "green", "orange"]  # ['green','orange'] #
car_price_df.loc[:, "paint_color"] = np.where(
    car_price_df.paint_color.isin(minority_colors), "other", car_price_df.paint_color
)

# drop hybrids and electrics
car_price_df.drop(
    car_price_df[car_price_df.fuel.isin(["hybrid_petrol", "electro"])].index,
    inplace=True,
)

# group model keys by series
for i in range(1, 8):
    model_mask = (
        car_price_df.model_key.str.startswith(f"{i}")
        | car_price_df.model_key.str.startswith(f"M{i}")
        | car_price_df.model_key.str.startswith(f"Z{i}")
    )
    car_price_df.loc[model_mask, "model_key"] = f"{i}00 series"
    car_price_df.loc[
        car_price_df.model_key.str.startswith(f"X{i}"), "model_key"
    ] = f"X{i} series"

# %%
# fit model with less features
red_feature_df = car_price_df[
    ["model_key", "car_type", "feature_2", "feature_5", "feature_7", "feature_8"]
    + num_var
]
y = car_price_df[target].values.reshape(-1)

# %%
categorical_columns_selector = make_column_selector(dtype_exclude="int64")
set_unknown_value = np.nan

ordinal_encoder = make_column_transformer(
    (
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=set_unknown_value
        ),
        categorical_columns_selector,
    ),
    remainder="passthrough",
)

rand_forest = Pipeline(
    [
        ("preprocessor", ordinal_encoder),
        (
            "regressor",
            RandomForestRegressor(
                n_jobs=2,
                # random_state=42,
            ),
        ),
    ]
)

# fit rand forest on reduced features data_set to explore solution in web app
_ = rand_forest.fit(red_feature_df, y)

# %%
# dash date picker
min_date_allowed = 2005
max_date_allowed = 2016
dd_date_ini = dbc.Select(
    id="my-date-ini",
    options=[
        {"label": l, "value": l} for l in range(min_date_allowed, max_date_allowed + 1)
    ],
    value=list(range(min_date_allowed, max_date_allowed + 1))[0],
)
dd_date_end = dbc.Select(
    id="my-date-end",
    options=[
        {"label": l, "value": l} for l in range(min_date_allowed, max_date_allowed + 1)
    ],
    value=list(range(min_date_allowed, max_date_allowed + 1))[-1],
)

# Input Engine Power
power_input = dcc.Input(
    id="input_power", type="range", min=70, max=320, step=10, value=190
)

# Input Mileage
mil_input = dcc.Input(
    id="input_mileage", type="range", min=5000, max=300000, step=5000, value=150000
)

# Input Month
month_input = dcc.Input(
    id="input_sold_month", type="range", min=1, max=9, step=1, value=5
)

# dbc select: car model
available_models = sorted(red_feature_df.model_key.unique(), key=str.lower)
dd_car_model = dbc.Select(
    id="my-car-model",
    options=[{"label": l, "value": l} for l in available_models],
    value=available_models[0],
)

# dbc select: car type
available_types = sorted(red_feature_df.car_type.unique(), key=str.lower)
dd_car_type = dbc.Select(
    id="my-car-type",
    options=[{"label": l, "value": l} for l in available_types],
    value=available_types[0],
)

# # dbc select: car type (optional reactive with callback)
# dd_car_type = dbc.Select(
#     id="my-car-type",
#     options=[],
#     value="",
# )

# feature switches
switch_features = html.Div(
    [
        dbc.Checklist(
            options=[
                {"label": "Feature 2", "value": 2},
                {"label": "Feature 5", "value": 5},
                {"label": "Feature 7", "value": 7},
                {"label": "Feature 8", "value": 8},
            ],
            value=[],
            id="switches-input",
            switch=True,
        ),
    ]
)

# %%
# input row
# dbc data upload row
input_row = dbc.Container(
    dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.P(
                            "Select Init year",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "18px",
                                "marginBottom": "20px",
                                "textAlign": "center",
                                "color": "DeepSkyBlue",
                            },
                        ),
                        dd_date_ini,
                    ]
                ),
                width="auto",
            ),
            dbc.Col(
                html.Div(
                    [
                        html.P(
                            "Select End year",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "18px",
                                "marginBottom": "20px",
                                "textAlign": "center",
                                "color": "DeepSkyBlue",
                            },
                        ),
                        dd_date_end,
                    ]
                ),
                width="auto",
            ),
            dbc.Col(
                html.Div(
                    [
                        html.P(
                            "Select Engine Power",
                            style={
                                "fontWeight": "bold",  # 'normal', #
                                "textAlign": "left",  # 'center', #
                                # 'paddingTop': '25px',
                                "color": "DeepSkyBlue",
                                "fontSize": "18px",
                                "marginBottom": "10px",
                            },
                        ),
                        power_input,
                        html.Div(
                            id="power-state",
                            style={
                                "color": "DarkBlue",
                                "textAlign": "center",
                                "fontWeight": "bold",
                            },
                        ),
                    ]
                ),
                width="auto",
            ),
            dbc.Col(
                html.Div(
                    [
                        html.P(
                            "Select Car Model",
                            style={
                                "fontWeight": "bold",  # 'normal', #
                                "textAlign": "left",  # 'center', #
                                # 'paddingTop': '25px',
                                "color": "DeepSkyBlue",
                                "fontSize": "18px",
                                "marginBottom": "10px",
                            },
                        ),
                        dd_car_model,
                    ]
                ),
                width="auto",
            ),
            dbc.Col(
                html.Div(
                    [
                        html.P(
                            "Select Car Type",
                            style={
                                "fontWeight": "bold",  # 'normal', #
                                "textAlign": "left",  # 'center', #
                                # 'paddingTop': '25px',
                                "color": "DeepSkyBlue",
                                "fontSize": "18px",
                                "marginBottom": "10px",
                            },
                        ),
                        dd_car_type,
                    ]
                ),
                width="auto",
            ),
            dbc.Col(
                html.Div(
                    [
                        html.P(
                            "Toggle Car Features",
                            style={
                                "fontWeight": "bold",  # 'normal', #
                                "textAlign": "left",  # 'center', #
                                # 'paddingTop': '25px',
                                "color": "DeepSkyBlue",
                                "fontSize": "18px",
                                "marginBottom": "10px",
                            },
                        ),
                        switch_features,
                    ]
                ),
                width="auto",
            ),
        ],
        justify="evenly",
        align="start",
    ),
    fluid=True,
)

# %%
# dictionary for plotly: label with no figure
label_no_fig = {
    "layout": {
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "annotations": [
            {
                "text": "No matching data",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 28},
            }
        ],
    }
}

# dbc plot row
plot_row = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.P(
                                "Select Mileage",
                                style={
                                    "fontWeight": "bold",  # 'normal', #
                                    "textAlign": "left",  # 'center', #
                                    # 'paddingTop': '25px',
                                    "color": "DeepSkyBlue",
                                    "fontSize": "18px",
                                    "marginBottom": "10px",
                                },
                            ),
                            mil_input,
                            html.Div(
                                id="mil-state",
                                style={
                                    "color": "DarkBlue",
                                    "textAlign": "center",
                                    "fontWeight": "bold",
                                },
                            ),
                        ]
                    ),
                    width="auto",
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.P(
                                "Select Selling Month",
                                style={
                                    "fontWeight": "bold",  # 'normal', #
                                    "textAlign": "left",  # 'center', #
                                    # 'paddingTop': '25px',
                                    "color": "DeepSkyBlue",
                                    "fontSize": "18px",
                                    "marginBottom": "10px",
                                },
                            ),
                            month_input,
                            html.Div(
                                id="month-state",
                                style={
                                    "color": "DarkBlue",
                                    "textAlign": "center",
                                    "fontWeight": "bold",
                                },
                            ),
                        ]
                    ),
                    width="auto",
                ),
            ],
            justify="around",
            align="start",
            style={
                "paddingLeft": "0px",
                "marginBottom": "0",
            },
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="price-plot-time", figure=label_no_fig), width=6),
                dbc.Col(dcc.Graph(id="price-plot-mil", figure=label_no_fig), width=6),
            ],
            justify="evenly",
            align="start",
        ),
    ],
    fluid=True,
)

# %%
fontawesome_stylesheet = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
app = Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP, fontawesome_stylesheet]
)

# to deploy using WSGI server
server = app.server
# app tittle for web browser
app.title = "BMW Regression Explorer"

# title row
title_row = dbc.Container(
    dbc.Row(
        [
            dbc.Col(
                html.Img(src="assets/bmw_logo.png", style={"width": "30%"}),
                width=3,
                # width={"size": 3, "offset": 1},
                style={"paddingLeft": "100px", "paddingTop": "30px"},
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H6(
                            "BMW Regression Explorer",
                            style={
                                "fontWeight": "bold",
                                "textAlign": "center",
                                "paddingTop": "25px",
                                "color": "white",
                                "fontSize": "32px",
                            },
                        ),
                    ]
                ),
                # width='auto',
                width={"size": "auto", "offset": 1},
            ),
        ],
        justify="start",
        align="center",
    ),
    fluid=True,
)

# %%
# App Layout
app.layout = html.Div(
    [
        # title Div
        html.Div(
            [title_row],
            style={
                "height": "150px",
                "width": "100%",
                "backgroundColor": "DeepSkyBlue",
                "margin-left": "auto",
                "margin-right": "auto",
                "margin-top": "15px",
            },
        ),
        # div input row
        html.Div(
            [input_row],
            style={
                "paddingTop": "20px",
                "paddingBottom": "20px",
            },
        ),
        # div plot row
        dcc.Loading(
            html.Div(
                [plot_row],
            ),
            id="loading-surf",
            type="circle",
            fullscreen=True,
        ),
        html.Hr(
            style={
                "color": "DeepSkyBlue",
                "height": "3px",
                "margin-top": "0",
                "margin-bottom": "0",
            }
        ),
    ]
)

# %%
@app.callback(
    Output("power-state", "children"),
    Input("input_power", "value"),
)
def power_render(p_val):
    return f"{p_val}"


@app.callback(
    Output("mil-state", "children"),
    Input("input_mileage", "value"),
)
def mil_render(mil_val):
    return f"{mil_val}"


@app.callback(
    Output("month-state", "children"),
    Input("input_sold_month", "value"),
)
def month_render(m_val):
    return f"{m_val}"


# %%
@app.callback(
    Output("my-date-end", "options"),
    Output("my-date-end", "value"),
    Input("my-date-ini", "value"),
    State("my-date-end", "value"),
    prevent_initial_call=True,
)
def update_end_options(init_date, end_date):
    end_date = init_date if int(init_date) > int(end_date) else end_date
    end_date_options = [
        {"label": l, "value": l} for l in range(int(init_date), max_date_allowed + 1)
    ]
    return end_date_options, end_date


# %%
# @app.callback(
#     Output("my-car-type", "options"),
#     Output("my-car-type", "value"),
#     Input("my-car-model", "value"),
# )
# def update_car_type_options(car_model):
#     available_types = sorted(red_feature_df.query(
#         "model_key == @car_model"
#     ).car_type.unique(), key=str.lower)
#     car_type_options = [
#         {"label": l, "value": l} for l in available_types
#     ]
#     car_type_val = available_types[0]
#     return car_type_options, car_type_val

# %%
@app.callback(
    Output("price-plot-time", "figure"),
    Input("my-date-ini", "value"),
    Input("my-date-end", "value"),
    Input("power-state", "children"),
    Input("mil-state", "children"),
    Input("my-car-model", "value"),
    Input("my-car-type", "value"),
    Input("switches-input", "value"),
    prevent_initial_call=True,
)
def update_price_time(
    year_ini, year_end, power_val, mil_val, car_model, car_type, car_features
):

    year_range = range(int(year_ini), int(year_end) + 1)
    time_range = range(1, 10)
    year_time_arrays = [[yr, tm] for yr in year_range for tm in time_range]
    year_array = [elem[0] for elem in year_time_arrays]
    time_array = [elem[1] for elem in year_time_arrays]
    plot_matrix_len = len(year_time_arrays)

    input_df = pd.DataFrame.from_dict(
        {
            "model_key": [car_model for i in range(plot_matrix_len)],
            "car_type": [car_type for i in range(plot_matrix_len)],
            "feature_2": [
                True if 2 in car_features else False for i in range(plot_matrix_len)
            ],
            "feature_5": [
                True if 5 in car_features else False for i in range(plot_matrix_len)
            ],
            "feature_7": [
                True if 7 in car_features else False for i in range(plot_matrix_len)
            ],
            "feature_8": [
                True if 8 in car_features else False for i in range(plot_matrix_len)
            ],
            "mileage": [int(mil_val) for i in range(plot_matrix_len)],
            "engine_power": [int(power_val) for i in range(plot_matrix_len)],
            "registration_date": year_array,
            "sold_at": time_array,
        }
    )

    z_data = rand_forest.predict(input_df)

    fig = go.Figure(
        data=[
            go.Surface(
                z=z_data.reshape(len(year_range), len(time_range)),
                x=list(time_range),
                y=list(year_range),
            )
        ]
    )
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.update_layout(
        title="Car Selling Price Predictions by Year and Selling Month",
        scene=dict(
            xaxis_title="Selling Month",
            yaxis_title="Registration Year",
            zaxis_title="Price",
        ),
        autosize=False,
        # scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=65),
    )

    return fig


# %%
@app.callback(
    Output("price-plot-mil", "figure"),
    Input("my-date-ini", "value"),
    Input("my-date-end", "value"),
    Input("power-state", "children"),
    Input("month-state", "children"),
    Input("my-car-model", "value"),
    Input("my-car-type", "value"),
    Input("switches-input", "value"),
    prevent_initial_call=True,
)
def update_price_mile(
    year_ini, year_end, power_val, month_val, car_model, car_type, car_features
):

    year_range = range(int(year_ini), int(year_end) + 1)
    mile_range = range(5000, 305000, 5000)

    year_mile_arrays = [[yr, ml] for yr in year_range for ml in mile_range]
    year_array = [elem[0] for elem in year_mile_arrays]
    mile_array = [elem[1] for elem in year_mile_arrays]
    plot_matrix_len = len(year_mile_arrays)

    input_df = pd.DataFrame.from_dict(
        {
            "model_key": [car_model for i in range(plot_matrix_len)],
            "car_type": [car_type for i in range(plot_matrix_len)],
            "feature_2": [
                True if 2 in car_features else False for i in range(plot_matrix_len)
            ],
            "feature_5": [
                True if 5 in car_features else False for i in range(plot_matrix_len)
            ],
            "feature_7": [
                True if 7 in car_features else False for i in range(plot_matrix_len)
            ],
            "feature_8": [
                True if 8 in car_features else False for i in range(plot_matrix_len)
            ],
            "mileage": mile_array,
            "engine_power": [int(power_val) for i in range(plot_matrix_len)],
            "registration_date": year_array,
            "sold_at": [int(month_val) for i in range(plot_matrix_len)],
        }
    )

    z_data = rand_forest.predict(input_df)

    fig = go.Figure(
        data=[
            go.Surface(
                z=z_data.reshape(len(year_range), len(mile_range)),
                x=list(mile_range),
                y=list(year_range),
            )
        ]
    )
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.update_layout(
        title="Car Selling Price Predictions by Year and Mileage",
        scene=dict(
            xaxis_title="Mileage",
            yaxis_title="Registration Year",
            zaxis_title="Price",
        ),
        autosize=False,
        # scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=65),
    )

    return fig


# %%
if __name__ == "__main__":
    app.run_server(debug=True)
