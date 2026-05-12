import webbrowser
import numpy as np
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

DEFAULT_WINDOW_SEC = 20.0
WINDOW_OPTIONS = [5, 10, 20, 30, 60]

LE_X_COLOR  = "#1f4e79"
LE_Y_COLOR  = "#6baed6"
RE_X_COLOR  = "#7f0000"
RE_Y_COLOR  = "#fc8d59"

SKULL_VEL_OPTIONS = [
    {"label": " yaw ",   "value": "yaw_v"},
    {"label": " pitch ", "value": "pitch_v"},
    {"label": " roll ",  "value": "roll_v"},
]

SUBPLOT_TITLES = [
    "Left Eye Position (deg)",
    "Right Eye Position (deg)",
    "Eye Velocity (deg/s)",
    "Skull Rotation (deg)",
    "Skull Angular Velocity (deg/s)",
    "Locomotion Speed (mm/s)",
    "Pupil Size (mm)",
]

# fixed y-ranges (None = auto)
Y_RANGES = {
    1: [-30, 30],
    2: [-30, 30],
    3: [0, 300],
    4: None,
    5: None,       # dynamic — set per window based on selected component
    6: [0, 800],
    7: [1, 5],
}


def _window_mask(ts, t_start, window_sec):
    return (ts >= t_start) & (ts < t_start + window_sec)


def launch_viewer(Results):
    eq_ts = Results.EQtimestamps
    leq   = Results.LEQ.astype(float)[:len(eq_ts)]
    req   = Results.REQ.astype(float)[:len(eq_ts)]

    t_min = float(Results.LE_timestamps[0])
    t_max = float(Results.LE_timestamps[-1])

    skull_vel_arrays = {
        "yaw_v":   Results.yaw_v,
        "pitch_v": Results.pitch_v,
        "roll_v":  Results.roll_v,
    }
    skull_vel_colors = {
        "yaw_v":   "#9467bd",
        "pitch_v": "#ff7f0e",
        "roll_v":  "#2ca02c",
    }

    app = Dash(__name__)

    controls = html.Div([
        # Window selector
        html.Span("Window: ", style={"color": "#333", "fontFamily": "sans-serif",
                                     "marginRight": "6px"}),
        dcc.RadioItems(
            id="window-select",
            options=[{"label": f" {w}s ", "value": w} for w in WINDOW_OPTIONS],
            value=DEFAULT_WINDOW_SEC,
            inline=True,
            style={"color": "#333", "fontFamily": "sans-serif",
                   "display": "inline-block", "marginRight": "32px"},
        ),
        # Skull velocity component selector
        html.Span("Head vel: ", style={"color": "#333", "fontFamily": "sans-serif",
                                       "marginRight": "6px"}),
        dcc.RadioItems(
            id="skull-vel-select",
            options=SKULL_VEL_OPTIONS,
            value="yaw_v",
            inline=True,
            style={"color": "#333", "fontFamily": "sans-serif",
                   "display": "inline-block", "marginRight": "32px"},
        ),
        # Time slider
        html.Span("Start: ", style={"color": "#333", "fontFamily": "sans-serif",
                                    "marginRight": "6px"}),
        html.Div(
            dcc.Slider(
                id="time-slider",
                min=t_min,
                max=t_max,
                step=0.5,
                value=t_min,
                marks={
                    int(t_min): f"{t_min:.0f}s",
                    int(t_min + (t_max - t_min) / 2): f"{t_min + (t_max - t_min) / 2:.0f}s",
                    int(t_max): f"{t_max:.0f}s",
                },
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            style={"flex": "1", "minWidth": "300px"},
        ),
    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
              "padding": "6px 40px 16px", "background": "white"})

    app.layout = html.Div([
        dcc.Graph(id="main-plot", style={"height": "90vh"},
                  config={"scrollZoom": False}),
        controls,
    ], style={"background": "white"})

    @app.callback(
        Output("main-plot", "figure"),
        Input("time-slider", "value"),
        Input("window-select", "value"),
        Input("skull-vel-select", "value"),
    )
    def update(t_start, window_sec, skull_vel_key):
        window_sec = float(window_sec)

        fig = make_subplots(
            rows=7, cols=1,
            shared_xaxes=True,
            subplot_titles=SUBPLOT_TITLES,
            vertical_spacing=0.035,
            row_heights=[1, 1, 1, 0.9, 0.9, 0.8, 0.7],
        )

        # eye mask
        me = _window_mask(Results.LE_timestamps, t_start, window_sec)
        ts_eye = Results.LE_timestamps[me]

        # quality mask (EQ timestamps)
        mq = _window_mask(eq_ts, t_start, window_sec)
        ts_eq = eq_ts[mq]

        # skull mask
        ms = _window_mask(Results.skull_timestamps, t_start, window_sec)
        ts_skull = Results.skull_timestamps[ms]

        # --- Row 1: Left Eye Position + LEQ quality ---
        for arr, name, color in [
            (Results.LE_x, "LE x", LE_X_COLOR),
            (Results.LE_y, "LE y", LE_Y_COLOR),
        ]:
            fig.add_trace(go.Scatter(x=ts_eye, y=arr[me], name=name,
                                     line=dict(color=color, width=1.2),
                                     legendgroup="le_pos"), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ts_eq, y=leq[mq] * 20,
            name="LE quality", line=dict(color="rgba(200,200,200,0.6)", width=1),
            fill="tozeroy", fillcolor="rgba(200,200,200,0.15)",
            legendgroup="quality",
        ), row=1, col=1)

        # --- Row 2: Right Eye Position + REQ quality ---
        for arr, name, color in [
            (Results.RE_x, "RE x", RE_X_COLOR),
            (Results.RE_y, "RE y", RE_Y_COLOR),
        ]:
            fig.add_trace(go.Scatter(x=ts_eye, y=arr[me], name=name,
                                     line=dict(color=color, width=1.2),
                                     legendgroup="re_pos"), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=ts_eq, y=req[mq] * 20,
            name="RE quality", line=dict(color="rgba(200,200,200,0.6)", width=1),
            fill="tozeroy", fillcolor="rgba(200,200,200,0.15)",
            legendgroup="quality", showlegend=False,
        ), row=2, col=1)

        # --- Row 3: Eye velocity magnitudes ---
        le_vel = np.sqrt(Results.LE_vx ** 2 + Results.LE_vy ** 2)
        re_vel = np.sqrt(Results.RE_vx ** 2 + Results.RE_vy ** 2)
        for arr, name, color in [
            (le_vel, "LE vel", LE_X_COLOR),
            (re_vel, "RE vel", RE_X_COLOR),
        ]:
            fig.add_trace(go.Scatter(x=ts_eye, y=arr[me], name=name,
                                     line=dict(color=color, width=1.2),
                                     legendgroup="eye_vel"), row=3, col=1)

        # --- Row 4: Skull rotation ---
        for arr, name, color in [
            (Results.roll,  "roll",  "#2ca02c"),
            (Results.pitch, "pitch", "#ff7f0e"),
            (Results.yaw,   "yaw",   "#9467bd"),
        ]:
            fig.add_trace(go.Scatter(x=ts_skull, y=arr[ms], name=name,
                                     line=dict(color=color, width=1.2),
                                     legendgroup="skull"), row=4, col=1)

        # --- Row 5: Skull angular velocity (selected component) ---
        skull_arr = skull_vel_arrays[skull_vel_key]
        skull_color = skull_vel_colors[skull_vel_key]
        windowed_vel = skull_arr[ms]
        fig.add_trace(go.Scatter(x=ts_skull, y=windowed_vel, name=skull_vel_key,
                                 line=dict(color=skull_color, width=1.2),
                                 legendgroup="skull_vel"), row=5, col=1)

        # --- Row 6: Locomotion speed ---
        loco = np.sqrt(Results.linearVel_x ** 2 + Results.linearVel_y ** 2)
        fig.add_trace(go.Scatter(x=ts_skull, y=loco[ms], name="loco speed",
                                 line=dict(color="#8c564b", width=1.2),
                                 legendgroup="loco"), row=6, col=1)

        # --- Row 7: Pupil size ---
        for arr, name, color in [
            (Results.LE_pupil_major, "LE pupil", LE_X_COLOR),
            (Results.RE_pupil_major, "RE pupil", RE_X_COLOR),
        ]:
            fig.add_trace(go.Scatter(x=ts_eye, y=arr[me], name=name,
                                     line=dict(color=color, width=1.2),
                                     legendgroup="pupil"), row=7, col=1)

        # --- Axis ranges ---
        fig.update_xaxes(range=[t_start, t_start + window_sec])
        for row, yrange in Y_RANGES.items():
            if yrange:
                fig.update_yaxes(range=yrange, row=row, col=1)

        # auto-scale skull velocity to current window
        if len(windowed_vel) > 0:
            v_max = np.nanmax(np.abs(windowed_vel)) * 1.15
            fig.update_yaxes(range=[-v_max, v_max], row=5, col=1)

        fig.update_layout(
            margin=dict(l=60, r=20, t=40, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#333", size=11),
            legend=dict(groupclick="toggleitem", bgcolor="rgba(0,0,0,0)"),
            showlegend=True,
            uirevision="static",
        )
        fig.update_xaxes(showgrid=False, zerolinecolor="#aaa")
        fig.update_yaxes(showgrid=False, zerolinecolor="#aaa")

        return fig

    webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=False)
