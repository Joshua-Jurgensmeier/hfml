# app.py
import os
import time
import threading
import numpy as np
from collections import deque
import asyncio

from scipy.signal import spectrogram

from dash import Dash, dcc, html, Input, Output, State
import dash
import plotly.graph_objs as go

# -----------------------
# Configuration
# -----------------------
SAMPLE_RATE = 48000
CHUNK_SECONDS = 0.5          # how many seconds per chunk read
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_SECONDS)
NFFT = 1024                  # FFT size for spectrogram
OVERLAP = NFFT // 2
MAX_TIME_SLICES = 200        # how many time columns to keep in waterfall

async def get_next(chunk):
    # compute spectrogram for the chunk and push a time-slice (magnitude vs freq)
    f, t, Sxx = spectrogram(chunk, fs=SAMPLE_RATE, window='hann',

# -----------------------
# Dash app
# -----------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H3("Spectral Waterfall Dashboard"),
    html.Div([
        html.Label("Source:"),
        dcc.Dropdown(
            id='source-dropdown',
            options=[
                {'label': 'Simulated signal', 'value': 'simulate'},
                {'label': 'Microphone (requires sounddevice)', 'value': 'mic'},
                {'label': 'Load WAV file', 'value': 'wav'}
            ],
            value='simulate',
            clearable=False,
            style={'width': '300px'}
        ),
        dcc.Input(id='wav-path', type='text', placeholder='Path to WAV file (optional)', style={'width': '400px'}),
        html.Button('Start', id='start-btn', n_clicks=0),
        html.Button('Stop', id='stop-btn', n_clicks=0),
    ], style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}),
    dcc.Graph(id='waterfall-graph', style={'height': '600px'}),
    dcc.Interval(id='interval', interval=500, n_intervals=0, disabled=True),
    html.Div(id='status', style={'marginTop': '8px', 'fontFamily': 'monospace'})
])

# Global source object
SRC = None

@app.callback(
    Output('interval', 'disabled'),
    Output('status', 'children'),
    Input('start-btn', 'n_clicks'),
    Input('stop-btn', 'n_clicks'),
    State('source-dropdown', 'value'),
    State('wav-path', 'value'),
    prevent_initial_call=True
)
def start_stop(start_clicks, stop_clicks, source_value, wav_path):
    global SRC
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'start-btn':
        if SRC:
            SRC.stop()
        SRC = Source(mode=source_value, wav_path=wav_path if wav_path else None)
        SRC.start()
        return False, f"Running: mode={source_value}"
    else:
        if SRC:
            SRC.stop()
            SRC = None
        return True, "Stopped"

@app.callback(
    Output('waterfall-graph', 'figure'),
    Input('interval', 'n_intervals'),
    State('waterfall-graph', 'figure')
)
def update_graph(n_intervals, existing_fig):
    global SRC
    if not SRC or len(SRC.buffer) == 0:
        # empty placeholder
        fig = go.Figure()
        fig.update_layout(
            xaxis={'title': 'Time (s)'},
            yaxis={'title': 'Frequency (Hz)'},
            template='plotly_dark'
        )
        return fig

    # Build waterfall matrix: columns are time slices, rows are frequency bins
    # All entries in buffer share the same frequency vector f
    f = SRC.buffer[0][0]
    mags = np.stack([col[1] for col in SRC.buffer], axis=1)  # shape (freq_bins, time_slices)
    # x axis: relative time indices (older -> left)
    t_axis = np.arange(-mags.shape[1]+1, 1) * CHUNK_SECONDS

    # Create heatmap (flip freq axis so low freq at bottom)
    fig = go.Figure(data=go.Heatmap(
        z=np.flipud(mags),
        x=t_axis,
        y=np.flipud(f),
        colorscale='Viridis',
        colorbar=dict(title='dB'),
        zmin=np.max(mags) - 80,
        zmax=np.max(mags)
    ))
    fig.update_layout(
        xaxis_title='Time (s, relative)',
        yaxis_title='Frequency (Hz)',
        template='plotly_dark',
        margin=dict(l=60, r=10, t=30, b=60)
    )
    return fig

if __name__ == '__main__':
    # Start with a simulated source by default
    SRC = Source(mode='simulate')
    SRC.start()
    # Enable Dash debug mode off for production; set host='0.0.0.0' to expose externally
    app.run(debug=True, port=8050)
