# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_session_data, removeBadData, extract_saccades

SESSION = "session_2025-07-07_ferret_757_EyeCameras_P39_E11_analyzable_output"

Results = load_session_data(SESSION)
removeBadData(Results)

# df = extract_saccades(Results, 'skull', velocity_threshold=20.0)

# %%
import plotly.graph_objects as go
import numpy as np

df = extract_saccades(Results,
                      'eye', 
                      eye='LE', 
                      velocity_threshold=100,
                      min_duration=12)

y1 = Results.LE_x #Results.LE_gaze_horizontal_deg[win]
y2 = np.full(len(y1), np.nan)
y3 = np.full(len(y1), np.nan)

for j in range(len(df)):
        y2[df['onset'][j]] = y1[df['onset'][j]]
        y3[df['peak'][j]] = y1[df['peak'][j]]

win = np.arange(15 * 1e3, 25 * 1e3).astype(int)
y1 = y1[win]
y2 = y2[win]
y3 = y3[win]

fig = go.Figure()
fig.add_trace(go.Scatter(x=win , y=y1))
fig.add_trace(go.Scatter(x=win , y=y2, mode='markers', marker=dict(color='orange', size=6)))
fig.add_trace(go.Scatter(x=win , y=y3, mode='markers', marker=dict(color='red', size=6)))


# %%

from data_viewer import launch_viewer

launch_viewer(Results)

