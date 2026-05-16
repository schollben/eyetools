# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_session_data, removeBadData

SESSION = "session_2025-07-07_ferret_757_EyeCameras_P39_E11_analyzable_output"

Results = load_session_data(SESSION)
removeBadData(Results)

# %%

import plotly.graph_objects as go
import numpy as np

win = np.arange(15 * 1e3, 20 * 1e3).astype(int)

LEQ = Results.LEQ.astype(bool)
EQ = np.zeros(LEQ[win].shape)
EQ[LEQ[win]] = 100

y1 = Results.LE_gaze_horizontal_deg[win]

y2 = Results.LE_x[win]

fig = go.Figure()
fig.add_trace(go.Scatter(x=win , y=y1))
fig.add_trace(go.Scatter(x=win , y=y2))
fig.add_trace(go.Scatter(x=win, y=EQ))


# %%

from data_viewer import launch_viewer

launch_viewer(Results)

