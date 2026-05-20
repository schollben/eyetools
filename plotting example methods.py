# %% Multiple plotting methods
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scratch import Results

window = slice(100, 1000)  # Example window of frames to plot
window_times = Results['skull_timestamps'][window]

# Method 2: Plotly Graph Objects (more control)

fig_go = go.Figure()
fig_go.add_trace(go.Scatter(x=window_times, y=Results['roll'][window], mode='lines', name='roll'))
fig_go.add_trace(go.Scatter(x=window_times, y=Results['pitch'][window], mode='lines', name='pitch'))
fig_go.add_trace(go.Scatter(x=window_times, y=Results['yaw'][window], mode='lines', name='yaw'))
fig_go.update_layout(
    title="Skull Rotations (Plotly Graph Objects)",
    xaxis_title="Time (s)",
    yaxis_title="Degrees",
    hovermode='x unified'
)
fig_go.show()

# %%
window = slice(11000, 20000)  # Example window of frames to plot
window_times = Results['LE_timestamps'][window]
window_times = window_times - window_times[0]  # Normalize to start at 0

fig_go = go.Figure()
fig_go.add_trace(go.Histogram(x=Results['LE_pupil_major'][window], nbinsx=30, name='LE_pupil_major'))
fig_go.add_trace(go.Histogram(x=Results['RE_pupil_major'][window], nbinsx=30, name='RE_pupil_major'))
fig_go.update_layout(
    title="Pupil Sizes Distribution (Plotly Graph Objects)",
    xaxis_title="mm",
    yaxis_title="Frequency",
    width=600,
    height=400,
    barmode='overlay'
)
fig_go.show()


# %% Distribution of LE_pupil_major with custom bins
window = slice(11000, 20000)
bins = np.linspace(0, 5, 40)  # Example bins from 0 to 5 mm
fig_mpl, ax = plt.subplots(figsize=(10, 6))
ax.hist(Results['LE_pupil_major'][window], bins=bins, label='LE_pupil_major', edgecolor='white', linewidth=.25, alpha=0.7)
ax.hist(Results['RE_pupil_major'][window], bins=bins, label='RE_pupil_major', edgecolor='white', linewidth=.25, alpha=0.7)
plt.tight_layout()
plt.show()


# %% Method 3: Matplotlib with file save (non-interactive)
fig_mpl, ax = plt.subplots(figsize=(10, 6))
ax.plot(window_times, Results['roll'][window], label="roll", linewidth=2)
ax.plot(window_times, Results['pitch'][window], label="pitch", linewidth=2)
ax.plot(window_times, Results['yaw'][window], label="yaw", linewidth=2)
ax.set_xlabel("Time (s)", fontfamily="Arial", fontsize=16)
ax.set_ylabel("Degrees", fontfamily="Arial", fontsize=16)
ax.set_title("Skull Rotation", fontfamily="Arial", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily("Arial")
    label.set_fontsize(16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.legend()
plt.tight_layout()
# Save to file instead of show() for non-interactive use?
# plt.savefig("/tmp/skull_rotations.png", dpi=150)
# plt.close()  # Close to free memory
# print("Matplotlib plot saved to /tmp/skull_rotations.png")



# %% Method 4: Altair (declarative, good for web)
try:
    import altair as alt
    df_plot = pd.DataFrame({
        'Time': np.tile(window_times, 3),
        'Degrees': np.concatenate([Results['roll'][window], Results['pitch'][window], Results['yaw'][window]]),
        'Component': ['roll']*len(window_times) + ['pitch']*len(window_times) + ['yaw']*len(window_times)
    })
    chart = alt.Chart(df_plot).mark_line().encode(
        x='Time:Q',
        y='Degrees:Q',
        color='Component:N'
    ).properties(title="Skull Rotations (Altair)", width=800, height=400)
    chart.show()
except ImportError:
    print("Altair not installed. Install with: pip install altair")
    
# %%

# other examples: 

    # sns.scatterplot(x=eos, y=pupil_m,   ax=axes[0], 
#                 color="#72B3E9",marker='o', s=40, edgecolor='black')


# fig, ax = plt.subplots(figsize=(4, 3))
# sns.kdeplot(Y1, ax=ax, color='green', label='LE', fill=True, alpha=0.3)
# sns.kdeplot(Y2, ax=ax, color='red',   label='RE', fill=True, alpha=0.3)

# %% SAVING SVG METHODS:

loc = "/Users/benjaminscholl/Library/CloudStorage/Dropbox/projects/2poptostim/rawfigs/"
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(f'{loc}/XXX.svg', format='svg', bbox_inches='tight')
