
1. NaN out bad frames from quality. convolve with box filter to make sure there are not single bad frames?
difference between low/high threshold?

2. saccade detection: velocity and acceleration peak (with threshold), binocular coordination? 

3. head saccade detection: same principle as (2) but may need different parameters. combine yaw/pitch? maybe just examine yaw.

4. gaze shift detection: same principle for 2 & 3 but only consider horizontal gaze (head + eye). is gaze collected?

5. combine data across ages/EOs in the same data structure? or build a new 'processed' data structure from loaded?

# problems?: 
RE 757 and 753 always 100% bad?session_2026-02-28_ferret_405_EO0_analyzable_output
