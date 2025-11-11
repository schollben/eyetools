import numpy as np
import matplotlib.pyplot as plt

def quickPlot(xPosLeye, yPosLeye,
              xPosReye, yPosReye,
              yaw, pitch, roll,
              speed,
              window=[1000, 4000]):
    
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(12, 8))
    ax_eyeL, ax_eyeR, ax_rot1,ax_rot2, ax_rot3, ax_speed = axes

    fs = 120
    t = np.arange(int(window[0]), int(window[1]))
    time_sec = (t - t[0]) / fs

    # Left Eye
    ax_eyeL.plot(time_sec, xPosLeye[t], label="X Position", color="tab:blue", alpha=0.7)
    ax_eyeL.plot(time_sec, yPosLeye[t], label="Y Position", color="tab:red", alpha=0.7)
    
    ax_eyeL.set_ylim([-60, 60])
    ax_eyeL.set_yticks([-60, 0, 60])
    ax_eyeR.set_ylabel("Degrees")
    ax_eyeL.legend(loc="upper right")

    # Right Eye
    ax_eyeR.plot(time_sec, xPosReye[t], label="X Position", color="tab:blue", alpha=0.7)
    ax_eyeR.plot(time_sec, yPosReye[t], label="Y Position", color="tab:red", alpha=0.7)
    
    ax_eyeR.set_ylim([-60, 60])
    ax_eyeR.set_yticks([-60, 0, 60])
    ax_eyeR.set_ylabel("Degrees")
    ax_eyeR.legend(loc="upper right")

    # Head rotation
    ax_rot1.plot(time_sec, yaw[t], label="Yaw", color="tab:green")
    ax_rot1.legend(loc="upper right")

    ax_rot1.set_ylim([-180, 180])
    ax_rot1.set_yticks([-180, 0, 180])
    ax_rot1.set_ylabel("Degrees")

    ax_rot2.plot(time_sec, pitch[t], label="Pitch", color="tab:orange")
    ax_rot2.legend(loc="upper right")

    ax_rot2.set_ylim([-60, 60])
    ax_rot2.set_ylabel("Degrees")

    ax_rot3.plot(time_sec, roll[t], label="Roll", color="tab:purple")
    ax_rot3.legend(loc="upper right")
    ax_rot3.set_ylim([-60, 60])
    ax_rot3.set_ylabel("Degrees")

    # Speed
    ax_speed.plot(time_sec, speed[t], color="black", label="Speed")
    ax_speed.plot([4, 5], [500, 500], color='black', linestyle='-', linewidth=5)
    ax_speed.text(3.5, 375, "1 Second", fontsize=12)

    ax_speed.legend(loc="upper right")
    ax_speed.set_ylim([0, 100])
    ax_speed.set_yticks([0, 20, 40, 60, 80, 100])
    ax_speed.set_ylabel("cm/s")

    for ax in axes:
        ax.tick_params(direction='out')
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()



def quickPlot_wBools(xPosLeye, yPosLeye,
              xPosReye, yPosReye,
              eyeConfL,eyeConfR,
              blinkThreshL, blinkThreshR,
              window=[1000, 4000]):
    
    # plotStuff.quickPlot_wBools(
    #     xPosLeye=results[ID].xL,
    #     yPosLeye=results[ID].yL,
    #     xPosReye=results[ID].xR,
    #     yPosReye=results[ID].yR,
    #     eyeConfL=results[ID].eyeConfL,
    #     eyeConfR=results[ID].eyeConfR,
    #     blinkThreshL=results[ID].blinkThreshL,
    #     blinkThreshR=results[ID].blinkThreshR,
    #     window=[start, start + 120*dur]
    # )
    
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax_eyeL, ax_eyeR = axes

    fs = 120
    t = np.arange(int(window[0]), int(window[1]))
    time_sec = (t - t[0]) / fs

    # Left Eye
    ax_eyeL.plot(time_sec, xPosLeye[t], label="X Position", color="tab:blue", alpha=0.7)
    ax_eyeL.plot(time_sec, yPosLeye[t], label="Y Position", color="tab:red", alpha=0.7)
    # ax_eyeL.plot(time_sec, eyeConfL[t]*60, label="Eye Conf", color="gray", linestyle='--', linewidth=2)
    ax_eyeL.plot(time_sec, blinkThreshL[t]*60, label="Blink Thresh", color="gray", linestyle='-', linewidth=2)

    ax_eyeL.set_ylim([-60, 60])
    ax_eyeL.set_yticks([-60, 0, 60])
    ax_eyeR.set_ylabel("Degrees")
    ax_eyeL.legend(loc="upper right")

    # Right Eye
    ax_eyeR.plot(time_sec, xPosReye[t], label="X Position", color="tab:blue", alpha=0.7)
    ax_eyeR.plot(time_sec, yPosReye[t], label="Y Position", color="tab:red", alpha=0.7)
    # ax_eyeR.plot(time_sec, eyeConfR[t]*60, label="Eye Conf", color="gray", linestyle='--', linewidth=2, alpha=0.5)
    ax_eyeR.plot(time_sec, blinkThreshR[t]*60, label="Blink Thresh", color="gray", linestyle='-', linewidth=2)

    ax_eyeR.set_ylim([-60, 60])
    ax_eyeR.set_yticks([-60, 0, 60])
    ax_eyeR.set_ylabel("Degrees")
    ax_eyeR.legend(loc="upper right")

    for ax in axes:
        ax.tick_params(direction='out')
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

