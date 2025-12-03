"""
Rigid Body Transforms: World vs Local Frame Angular Velocity
=============================================================

This script demonstrates the mathematics of:
1. Quaternion representation of 3D rotations
2. Computing angular velocity from quaternion time series
3. The critical difference between world-frame and body-local-frame angular velocity
4. Why this distinction matters for inertial sensors (IMUs) attached to moving bodies

MATHEMATICAL BACKGROUND
=======================

1. QUATERNIONS FOR ROTATION
---------------------------
A unit quaternion q = (w, x, y, z) represents a 3D rotation where:
  - w = cos(θ/2)           [scalar part]
  - (x, y, z) = sin(θ/2) * n̂  [vector part, where n̂ is the rotation axis]

This comes from Euler's rotation theorem: any 3D rotation can be represented
as a single rotation of angle θ about some axis n̂.

Key quaternion operations:
  - Conjugate: q* = (w, -x, -y, -z)
  - For unit quaternions: q⁻¹ = q* (inverse equals conjugate)
  - Composition: q_total = q₂ * q₁ means "apply q₁ first, then q₂"
  - Rotate vector v: v' = q * v * q⁻¹ (treating v as pure quaternion (0, vx, vy, vz))

2. ANGULAR VELOCITY
-------------------
Angular velocity ω is a 3D vector representing instantaneous rotation:
  - Direction: axis of rotation
  - Magnitude: rotation rate in rad/s

Given two orientations q₁ and q₂ separated by time dt:
  - The incremental rotation is: Δq = q₂ * q₁⁻¹
  - Convert Δq to axis-angle: (n̂, θ)
  - Angular velocity: ω = (θ/dt) * n̂

3. WORLD FRAME vs LOCAL (BODY) FRAME
------------------------------------
This is the KEY insight this script demonstrates:

WORLD FRAME (also called "inertial frame" or "space frame"):
  - Fixed coordinate axes that don't move
  - Example: X=East, Y=North, Z=Up
  - ω_world tells you how the object rotates relative to the fixed world

LOCAL FRAME (also called "body frame" or "material frame"):
  - Coordinate axes attached to and moving with the body
  - Example: X=forward, Y=left, Z=up (from the body's perspective)
  - ω_local tells you what an accelerometer/gyroscope ON THE BODY would measure

The relationship:
  - ω_local = q⁻¹ * ω_world  (rotate world ω by inverse of body orientation)
  - ω_world = q * ω_local    (rotate local ω by body orientation)

WHY THIS MATTERS:
  - IMU sensors (gyroscopes) measure ω_local because they're attached to the body
  - Physics simulations often work in world frame
  - Navigation algorithms must convert between frames correctly
  - Getting this wrong causes "sensor fusion" bugs!

4. INTUITION EXAMPLE
--------------------
Imagine a spinning top that's also tilted:
  - World frame: "The top is rotating about the vertical Z axis"
  - Local frame: "From the top's perspective, it's rotating about its own tilted axis"

These are DIFFERENT vectors! The world frame ω points straight up, but the
local frame ω points along the top's (tilted) symmetry axis.

Run with: python rigid_body_quaternion_play.py

Outputs:
  - world_vs_local_omega.html: Interactive animated visualization
  - angular_velocity_frames.csv: Time series data for analysis
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# QUATERNION CLASS
# =============================================================================

@dataclass
class Quaternion:
    """
    Unit quaternion for rotations. Convention: scalar-first [w, x, y, z].

    MATHEMATICAL NOTES:
    -------------------
    Quaternions extend complex numbers to 4D: q = w + xi + yj + zk
    where i² = j² = k² = ijk = -1

    For rotations, we use UNIT quaternions (|q| = 1), which form the
    group S³ (3-sphere in 4D space). There's a 2-to-1 mapping from
    quaternions to rotations: q and -q represent the same rotation.

    The formula for rotating vector v by quaternion q is:
        v' = q * v * q⁻¹
    where v is embedded as a "pure quaternion" (0, vx, vy, vz).

    This is equivalent to the rotation matrix R(q), but quaternions:
    - Use only 4 numbers instead of 9
    - Don't suffer from gimbal lock
    - Interpolate smoothly (SLERP)
    - Compose efficiently
    """
    w: float  # Scalar part: cos(θ/2)
    x: float  # Vector part x: sin(θ/2) * axis_x
    y: float  # Vector part y: sin(θ/2) * axis_y
    z: float  # Vector part z: sin(θ/2) * axis_z

    def __post_init__(self) -> None:
        """Normalize to ensure unit quaternion (required for rotations)."""
        norm = np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        if norm < 1e-10:
            raise ValueError("Cannot normalize zero quaternion")
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm

    @classmethod
    def identity(cls) -> "Quaternion":
        """Return identity quaternion (no rotation): q = (1, 0, 0, 0)."""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @classmethod
    def from_axis_angle(cls, axis: NDArray[np.float64], angle_rad: float) -> "Quaternion":
        """
        Create quaternion from axis-angle representation.

        MATH: Given axis n̂ (unit vector) and angle θ:
            q = (cos(θ/2), sin(θ/2) * n̂)

        The half-angle appears because quaternion multiplication
        applies the rotation TWICE (q * v * q⁻¹), so we need θ/2
        to get total rotation θ.

        Args:
            axis: Rotation axis (will be normalized)
            angle_rad: Rotation angle in radians (right-hand rule)
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return cls.identity()
        axis = axis / axis_norm
        half_angle = angle_rad / 2.0
        return cls(
            w=np.cos(half_angle),
            x=np.sin(half_angle) * axis[0],
            y=np.sin(half_angle) * axis[1],
            z=np.sin(half_angle) * axis[2],
        )

    @classmethod
    def from_euler_xyz(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        """
        Create quaternion from Euler angles (XYZ intrinsic convention).

        MATH: Euler angles decompose rotation into three sequential rotations
        about coordinate axes. "XYZ intrinsic" means:
            1. Roll (rotation about X axis)
            2. Pitch (rotation about NEW Y axis after roll)
            3. Yaw (rotation about NEW Z axis after roll and pitch)

        As quaternion multiplication: q_total = q_z * q_y * q_x
        (rightmost applied first)

        WARNING: Euler angles suffer from "gimbal lock" when pitch = ±90°
        (one degree of freedom is lost). Quaternions avoid this!

        Args:
            roll: Rotation about X axis (radians)
            pitch: Rotation about Y axis (radians)
            yaw: Rotation about Z axis (radians)
        """
        qx = cls.from_axis_angle(axis=np.array([1, 0, 0]), angle_rad=roll)
        qy = cls.from_axis_angle(axis=np.array([0, 1, 0]), angle_rad=pitch)
        qz = cls.from_axis_angle(axis=np.array([0, 0, 1]), angle_rad=yaw)
        # Composition order: apply X first, then Y, then Z
        return qz * qy * qx

    def conjugate(self) -> "Quaternion":
        """
        Return conjugate q* = (w, -x, -y, -z).

        MATH: The conjugate reverses the rotation direction.
        For unit quaternions, conjugate equals inverse.
        """
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def inverse(self) -> "Quaternion":
        """
        Return inverse q⁻¹ such that q * q⁻¹ = identity.

        MATH: For unit quaternions, q⁻¹ = q* (conjugate).
        For non-unit: q⁻¹ = q* / |q|² (but we always normalize).
        """
        return self.conjugate()

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """
        Quaternion multiplication (Hamilton product).

        MATH: Using i² = j² = k² = ijk = -1:

        (w₁ + x₁i + y₁j + z₁k)(w₂ + x₂i + y₂j + z₂k) =
            (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂) +
            (w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂)i +
            (w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂)j +
            (w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)k

        NOTE: Quaternion multiplication is NOT commutative!
        q₁ * q₂ ≠ q₂ * q₁ (just like rotations)

        For composing rotations: q_total = q₂ * q₁ applies q₁ first, then q₂.
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def rotate_vector(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Rotate vector v by this quaternion: v' = q * v * q⁻¹

        MATH: We embed the 3D vector v as a "pure quaternion" (0, vx, vy, vz),
        then apply the sandwich product q * v * q⁻¹.

        The result is another pure quaternion whose vector part is the
        rotated vector. This is equivalent to multiplying by the 3x3
        rotation matrix R(q), but more efficient.

        Args:
            v: 3D vector to rotate

        Returns:
            Rotated 3D vector
        """
        v = np.asarray(v, dtype=np.float64)
        # Embed vector as pure quaternion (w=0)
        v_quat = Quaternion(w=0.0, x=v[0], y=v[1], z=v[2])
        # Apply rotation: q * v * q⁻¹
        rotated = self * v_quat * self.conjugate()
        # Extract vector part (w component should be ~0)
        return np.array([rotated.x, rotated.y, rotated.z])

    def to_axis_angle(self) -> tuple[NDArray[np.float64], float]:
        """
        Extract axis-angle representation from quaternion.

        MATH: Given q = (cos(θ/2), sin(θ/2) * n̂), we recover:
            θ = 2 * arccos(|w|)
            n̂ = (x, y, z) / sin(θ/2)

        We use |w| and handle the sign to ensure θ ∈ [0, π].

        Returns:
            tuple: (axis unit vector, angle in radians)
        """
        # Clamp w to [-1, 1] for numerical stability in arccos
        w_clamped = np.clip(self.w, -1.0, 1.0)
        angle = 2.0 * np.arccos(abs(w_clamped))

        sin_half = np.sqrt(1.0 - w_clamped ** 2)
        if sin_half < 1e-10:
            # Near-zero rotation: axis is arbitrary
            return np.array([1.0, 0.0, 0.0]), 0.0

        axis = np.array([self.x, self.y, self.z]) / sin_half
        # Ensure consistent axis direction
        if self.w < 0:
            axis = -axis
        return axis, angle


def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.

    MATH: SLERP traces the shortest arc on the 4D unit sphere between
    q1 and q2 at constant angular velocity. The formula is:

        slerp(q1, q2, t) = sin((1-t)θ)/sin(θ) * q1 + sin(tθ)/sin(θ) * q2

    where θ = arccos(q1 · q2) is the angle between quaternions.

    WHY SLERP?
    - Linear interpolation (lerp) doesn't maintain unit length
    - SLERP gives constant angular velocity (no acceleration)
    - Essential for smooth animation and motion planning

    Args:
        q1: Starting quaternion (t=0)
        q2: Ending quaternion (t=1)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Compute dot product (cosine of angle between quaternions)
    dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

    # If dot < 0, quaternions are on opposite hemispheres
    # Negate one to take the shorter arc (q and -q are same rotation)
    q2_adj = q2
    if dot < 0:
        q2_adj = Quaternion(w=-q2.w, x=-q2.x, y=-q2.y, z=-q2.z)
        dot = -dot

    # If quaternions are very close, use linear interpolation to avoid division by zero
    if dot > 0.9995:
        return Quaternion(
            w=q1.w + t * (q2_adj.w - q1.w),
            x=q1.x + t * (q2_adj.x - q1.x),
            y=q1.y + t * (q2_adj.y - q1.y),
            z=q1.z + t * (q2_adj.z - q1.z),
        )

    # Standard SLERP formula
    theta = np.arccos(dot)  # Angle between quaternions
    sin_theta = np.sin(theta)
    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return Quaternion(
        w=s1 * q1.w + s2 * q2_adj.w,
        x=s1 * q1.x + s2 * q2_adj.x,
        y=s1 * q1.y + s2 * q2_adj.y,
        z=s1 * q1.z + s2 * q2_adj.z,
    )


# =============================================================================
# ANGULAR VELOCITY COMPUTATION
# =============================================================================

def compute_omega_world(q1: Quaternion, q2: Quaternion, dt: float) -> NDArray[np.float64]:
    """
    Compute angular velocity in WORLD frame from two quaternions.

    MATH DERIVATION:
    ----------------
    The incremental rotation from q1 to q2 is:
        Δq = q2 * q1⁻¹

    This represents "what rotation, applied in world frame, takes q1 to q2?"

    Converting Δq to axis-angle (n̂, θ):
        ω_world = (θ / dt) * n̂

    This gives angular velocity expressed in WORLD frame coordinates.

    IMPORTANT: This is the "spatial" or "left" angular velocity in
    robotics terminology. It describes rotation as seen by a stationary
    observer in the world frame.

    Args:
        q1: Orientation at time t
        q2: Orientation at time t + dt
        dt: Time step (must be positive)

    Returns:
        Angular velocity vector in world frame [rad/s]
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Compute incremental rotation in world frame: Δq = q2 * q1⁻¹
    delta_q = q2 * q1.inverse()

    # Ensure we take the short path (both q and -q represent same rotation)
    if delta_q.w < 0:
        delta_q = Quaternion(w=-delta_q.w, x=-delta_q.x, y=-delta_q.y, z=-delta_q.z)

    # Convert to axis-angle and compute angular velocity
    axis, angle = delta_q.to_axis_angle()
    return (angle / dt) * axis


def omega_world_to_local(omega_world: NDArray[np.float64], q: Quaternion) -> NDArray[np.float64]:
    """
    Transform angular velocity from world frame to body-local frame.

    MATH:
    -----
    To express a world-frame vector in body-local coordinates, we rotate
    by the INVERSE of the body's orientation:

        ω_local = q⁻¹ * ω_world

    Equivalently: ω_local = Rᵀ * ω_world (where R is the rotation matrix)

    PHYSICAL MEANING:
    -----------------
    This is what a gyroscope sensor attached to the body would measure!

    Example: If the body is tilted 45° and rotating about the world Z axis:
    - ω_world = (0, 0, ωz)  [pointing straight up in world]
    - ω_local = q⁻¹ * ω_world  [pointing diagonally in body frame]

    The gyroscope "sees" the rotation axis as diagonal because the
    sensor itself is tilted.

    Args:
        omega_world: Angular velocity in world frame [rad/s]
        q: Current body orientation (world-to-body rotation)

    Returns:
        Angular velocity in body-local frame [rad/s]
    """
    return q.inverse().rotate_vector(omega_world)


def omega_local_to_world(omega_local: NDArray[np.float64], q: Quaternion) -> NDArray[np.float64]:
    """
    Transform angular velocity from body-local frame to world frame.

    MATH:
    -----
    To express a body-local vector in world coordinates, we rotate
    by the body's orientation:

        ω_world = q * ω_local

    Equivalently: ω_world = R * ω_local (where R is the rotation matrix)

    PHYSICAL MEANING:
    -----------------
    If you have gyroscope measurements (ω_local) and want to know the
    rotation in world coordinates, use this function.

    This is essential for:
    - Dead reckoning navigation
    - Sensor fusion algorithms
    - Physics simulation in world coordinates

    Args:
        omega_local: Angular velocity in body frame [rad/s] (e.g., from gyroscope)
        q: Current body orientation (world-to-body rotation)

    Returns:
        Angular velocity in world frame [rad/s]
    """
    return q.rotate_vector(omega_local)


# =============================================================================
# TRAJECTORY GENERATION
# =============================================================================

@dataclass
class Transform:
    """
    Rigid body pose: position and orientation at a given time.

    A "transform" or "pose" fully describes where a rigid body is in space:
    - translation: position of body origin in world coordinates
    - rotation: orientation of body axes relative to world axes
    """
    translation: NDArray[np.float64]  # Position [x, y, z] in world frame
    rotation: Quaternion              # Orientation (world-to-body rotation)
    time: float                       # Timestamp in seconds


def generate_trajectory(n_frames: int, duration: float) -> list[Transform]:
    """
    Generate a trajectory with interesting rotation to show frame differences.

    The trajectory combines:
    1. Circular translation in XY plane with Z oscillation
    2. Tilting (roll/pitch) that varies with position
    3. Spinning (yaw) that accumulates over time

    This creates a complex motion where world and local angular velocities
    are clearly different - demonstrating why the distinction matters.

    Args:
        n_frames: Number of frames in the trajectory
        duration: Total time duration in seconds

    Returns:
        List of Transform objects representing the trajectory
    """
    transforms = []

    for i in range(n_frames):
        t = duration * i / (n_frames - 1)
        theta = 2 * np.pi * t / duration  # Phase: one full loop over duration

        # Circular path with vertical oscillation
        translation = np.array([
            3 * np.cos(theta),       # X: circular motion
            3 * np.sin(theta),       # Y: circular motion
            0.5 * np.sin(2 * theta), # Z: oscillates twice per loop
        ])

        # Rotation: body tilts AND spins
        # This creates motion where world ≠ local frame is obvious
        rotation = Quaternion.from_euler_xyz(
            roll=0.4 * np.sin(theta),   # Tilting side to side (±23°)
            pitch=0.3 * np.cos(theta),  # Nodding up/down (±17°)
            yaw=theta,                  # Continuous spinning
        )

        transforms.append(Transform(translation=translation, rotation=rotation, time=t))

    return transforms


# =============================================================================
# RIGID BODY GEOMETRY
# =============================================================================

def create_cube_with_spike(size: float = 1.0, spike_length: float = 0.5) -> tuple[NDArray, list]:
    """
    Create vertices and edges for a cube with a spike on top.

    The spike helps visualize the body's orientation - it always
    points along the body's local +Z axis.

    Args:
        size: Side length of the cube
        spike_length: Length of the spike extending from top face

    Returns:
        tuple: (vertices array [N x 3], list of edge pairs)
    """
    h = size / 2
    # Cube vertices (centered at origin)
    vertices = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],  # Bottom face
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],       # Top face
        [0, 0, h], [0, 0, h + spike_length],                  # Spike
    ], dtype=np.float64)

    # Edges as vertex index pairs
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Spike
        (8, 9),
    ]
    return vertices, edges


def transform_vertices(vertices: NDArray, translation: NDArray, rotation: Quaternion) -> NDArray:
    """
    Apply rigid transform to a set of vertices.

    The transformation order is: rotate first, then translate.
    This is the standard convention for rigid body transforms:
        v_world = R * v_local + t

    Args:
        vertices: Array of vertices in local coordinates [N x 3]
        translation: Position offset [x, y, z]
        rotation: Orientation quaternion

    Returns:
        Transformed vertices in world coordinates [N x 3]
    """
    result = np.zeros_like(vertices)
    for i, v in enumerate(vertices):
        result[i] = rotation.rotate_vector(v) + translation
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_animated_visualization(
    transforms: list[Transform],
    omega_world: NDArray[np.float64],
    omega_local: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> go.Figure:
    """
    Create an interactive animated Plotly visualization.

    Features:
    - Play/pause button for animation
    - Slider to scrub through time
    - 3D view showing trajectory, body orientation, and angular velocity vectors
    - Merged top row: single 3D scene with body, axes, and omega vectors
    - Bottom row: 2D plots of omega world and omega local

    Args:
        transforms: List of rigid body poses
        omega_world: Angular velocity in world frame [N x 3]
        omega_local: Angular velocity in local frame [N x 3]
        timestamps: Time values [N]

    Returns:
        Plotly Figure object
    """
    n = len(transforms)
    translations = np.array([tf.translation for tf in transforms])
    vertices_base, edges = create_cube_with_spike()

    # Precompute transformed vertices for all frames
    all_vertices = []
    for tf in transforms:
        all_vertices.append(transform_vertices(vertices_base, tf.translation, tf.rotation))

    # Create figure with subplots: 1 3D scene on top, 2 2D plots on bottom
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D View: Body, Local Axes, and Angular Velocity Vectors",
            "ω World Frame (rad/s)",
            "ω Local Frame (rad/s) — Gyroscope Measurement",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        row_heights=[0.6, 0.4],
    )

    # =========================================================================
    # STATIC TRACES
    # =========================================================================

    # Trajectory path (static, faded)
    fig.add_trace(go.Scatter3d(
        x=translations[:, 0],
        y=translations[:, 1],
        z=translations[:, 2],
        mode="lines",
        line=dict(color="rgba(100, 100, 255, 0.4)", width=3),
        name="Path",
        hoverinfo="skip",
    ), row=1, col=1)

    # Static omega plots (bottom row)
    colors = ["red", "green", "blue"]
    labels = ["ωx", "ωy", "ωz"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(go.Scatter(
            x=timestamps, y=omega_world[:, i],
            mode="lines", line=dict(color=color, width=2),
            name=f"{label} (world)",
            legendgroup="world",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=timestamps, y=omega_local[:, i],
            mode="lines", line=dict(color=color, width=2),
            name=f"{label} (local)",
            legendgroup="local",
            showlegend=False,
        ), row=2, col=2)

    # =========================================================================
    # ANIMATED TRACES (3D scene)
    # =========================================================================

    initial_idx = 0
    tf = transforms[initial_idx]
    verts = all_vertices[initial_idx]

    # Body edges (cube with spike)
    edge_x, edge_y, edge_z = [], [], []
    for e in edges:
        v1, v2 = verts[e[0]], verts[e[1]]
        edge_x.extend([v1[0], v2[0], None])
        edge_y.extend([v1[1], v2[1], None])
        edge_z.extend([v1[2], v2[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="darkred", width=5),
        name="Body",
    ), row=1, col=1)

    # Body local axes
    axis_colors = ["red", "green", "blue"]
    axis_names = ["Body X", "Body Y", "Body Z"]
    axis_length = 1.5

    for ax_idx in range(3):
        local_axis = np.zeros(3)
        local_axis[ax_idx] = axis_length
        world_axis = tf.rotation.rotate_vector(local_axis)
        origin = tf.translation
        tip = origin + world_axis

        fig.add_trace(go.Scatter3d(
            x=[origin[0], tip[0]],
            y=[origin[1], tip[1]],
            z=[origin[2], tip[2]],
            mode="lines",
            line=dict(color=axis_colors[ax_idx], width=6),
            name=axis_names[ax_idx],
        ), row=1, col=1)

    # Angular velocity vector (world frame)
    omega_scale = 0.7
    ow = omega_world[initial_idx]
    origin = tf.translation
    tip = origin + omega_scale * ow

    fig.add_trace(go.Scatter3d(
        x=[origin[0], tip[0]],
        y=[origin[1], tip[1]],
        z=[origin[2], tip[2]],
        mode="lines+markers",
        line=dict(color="magenta", width=10),
        marker=dict(size=[0, 8], color="magenta"),
        name="ω_world",
    ), row=1, col=1)

    # Angular velocity vector (local frame, shown in world coords)
    ol = omega_local[initial_idx]
    ol_world = tf.rotation.rotate_vector(ol)
    tip_local = origin + omega_scale * ol_world

    fig.add_trace(go.Scatter3d(
        x=[origin[0], tip_local[0]],
        y=[origin[1], tip_local[1]],
        z=[origin[2], tip_local[2]],
        mode="lines+markers",
        line=dict(color="cyan", width=10),
        marker=dict(size=[0, 8], color="cyan"),
        name="ω_local (in world coords)",
    ), row=1, col=1)

    # Time indicator lines on 2D plots
    fig.add_trace(go.Scatter(
        x=[timestamps[initial_idx], timestamps[initial_idx]],
        y=[-3, 3],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Current time",
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[timestamps[initial_idx], timestamps[initial_idx]],
        y=[-3, 3],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=False,
    ), row=2, col=2)

    # =========================================================================
    # CREATE ANIMATION FRAMES
    # =========================================================================

    # Trace indices for animation (only the dynamic ones)
    # 0: path (static)
    # 1-6: omega world/local 2D lines (static)
    # 7: body edges
    # 8, 9, 10: body X, Y, Z axes
    # 11: omega_world vector
    # 12: omega_local vector
    # 13: time indicator line (world plot)
    # 14: time indicator line (local plot)

    frames = []
    frame_indices = list(range(0, n, 2))  # Every 2nd frame

    for idx in frame_indices:
        tf = transforms[idx]
        verts = all_vertices[idx]

        # Body edges
        edge_x, edge_y, edge_z = [], [], []
        for e in edges:
            v1, v2 = verts[e[0]], verts[e[1]]
            edge_x.extend([v1[0], v2[0], None])
            edge_y.extend([v1[1], v2[1], None])
            edge_z.extend([v1[2], v2[2], None])

        frame_data = []

        # Path (unchanged)
        frame_data.append(go.Scatter3d(
            x=translations[:, 0], y=translations[:, 1], z=translations[:, 2]
        ))

        # 2D omega lines (unchanged)
        for i in range(3):
            frame_data.append(go.Scatter(x=timestamps, y=omega_world[:, i]))
            frame_data.append(go.Scatter(x=timestamps, y=omega_local[:, i]))

        # Body edges
        frame_data.append(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z))

        # Body axes
        for ax_idx in range(3):
            local_axis = np.zeros(3)
            local_axis[ax_idx] = axis_length
            world_axis = tf.rotation.rotate_vector(local_axis)
            origin = tf.translation
            tip = origin + world_axis
            frame_data.append(go.Scatter3d(
                x=[origin[0], tip[0]],
                y=[origin[1], tip[1]],
                z=[origin[2], tip[2]],
            ))

        # Omega world vector
        ow = omega_world[idx]
        origin = tf.translation
        tip = origin + omega_scale * ow
        frame_data.append(go.Scatter3d(
            x=[origin[0], tip[0]],
            y=[origin[1], tip[1]],
            z=[origin[2], tip[2]],
        ))

        # Omega local vector (transformed to world for display)
        ol = omega_local[idx]
        ol_world = tf.rotation.rotate_vector(ol)
        tip_local = origin + omega_scale * ol_world
        frame_data.append(go.Scatter3d(
            x=[origin[0], tip_local[0]],
            y=[origin[1], tip_local[1]],
            z=[origin[2], tip_local[2]],
        ))

        # Time indicator lines
        t_current = timestamps[idx]
        frame_data.append(go.Scatter(x=[t_current, t_current], y=[-3, 3]))
        frame_data.append(go.Scatter(x=[t_current, t_current], y=[-3, 3]))

        frames.append(go.Frame(
            data=frame_data,
            name=str(idx),
            traces=list(range(15)),
        ))

    fig.frames = frames

    # =========================================================================
    # SLIDER AND PLAY BUTTON
    # =========================================================================

    slider_steps = []
    for idx in frame_indices:
        step = dict(
            method="animate",
            args=[
                [str(idx)],
                dict(
                    mode="immediate",
                    frame=dict(duration=50, redraw=True),
                    transition=dict(duration=0),
                )
            ],
            label=f"{timestamps[idx]:.1f}s",
        )
        slider_steps.append(step)

    fig.update_layout(
        title=dict(
            text="<b>World vs Local Frame Angular Velocity</b><br>" +
                 "<sup>Magenta=ω_world | Cyan=ω_local (shown in world coords) | RGB=Body XYZ axes</sup>",
            x=0.5,
            font=dict(size=18),
        ),
        scene=dict(
            xaxis=dict(range=[-5, 5], title="X"),
            yaxis=dict(range=[-5, 5], title="Y"),
            zaxis=dict(range=[-2, 2], title="Z"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.8)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.05,
                x=0.0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            )
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            )
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=14),
                    prefix="Time: ",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=60),
                len=0.85,
                x=0.15,
                y=-0.05,
                steps=slider_steps,
            )
        ],
        height=900,
        width=1100,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.9)",
        ),
    )

    # Update 2D plot axes
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="ω (rad/s)", range=[-2.5, 2.5], row=2, col=1)
    fig.update_yaxes(title_text="ω (rad/s)", range=[-2.5, 2.5], row=2, col=2)

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Main function: generate trajectory, compute angular velocities, create visualization.
    """
    print("=" * 70)
    print("RIGID BODY ANGULAR VELOCITY: WORLD vs LOCAL FRAME")
    print("=" * 70)

    # Generate trajectory
    print("\n1. Generating trajectory...")
    n_frames = 200
    duration = 8.0
    transforms = generate_trajectory(n_frames=n_frames, duration=duration)

    # Compute angular velocities
    print("2. Computing angular velocities...")
    n = len(transforms)
    timestamps = np.array([tf.time for tf in transforms])
    omega_world = np.zeros((n, 3))
    omega_local = np.zeros((n, 3))

    for i in range(n):
        # Use central differences for interior points, forward/backward at boundaries
        if i == 0:
            dt = timestamps[1] - timestamps[0]
            ow = compute_omega_world(transforms[0].rotation, transforms[1].rotation, dt)
        elif i == n - 1:
            dt = timestamps[-1] - timestamps[-2]
            ow = compute_omega_world(transforms[-2].rotation, transforms[-1].rotation, dt)
        else:
            # Central difference: more accurate
            dt = timestamps[i + 1] - timestamps[i - 1]
            ow = compute_omega_world(transforms[i - 1].rotation, transforms[i + 1].rotation, dt)

        omega_world[i] = ow
        # Convert to local frame (what a gyroscope would measure)
        omega_local[i] = omega_world_to_local(ow, transforms[i].rotation)

    # Create visualization
    print("3. Creating interactive animated visualization...")
    fig = create_animated_visualization(transforms, omega_world, omega_local, timestamps)

    # Save
    output_path = Path("world_vs_local_omega.html")
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    print(f"   Saved: {output_path}")

    # Also save CSV
    df = pd.DataFrame({
        "time": timestamps,
        "omega_world_x": omega_world[:, 0],
        "omega_world_y": omega_world[:, 1],
        "omega_world_z": omega_world[:, 2],
        "omega_local_x": omega_local[:, 0],
        "omega_local_y": omega_local[:, 1],
        "omega_local_z": omega_local[:, 2],
    })
    csv_path = Path("angular_velocity_frames.csv")
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Print key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    WORLD FRAME ANGULAR VELOCITY (ω_world):
    - Rotation expressed in fixed world axes (e.g., East-North-Up)
    - Used in physics simulations, navigation equations
    - Independent of body orientation
    
    LOCAL FRAME ANGULAR VELOCITY (ω_local):  
    - Rotation as experienced BY the body
    - THIS IS WHAT GYROSCOPES MEASURE!
    - Changes with body orientation
    
    THE RELATIONSHIP:
        ω_local = q⁻¹ * ω_world   (rotate by inverse orientation)
        ω_world = q * ω_local     (rotate by orientation)
    
    WHY IT MATTERS:
    - IMU sensors output ω_local
    - To integrate into world-frame pose, you must convert!
    - Getting this wrong is a common bug in sensor fusion
    
    OBSERVE IN THE ANIMATION:
    - Magenta vector (ω_world) points in consistent world directions
    - Cyan vector (ω_local) follows body orientation  
    - When body is tilted, these clearly differ!
    """)


if __name__ == "__main__":
    main()