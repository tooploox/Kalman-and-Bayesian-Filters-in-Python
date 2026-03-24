# We want to track an airplane in 2D (x, height) assuming a motion model that includes positions, velocities and accelerations:
# x
# x’
# x’’
# y
# y’
# y’’

# We want to measure:
# The angle and range to two radar stations, one behind and one in front of the aircraft
# Acceleration from and accelerometer

# We want to model the motion of the airplane with a standard KF, EKF, UKF and compare results.

# We want to model the moment from when the aircraft starts accelerating on the runway and takes off, and some time when it is flying up.

# To begin with we want to create a simulation that will provide the required inputs for the Kalman filter. We can try to simulate the motion as:
# Fixed acceleration in the horizontal direction (start from 0)
# Force lifting the airplane proportional to the horizontal speed, assuming known mass of plane, we can estimate the moment when the lifting force exceeds gravity



import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import norm
from math import atan2
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise


# --- Simulation helper classes ---

class RadarStation:
    def __init__(self, pos, range_std, elev_angle_std):
        self.pos = np.asarray(pos)
        self.range_std = range_std
        self.elev_angle_std = elev_angle_std

    def reading_of(self, ac_pos):
        diff = np.subtract(ac_pos, self.pos)
        range = norm(diff)
        bearing = atan2(diff[1], diff[0])
        return range, bearing

    def noisy_reading(self, ac_pos):
        range, bearing = self.reading_of(ac_pos)
        range += randn() * self.range_std
        bearing += randn() * self.elev_angle_std
        return range, bearing


class ACSim:
    def __init__(self, initial_pos, initial_vel, initial_acc):
        self.pos = np.asarray(initial_pos, dtype=float)
        self.vel = np.asarray(initial_vel, dtype=float)
        self.acc = np.asarray(initial_acc, dtype=float)
        self.initial_acc = initial_acc

    def update(self, dt):
        MASS, GRAVITY, LIFT_COEFF = 20000., 9.81, 30.0
        lift_force = LIFT_COEFF * self.vel[0] ** 2 / 2
        gravity_force = MASS * GRAVITY

        self.acc[0] = self.initial_acc[0]
        self.acc[1] = (lift_force - gravity_force) / MASS if lift_force > gravity_force else 0.0

        self.vel += self.acc * dt
        self.pos += self.vel * dt
        return self.pos


# --- UKF & EKF shared Logic ---

def h_radar(x):
    measurements = []
    for radar in radars:
        dx = x[0] - radar.pos[0]
        dy = x[3] - radar.pos[1]
        range_val = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx)
        measurements.extend([range_val, angle])
    return np.array(measurements)


def f_radar(x, dt):
    F = np.array([[1., dt, .5*dt*dt, 0,   0,        0],
                  [0., 1.,       dt, 0,   0,        0],
                  [0., 0.,       1., 0,   0,        0],
                  [0,  0,        0., 1., dt, .5*dt*dt],
                  [0,  0,        0,  0,  1.,       dt],
                  [0,  0,        0,  0,   0,       1.]], dtype=float)
    return F @ x


# --- EKF Jacobian ---

def H_jacobian(x):
    """ Partial derivatives of h(x) with respect to the state """
    def jac_for_radar(rx, ry):
        dx, dy = x[0] - rx, x[3] - ry
        dist_sq = dx**2 + dy**2
        dist = math.sqrt(dist_sq)
        # Row 1: Range derivatives [d/dx, d/dvx, d/dax, d/dy, d/dvy, d/day]
        r_row = [dx/dist, 0, 0, dy/dist, 0, 0]
        # Row 2: Angle derivatives
        a_row = [-dy/dist_sq, 0, 0, dx/dist_sq, 0, 0]
        return r_row, a_row

    jacobian_rows = []
    for radar in radars:
        r_row, a_row = jac_for_radar(radar.pos[0], radar.pos[1])
        jacobian_rows.extend([r_row, a_row])
    return np.array(jacobian_rows)


# --- Plotting Helpers ---

def plot_radar_readings(radar_readings, time):
    num_radars = len(radars)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(num_radars):
        ranges = [r[i][0] for r in radar_readings]
        angles = [np.degrees(r[i][1]) for r in radar_readings]
        axes[0].plot(time, ranges, label=f'Radar {i+1} Range')
        axes[1].plot(time, angles, label=f'Radar {i+1} Angle')
    
    axes[0].set_title('Radar Ranges')
    axes[0].legend()
    axes[1].set_title('Radar Elevation Angles')
    axes[1].legend()
    plt.show()

# --- Simulation ---

# Init simulation structures
dt, range_std, angle_std = 1.0, 50.0, math.radians(1.0)
radars = [
    RadarStation(pos=(-1000, 0), range_std=range_std, elev_angle_std=angle_std),
    RadarStation(pos=(1000, 0), range_std=range_std, elev_angle_std=angle_std),
    RadarStation(pos=(20000, 0), range_std=range_std, elev_angle_std=angle_std),
    RadarStation(pos=(15_000, 100_000), range_std=range_std, elev_angle_std=angle_std)
]
ac = ACSim(initial_pos=(0, 100), initial_vel=(0, 0), initial_acc=(5, 0))

# Calculate measurement dimension (2 measurements per radar: range and angle)
dim_z = len(radars) * 2

# Init Filters
points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=0.)
ukf = UKF(dim_x=6, dim_z=dim_z, fx=f_radar, hx=h_radar, dt=dt, points=points)
ekf = EKF(dim_x=6, dim_z=dim_z)

for f in [ukf, ekf]:
    f.x = np.zeros(6)
    f.P *= 100.
    # Create R matrix for all radars
    R_diag = []
    for _ in radars:
        R_diag.extend([range_std**2, angle_std**2])
    f.R = np.diag(R_diag)
    q_mat = Q_discrete_white_noise(3, dt=dt, var=0.1)
    f.Q[0:3, 0:3], f.Q[3:6, 3:6] = q_mat, q_mat

ekf.F = np.array([[1., dt, .5*dt*dt, 0,   0,        0],
                  [0., 1.,       dt, 0,   0,        0],
                  [0., 0.,       1., 0,   0,        0],
                  [0,  0,        0., 1., dt, .5*dt*dt],
                  [0,  0,        0,  0,  1.,       dt],
                  [0,  0,        0,  0,   0,       1.]], dtype=float)

time = np.arange(0, 100, dt)
radar_readings = []
actual_pos, ukf_est, ekf_est = [], [], []
observed_positions = [[] for _ in radars]

# Run simulation and filter
for t in time:
    pos = ac.update(dt)
    actual_pos.append(pos.copy())

    # Get readings from all radars
    radar_measurements = []
    z = []
    for i, radar in enumerate(radars):
        measurement = radar.noisy_reading(pos)
        radar_measurements.append(measurement)
        z.extend([measurement[0], measurement[1]])
        
        # Convert noisy polar back to Cartesian for "Observed Trajectory"
        obs_x = radar.pos[0] + measurement[0] * math.cos(measurement[1])
        obs_y = radar.pos[1] + measurement[0] * math.sin(measurement[1])
        observed_positions[i].append([obs_x, obs_y])
    
    radar_readings.append(radar_measurements)

    ukf.predict()
    ukf.update(z)
    ukf_est.append(ukf.x.copy())

    ekf.predict()
    ekf.update(z, H_jacobian, h_radar)
    ekf_est.append(ekf.x.copy())

# Final Visuals
plot_radar_readings(radar_readings, time)

plt.figure(figsize=(12, 6))
actual_pos, ukf_est, ekf_est = np.array(actual_pos), np.array(ukf_est), np.array(ekf_est)
plt.plot(actual_pos[:, 0], actual_pos[:, 1], 'k-', label='True Path', linewidth=2)
plt.plot(ukf_est[:, 0], ukf_est[:, 3], 'r--', label='UKF Estimate')
plt.plot(ekf_est[:, 0], ekf_est[:, 3], 'b:', label='EKF Estimate')

# Plot observed positions from all radars
for i, obs_pos in enumerate(observed_positions):
    obs_pos = np.array(obs_pos)
    plt.scatter(obs_pos[:, 0], obs_pos[:, 1], s=3, alpha=0.7, label=f'Observed (Radar {i+1})')

# Plot radar positions
for i, radar in enumerate(radars):
    plt.scatter(radar.pos[0], radar.pos[1], marker='*', s=200, color='orange', edgecolor='k', label=f'Radar {i+1} Position')

plt.title("Comparison: UKF vs EKF Trajectory Tracking")
plt.legend()
plt.show()