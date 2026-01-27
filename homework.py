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
    # Radar 1
    dx1, dy1 = x[0] - radar1.pos[0], x[3] - radar1.pos[1]
    # Radar 2
    dx2, dy2 = x[0] - radar2.pos[0], x[3] - radar2.pos[1]

    return np.array([math.sqrt(dx1 ** 2 + dy1 ** 2), math.atan2(dy1, dx1),
                     math.sqrt(dx2 ** 2 + dy2 ** 2), math.atan2(dy2, dx2)])


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

    r1_rows = jac_for_radar(radar1.pos[0], radar1.pos[1])
    r2_rows = jac_for_radar(radar2.pos[0], radar2.pos[1])
    return np.array([r1_rows[0], r1_rows[1], r2_rows[0], r2_rows[1]])


# --- Plotting Helpers ---

def plot_radar_readings(radar_readings, time):
    ranges1 = [r[0][0] for r in radar_readings]
    angles1 = [np.degrees(r[0][1]) for r in radar_readings]
    ranges2 = [r[1][0] for r in radar_readings]
    angles2 = [np.degrees(r[1][1]) for r in radar_readings]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time, ranges1, label='Radar 1 Range')
    plt.plot(time, ranges2, label='Radar 2 Range')
    plt.title('Radar Ranges')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(time, angles1, label='Radar 1 Angle')
    plt.plot(time, angles2, label='Radar 2 Angle')
    plt.title('Radar Elevation Angles')
    plt.legend()
    plt.show()

# --- Simulation ---

# Init simulation structures
dt, range_std, angle_std = 1.0, 50.0, math.radians(1.0)
radar1 = RadarStation(pos=(-1000, 0), range_std=range_std, elev_angle_std=angle_std)
radar2 = RadarStation(pos=(1000, 0), range_std=range_std, elev_angle_std=angle_std)
ac = ACSim(initial_pos=(0, 0), initial_vel=(0, 0), initial_acc=(5, 0))

# Init Filters
points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=0.)
ukf = UKF(dim_x=6, dim_z=4, fx=f_radar, hx=h_radar, dt=dt, points=points)
ekf = EKF(dim_x=6, dim_z=4)

for f in [ukf, ekf]:
    f.x = np.zeros(6)
    f.P *= 100.
    f.R = np.diag([range_std**2, angle_std**2, range_std**2, angle_std**2])
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
obs1, obs2 = [], []

# Run simulation and filter
for t in time:
    pos = ac.update(dt)
    actual_pos.append(pos.copy())

    r1_z = radar1.noisy_reading(pos)
    r2_z = radar2.noisy_reading(pos)
    radar_readings.append((r1_z, r2_z))
    z = [r1_z[0], r1_z[1], r2_z[0], r2_z[1]]

    # Convert noisy polar back to Cartesian for "Observed Trajectory"
    obs1.append([radar1.pos[0] + r1_z[0] * math.cos(r1_z[1]), radar1.pos[1] + r1_z[0] * math.sin(r1_z[1])])
    obs2.append([radar2.pos[0] + r2_z[0] * math.cos(r2_z[1]), radar2.pos[1] + r2_z[0] * math.sin(r2_z[1])])

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
obs1, obs2 = np.array(obs1), np.array(obs2)
plt.plot(actual_pos[:, 0], actual_pos[:, 1], 'k-', label='True Path', linewidth=2)
plt.plot(ukf_est[:, 0], ukf_est[:, 3], 'r--', label='UKF Estimate')
plt.plot(ekf_est[:, 0], ekf_est[:, 3], 'b:', label='EKF Estimate')
plt.scatter(obs1[:, 0], obs1[:, 1], s=3, alpha=0.7, label='Observed (Radar 1)')
plt.scatter(obs2[:, 0], obs2[:, 1], s=3, alpha=0.7, label='Observed (Radar 2)')
plt.title("Comparison: UKF vs EKF Trajectory Tracking")
plt.legend()
plt.show()