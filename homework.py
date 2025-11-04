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



from kf_book.book_plots import set_figsize, figsize
import matplotlib.pyplot as plt
from kf_book.nonlinear_plots import plot_nonlinear_func
from numpy.random import normal
import numpy as np
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from numpy.linalg import norm
from math import atan2
import math
from kf_book.ukf_internal import plot_radar
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from random import randint



class RadarStation:
    def __init__(self, pos, range_std, elev_angle_std):
        self.pos = np.asarray(pos)
        self.range_std = range_std
        self.elev_angle_std = elev_angle_std


    def reading_of(self, ac_pos):
        """ Returns (range, elevation angle) to aircraft.
        Elevation angle is in radians.
        """

        diff = np.subtract(ac_pos, self.pos)
        rng = norm(diff)
        brg = atan2(diff[1], diff[0])
        return rng, brg


    def noisy_reading(self, ac_pos):
        """ Compute range and elevation angle to aircraft with
        simulated noise"""

        rng, brg = self.reading_of(ac_pos)
        rng += randn() * self.range_std
        brg += randn() * self.elev_angle_std
        return rng, brg


class ACSim:
    def __init__(self, initial_pos, initial_vel, initial_acc, acc_std):
        self.pos = np.asarray(initial_pos, dtype=float)
        self.vel = np.asarray(initial_vel, dtype=float)
        self.acc = np.asarray(initial_acc, dtype=float)
        self.initial_acc = initial_acc
        self.acc_std = acc_std

    def update(self, dt):
        """ Compute and returns next position. Incorporates
        random variation in velocity. """

        MASS = 20000. # kg
        GRAVITY = 9.81 # m/s^2
        LIFT_COEFF = 30.0 # made up

        lift_force = LIFT_COEFF * self.vel[0]**2 / 2
        gravity_force = MASS * GRAVITY

        # Constant acceleration in x direction
        self.acc[0] = self.initial_acc[0]

        # Vertical acceleration proportional to lifting force
        if lift_force > gravity_force:
            self.acc[1] = (lift_force - gravity_force) / MASS

        # Add noise to acceleration
        r = randn()
        # print(r)
        self.acc[0] = self.acc[0] + (r * self.acc_std)

        # Update velocity
        self.vel += self.acc * dt

        # Update position
        self.pos += self.vel * dt

        # print(f"pos: {self.pos}, vel: {self.vel}, acc: {self.acc}, lift: {lift_force}, gravity: {gravity_force}")

        return self.pos


    def get_acceleration(self, sensor_std):
        return self.acc + randn() * sensor_std



def plot_radar_readings(radar_readings, time):
    ranges1 = [r[0][0] for r in radar_readings]
    angles1 = [np.degrees(r[0][1]) for r in radar_readings]

    ranges2 = [r[1][0] for r in radar_readings]
    angles2 = [np.degrees(r[1][1]) for r in radar_readings]

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(time, ranges1, label='Radar 1 Range')
    plt.plot(time, ranges2, label='Radar 2 Range')
    plt.xlabel('Time Step')
    plt.ylabel('Range (m)')
    plt.title('Radar Ranges Over Time')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(time, angles1, label='Radar 1 Elevation Angle')
    plt.plot(time, angles2, label='Radar 2 Elevation Angle')
    plt.xlabel('Time Step')
    plt.ylabel('Elevation Angle (degrees)')
    plt.title('Radar Elevation Angles Over Time')
    plt.legend()

    plt.tight_layout()

def plot_filter_estimates(positions, filter_estimates, radar1_pos, radar2_pos, time):
    plt.figure(figsize=(10,5))
    for idx, pos_list in enumerate(positions):
        pos_list = np.array(pos_list)
        print(pos_list.shape)
        print(pos_list)
        plt.plot(pos_list[:,0], pos_list[:,1], label=f'{idx}', c='green')
    for idx, flt_list in enumerate(filter_estimates):
        flt_list = np.array(flt_list)
        print(flt_list.shape)
        print(flt_list)
        plt.plot(flt_list[:,0], flt_list[:,1], label=f'filter {idx}', c='blue')    

    plt.scatter(radar1_pos[0], radar1_pos[1], c='red', marker='x', label='Radar 1')
    plt.scatter(radar2_pos[0], radar2_pos[1], c='green', marker='x', label='Radar 2')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.legend()
    plt.title('ACSim Position Over Time')


# stds = [0, 5, 5, 5, 5, 5, 5, 5, 5, 5]

def plot_acsim_position(positions, radar1_pos, radar2_pos):
    plt.figure(figsize=(10,5))
    for idx, pos_list in enumerate(positions):
        pos_list = np.array(pos_list)
        print(pos_list.shape)
        print(pos_list)
        plt.plot(pos_list[:,0], pos_list[:,1], label=f'{idx}')

    plt.scatter(radar1_pos[0], radar1_pos[1], c='red', marker='x', label='Radar 1')
    plt.scatter(radar2_pos[0], radar2_pos[1], c='green', marker='x', label='Radar 2')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.legend()
    plt.title('ACSim Position Over Time')


# from filterpy.kalman import KalmanFilter

# def SecondOrderKF(R_std, Q, dt, P=100):
#     """ Create second order Kalman filter. 
#     Specify R and Q as floats."""
    
#     kf = KalmanFilter(dim_x=6, dim_z=2)
#     kf.x = np.zeros(6)
#     kf.P[0, 0] = P
#     kf.P[1, 1] = 1
#     kf.P[2, 2] = 1
#     kf.P[3, 3] = P
#     kf.P[4, 4] = 1
#     kf.P[5, 5] = 1
#     kf.R *= R_std**2
#     kf.Q = Q_discrete_white_noise(6, dt, Q)
#     kf.F = np.array([[1., dt, .5*dt*dt, 0,   0,        0],
#                      [0., 1.,       dt, 0,   0,        0],
#                      [0., 0.,       1., 0,   0,        0],
#                      [0,  0,        0., 1., dt, .5*dt*dt],
#                      [0,  0,        0,  0,  1.,       dt],
#                      [0,  0,        0,  0,   0,       1.]])
#     kf.H = np.array([[1., 0., 0., 1., 0., 0.]])
#     return kf

# kf2 = SecondOrderKF(R, 0, dt=1)

dt = 1. # 12 seconds between readings
range_std = 5 # meters
elevation_angle_std = math.radians(0.5)

radar1 = RadarStation(pos=(-10000, 0), range_std=range_std, elev_angle_std=elevation_angle_std)
radar2 = RadarStation(pos=(10000, 0), range_std=range_std, elev_angle_std=elevation_angle_std)
ac = ACSim(initial_pos=(0, 0), initial_vel=(0, 0), initial_acc=(5, 0), acc_std=5)

def h_radar(x):
    
    dx = x[0] - radar1.pos[0]
    dy = x[3] - radar1.pos[1]

    slant_range = math.sqrt(dx**2 + dy**2)
    elevation_angle = math.atan2(dy, dx)
    return [slant_range, elevation_angle]

def f_radar(x, dt):
    """ state transition function for a constant velocity 
    aircraft"""
    F = np.array([[1., dt, .5*dt*dt, 0,   0,        0],
                  [0., 1.,       dt, 0,   0,        0],
                  [0., 0.,       1., 0,   0,        0],
                  [0,  0,        0., 1., dt, .5*dt*dt],
                  [0,  0,        0,  0,  1.,       dt],
                  [0,  0,        0,  0,   0,       1.]], dtype=float)
    return F @ x

def create_UKF(R_std):
    points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=-1.)
    kf = UKF(6, len(R_std), dt, fx=f_radar, hx=h_radar, points=points)

    kf.Q[0:3, 0:3] = Q_discrete_white_noise(3, dt=dt, var=0.1)
    kf.Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=dt, var=0.1)
    kf.R = np.diag(R_std)
    kf.R = kf.R @ kf.R  # square to get variance
    kf.x = np.array([0., 0., 0., 0., 0., 0.])
    P_initial = 100**2
    kf.P = np.diag([P_initial, P_initial, P_initial, P_initial, P_initial, P_initial])
    return kf


range_std = 5 # meters
elevation_angle_std = math.radians(0.5)
KF_UKF = create_UKF(R_std=[range_std, elevation_angle_std]) # TODO expand R_std to 4 measurements

time = np.arange(0, 100, dt)
xs = []
positions = []
radar_readings = []
for _ in time:
    ac.update(dt)
    r1_reading = radar1.noisy_reading(ac.pos)
    r2_reading = radar2.noisy_reading(ac.pos)
    acc_reading = ac.get_acceleration(sensor_std=0.2)

    positions.append(ac.pos.copy())
    radar_readings.append((r1_reading, r2_reading))

    # Process data from this iteation through the filter
    KF_UKF.update(radar_readings[-1])
    KF_UKF.predict()
    xs.append(KF_UKF.x)

plot_radar_readings(radar_readings, time)
plot_filter_estimates(positions, xs, radar1.pos, radar2.pos, time)
plot_acsim_position([positions], radar1.pos, radar2.pos)
plt.show()


#data2 = filter_data(kf2, radar_readings)