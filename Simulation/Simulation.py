# Add main directory to path
import os
import pdb
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Import
from dataclasses import dataclass
import argparse
import math
from pylie import SO3
import progressbar
import numpy as np
import pandas as pd
from Filters.Calibrated.SE3_se3_R3_EqF import SE3_se3_R3_EqF
from Filters.Calibrated.IEKF import IEKF
from Filters.Calibrated.LIEKF import LIEKF
from Filters.Calibrated.TF_IEKF import TF_IEKF
from Filters.Calibrated.TF_LIEKF import TF_LIEKF
from Filters.Calibrated.MEKF import MEKF
# from Filters.Calibrated.UKF import UKF
from Filters.Calibrated.SE23_se3_EqF import SE23_se3_EqF
from Filters.Calibrated.SE23_se23_EqF import SE23_se23_EqF
from Filters.Calibration.SE3_se3_R3_cal_EqF import SE3_se3_R3_cal_EqF
from Filters.Calibration.cal_IEKF import cal_IEKF
from Filters.Calibration.cal_TF_IEKF import cal_TF_IEKF
from Filters.Calibration.SE23_se23_cal_EqF import SE23_se23_cal_EqF
from Filters.Calibration.SE23_se3_cal_EqF import SE23_se3_cal_EqF
from Filters.Calibration.cal_MEKF import cal_MEKF
from Utils.utils import *
from scipy.spatial.transform import Rotation
from scipy.io import savemat
from threadpoolctl import threadpool_limits
import warnings
import yaml

# Argument parser
parser = argparse.ArgumentParser("Load dataset(s) for SE23_se23_EqF navigation.")
parser.add_argument("data_path", metavar='m', help="The dataset file name or the folder name.")
parser.add_argument("result_path", metavar='m', help="The folder name where result are stored.")
parser.add_argument("--ct", action='store_true', help="Run SE3_se3_R3 EqF.")
parser.add_argument("--ctex", action='store_true', help="Run SE23_se23 EqF.")
parser.add_argument("--ctnew", action='store_true', help="Run SE23_se3 EqF.")
parser.add_argument("--iekf", action='store_true', help="Run IEKF and LIEKF.")
parser.add_argument("--tfiekf", action='store_true', help="Run Two_Frames IEKF and LIEKF.")
parser.add_argument("--mekf", action='store_true', help="Run MEKF.")
parser.add_argument("--ct_cal", action='store_true', help="Run SE3_se3_R3_cal EqF.")
parser.add_argument("--ctex_cal", action='store_true', help="Run SE23_se23_cal EqF.")
parser.add_argument("--ctnew_cal", action='store_true', help="Run SE23_se3_cal EqF.")
parser.add_argument("--iekf_cal", action='store_true', help="Run cal_IEKF.")
parser.add_argument("--tfiekf_cal", action='store_true', help="Run Two_Frames_cal IEKF.")
parser.add_argument("--mekf_cal", action='store_true', help="Run cal_MEKF.")
parser.add_argument("--correctinit", action='store_true', help="Initialize filter with ground-truth.")
parser.add_argument("--propagationonly", action='store_true', help="Avoid filter updates")
parser.add_argument("--curvature_correction", action='store_true', help="Apply covariance curvature correction")
parser.add_argument("--equivariant_output", action='store_true', help="Reformulate output as equivariant whnever possible")
parser.add_argument("--threads", type=int, default=1, help="number of threads for NumPy to use")
parser.add_argument("--insane", action='store_true', help="Run Insane simulation")
args = parser.parse_args()


if (args.insane):
    @dataclass
    class Imu:
        t: float = 0.0
        w: np.ndarray = np.zeros((3, 1))
        a: np.ndarray = np.zeros((3, 1))


    @dataclass
    class Gps:
        t: float = 0.0
        yp: np.ndarray = np.zeros((3, 1))


    @dataclass
    class Groundtruth:
        t: float = 0.0
        R: SO3 = SO3.identity()
        p: np.ndarray = np.zeros((3, 1))
        v: np.ndarray = np.zeros((3, 1))
        bw: np.ndarray = np.zeros((3, 1))
        ba: np.ndarray = np.zeros((3, 1))
        cal: np.ndarray = np.zeros((3, 1))

        def getStateTuple(self):
            return (self.R.as_matrix(), self.v, self.p, self.bw, self.ba, self.cal)

        def getStateList(self):
            return np.concatenate((
                self.R.as_quaternion().reshape((4, 1)),
                self.v,
                self.p,
                self.bw,
                self.ba,
                self.cal))


    class Data:
        def __init__(self, data_class_type):
            self.data_class_type = data_class_type
            self.data_dict = {}

        def append(self, data):
            if isinstance(data, self.data_class_type):
                self.data_dict[data.t] = data
            else:
                raise ValueError("Wrong data type")

        def at(self, t):
            return self.data_dict.get(t)

        def closest(self, t):
            if not self.data_dict:
                raise BufferError("Data is empty")
            closest_t = min(self.data_dict.keys(), key=lambda x: abs(x - t))
            return self.data_dict[closest_t]

        def timestamps(self):
            if not self.data_dict:
                raise BufferError("Data is empty")
            return list(self.data_dict.keys())

        def count(self):
            return len(self.data_dict.keys())


    class ImuData(Data):
        def __init__(self):
            super().__init__(Imu)


    class GpsData(Data):
        def __init__(self):
            super().__init__(Gps)


    class GroundtruthData(Data):
        def __init__(self):
            super().__init__(Groundtruth)


    # setup global variables
    b_gnss_reference_is_set = False  # determines if a GNSS reference has been set
    gnss_reference = Gps()  # global GNSS reference frame
    lever_arm = np.zeros((3, 1))  # GNSS lever arm


    def readCSV(path: str,
                t_start: float = 36.0,
                t_duration: float = float(sys.maxsize),
                num_gnss: int = 2):

        # Define data
        gt_data = GroundtruthData()
        imu_data = ImuData()
        gps_data = GpsData()

        t_ref = 0.0
        time_info: dict

        # output visualization
        bar = progressbar.ProgressBar()

        # read configuration from yaml
        print("[LOAD] Loading time configuration", flush=True)
        if os.path.isfile(path + "/time_info.yaml"):
            fh = open(f'{path}/time_info.yaml', 'r')
            time_info = yaml.safe_load(fh)
            print(float(time_info['t_pximu_imugt']))
            fh.close()
        else:
            raise FileNotFoundError("INSANE 'time_info.yaml' not found.")

        # read IMU Data
        print("[LOAD] Loading IMU Data", flush=True)
        if os.path.isfile(path + "/px4_imu.csv"):
            df = pd.read_csv(path + "/px4_imu.csv", skipinitialspace=True)
            df = df.reset_index()

            # setup t_ref to be the initial IMU measurement
            t_ref = float(df['t'][0])

            # update progressbar
            bar.maxval = len(df)
            bar.start()

            # read data from frame
            index: int = 0
            for index, row in df.iterrows():
                data = Imu()
                data.t = float(row['t']) + float(time_info['t_pximu_imugt'])

                # check if data (assumed sorted) is out after duration
                if data.t - t_ref > t_start + t_duration:
                    break

                # only add data if it is after t_start
                elif not data.t - t_ref < t_start:
                    # Load IMU inputs from csv row as vectors
                    data.w = np.array([float(row['w_x']), float(row['w_y']), float(row['w_z'])]).reshape((3, 1))
                    data.a = np.array([float(row['a_x']), float(row['a_y']), float(row['a_z'])]).reshape((3, 1))

                    # append to unsorted data list
                    imu_data.append(data)

                # finally update index
                bar.update(index)

            # stop progressbar
            bar.finish()
        else:
            raise FileNotFoundError("INSANE 'px4_imu.csv' not found.")

        # read GT Data
        print("[LOAD] Loading Ground-Truth Data", flush=True)
        if os.path.isfile(path + "/ground_truth/ground_truth_80hz.csv"):
            df = pd.read_csv(path + "/ground_truth/ground_truth_80hz.csv", skipinitialspace=True)
            df = df.reset_index()

            # update progressbar
            bar.maxval = len(df)
            bar.start()

            # read data from frame
            for index, row in df.iterrows():
                data = Groundtruth()
                data.t = float(row['t']) + float(time_info['t_pximu_imugt'])

                # check if data (assumed sorted) is out after duration
                if data.t - t_ref > t_start + t_duration:
                    break

                # only add data if it is after t_start
                elif not data.t - t_ref < t_start:
                    # Load GT Data from csv row as vectors
                    data.p = np.array([float(row['p_x']), float(row['p_y']), float(row['p_z'])]).reshape((3, 1))
                    quat = np.array([float(row['q_x']), float(row['q_y']), float(row['q_z']), float(row['q_w'])])
                    data.R = SO3.from_matrix(Rotation.from_quat(quat).as_matrix())

                    # add velocity GT if existent
                    if 'v_x' in row:
                        data.v = np.array([float(row['v_x']), float(row['v_y']), float(row['v_z'])]).reshape((3, 1))

                    # append to unsorted data list
                    gt_data.append(data)

                # finally update index
                bar.update(index)

            # stop progressbar
            bar.finish()
        else:
            raise FileNotFoundError("INSANE 'ground_truth/ground_truth_80hz.csv' not found.")

        global b_gnss_reference_is_set
        global gnss_reference
        global lever_arm

        print("[LOAD] Loading GNSS" + str(num_gnss) + " Data", flush=True)

        if num_gnss == 1 or num_gnss == 2:
            filename = "/rtk_gps" + str(num_gnss) + ".csv"
        elif num_gnss == 3:
            filename = "/px4_gps.csv"
        else:
            raise ValueError("Non existing GNSS")

        if os.path.isfile(path + filename):
            df = pd.read_csv(path + filename, skipinitialspace=True)
            df = df.reset_index()

            if num_gnss == 1:
                lever_arm = np.array([0.350121933088198, 0.410121933088197, 0.0]).reshape((3, 1))
            elif num_gnss == 2:
                lever_arm = np.array([-0.470121933088198, -0.410121933088197, 0.0]).reshape((3, 1))
            elif num_gnss == 3:
                lever_arm = np.array([0.0, 0.0, 0.21]).reshape((3, 1))
            else:
                raise ValueError("Non existing GNSS")

            # update progressbar
            bar.maxval = len(df)
            bar.start()

            # read data from frame
            for index, row in df.iterrows():
                data = Gps()
                data.t = float(row['t'])
                if num_gnss == 3:
                    data.t += float(time_info['t_pximu_imugt'])

                # check if data (assumed sorted) is out after duration
                if data.t - t_ref > t_start + t_duration:
                    break

                # only add data if it is after t_start
                elif not data.t - t_ref < t_start:
                    if not b_gnss_reference_is_set:
                        gnss_reference.t = data.t
                        closestgt = gt_data.closest(gnss_reference.t)
                        if closestgt is not None and abs(closestgt.t - gnss_reference.t) < 0.2:
                            gnss_reference.yp = np.array(
                                [float(row['p_x']), float(row['p_y']), float(row['p_z'])]).reshape((3, 1)) - (
                                                            closestgt.R @ lever_arm)
                            b_gnss_reference_is_set = True
                        else:
                            raise ValueError("R not found for compensating GNSS ref with lever arm")
                    else:
                        closestgt = gt_data.closest(gnss_reference.t)
                        if closestgt is not None and abs(closestgt.t - gnss_reference.t) < 0.2:
                            data.yp = (np.array([float(row['p_x']), float(row['p_y']), float(row['p_z'])]).reshape(
                                (3, 1))) - (closestgt.R @ lever_arm) - gnss_reference.yp
                        else:
                            raise ValueError("R not found for compensating GNSS measurement with lever arm")

                    # append to unsorted data list
                    gps_data.append(data)

                # finally update index
                bar.update(index)

            # stop progressbar
            bar.finish()
        else:
            raise FileNotFoundError("INSANE '/rtk_gps" + str(num_gnss) + ".csv' not found.")
        pass

        # Compensate gt position
        offset = np.zeros((3, 1))
        # offset = gt_data.at(gt_data.timestamps()[0]).p - gnss_reference.yp
        for stamp in gt_data.timestamps():
            gt_data.at(stamp).p -= (gnss_reference.yp + offset)

        return gt_data, imu_data, gps_data


    def run_filter(filter_class, filter_args, data, mat_filepath):

        gt_data, imu_data, gps_data = data

        # Define progressbar
        p = progressbar.ProgressBar()

        # Define Filter
        filter = filter_class(*filter_args)

        # timestamps
        timestamps = sorted(list(set([*imu_data.timestamps(), *gps_data.timestamps()])), key=float)

        # Preallocate gt list
        time = []
        groundtruth = []

        # Preallocate estimated state list
        R_est = []
        p_est = []
        v_est = []
        bw_est = []
        ba_est = []
        bmu_est = []
        cal_est = []

        # Preallocate filter energy list
        filter_energy = []
        filter_energy_nav = []
        filter_energy_pose = []
        filter_energy_bias = []

        # Preallocate errors
        attitude_error = []
        position_error = []
        velocity_error = []
        bw_error = []
        ba_error = []
        bmu_error = []
        cal_error = []

        # check var
        last_imu_t = None

        global gnss_reference

        # Filter loop
        for t in p(timestamps):

            imu = imu_data.at(t)
            if imu is not None:
                last_imu_t = t
                vel = np.vstack((imu.w, imu.a))
                try:
                    filter.propagate(t, vel, omega_noise, acc_noise, tau_noise, 0.0)
                except ValueError as e:
                    print('Filter propagation Error')

            gps = gps_data.at(t)
            if gps is not None:
                if abs(t - last_imu_t) < 0.0075:
                    try:
                        nis = filter.update(np.array([]), omega_noise, acc_noise, tau_noise, 0.0, gps.yp, meas_noise, 0.0, False)
                        # print(nis)
                        filter_energy.append(nis)
                    except ValueError as e:
                        print('Filter GNSS update Error')
                else:
                    print('Out of order GNSS measurement')

                gt = gt_data.closest(t)
                if gt is not None and abs(gt.t - t) < 0.2:
                    # Get gt
                    time.append(gt.t)
                    groundtruth.append(gt.getStateList())

                    # Get estimate, filter energy and errors
                    R, p, v, bw, ba, bmu, cal = filter.getEstimate()

                    # Assign estimate
                    R_est.append(R)
                    p_est.append(p)
                    v_est.append(v)
                    bw_est.append(bw)
                    ba_est.append(ba)
                    bmu_est.append(bmu)
                    cal_est.append(cal)

                    # # Get and assign filter energy
                    # nees, nav_ness, pose_nsee, bias_ness = filter.computeNEES(gt.getStateTuple())
                    # filter_energy.append(nees / filter.dof)
                    # filter_energy_nav.append(nav_ness / 9)
                    # filter_energy_pose.append(pose_nsee / 6)
                    # filter_energy_bias.append(bias_ness / (filter.dof - 9))

                    # Compute errors
                    attitude_error.append(np.linalg.norm(SO3.log(SO3.from_matrix(gt.R @ R.T))))
                    position_error.append(np.linalg.norm(gt.p - p))
                    velocity_error.append(np.linalg.norm(gt.v - v))
                    bw_error.append(np.linalg.norm(gt.bw - bw))
                    ba_error.append(np.linalg.norm(gt.ba - ba))
                    bmu_error.append(np.linalg.norm(np.zeros((3, 1)) - bmu))
                    cal_error.append(np.linalg.norm(gt.cal - cal))

        # Delete if file exist
        if os.path.exists(mat_filepath):
            print("Deleting existing file\n")
            os.remove(mat_filepath)

        # Save as mat file and return
        print(f"Saving results to {mat_filepath} ...\n")
        savemat(mat_filepath, dict(t=time,
                                   groundtruth=groundtruth,
                                   R_est=listArrayToListOfLists(R_est),
                                   p_est=listArrayToListOfLists(p_est),
                                   v_est=listArrayToListOfLists(v_est),
                                   bw_est=listArrayToListOfLists(bw_est),
                                   ba_est=listArrayToListOfLists(ba_est),
                                   bmu_est=listArrayToListOfLists(bmu_est),
                                   cal_est=listArrayToListOfLists(cal_est),
                                   attitude_error=listArrayToListOfLists(attitude_error),
                                   position_error=listArrayToListOfLists(position_error),
                                   velocity_error=listArrayToListOfLists(velocity_error),
                                   bw_error=listArrayToListOfLists(bw_error),
                                   ba_error=listArrayToListOfLists(ba_error),
                                   bmu_error=listArrayToListOfLists(bmu_error),
                                   cal_error=listArrayToListOfLists(cal_error),
                                   filter_energy=filter_energy,
                                   filter_energy_nav=filter_energy_nav,
                                   filter_energy_bias=filter_energy_bias))

        return data, R_est, p_est, v_est, bw_est, ba_est, bmu_est, cal_est, attitude_error, position_error, velocity_error, bw_error, ba_error, bmu_error, cal_error, filter_energy

else:
    @dataclass
    class Data:
        # Ground-truth state
        R: SO3 = SO3.identity()
        p: np.ndarray = np.zeros((3, 1))
        v: np.ndarray = np.zeros((3, 1))
        bw: np.ndarray = np.zeros((3, 1))
        ba: np.ndarray = np.zeros((3, 1))
        cal: np.ndarray = np.zeros((3, 1))

        # Input measurements
        w: np.ndarray = np.zeros((3, 1))
        a: np.ndarray = np.zeros((3, 1))

        # Output measurements
        yp: np.ndarray = np.zeros((3, 1))

        # Time
        t: float = 0.0
        dt: float = 0.0

        def getStateVec(self) -> np.ndarray:
            return np.vstack((self.R.as_euler().reshape(3, 1), self.p, self.v, self.bw, self.ba, self.cal))

        def getStateTuple(self):
            return (self.R.as_matrix(), self.v, self.p, self.bw, self.ba, self.cal)

        def getStateList(self):
            return np.concatenate((self.R.as_quaternion().reshape((4, 1)),
                                   self.v,
                                   self.p,
                                   self.bw,
                                   self.ba,
                                   self.cal))
        def getOutputVec(self) -> np.ndarray:
            return np.vstack((self.yp))

        def getInputVec(self) -> np.ndarray:
            return np.vstack((self.w, self.a))

    # Read data from csv and parse into common Data structure
    def readCSV(p_name):
        # read .csv file into pandas dataframe
        df = pd.read_csv(p_name)
        df = df.reset_index()

        # Define data_list as list
        data_list = []

        df['tnext'] = df['t'].shift(-1)

        # Check for existance of bias groundtruth into data_list
        bias_exist = False
        if {'b_a_x', 'b_a_y', 'b_a_z', 'b_w_x', 'b_w_y', 'b_w_z'}.issubset(df.columns):
            bias_exist = True

        # Check for existance of calibration groundtruth into data_list
        cal_exist = False
        if {'cal_x', 'cal_y', 'cal_z'}.issubset(df.columns):
            cal_exist = True

        rowit = df.iterrows()
        for index, row in rowit:

            # Define Data structure
            d = Data()

            # Load timestamps and record dt
            d.t = float(row['t'])
            d.tnext = float(row['tnext'])
            d.dt = d.tnext - d.t

            # Skip data_list if dt is smaller than a micro second
            if d.dt < 1e-6 or d.dt > 1e6 or np.isnan(d.dt):
                continue

            # Load groundtruth values from csv row as vectors
            d.p = np.array([float(row['p_x']), float(row['p_y']), float(row['p_z'])]).reshape((3, 1))
            d.v = np.array([float(row['v_x']), float(row['v_y']), float(row['v_z'])]).reshape((3, 1))
            quat = np.array([float(row['q_x']), float(row['q_y']), float(row['q_z']), float(row['q_w'])])
            d.R = SO3.from_matrix(Rotation.from_quat(quat).as_matrix())

            # Load IMU biases
            if bias_exist:
                d.bw = np.array([float(row['b_w_x']), float(row['b_w_y']), float(row['b_w_z'])]).reshape((3, 1))
                d.ba = np.array([float(row['b_a_x']), float(row['b_a_y']), float(row['b_a_z'])]).reshape((3, 1))

            # Load GNSS calibration
            if cal_exist:
                d.cal = np.array([float(row['cal_x']), float(row['cal_y']), float(row['cal_z'])]).reshape((3, 1))

            # Load IMU inputs from csv row as vectors
            d.w = np.array([float(row['w_x']), float(row['w_y']), float(row['w_z'])]).reshape((3, 1))
            d.a = np.array([float(row['a_x']), float(row['a_y']), float(row['a_z'])]).reshape((3, 1))

            # Load measurements from csv row as vector
            d.yp = np.array([float(row['y_x']), float(row['y_y']), float(row['y_z'])]).reshape((3, 1))

            # Append to data_list list
            data_list.append(d)

        return data_list

    def run_filter(filter_class, filter_args, data, mat_filepath):

        # Define progressbar
        p = progressbar.ProgressBar()

        # Define Filter
        filter = filter_class(*filter_args)

        # Preallocate gt list
        t = []
        gt = []

        # Preallocate estimated state list
        R_est = []
        p_est = []
        v_est = []
        bw_est = []
        ba_est = []
        bmu_est = []
        cal_est = []

        # Preallocate filter energy list
        filter_energy = []
        filter_energy_nav = []
        filter_energy_bias = []

        # Preallocate errors
        attitude_error = []
        position_error = []
        velocity_error = []
        bw_error = []
        ba_error = []
        bmu_error = []
        cal_error = []

        # Filter loop
        for d in p(data):

            # Get gt
            t.append(d.t)
            gt.append(d.getStateList())

            # Get estimate, filter energy and errors
            R, p, v, bw, ba, bmu, cal = filter.getEstimate()

            # Assign estimate
            R_est.append(R)
            p_est.append(p)
            v_est.append(v)
            bw_est.append(bw)
            ba_est.append(ba)
            bmu_est.append(bmu)
            cal_est.append(cal)

            # Get and assign filter energy
            nees, nav_ness, bias_ness = filter.computeNEES(d.getStateTuple())
            filter_energy.append(nees / filter.dof)
            filter_energy_nav.append(nav_ness / 9)
            filter_energy_bias.append(bias_ness / (filter.dof - 9))

            # Compute errors
            attitude_error.append(np.linalg.norm(SO3.log(SO3.from_matrix(d.R @ R.T))))
            position_error.append(np.linalg.norm(d.p - p))
            velocity_error.append(np.linalg.norm(d.v - v))
            bw_error.append(np.linalg.norm(d.bw - bw))
            ba_error.append(np.linalg.norm(d.ba - ba))
            bmu_error.append(np.linalg.norm(np.zeros((3, 1)) - bmu))
            cal_error.append(np.linalg.norm(d.cal - cal))

            # Run filter
            try:
                filter.update(np.vstack((d.w, d.a)), omega_noise, acc_noise, tau_noise, virtual_noise, d.yp, meas_noise, d.dt)
            except ValueError as e:
                print('Filter.update Error\n')

        # Delete if file exist
        if os.path.exists(mat_filepath):
            print("Deleting existing file\n")
            os.remove(mat_filepath)

        # Save as mat file and return
        print(f"Saving results to {mat_filepath} ...\n")
        savemat(mat_filepath, dict(t=t,
                                   groundtruth=gt,
                                   R_est=listArrayToListOfLists(R_est),
                                   p_est=listArrayToListOfLists(p_est),
                                   v_est=listArrayToListOfLists(v_est),
                                   bw_est=listArrayToListOfLists(bw_est),
                                   ba_est=listArrayToListOfLists(ba_est),
                                   bmu_est=listArrayToListOfLists(bmu_est),
                                   cal_est=listArrayToListOfLists(cal_est),
                                   attitude_error=listArrayToListOfLists(attitude_error),
                                   position_error=listArrayToListOfLists(position_error),
                                   velocity_error=listArrayToListOfLists(velocity_error),
                                   bw_error=listArrayToListOfLists(bw_error),
                                   ba_error=listArrayToListOfLists(ba_error),
                                   bmu_error=listArrayToListOfLists(bmu_error),
                                   cal_error=listArrayToListOfLists(cal_error),
                                   filter_energy=filter_energy,
                                   filter_energy_nav=filter_energy_nav,
                                   filter_energy_bias=filter_energy_bias))

        return data, R_est, p_est, v_est, bw_est, ba_est, bmu_est, cal_est, attitude_error, position_error, velocity_error, bw_error, ba_error, bmu_error, cal_error, filter_energy


if __name__ == '__main__':

    # Fixed noise parameters
    np.random.seed(0)

    if args.insane:
        initial_att_noise = 2.0
        initial_vel_noise = 2.0
        initial_pos_noise = 2.0
        initial_bias_noise = 0.05
        omega_noise = 1.0e-1 * math.pi / 180
        acc_noise = 5.0e-2
        tau_noise = 1.0e-3
        virtual_noise = 1e-9
        meas_noise = 0.2
    else:
        initial_att_noise = 0.25
        initial_vel_noise = 0.1
        initial_pos_noise = 1.0
        initial_bias_noise = 0.01
        # omega_noise = 1.5e-3 * math.pi / 180
        # acc_noise = 1.5e-2
        omega_noise = 1.0e-3 * math.pi / 180  # Automatica
        acc_noise = 1.2e-2  # Automatica
        tau_noise = 1.0e-4
        virtual_noise = 1e-9
        meas_noise = 0.2

    # Special filter parameters
    measure_bv = False

    if args.correctinit:
        warnings.warn("xi_0 is different from identity! Hardcoded matrices works with xi_0 = identity!\n")

    if (args.ct == args.iekf == args.ctex == args.ctnew == args.tfiekf == args.mekf == args.ct_cal == args.iekf_cal == args.mekf_cal == args.ctex_cal == args.ctnew_cal == args.tfiekf_cal == args.ctnew_cal_equi == False):
        raise ValueError("Please specify which filter to be run")

    # Create the folder for saving results
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    filter_args = [initial_att_noise, initial_vel_noise, initial_pos_noise, initial_bias_noise, args.propagationonly, args.equivariant_output, args.curvature_correction]

    print("Read data from .csv file\n")
    if not (args.insane or os.path.isfile(args.data_path)):
        for file in os.listdir(args.data_path):
            if file.endswith(".csv"):
                print(f"Loading dataset: {file}\n")
                data = readCSV(os.path.join(args.data_path, file))
            else:
                raise ValueError("Please make sure the specified folder contains .csv files\n")

            result_path = args.result_path + '/' + os.path.splitext(file)[0]
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            else:
                cond_1 = (os.path.exists(result_path + '/result_MEKF.mat') or not args.iekf) and \
                         (os.path.exists(result_path + '/result_IEKF.mat') or os.path.exists(result_path + '/result_IEKF_equi.mat') or os.path.exists(result_path + '/result_LIEKF.mat') or not args.iekf) and \
                         (os.path.exists(result_path + '/result_TF_IEKF.mat') or os.path.exists(result_path + '/result_TF_IEKF_equi.mat') or os.path.exists(result_path + '/result_TF_LIEKF.mat') or not args.tfiekf) and \
                         (os.path.exists(result_path + '/result_SE3_se3_R3_EqF.mat') or not args.ct) and \
                         (os.path.exists(result_path + '/result_SE23_se23_EqF.mat') or os.path.exists(result_path + '/result_SE23_se23_equi_EqF.mat') or not args.ctex) and \
                         (os.path.exists(result_path + '/result_SE23_se3_EqF.mat') or os.path.exists(result_path + '/result_SE23_se3_equi_EqF.mat') or not args.ctnew)
                cond_2 = (os.path.exists(result_path + '/result_cal_MEKF.mat') or not args.mekf_cal) and \
                         (os.path.exists(result_path + '/result_cal_IEKF_equi.mat') or os.path.exists(result_path + '/result_cal_IEKF.mat') or not args.iekf_cal) and \
                         (os.path.exists(result_path + '/result_cal_TF_IEKF_equi.mat') or os.path.exists(result_path + '/result_cal_TF_IEKF.mat') or not args.tfiekf_cal) and \
                         (os.path.exists(result_path + '/result_SE3_se3_R3_cal_EqF.mat') or not args.ct_cal) and \
                         (os.path.exists(result_path + '/result_SE23_se23_cal_EqF.mat') or os.path.exists(result_path + '/result_SE23_se23_cal_equi_EqF.mat') or not args.ctex_cal) and \
                         (os.path.exists(result_path + '/result_SE23_se3_cal_EqF.mat') or os.path.exists(result_path + '/result_SE23_se3_cal_equi_EqF.mat') or not args.ctnew_cal)
                if cond_1 or cond_2:
                    continue

            if args.mekf:
                print("\nSimulating MEKF")
                name = '/result_MEKF.mat'
                with threadpool_limits(args.threads):
                    run_filter(MEKF, filter_args[0:-2], data, result_path + name)
                    # run_filter(UKF, filter_args[0:-2], data, result_path + name)

            if args.iekf:
                print("\nSimulating IEKF")
                if args.equivariant_output:
                    name = '/result_IEKF_equi.mat'
                else:
                    name = '/result_IEKF.mat'
                with threadpool_limits(args.threads):
                    # run_filter(IEKF, filter_args[0:-1], data, result_path + name)
                    run_filter(LIEKF, filter_args[0:-1], data, result_path + '/result_LIEKF.mat')

            if args.tfiekf:
                print("\nSimulating Two Frames IEKF")
                if args.equivariant_output:
                    name = '/result_TF_IEKF_equi.mat'
                else:
                    name = '/result_TF_IEKF.mat'
                with threadpool_limits(args.threads):
                    run_filter(TF_IEKF, filter_args[0:-1], data, result_path + name)
                    # run_filter(TF_LIEKF, filter_args[0:-1], data, result_path + '/result_TF_LIEKF.mat')

            if args.ct:
                print("\nSimulating SE3_se3_R3 EqF")
                name = '/result_SE3_se3_R3_EqF.mat'
                with threadpool_limits(args.threads):
                    run_filter(SE3_se3_R3_EqF, filter_args[0:-2] + [filter_args[-1]], data, result_path + name)

            if args.ctex:
                print("\nSimulating SE23_se23 EqF")
                if args.equivariant_output:
                    name = '/result_SE23_se23_EqF_equi.mat'
                else:
                    name = '/result_SE23_se23_EqF.mat'
                with threadpool_limits(args.threads):
                    run_filter(SE23_se23_EqF, filter_args + [measure_bv], data, result_path + name)

            if args.ctnew:
                print("\nSimulating SE23_se3 EqF")
                if args.equivariant_output:
                    name = '/result_SE23_se3_EqF_equi.mat'
                else:
                    name = '/result_SE23_se3_EqF.mat'
                with threadpool_limits(args.threads):
                    run_filter(SE23_se3_EqF, filter_args, data, result_path + name)

            if args.mekf_cal:
                print("\nSimulating MEKF with calibration states")
                name = '/result_cal_MEKF.mat'
                with threadpool_limits(args.threads):
                    run_filter(cal_MEKF, filter_args[0:-2], data, result_path + name)

            if args.iekf_cal:
                print("\nSimulating IEKF with calibration states")
                if args.equivariant_output:
                    name = '/result_cal_IEKF_equi.mat'
                else:
                    name = '/result_cal_IEKF.mat'
                with threadpool_limits(args.threads):
                    run_filter(cal_IEKF, filter_args[0:-1], data, result_path + name)

            if args.tfiekf_cal:
                print("\nSimulating Two Frames IEKF with calibration states")
                if args.equivariant_output:
                    name = '/result_cal_TF_IEKF_equi.mat'
                else:
                    name = '/result_cal_TF_IEKF.mat'
                with threadpool_limits(args.threads):
                    run_filter(cal_TF_IEKF, filter_args[0:-1], data, result_path + name)

            if args.ct_cal:
                print("\nSimulating SE3_se3_R3 EqF with calibration states")
                name = '/result_SE3_se3_R3_cal_EqF.mat'
                with threadpool_limits(args.threads):
                    run_filter(SE3_se3_R3_cal_EqF, filter_args[0:-2] + [filter_args[-1]], data, result_path + name)

            if args.ctex_cal:
                print("\nSimulating SE23_se23 EqF with calibration states")
                if args.equivariant_output:
                    name = '/result_SE23_se23_cal_EqF_equi.mat'
                else:
                    name = '/result_SE23_se23_cal_EqF.mat'
                with threadpool_limits(args.threads):
                    run_filter(SE23_se23_cal_EqF, filter_args + [measure_bv], data, result_path + name)

            if args.ctnew_cal:
                print("\nSimulating SE23_se3 EqF with calibration states")
                if args.equivariant_output:
                    name = '/result_SE23_se3_cal_EqF_equi.mat'
                else:
                    name = '/result_SE23_se3_cal_EqF.mat'
                with threadpool_limits(args.threads):
                    run_filter(SE23_se3_cal_EqF, filter_args, data, result_path + name)

    else:

        print(f"Loading dataset: {args.data_path}\n")
        if args.insane:
            gt_data, imu_data, gps_data = readCSV(args.data_path, 50.0, float(sys.maxsize), 1)
            data = (gt_data, imu_data, gps_data)
        else:
            data = readCSV(args.data_path)

        if args.mekf:
            print("\nSimulating MEKF")
            name = '/result_MEKF.mat'
            with threadpool_limits(args.threads):
                run_filter(MEKF, filter_args[0:-2], data, args.result_path + name)
                # run_filter(UKF, filter_args[0:-2], data, args.result_path + name)

        if args.iekf:
            print("\nSimulating IEKF")
            if args.equivariant_output:
                name = '/result_IEKF_equi.mat'
            else:
                name = '/result_IEKF.mat'
            with threadpool_limits(args.threads):
                # run_filter(IEKF, filter_args[0:-1], data, args.result_path + name)
                run_filter(LIEKF, filter_args[0:-1], data, args.result_path + '/result_LIEKF.mat')

        if args.tfiekf:
            print("\nSimulating Two Frames IEKF")
            if args.equivariant_output:
                name = '/result_TF_IEKF_equi.mat'
            else:
                name = '/result_TF_IEKF.mat'
            with threadpool_limits(args.threads):
                run_filter(TF_IEKF, filter_args[0:-1], data, args.result_path + name)
                # run_filter(TF_LIEKF, filter_args[0:-1], data, args.result_path + '/result_TF_LIEKF.mat')

        if args.ct:
            print("\nSimulating SE3_se3_R3 EqF")
            name = '/result_SE3_se3_R3_EqF.mat'
            with threadpool_limits(args.threads):
                run_filter(SE3_se3_R3_EqF, filter_args[0:-2] + [filter_args[-1]], data, args.result_path + name)

        if args.ctex:
            print("\nSimulating SE23_se23 EqF")
            if args.equivariant_output:
                name = '/result_SE23_se23_EqF_equi.mat'
            else:
                name = '/result_SE23_se23_EqF.mat'
            with threadpool_limits(args.threads):
                run_filter(SE23_se23_EqF, filter_args + [measure_bv], data, args.result_path + name)

        if args.ctnew:
            print("\nSimulating SE23_se3 EqF")
            if args.equivariant_output:
                name = '/result_SE23_se3_EqF_equi.mat'
            else:
                name = '/result_SE23_se3_EqF.mat'
            with threadpool_limits(args.threads):
                run_filter(SE23_se3_EqF, filter_args, data, args.result_path + name)

        if args.mekf_cal:
            print("\nSimulating MEKF with calibration states")
            name = '/result_cal_MEKF.mat'
            with threadpool_limits(args.threads):
                run_filter(cal_MEKF, filter_args[0:-2], data, args.result_path + name)

        if args.iekf_cal:
            print("\nSimulating IEKF with calibration states")
            if args.equivariant_output:
                name = '/result_cal_IEKF_equi.mat'
            else:
                name = '/result_cal_IEKF.mat'
            with threadpool_limits(args.threads):
                run_filter(cal_IEKF, filter_args[0:-1], data, args.result_path + name)

        if args.tfiekf_cal:
            print("\nSimulating Two Frames IEKF with calibration states")
            if args.equivariant_output:
                name = '/result_cal_TF_IEKF_equi.mat'
            else:
                name = '/result_cal_TF_IEKF.mat'
            with threadpool_limits(args.threads):
                 run_filter(cal_TF_IEKF, filter_args[0:-1], data, args.result_path + name)

        if args.ct_cal:
            print("\nSimulating SE3_se3_R3 EqF with calibration states")
            name = '/result_SE3_se3_R3_cal_EqF.mat'
            with threadpool_limits(args.threads):
                run_filter(SE3_se3_R3_cal_EqF, filter_args[0:-2] + [filter_args[-1]], data, args.result_path + name)

        if args.ctex_cal:
            print("\nSimulating SE23_se23 EqF with calibration states")
            if args.equivariant_output:
                name = '/result_SE23_se23_cal_EqF_equi.mat'
            else:
                name = '/result_SE23_se23_cal_EqF.mat'
            with threadpool_limits(args.threads):
                run_filter(SE23_se23_cal_EqF, filter_args + [measure_bv], data, args.result_path + name)

        if args.ctnew_cal:
            print("\nSimulating SE23_se3 EqF with calibration states")
            if args.equivariant_output:
                name = '/result_SE23_se3_cal_EqF_equi.mat'
            else:
                name = '/result_SE23_se3_cal_EqF.mat'
            with threadpool_limits(args.threads):
                run_filter(SE23_se3_cal_EqF, filter_args, data, args.result_path + name)