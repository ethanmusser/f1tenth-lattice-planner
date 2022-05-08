#!/usr/bin/env python3
import sys
import os
import datetime
import numpy as np
import graph_ltpl


def get_path_dict(toppath, track_specifier):
    sys.path.append(toppath)
    path_dict = {'globtraj_input_path': toppath + "/inputs/traj_ltpl_cl/traj_ltpl_cl_" + track_specifier + ".csv",
                 'graph_store_path': toppath + "/inputs/graphs/stored_graph.pckl",
                 'ltpl_offline_param_path': toppath + "/config/graph_ltpl/ltpl_config_offline.ini",
                 'ltpl_online_param_path': toppath + "/config/graph_ltpl/ltpl_config_online.ini",
                 'log_path': toppath + "/log/graph_ltpl/",
                 'graph_log_id': datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                 }
    return path_dict


def import_global_traj(import_path):
    # Read File
    csv_data_temp = np.loadtxt(import_path, delimiter=';')

    # Gather Trajectory Data
    refline = csv_data_temp[:-1, 0:2]
    width_right = csv_data_temp[:-1, 2]
    width_left = csv_data_temp[:-1, 3]
    norm_vec = csv_data_temp[:-1, 4:6]
    alpha = csv_data_temp[:-1, 6]
    s = csv_data_temp[:-1, 7]
    length_rl = np.diff(csv_data_temp[:, 7])
    psi = csv_data_temp[:-1, 8]
    kappa_rl = csv_data_temp[:-1, 9]
    vel_rl = csv_data_temp[:-1, 10]
    acc_rl = csv_data_temp[:-1, 11]

    return refline, width_right, width_left, norm_vec, alpha, s, psi, kappa_rl, vel_rl, acc_rl


def get_traj_line(refline, norm_vec, alpha):
    assert len(refline) == len(norm_vec), "Lengths of all inputs must be equal."
    assert len(refline) == len(alpha), "Lengths of all inputs must be equal."
    return [refline[i] + alpha[i] * norm_vec[i] for i in range(len(refline))]

