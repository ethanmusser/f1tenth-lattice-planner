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

