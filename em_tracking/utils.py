import numpy as np
from em_tracking.calibration_config.itob_quanzhou import itob1_factor,itob2_factor,itob3_factor

def convert_i_to_b(i):
    i1 = i[:,0:3][...,np.newaxis]
    i2 = i[:,3:6][...,np.newaxis]
    i3 = i[:,6:9][...,np.newaxis]

    b1 = np.matmul(itob1_factor, i1).squeeze()
    b2 = np.matmul(itob2_factor, i2).squeeze()
    b3 = np.matmul(itob3_factor, i3).squeeze()

    return np.concatenate([b1,b2,b3],axis=-1)

def get_data(p):
    '''
    Load data from txt.
    The data format in the txt file is:
    0-8: magnetic induced field sensed by coils. b1,b2,b3
    9-17: targets (xyz) position of receiver by optical tracking: left,right,front
    18-26: targets (xyz) position of emitter by optical tracking: left,right,front
    27-32: acceleration from imu: receiver, emitter
    33-38: angular velocity from imu: receiver, emitter
    39-46: quaternion from imu: receiver, emitter
    param p: txt file path
    return: data dictionary
    '''
    data_raw = np.loadtxt(p)
    curr = data_raw[:,0:9]
    curr = np.roll(curr, -3, axis = -1)

    b = convert_i_to_b(curr)
    pos_rx_left = data_raw[:,9:12]
    pos_rx_right = data_raw[:,12:15]
    pos_rx_front = data_raw[:,15:18]

    pos_tx_left = data_raw[:,18:21]
    pos_tx_right = data_raw[:,21:24]
    pos_tx_front = data_raw[:,24:27]

    imu_rx_acc = data_raw[:, 27:30]
    imu_tx_acc = data_raw[:, 30:32]
    imu_rx_ang = data_raw[:, 33:36]
    imu_tx_ang = data_raw[:, 36:39]
    imu_rx_qtn = data_raw[:, 39:43]
    imu_tx_qtn = data_raw[:, 43:47]


    pos_rx = (51.67396739673968 / 93.8176818448948 * (pos_rx_right - pos_rx_left)) + pos_rx_left
    pos_tx = (73.82978297829783 / 168.44277858900054 * (pos_tx_right - pos_tx_left)) + pos_tx_left

    transmit_x = pos_tx_right - pos_tx
    transmit_y = pos_tx_front - pos_tx
    t_axis_x = transmit_x / np.linalg.norm(transmit_x, axis=-1)[:, np.newaxis]
    t_axis_y = transmit_y / np.linalg.norm(transmit_y, axis=-1)[:, np.newaxis]
    t_axis_z = np.cross(t_axis_x, t_axis_y)
    pos = np.matmul(np.stack([t_axis_x,t_axis_y,t_axis_z],axis=1), (pos_rx - pos_tx)[..., np.newaxis]) / 1000

    receiver_x = pos_rx_right - pos_rx
    receiver_y = pos_rx_front - pos_rx
    r_axis_x = receiver_x / np.linalg.norm(receiver_x, axis=-1)[:, np.newaxis]
    r_axis_y = receiver_y / np.linalg.norm(receiver_y, axis=-1)[:, np.newaxis]
    r_axis_z = np.cross(r_axis_x, r_axis_y)

    rot_mat =  np.matmul(np.stack([t_axis_x,t_axis_y,t_axis_z],axis=1), np.stack([r_axis_x,r_axis_y,r_axis_z],axis=-1))

    ret = {'position':pos.squeeze(),
           'rotation_matrix':rot_mat,
           'magnetic_intensity':b,
           'imu_tx_acc':imu_tx_acc,
           'imu_rx_acc':imu_rx_acc,
           'imu_tx_ang': imu_tx_ang,
           'imu_rx_ang': imu_rx_ang,
           'imu_tx_qtn': imu_tx_qtn,
           'imu_rx_qtn': imu_rx_qtn}

    return ret


def mse(a,b):
    return np.mean(np.linalg.norm(a - b, axis=-1), axis=0)

