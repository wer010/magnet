import numpy as np

def convert_i_to_b(i):
    itob1_factor = np.array([[9.91799111e-10, -1.18536119e-11,  1.65072009e-11],
                             [-1.40363327e-11, 7.76394898e-10, 3.99175667e-12],
                             [-4.61721415e-12, -3.83052461e-13, 7.92748170e-10]])

    itob2_factor = np.array([[1.00495013e-09, -5.49719341e-12, -1.13997557e-11],
                             [2.64988303e-11,  8.09614956e-10, -2.99294999e-11],
                             [-2.99018981e-12,  2.18980608e-12,  8.41045631e-10]])

    itob3_factor = np.array([[1.03099047e-09, -2.22568015e-13, -6.35683494e-12],
                             [-8.55517979e-12,  8.09060679e-10,  2.88504804e-13],
                             [4.11595239e-12, 6.99139678e-12, 8.01764283e-10]])

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

    imu_tx_acc = data_raw[:, 27:30]
    imu_tx_ang = data_raw[:, 30:33]
    imu_tx_qtn = data_raw[:, 33:37]

    imu_rx_acc = data_raw[:, 37:40]
    imu_rx_ang = data_raw[:, 40:43]
    imu_rx_qtn = data_raw[:, 43:47]

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
           'imu_rx_qtn': imu_rx_qtn
           }

    return ret

def get_octant_by_imu(b_a, bs_a, rm):
    '''
    3*3 tensor:param b_a: the magnetic vector in 1st octant. b1,b2,b3 are row vectors
    3*3 tensor:param bs_a: the absolute magnetic vector sensed by magnetic induced coils
    3*3 tensor:param rm: rotation matrix. bs = rm@b
    int:return: the index of octant (1-4)
    '''
    sign_b = [[[1,1,1],[1,1,1],[1,1,1]],
              [[1,-1,-1],[-1,1,1],[-1,1,1]],
              [[1,1,-1],[1,1,-1],[-1,-1,1]],
              [[1,-1,1],[-1,1,-1],[1,-1,1]]]

    e_list = []
    for i in range(4):
        b = b_a*sign_b[i]
        bs = np.abs(np.matmul(rm, b.T))
        e = bs.T - bs_a
        e_list.append(np.sum(e**2))
    e = np.concatenate([e_list])
    # print(e)
    oct = np.argmin(e)+1
    return oct

def mse(a,b):
    return np.mean(np.linalg.norm(a - b,axis=-1), axis=0)

