import numpy as np


def cal_pos(matrix):
    offset = 0
    r_a = matrix[9 - offset:12 - offset]
    r_b = matrix[12 - offset:15 - offset]
    r_c = matrix[15 - offset:18 - offset]
    t_a = matrix[18 - offset:21 - offset]
    t_b = matrix[21 - offset:24 - offset]
    t_c = matrix[24 - offset:27 - offset]
    r_d = (51.67396739673968 / 93.8176818448948 * (r_b - r_a)) + r_a
    t_d = (73.82978297829783 / 168.44277858900054 * (t_b - t_a)) + t_a
    receive_x = r_b - r_d
    receive_y = r_c - r_d
    transmit_x = t_b - t_d
    transmit_y = t_c - t_d

    r_axis_x = receive_x / np.linalg.norm(receive_x)
    r_axis_y = receive_y / np.linalg.norm(receive_y)
    r_axis_z = np.cross(r_axis_x, r_axis_y)
    r_axis = np.hstack((r_axis_x, r_axis_y))
    r_axis = np.hstack((r_axis, r_axis_z))
    r_axis = np.reshape(r_axis, (3, 3))

    t_axis_x = transmit_x / np.linalg.norm(transmit_x)
    t_axis_y = transmit_y / np.linalg.norm(transmit_y)
    t_axis_z = np.cross(t_axis_x, t_axis_y)
    t_axis = np.hstack((t_axis_x, t_axis_y))
    t_axis = np.hstack((t_axis, t_axis_z))
    t_axis = np.reshape(t_axis, (3, 3))
    relative_r = r_axis.dot(np.linalg.inv(t_axis))
    #     relative_r = normalize_rotation_matrix(relative_r)
    pos = np.dot(t_axis, (r_d - t_d)) / 1000
    #     matrix_r = R.from_matrix(relative_r)
    #     euler = matrix_r.as_euler("xyz",degrees = True)
    relative_r = np.reshape(relative_r, (9,))
    return pos, relative_r


def main():
    data_raw = np.loadtxt("duijiaoxian.txt")

    mag1 = data_raw[:,0:3]
    mag2 = data_raw[:,3:6]
    mag3 = data_raw[:,6:9]
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




    r_all = []
    for i in range(np.shape(data_raw)[0]):
        pos, relative_r = cal_pos(data_raw[i])
        r = np.sqrt(np.sum(np.square(pos)))
        r_all.append(r)
        if i == 0:
            pos_all = pos
            relative_r_all = relative_r
            data = data_raw[i][0:9]
        else:
            pos_all = np.vstack((pos_all, pos))
            relative_r_all = np.vstack((relative_r_all, relative_r))
            data = np.vstack((data, data_raw[i][0:9]))
    r_all = np.array(r_all)
    return 0

if __name__ == '__main__':
    main()
