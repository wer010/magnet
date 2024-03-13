import numpy as np
from matplotlib import pyplot as plt

def convert_i_to_b(i):
    itob1_factor = np.array([[3.09905629e-10, -3.58266440e-12, 1.39214392e-11],
                              [4.10656032e-12, 3.05311418e-10, -4.32355817e-12],
                              [-4.01117467e-12, -1.97456465e-12, 3.15281188e-10]])

    itob2_factor = np.array([[3.15954185e-10, -5.29419522e-13, 6.65786439e-12],
                            [-4.88240345e-12, 3.04687562e-10, 5.71936018e-12],
                            [-6.03156406e-12, 1.34914572e-12, 3.11859070e-10]])

    itob3_factor = np.array([[3.04114453e-10, 8.66825106e-12, -6.99615622e-12],
                            [-1.11279122e-12, 2.92344594e-10, -6.26868840e-12],
                            [-1.29690531e-13, 2.87699996e-12, 2.99422857e-10]])

    i1 = np.stack([i[:, 5], i[:, 4], i[:, 3]], axis=-1)[...,np.newaxis]
    i2 = np.stack([i[:, 8], i[:, 7], i[:, 6]], axis=-1)[...,np.newaxis]
    i3 = np.stack([i[:, 2], i[:, 1], i[:, 0]], axis=-1)[...,np.newaxis]

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

    b = convert_i_to_b(data_raw[:,0:9])
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


def main():
    data = get_data("duijiaoxian.txt")
    theory_bt = 2.42e-9
    b_sensor = data['magnetic_intensity']

    b1_norm = np.sum(b_sensor[:,0:3]**2,axis=-1)
    b2_norm = np.sum(b_sensor[:,3:6]**2,axis=-1)
    b3_norm = np.sum(b_sensor[:,6:9]**2,axis=-1)
    r = np.power(theory_bt * theory_bt * 6 / (b1_norm + b2_norm + b3_norm), 1 / 6)
    x2 = b1_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3
    y2 = b2_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3
    z2 = b3_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3

    pos_label = data['position']
    r_label = np.linalg.norm(pos_label,axis=-1)

    ax1 = plt.subplot(141)
    ax1.plot(r)
    ax1.plot(r_label)
    ax1.set_title('Distance')

    ax2 = plt.subplot(142)
    ax2.plot(np.sqrt(np.abs(x2)))
    ax2.plot(pos_label[:,0])
    ax2.set_title('x')

    ax3 = plt.subplot(143)
    ax3.plot(np.sqrt(np.abs(y2)))
    ax3.plot(pos_label[:, 1])
    ax3.set_title('y')

    ax4 = plt.subplot(144)
    ax4.plot(np.sqrt(np.abs(z2)))
    ax4.plot(pos_label[:, 2])
    ax4.set_title('z')


    plt.show()





if __name__ == '__main__':
    main()
