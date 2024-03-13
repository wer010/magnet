import numpy as np
from matplotlib import pyplot as plt
from em_tracking.utils import get_data
from magnetic_field_models import Magnetic_dipole
from datetime import datetime

def get_pos_by_analytical_solution(b_sensor):
    theory_bt = 2.42e-9
    b1_norm = np.sum(b_sensor[:, 0:3] ** 2, axis=-1)
    b2_norm = np.sum(b_sensor[:, 3:6] ** 2, axis=-1)
    b3_norm = np.sum(b_sensor[:, 6:9] ** 2, axis=-1)
    r = np.power(theory_bt * theory_bt * 6 / (b1_norm + b2_norm + b3_norm), 1 / 6)
    x2 = b1_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3
    y2 = b2_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3
    z2 = b3_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3

    return np.stack([np.sqrt(np.abs(x2)),np.sqrt(np.abs(y2)),np.sqrt(np.abs(z2))],axis=-1),r


def get_pos_by_matrix_analysis(b_sensor):
    theory_bt = 2.42e-9
    y = b_sensor.reshape(-1,3,3).transpose([0,2,1])
    pos = []
    for i in range(y.shape[0]):
        u,s,v = np.linalg.svd(y[i])
        p = np.power(np.sqrt(6)* theory_bt/np.linalg.norm(y[i]),1/3)* v[0]
        pos.append(p)
    return np.stack(pos), np.linalg.norm(pos,axis=-1)

def main():
    theory_bt = 2.42e-9
    data = get_data("20240311_test3.txt")
    b_sensor = data['magnetic_intensity']

    pos_label = data['position']
    r_label = np.linalg.norm(pos_label,axis=-1)

    rm = data['rotation_matrix']

    m1 = Magnetic_dipole(theory_bt, np.array([1, 0, 0]))
    m2 = Magnetic_dipole(theory_bt, np.array([0, 1, 0]))
    m3 = Magnetic_dipole(theory_bt, np.array([0, 0, 1]))

    theoretical_b1 = m1.get_bvector(pos_label)
    theoretical_b1s = np.matmul(np.transpose(rm, axes = [0,2,1]), theoretical_b1[..., np.newaxis]).squeeze()

    theoretical_b2 = m2.get_bvector(pos_label)
    theoretical_b2s = np.matmul(np.transpose(rm, axes = [0,2,1]), theoretical_b2[..., np.newaxis]).squeeze()

    theoretical_b3 = m3.get_bvector(pos_label)
    theoretical_b3s = np.matmul(np.transpose(rm, axes = [0,2,1]), theoretical_b3[..., np.newaxis]).squeeze()

    sign_tbs = np.sign(np.concatenate([theoretical_b1s,theoretical_b2s,theoretical_b3s],axis =-1))
    b_sensor_with_sign = b_sensor*sign_tbs


    start = datetime.now()
    pos_ana,r_ana = get_pos_by_analytical_solution(b_sensor)
    end = datetime.now()
    print(f'Time consumpts {end-start}s.')

    start = datetime.now()
    pos_mat,r_mat = get_pos_by_matrix_analysis(b_sensor_with_sign)
    end = datetime.now()
    print(f'Time consumpts {end - start}s.')

    ax1 = plt.subplot(141)
    ax1.plot(r_ana,label='EM ana')
    ax1.plot(r_mat,label='EM mat')
    ax1.plot(r_label,label='Optical tracking')
    ax1.set_title('Distance')
    ax1.legend()

    ax2 = plt.subplot(142)
    ax2.plot(pos_ana[:,0],label='EM ana')
    ax2.plot(pos_mat[:,0],label='EM mat')
    ax2.plot(pos_label[:,0],label='Optical tracking')
    ax2.set_title('x')
    ax2.legend()

    ax3 = plt.subplot(143)
    ax3.plot(-1*pos_ana[:,1],label='EM ana')
    ax3.plot(pos_mat[:,1],label='EM mat')
    ax3.plot(pos_label[:, 1],label='Optical tracking')
    ax3.set_title('y')
    ax3.legend()

    ax4 = plt.subplot(144)
    ax4.plot(pos_ana[:,2],label='EM ana')
    ax4.plot(pos_mat[:,2],label='EM mat')
    ax4.plot(pos_label[:, 2],label='Optical tracking')
    ax4.set_title('z')
    ax4.legend()


    plt.show()





if __name__ == '__main__':
    main()
