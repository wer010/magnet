# TODO: the entrance for the algorithm testing
from magnetic_field_models import Magnetic_datadriven,Magnetic_dipole
from rotation import Quaternion
from LM import lm
import numpy as np


def main():
    # 1. Load the ground truth data and initialize the model
    mf_gt = Magnetic_datadriven((1e5,np.array([0,0,1]),'/home/lanhai/restore/dataset/magnetic tracking/output.fld'))
    mf_dipole_model = Magnetic_dipole(1e5,np.array([0,0,1]))
    # 2. Give some points, get the magnetic induction intensity
    p = np.array([0.2,0.2,0,2])
    b_gt = mf_gt.get_bvector(p)
    # 3. now the ground truth data is ready, let's define a sensor array and simulate of sensor data
    s_array = np.array([[0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0]])
    # 4. the coordinate axis of sensor array is different from the world coordinate axis.
    # we only consider the rotation. quaternion is used to denote the rotation.
    q = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0, 0]
    q = Quaternion(q[0], q[1], q[2], q[3])
    # 5. get the position of every sensor, then get the magnetic vector corresponding to the positions
    # use quaternion to get sensor data along each sensor axis
    p_s_array = q.rotation(s_array)+p
    b_gt_array = mf_gt.get_bvector(p_s_array)
    b_s_array = np.linalg.inv(q.q_to_r())@b_gt_array
    # 6. use numerical method to solve the position p from sensor array data.
    p_c = lm(b_s_array)

    # 7. assess and visualize the error
    print(p-p_c)


    return 0




if __name__ == '__main__':
    main()



