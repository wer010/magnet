import numpy as np
import re
import time
from scipy.spatial.transform import Rotation as R

class Magnetic_field:
    def __init__(self,bt,m):
        assert m.shape==(3,)
        self.bt = bt
        self.m = m

    def get_bvector(self,p):
        raise NotImplementedError


    def get_sensor_b(self,p,q):
        # The axis of sensor are not parallel to the global coordinate axis, therefore
        # a rotation matrix need to be applied on the vector B
        r = self.q_to_r(q)
        B = self.get_mfield(p)
        bs = np.linalg.inv(r)@B
        return bs

    def jacobian(self, p):
        # Use numerical method to get the derivative
        return

class Magnetic_dipole(Magnetic_field):
    def __init__(self,bt,m,work_area):
        super(Magnetic_dipole,self).__init__(bt,m)

    def get_bvector(self,p):
        assert p.shape==(3,)
        mp=self.m@p
        r= np.linalg.norm(p)

        b = (self.bt/np.power(r,5)) * (3*mp*p-np.power(r,2)*self.m)
        return b

    def jacobian(self,p):


        return


class Magnetic_datadriven(Magnetic_field):
    def __init__(self, bt, m, path):
        super(Magnetic_datadriven, self).__init__(bt, m)
        self.pos, self.mag = self.read_txt(path)
        self.pos_start = self.pos[0, 0, 0, :]
        self.pos_end = self.pos[-1, -1, -1, :]
        self.pos_interval = self.pos[1, 1, 1, :] - self.pos[0, 0, 0, :]

    def get_bvector(self,p):
        ind = np.rint((p - self.pos_start)/self.pos_interval).astype(np.int)
        assert np.all(ind<=self.pos.shape[:3])
        print(self.pos[tuple(ind)])
        print(p)
        return self.mag[ind]

    def read_txt(self,path):
        with open(path) as f:
            start_time=time.time()
            ind = 0
            for i, line in enumerate(f):
                if i==0:
                    pattern = r'-?\d+cm'
                    print('Data loading...')
                    # Use re.findall to extract all matches
                    matches = re.findall(pattern, line)

                    # Convert the extracted strings to integers (assuming you want to work with numeric values)
                    numeric_values = [int(match[:-2]) for match in matches]

                    size = 1+ (np.array(numeric_values[3:6],dtype=np.int)-np.array(numeric_values[0:3],dtype=np.int))//np.array(numeric_values[6:9],dtype=np.int)
                    pos = np.empty(np.append(size,3))
                    mag = np.empty(pos.shape)

                elif i==1:
                    continue
                else:
                    num = line.split()
                    num = [float(n) for n in num]
                    pos[ind//(size[1]*size[2]),ind//(size[2])%size[1],ind%(size[2])] = num[:3]
                    mag[ind//(size[1]*size[2]),ind//(size[2])%size[1],ind%(size[2])] = np.array(num[3:])*1e6     # use ut as the magnetic induction density unit
                    ind+=1
                    if ind%(size[1]*size[2])==0:
                        print(f'\r{ind//(size[1]*size[2])}/{size[0]} completed',end='',flush=True)
            print('\r')
            assert size[0]*size[1]*size[2] == ind
            end_time = time.time()
            print(f'Data loaded! Took {end_time-start_time} seconds.')

            return pos, mag



if __name__ == '__main__':
    m = Magnetic_field(1e5,np.array([0,0,1]))
    m1 = Magnetic_dipole(1e5,np.array([0,0,1]))
    m2 = Magnetic_datadriven(1e5,np.array([0,0,1]),'/home/lanhai/restore/dataset/magnetic tracking/output.fld')
    b = m2.get_bvector([0,0,0])

