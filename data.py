import re
import numpy as np
from visualization import plot_2d,plot_3d_by_matplotlib

def read_txt(path):
    with open(path) as f:

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
        print('Data loaded!')

        return pos, mag






def main():
    pos,mag = read_txt('/home/lanhai/restore/dataset/magnetic tracking/output.fld')
    plot_2d(pos[::10,50,::10,:], mag[::10,50,::10,:])
    # plot_3d_by_matplotlib(pos[::10,::10,::10,:],mag[::10,::10,::10,:])

    return 0





if __name__ == '__main__':
    main()