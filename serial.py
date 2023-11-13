import serial
from serial.tools import list_ports
import time
import struct
import numpy as np
num_sensors = 16

ports_list = list(list_ports.comports())
if len(ports_list)<=0:
    print('No serial device')
else:
    for item in ports_list:
        print(item)
ser = serial.Serial('/dev/ttyUSB0', 115200,8,'N',1, timeout=None)

if ser.isOpen():
    while True:
        com_input = ser.read_until(b'\xde\xad\xbe\xef')
        if len(com_input)==(num_sensors*3*4+4):
            data_list = []
            for i in range(num_sensors*3):
                binary_string = com_input[4*i:4*(i+1)]
                float_value = struct.unpack('<f', binary_string)[0]
                data_list.append(float_value)

            # print(com_input.hex())
            data = np.array(data_list).reshape(num_sensors,3)
            print(data.shape)
        else:
            continue
        # time.sleep(0.001)


def main():
    return 0




if __name__ == '__main__':
    main()
