import sys
import serial
import threading
import numpy as np
import serial.tools.list_ports
from two_ports import Ui_Form
from PyQt5 import QtWidgets, QtGui, QtCore
import struct
from PyQt5.QtWidgets import QMessageBox
import pyqtgraph as pg
import warnings
import time
from vicon_dssdk.vicon_dssdk import ViconDataStream
warnings.filterwarnings("ignore")


class Pyqt5_Serial(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(Pyqt5_Serial, self).__init__()
        # 设定显示压力的最小值
        self.port_open_flag = False
        self.ui = Ui_Form()
        self.setupUi(self)
        self.init()
        self.setWindowTitle("足底压信息显示助手")
        # 初始化发射和接收串口
        self.ser_tx = serial.Serial()
        self.ser_rx = serial.Serial()

    def init(self):
        self.pi_unit = 180. / np.pi
        self.time_pre = 0
        self.euler_rx = np.array([0,0,0])
        self.euler_tx = np.array([0,0,0])
        self.linear_acc_tx = np.array([0,0,0])
        self.linear_acc_rx = np.array([0,0,0])
        self.linear_acc_earth_tx = np.array([0,0,0])
        self.linear_acc_earth_rx = np.array([0,0,0])
        # 设置主窗口的位置
        self.setGeometry(200, 120, 1600, 1050)
        # 串口检测按钮
        self.s1__box_1.clicked.connect(self.port_check_tx)
        self.s1__box_4.clicked.connect(self.port_check_rx)
        # 打开串口按钮
        self.open_button.clicked.connect(self.port_open)
        # 关闭串口按钮
        self.close_button.clicked.connect(self.port_close)
        self.close_button.setEnabled(False)
        self.saving_button.clicked.connect(self.start_saving_data)
        self.saving_data_flag = False
        self.stop_saving.clicked.connect(self.stop_saving_data)
        self.stop_saving.setEnabled(False)

        # 初始化绘图布局
        plot_size = 500
        self.x = np.linspace(0,plot_size-1,plot_size)
        self.plot_data_1 = np.zeros(plot_size)
        self.plot_data_2 = np.zeros(plot_size)
        self.plot_data_3 = np.zeros(plot_size)
        self.plot_data_4 = np.zeros(plot_size)
        self.plot_data_5 = np.zeros(plot_size)
        self.plot_data_6 = np.zeros(plot_size)
        self.plot_data_7 = np.zeros(plot_size)
        self.plot_data_8 = np.zeros(plot_size)
        self.plot_data_9 = np.zeros(plot_size)
        # 初始化x曲线
        self.plot_layout_x = QtWidgets.QGridLayout()  # 实例化一个网格布局层
        self.widget.setLayout(self.plot_layout_x)
        self.plot_pos_x = pg.PlotWidget()
        self.plot_pos_x.showGrid(x=True, y=True)  # 显示图形网格
        self.plot_layout_x.addWidget(self.plot_pos_x)
        self.plot_pos_x.addLegend()
        self.curve_pos_x_1 = self.plot_pos_x.plot(name = "tx_x",pen=pg.mkPen('r', width=1))
        self.curve_pos_y_1 = self.plot_pos_x.plot(name="rx_x",pen=pg.mkPen('g', width=1))
        self.curve_pos_z_1 = self.plot_pos_x.plot(name="f1_z",pen=pg.mkPen('b', width=1))
        # 初始化y曲线
        self.plot_layout_y = QtWidgets.QGridLayout()  # 实例化一个网格布局层
        self.widget_2.setLayout(self.plot_layout_y)
        self.plot_pos_y = pg.PlotWidget()
        self.plot_pos_y.showGrid(x=True, y=True)  # 显示图形网格
        self.plot_layout_y.addWidget(self.plot_pos_y)
        self.plot_pos_y.addLegend()
        self.curve_pos_x_2 = self.plot_pos_y.plot(name = "tx_y",pen=pg.mkPen('r', width=1))
        self.curve_pos_y_2 = self.plot_pos_y.plot(name="rx_y",pen=pg.mkPen('g', width=1))
        self.curve_pos_z_2 = self.plot_pos_y.plot(name="f2_z",pen=pg.mkPen('b', width=1))
        # 初始化z曲线
        self.plot_layout_z = QtWidgets.QGridLayout()  # 实例化一个网格布局层
        self.widget_3.setLayout(self.plot_layout_z)
        self.plot_pos_z = pg.PlotWidget()
        self.plot_pos_z.showGrid(x=True, y=True)  # 显示图形网格
        self.plot_layout_z.addWidget(self.plot_pos_z)
        self.plot_pos_z.addLegend()
        self.curve_pos_x_3 = self.plot_pos_z.plot(name="tx_z",pen=pg.mkPen('r', width=1))
        self.curve_pos_y_3 = self.plot_pos_z.plot(name="rz_z",pen=pg.mkPen('g', width=1))
        self.curve_pos_z_3 = self.plot_pos_z.plot(name="f3_z",pen=pg.mkPen('b', width=1))

        self.timer_plot_pos = QtCore.QTimer()
        self.timer_plot_pos.timeout.connect(self.updateData)

        #初始化vicon
        self.client = ViconDataStream.Client()
        self.client.Connect("192.168.10.1:801")
        self.client.EnableMarkerData()
        #计算加速度的时间间隔
        self.gap = 3
        self.vicon_pos = np.zeros([self.gap*2+3,3])
        self.mag_pos = np.zeros([self.gap*2+3,3])
        self.delt_t = 10.41/1000.
        #标定系数
        self.f1_x_factor = np.array([3.09905629e-10, -3.58266440e-12, 1.39214392e-11])
        self.f1_y_factor = np.array([4.10656032e-12, 3.05311418e-10, -4.32355817e-12])
        self.f1_z_factor = np.array([-4.01117467e-12, -1.97456465e-12, 3.15281188e-10])

        self.f2_x_factor = np.array([3.15954185e-10, -5.29419522e-13, 6.65786439e-12])
        self.f2_y_factor = np.array([-4.88240345e-12, 3.04687562e-10, 5.71936018e-12])
        self.f2_z_factor = np.array([-6.03156406e-12, 1.34914572e-12, 3.11859070e-10])

        self.f3_x_factor = np.array([3.04114453e-10, 8.66825106e-12, -6.99615622e-12])
        self.f3_y_factor = np.array([-1.11279122e-12, 2.92344594e-10, -6.26868840e-12])
        self.f3_z_factor = np.array([-1.29690531e-13, 2.87699996e-12, 2.99422857e-10])
        #初始化相对加速度
        self.relative_acc_mag = np.array([0,0,0])
        self.relative_acc_vicon = np.array([0,0,0])
        self.imu_gyro_tx = np.array([0,0,0])
        self.imu_gyro_rx = np.array([0,0,0])
        self.quater_tx = np.array([0,0,0,0])
        self.quater_rx = np.array([0,0,0,0])

    #计算加速度
    def cal_accel_fun(self,data):
        return (data[-1] - 2*data[self.gap+1] + data[0])/(self.delt_t**2)
    #计算电磁坐标
    def cal_mag_pos_fun(self,meaAmp):
        # 把接收电压值转为磁场
        f1_x = np.sum(np.array([meaAmp[5], meaAmp[4], meaAmp[3]]) * self.f1_x_factor)
        f1_y = np.sum(np.array([meaAmp[5], meaAmp[4], meaAmp[3]]) * self.f1_y_factor)
        f1_z = np.sum(np.array([meaAmp[5], meaAmp[4], meaAmp[3]]) * self.f1_z_factor)
        f2_x = np.sum(np.array([meaAmp[8], meaAmp[7], meaAmp[6]]) * self.f2_x_factor)
        f2_y = np.sum(np.array([meaAmp[8], meaAmp[7], meaAmp[6]]) * self.f2_y_factor)
        f2_z = np.sum(np.array([meaAmp[8], meaAmp[7], meaAmp[6]]) * self.f2_z_factor)
        f3_x = np.sum(np.array([meaAmp[2], meaAmp[1], meaAmp[0]]) * self.f3_x_factor)
        f3_y = np.sum(np.array([meaAmp[2], meaAmp[1], meaAmp[0]]) * self.f3_y_factor)
        f3_z = np.sum(np.array([meaAmp[2], meaAmp[1], meaAmp[0]]) * self.f3_z_factor)

        theory_bt = 2.42e-9
        f1_sum = f1_x ** 2 + f1_y ** 2 + f1_z ** 2
        f2_sum = f2_x ** 2 + f2_y ** 2 + f2_z ** 2
        f3_sum = f3_x ** 2 + f3_y ** 2 + f3_z ** 2
        dist = np.power(theory_bt * theory_bt * 6 / (f1_sum + f2_sum + f3_sum), 1 / 6)
        diff_x = f1_sum * (dist ** 8) / (3 * theory_bt * theory_bt) - dist * dist / 3
        diff_y = f2_sum * (dist ** 8) / (3 * theory_bt * theory_bt) - dist * dist / 3
        diff_z = f3_sum * (dist ** 8) / (3 * theory_bt * theory_bt) - dist * dist / 3
        if diff_x < 0:
            x = -np.sqrt(-diff_x)
        else:
            x = np.sqrt(diff_x)
        if diff_y < 0:
            y = -np.sqrt(-diff_y)
        else:
            y = np.sqrt(diff_y)
        if diff_z < 0:
            z = -np.sqrt(-diff_z)
        else:
            z = np.sqrt(diff_z)
        return np.array([x,y,z])

    def start_saving_data(self):
        self.fd = open("saving_data.txt", "w")
        self.saving_button.setEnabled(False)
        self.stop_saving.setEnabled(True)
        self.saving_data_flag = True
    def stop_saving_data(self):
        self.saving_data_flag = False
        time.sleep(0.1)
        self.fd.close()
        self.saving_button.setEnabled(True)
        self.stop_saving.setEnabled(False)

    def updateData(self):
        # print(self.linear_acc_earth_tx - self.linear_acc_earth_rx)

        self.curve_pos_x_1.setData(x=self.x, y=self.plot_data_1)
        # self.curve_pos_y_1.setData(x=self.x, y=self.plot_data_2)
        # self.curve_pos_z_1.setData(x=self.x, y=self.plot_data_3)

        self.curve_pos_x_2.setData(x=self.x, y=self.plot_data_4)
        # self.curve_pos_y_2.setData(x=self.x, y=self.plot_data_5)
        # self.curve_pos_z_2.setData(x=self.x, y=self.plot_data_6)

        self.curve_pos_x_3.setData(x=self.x, y=self.plot_data_7)
        # self.curve_pos_y_3.setData(x=self.x, y=self.plot_data_8)
        # self.curve_pos_z_3.setData(x=self.x, y=self.plot_data_9)


    def port_check_tx(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.s1__box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.s1__box_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")

    def port_check_rx(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict_rx = {}
        port_list = list(serial.tools.list_ports.comports())
        self.s1__box_3.clear()
        for port in port_list:
            self.Com_Dict_rx["%s" % port[0]] = "%s" % port[1]
            self.s1__box_3.addItem(port[0])
        if len(self.Com_Dict_rx) == 0:
            self.state_label.setText(" 无串口")

    # 打开串口
    def port_open(self):
        # 发射串口
        self.ser_tx.port = self.s1__box_2.currentText()
        self.ser_tx.baudrate = 115200
        # 接收串口
        self.ser_rx.port = self.s1__box_3.currentText()
        self.ser_rx.baudrate = 115200
        self.port_open_flag = True
        self.open_button.setEnabled(False)
        self.close_button.setEnabled(True)
        try:
            # 打开发射串口和创建数据读取的线程
            self.ser_tx.open()
            self.thread_read_ser_tx = threading.Thread(target=self.serial_read_tx)
            self.thread_read_ser_tx.start()
            # 打开接收串口和创建数据读取的线程
            self.ser_rx.open()
            self.thread_read_ser_rx = threading.Thread(target=self.serial_read_rx)
            self.thread_read_ser_rx.start()
            # 打开绘图的线程
            self.timer_plot_pos.start(10)
        except:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None

    # 串口读取
    def serial_read_tx(self):
        print("serial read")
        while self.port_open_flag:
            data_imu = self.ser_tx.read(33)
            count_time = struct.unpack("<L", data_imu[1:5])[0]
            self.time_pre = count_time
            temp_quaternion = np.array([0, 0, 0, 0])
            accel = np.array([0., 0., 0.])
            for i in range(4):
                quaternion_value_ = data_imu[(23 + i * 2):(25 + i * 2)]
                temp_quaternion[i] = struct.unpack("<h", quaternion_value_)[0]
                self.quater_tx[i] = temp_quaternion[i]
            for i in range(3):
                accel_ = data_imu[(5 + i * 2):(7 + i * 2)]
                accel[i] = round(struct.unpack("<h", accel_)[0], 2) / 100.0
            for i in range(3):
                gyro_ = data_imu[(11 + i * 2):(13 + i * 2)]
                self.imu_gyro_tx[i] = round(struct.unpack("<h", gyro_)[0], 2) / 16.0

            linear_acc = np.array([0., 0., 0.])
            for i in range(3):
                linear_acc_ = data_imu[(17 + i * 2):(19 + i * 2)]
                linear_acc[i] = round(struct.unpack("<h", linear_acc_)[0], 2)
            self.linear_acc_tx = linear_acc / 100.0

            # 计算欧拉角
            d_value = np.sqrt(
                temp_quaternion[0] ** 2 + temp_quaternion[1] ** 2 + temp_quaternion[2] ** 2 + temp_quaternion[
                    3] ** 2)
            q_w = temp_quaternion[0] / d_value
            q_x = temp_quaternion[1] / d_value
            q_y = temp_quaternion[2] / d_value
            q_z = temp_quaternion[3] / d_value
            # 根据四元数计算旋转矩阵
            matrix_tx = np.array(
                [[1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
                 [2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_w * q_x)],
                 [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x ** 2 + q_y ** 2)]
                 ])
            #求旋转矩阵的逆
            matrix_tx_inv = np.linalg.inv(matrix_tx)
            self.linear_acc_earth_tx = np.dot(matrix_tx_inv,self.linear_acc_tx)
            euler_x = round((np.arctan2(2 * (q_y * q_z + q_w * q_x),
                                        q_w * q_w - q_x * q_x - q_y * q_y + q_z * q_z)) * (
                                self.pi_unit), 2)
            euler_y = round(np.arcsin(-2 * (q_x * q_z - q_w * q_y)) * (self.pi_unit), 2)
            euler_z = round(np.arctan2(2 * (q_x * q_y + q_w * q_z),
                                       q_w * q_w + q_x * q_x - q_y * q_y - q_z * q_z) * (
                                self.pi_unit), 2)
            self.euler_tx = np.array([euler_x, euler_y, euler_z])

            # 画每一个线圈的值
            self.plot_data_1[:-1] = self.plot_data_1[1:]
            self.plot_data_1[-1] = self.linear_acc_earth_tx[0] - self.linear_acc_earth_rx[0]
            # self.plot_data_2[:-1] = self.plot_data_2[1:]
            # self.plot_data_2[-1] = meaAmp_[4]
            # self.plot_data_3[:-1] = self.plot_data_3[1:]
            # self.plot_data_3[-1] = meaAmp_[3]

            self.plot_data_4[:-1] = self.plot_data_4[1:]
            self.plot_data_4[-1] = self.linear_acc_earth_tx[1] - self.linear_acc_earth_rx[1]
            # self.plot_data_5[:-1] = self.plot_data_5[1:]
            # self.plot_data_5[-1] = meaAmp_[7]
            # self.plot_data_6[:-1] = self.plot_data_6[1:]
            # self.plot_data_6[-1] = meaAmp_[6]

            self.plot_data_7[:-1] = self.plot_data_7[1:]
            self.plot_data_7[-1] = self.linear_acc_earth_tx[2] - self.linear_acc_earth_rx[2]
            # self.plot_data_8[:-1] = self.plot_data_8[1:]
            # self.plot_data_8[-1] = meaAmp_[1]
            # self.plot_data_9[:-1] = self.plot_data_9[1:]
            # self.plot_data_9[-1] = meaAmp_[0]

    # 串口读取
    def serial_read_rx(self):
        print("serial read")
        while self.port_open_flag:
            # 六轴imu的一个数据包55字节
            # data_mag = self.ser.read(51)
            # 九轴imu带加速度、角速度和四元数的一个数据包63字节
            # 结构为 type struct {
            #     uint8_t head;
            #     uint32_t deltatime;
            #     float32_t meaAmp_f1[3];
            #     float32_t meaAmp_f2[3];
            #     float32_t meaAmp_f3[3];
            #     s16 x,y,z,ax,ay,az,gx,gy,gz;
            #     uint8_t checksum;
            #     uint8_t tail;
            # }
            #获取电磁的数据
            data_mag = self.ser_rx.read(63)
            # 0代表左手,1代表右手
            # delt_time = struct.unpack("<L", data_mag[1:5])[0] - self.pre_time
            # print(delt_time)
            # self.pre_time = struct.unpack("<L", data_mag[1:5])[0]
            meaAmp_ = np.zeros(9)
            #获取vicon的数据
            self.client.GetFrame()
            receive1_left = self.client.GetMarkerGlobalTranslation("receive", "receive1")
            receive3_right = self.client.GetMarkerGlobalTranslation("receive", "receive3")
            receive4_front = self.client.GetMarkerGlobalTranslation("receive", "receive4")
            ###2023.8.16
            transmit1_left = self.client.GetMarkerGlobalTranslation("transmit", "transmit1")
            transmit3_right = self.client.GetMarkerGlobalTranslation("transmit", "transmit3")
            transmit4_front = self.client.GetMarkerGlobalTranslation("transmit", "transmit4")

            # 电磁数据解码
            for i in range(9):
                floatdata_mag = [data_mag[i * 4 + 5], data_mag[i * 4 + 6], data_mag[i * 4 + 7], data_mag[i * 4 + 8]]
                meaAmp_[i] = struct.unpack('<f', struct.pack('4B', *floatdata_mag))[0]
            #计算坐标
            temp_mag_pos = self.cal_mag_pos_fun(meaAmp_)
            self.mag_pos[:-1] = self.mag_pos[1:]
            self.mag_pos[-1] = temp_mag_pos
            self.relative_acc_mag = self.cal_accel_fun(self.mag_pos)
            # 线性加速度解码
            temp_acc = np.zeros(3)
            for i in range(3):
                acc_value = data_mag[(49 + i * 2):(51 + i * 2)]
                temp_acc[i] = (struct.unpack("<h", acc_value)[0]) / 100.0
            self.linear_acc_rx = temp_acc
            # 角速度解码
            for i in range(3):
                gyro_value = data_mag[(55 + i * 2):(57 + i * 2)]
                self.imu_gyro_rx[i] = (struct.unpack("<h", gyro_value)[0]) / 16.0
            # 四元数数据解码
            temp_quaternion = np.zeros(4)
            for i in range(4):
                quaternion_value = data_mag[(41 + i * 2):(43 + i * 2)]
                temp_quaternion[i] = struct.unpack("<h", quaternion_value)[0]
                self.quater_rx[i] = temp_quaternion[i]
            d_value = np.sqrt(
                temp_quaternion[0] ** 2 + temp_quaternion[1] ** 2 + temp_quaternion[2] ** 2 + temp_quaternion[
                    3] ** 2)
            q_w = temp_quaternion[0] / d_value
            q_x = temp_quaternion[1] / d_value
            q_y = temp_quaternion[2] / d_value
            q_z = temp_quaternion[3] / d_value
            #根据四元数计算旋转矩阵
            matrix_rx = np.array([[1 - 2*(q_y**2 + q_z**2),2*(q_x*q_y-q_w*q_z),2*(q_x*q_z+q_w*q_y)],
                                 [2*(q_x*q_y+q_w*q_z),1-2*(q_x**2+q_z**2),2*(q_y*q_z-q_w*q_x)],
                                 [2*(q_x*q_z-q_w*q_y),2*(q_y*q_z+q_w*q_x),1-2*(q_x**2+q_y**2)]
                                 ])
            # 求旋转矩阵的逆
            matrix_rx_inv = np.linalg.inv(matrix_rx)
            self.linear_acc_earth_rx = np.dot(matrix_rx_inv, self.linear_acc_rx)
            # 四元数转欧拉角
            angle_x = round(
                (np.arctan2(2 * (q_y * q_z + q_w * q_x), q_w * q_w - q_x * q_x - q_y * q_y + q_z * q_z)) * (
                        180. / np.pi), 2)
            angle_y = round(np.arcsin(-2 * (q_x * q_z - q_w * q_y)) * (180. / np.pi), 2)
            angle_z = round(
                np.arctan2(2 * (q_x * q_y + q_w * q_z), q_w * q_w + q_x * q_x - q_y * q_y - q_z * q_z) * (
                            180. / np.pi),
                2)
            self.euler_rx = np.array([angle_x,angle_y,angle_z])

            relative_acc = self.linear_acc_earth_tx - self.linear_acc_earth_rx
            # 画每一个线圈的值
            # self.plot_data_1[:-1] = self.plot_data_1[1:]
            # self.plot_data_1[-1] = angle_x
            self.plot_data_2[:-1] = self.plot_data_2[1:]
            self.plot_data_2[-1] = self.linear_acc_earth_rx[0]
            # self.plot_data_3[:-1] = self.plot_data_3[1:]
            # self.plot_data_3[-1] = meaAmp_[3]

            # self.plot_data_4[:-1] = self.plot_data_4[1:]
            # self.plot_data_4[-1] = angle_y
            self.plot_data_5[:-1] = self.plot_data_5[1:]
            self.plot_data_5[-1] = self.linear_acc_earth_rx[1]
            # self.plot_data_6[:-1] = self.plot_data_6[1:]
            # self.plot_data_6[-1] = meaAmp_[6]

            # self.plot_data_7[:-1] = self.plot_data_7[1:]
            # self.plot_data_7[-1] = angle_z
            self.plot_data_8[:-1] = self.plot_data_8[1:]
            self.plot_data_8[-1] = self.linear_acc_earth_rx[2]
            # self.plot_data_9[:-1] = self.plot_data_9[1:]
            # self.plot_data_9[-1] = meaAmp_[0]

            if self.saving_data_flag:
                write_line = str(meaAmp_[5]) + " " + str(meaAmp_[4]) + " " + str(meaAmp_[3]) + " " + \
                             str(meaAmp_[8]) + " " + str(meaAmp_[7]) + " " + str(meaAmp_[6]) + " " + \
                             str(meaAmp_[2]) + " " + str(meaAmp_[1]) + " " + str(meaAmp_[0]) + " "
                #保存vicon的坐标数据
                out_s = str(receive1_left[0][0]) + " " + str(receive1_left[0][1]) + " " + str(receive1_left[0][2])
                out_s += " " + str(receive3_right[0][0]) + " " + str(receive3_right[0][1]) + " " + str(
                    receive3_right[0][2])
                out_s += " " + str(receive4_front[0][0]) + " " + str(receive4_front[0][1]) + " " + str(
                    receive4_front[0][2])
                ###2023.8.16
                out_s += " " + str(transmit1_left[0][0]) + " " + str(transmit1_left[0][1]) + " " + str(
                    transmit1_left[0][2])
                out_s += " " + str(transmit3_right[0][0]) + " " + str(transmit3_right[0][1]) + " " + str(
                    transmit3_right[0][2])
                out_s += " " + str(transmit4_front[0][0]) + " " + str(transmit4_front[0][1]) + " " + str(transmit4_front[0][2]) + " "
                out_s += str(self.linear_acc_tx[0]) + " " + str(self.linear_acc_tx[1]) + " " + str(self.linear_acc_tx[2])\
                         + " " + str(self.imu_gyro_tx[0]) + " " + str(self.imu_gyro_tx[1]) + " " + str(self.imu_gyro_tx[2])\
                         + " " + str(self.quater_tx[0]) + " " + str(self.quater_tx[1]) + " " + str(self.quater_tx[2]) + " " + str(self.quater_tx[3]) + " "
                out_s += str(self.linear_acc_rx[0]) + " " + str(self.linear_acc_rx[1]) + " " + str(self.linear_acc_rx[2])\
                         + " " + str(self.imu_gyro_rx[0]) + " " + str(self.imu_gyro_rx[1]) + " " + str(self.imu_gyro_rx[2])\
                         + " " + str(self.quater_rx[0]) + " " + str(self.quater_rx[1]) + " " + str(self.quater_rx[2]) + " " + str(self.quater_rx[3]) + "\n"
                write_line += out_s
                self.fd.write(write_line)

    # 关闭串口
    def port_close(self):

        # 串口标志位你置零
        self.port_open_flag = False
        self.timer_plot_pos.stop()
        print("cloes port")
        self.thread_read_ser_tx.join()
        self.thread_read_ser_rx.join()
        self.ser_tx.close()
        self.ser_rx.close()
        # self.fd.close()
        self.timer_plot_pos.stop()
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        # 接收数据和发送数据数目置零
        self.formGroupBox.setTitle("串口状态（已关闭）")

if __name__ == '__main__':
    # 异常获取模块
    _oldExceptionCatch = sys.excepthook

    def _exceptionCatch(exceptionType, value, traceback):
        _oldExceptionCatch(exceptionType, value, traceback)

    # 由于Qt界面中的异常捕获不到
    # 把系统的全局异常获取函数进行重定向
    sys.excepthook = _exceptionCatch
    # QT显示模块
    app = QtWidgets.QApplication(sys.argv)
    myshow = Pyqt5_Serial()
    myshow.show()

    sys.exit(app.exec_())
