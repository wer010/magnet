% 轴向磁化的圆柱永磁体，计算空间外一点P处的磁感应强度
% 主函数
Num_Legendre = 10000; % 勒让德函数的求和项
Legendre_Polynomials_L0=PL0(Num_Legendre); % PL(0)的值，储存在一维数组中
delta_small = 10^(-20); % 加一个小常数，防止除法出现无穷大

a=0.005;   % 圆柱体底面半径，单位m
H_Cylinder=0.010;   % 圆柱体高度，单位m
Br_Cylinder=1.15231;   % 剩磁，单位T

x_initial = 0+delta_small;   % 扫描的起始位置（单位m）
x_final = 0.03;  % 扫描的终止位置（单位m）
step_value_x = 0.001;  % 扫描步长值（单位m）
y_initial = 0+delta_small;   % 扫描的起始位置（单位m）
y_final = 0.03;  % 扫描的终止位置（单位m）
step_value_y = 0.001;  % 扫描步长值（单位m）
z_initial = 0+delta_small;   % 扫描的起始位置（单位m）
z_final = 0.03+delta_small;  % 扫描的终止位置（单位m）
step_value_z = 0.001;  % 扫描步长值

Num_Out_x=ceil((x_final-x_initial)/step_value_x)+1;  % 程序一共输出多少个值（x方向）
Num_Out_y=ceil((y_final-y_initial)/step_value_y)+1;  % 程序一共输出多少个值（y方向）
Num_Out_z=ceil((z_final-z_initial)/step_value_z)+1;  % 程序一共输出多少个值（z方向）
Num_Out=Num_Out_x*Num_Out_y*Num_Out_z;

% 初始设置
%x=-0.05; % 给一个小值，防止出现0/0的情形
%y=-0.05;
%z=0+delta_small; % 空间一点P到圆柱上表面所在平面的垂直距离 

% 以圆柱永磁体的中心为坐标原点，求空间一点P(x0, y0, z0)处的磁感应强度（单位：T）
x0 = linspace(0,0,Num_Out);   % 一维数组用来储x0的值
y0 = linspace(0,0,Num_Out);   % 一维数组用来储y0的值
z0 = linspace(0,0,Num_Out);   % 一维数组用来储z0的值
z1 = linspace(0,0,Num_Out);   % z1是点P到圆柱上表面的垂直距离（程序输出时使用）

% 以圆柱永磁体的中心为坐标原点，求空间一点P(x0, y0, z0)处的磁感应强度（单位：T）
B1 = linspace(0,0,Num_Out);   % 一维数组用来储x0的值
B2 = linspace(0,0,Num_Out);   % 一维数组用来储y0的值
B3 = linspace(0,0,Num_Out);   % 一维数组用来储z0的值

kk=0;  %指标kk用来标记储存位置

for k=1:Num_Out_z   % 对x0,y0,z0赋值   
    for j=1:Num_Out_y         
        for i=1:Num_Out_x   % 对x0,y0,z0赋值
        
            kk=i+Num_Out_x*(j-1)+Num_Out_x*Num_Out_y*(k-1);
        
            x0(kk)=x_initial+(i-1)*step_value_x+delta_small;
            y0(kk)=y_initial+(j-1)*step_value_y+delta_small;
            z0(kk)=z_initial+H_Cylinder/2+(k-1)*step_value_z+delta_small;
            z1(kk)=z_initial+(k-1)*step_value_z+delta_small;   % z1是点P到圆柱上表面的垂直距离（程序输出时使用）
            
            [B1(kk), B2(kk), B3(kk)]=Magnetic_Flux_Density_of_Cylinder(a,H_Cylinder,Br_Cylinder,x0(kk),y0(kk),z0(kk),Num_Legendre,Legendre_Polynomials_L0);
        end
    end
end

for p=1:Num_Out   % 单位变化，坐标的单位由米变为毫米，磁感应强度的单位由T变为mT
    x0(p)= x0(p)*1000;
    y0(p)= y0(p)*1000;
    z0(p)= z0(p)*1000;
    z1(p)= z1(p)*1000;   % z1是点P到圆柱上表面的垂直距离（程序输出时使用）
    B1(p)= B1(p)*1000;
    B2(p)= B2(p)*1000;
    B3(p)= B3(p)*1000;
end

data = [x0', y0', z0', z1', B1', B2', B3'];   % 将数据组集到data
[m, n] = size(data);            
data_cell = mat2cell(data, ones(m,1), ones(n,1));    % 将data切割成m*n的cell矩阵

title = {'x0 (mm)', 'y0 (mm)', 'z0 (mm)', 'z (mm)（点P到圆柱上表面的距离）', 'B1 (mT)', 'B2 (mT)', 'B3 (mT)'};    % 添加变量名称
result = [title; data_cell];    % 将变量名称和数值组集到result

string1='D:\Cylinder_圆柱永磁体外的磁感应强度3D-距离上表面高度';
string2=num2str(z_initial*1000);
string3=' mm.xls';
string_save=strcat(string1,string2,string3);
s = xlswrite(string_save, result);      % 将result写入到文件中