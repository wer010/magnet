% 以圆柱永磁体的中心为坐标原点，计算永磁体外部空间中一点(x0,y0,z0)处的磁感应强度(B1,B2,B3)
% 圆柱永磁体半径为a，高度为H，沿着轴向磁化，剩磁为Br
function [B1, B2, B3]=Magnetic_Flux_Density_of_Cylinder(a,H,Br,x0,y0,z0,Num_Legendre,Legendre_Polynomials_L0) 
[B1_up, B2_up, B3_up] = Magnetic_Flux_of_Circle(a,Br,x0,y0,z0-H/2,Num_Legendre,Legendre_Polynomials_L0);
[B1_down, B2_down, B3_down] = Magnetic_Flux_of_Circle(a,-Br,x0,y0,z0+H/2,Num_Legendre,Legendre_Polynomials_L0);
B1=B1_up+B1_down;
B2=B2_up+B2_down;
B3=B3_up+B3_down;
end