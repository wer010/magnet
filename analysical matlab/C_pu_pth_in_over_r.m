% 磁标势对theita的偏导数除以r（当r<a时）
function pu_pth_in_over_r=C_pu_pth_in_over_r(a,Br,r,Legendre_Polynomials_L0,Derivative_Legendre_costh,sinth,Num_Legendre)
c_sum = 0;
mu0 = 4*pi*10^(-7);
for k = 2:floor(Num_Legendre/2)   % 注意求和起点为k==2，floor为向下取整函数
    c_sum = c_sum +(1/(2*k-3)+1/(2*k)-1/(2*k-3)*(r/a)^(2*k-3))*Legendre_Polynomials_L0(2*k-2+1)*Derivative_Legendre_costh(2*k-2+1);
end
pu_pth_in_over_r=-Br/(2*mu0)*sinth*c_sum;