function [Legendre_Polynomials, Derivative_Legendre]=Legendre(x,Num_Legendre)
Legendre_Polynomials = linspace(0,0,Num_Legendre); % 一维数组Legendre_Function用来储存勒让德多项式的值
Derivative_Legendre = linspace(0,0,Num_Legendre); %一维数组Derivative_Legendre用来储存勒让德多项式一阶导数在x=x0位置的值
Legendre_Polynomials(1) = 1; % 数组标号从1开始，1号对应P0(x)=1
Legendre_Polynomials(2) = x; % P1(x)=x
Derivative_Legendre(1) = 0; % P0'(x)=0
for k=1:Num_Legendre-2   % Num_Legendre至少大于1000
    Legendre_Polynomials(k+2)=((2*k+1)*x*Legendre_Polynomials(k+1)-k*Legendre_Polynomials(k))/(k+1);
end
for k=1:Num_Legendre-1   % Num_Legendre至少大于1000
    Derivative_Legendre(k+1)=x*Derivative_Legendre(k)+k*Legendre_Polynomials(k);
end

end