function [A,K,var] = LSE(r)
N = 100;
x = [1:N];
A = [1:N];
% r = 1;
K = [1:N];
var = [1:N];
k_sum = 1 + 1/r;

% 生成信号值
for i = 1:N
    x(i) = 10 + randn() * sqrt(r^i);
end

A_0 = 10 + randn();
x_0 = A_0;
var_0 = 1;
K(1) = (1/r)/k_sum;
A(1) = A_0 + K(1)*(x(1)-A_0);
var(1) = (1-K(1))*var_0;
for i = 2:N
    k_sum = k_sum + (1/(r^i));
    K(i) = (1/(r^i)) / k_sum;
    A(i) = A(i-1) + K(i)*(x(i)-A(i-1));
    var(i) = (1-K(i))*var(i-1);
end

end