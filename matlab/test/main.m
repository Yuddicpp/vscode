
r = [1,0.95,1.05];
A = zeros(3,100);
K = zeros(3,100);
var = zeros(3,100);

for i = 1:3
    [A(i,:),K(i,:),var(i,:)] = LSE(r(i));
end

figure(1)
for i = 1:3
    plot(A(i,:));
    hold on;
end
legend('r=1','r=0.95','r=1.05');
title("A");

figure(2)
for i = 1:3
    plot(K(i,:));
    hold on;
end
legend('r=1','r=0.95','r=1.05');
title("K");

figure(3)
for i = 1:3
    plot(var(i,:));
    hold on;
end
legend('r=1','r=0.95','r=1.05');
title("Var");