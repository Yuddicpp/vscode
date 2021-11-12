function phase_12 = dual_polarization


% clear all;
% clc;
%%%%%%%%%%%仿真双极化天线阵列的理论值，用于校准%%%%%%%%%%%%%%%
%1.信号生成
% amp_s=10;
% phi0_s=pi/3;
% for ii=1:12
%     s(ii,1)=amp_s*sin(pi*ii/6+phi0_s);
% end
%2.给定相关参数
fw=[0];
fy=[90 0];
fd=[45 50];%信号源极化幅度特性0--90
xw=[90 60];%信号源极化相位特性0--360
Num=length(fw);%信号源数
radian=pi/180;%弧度单位
rfw=fw*radian;%方位角弧度
rfy=fy*radian;%俯仰角弧度
rfd=fd*radian;%信号源极化幅度特性弧度
rxw=xw*radian;%信号源极化相位特性弧度

x = [180 270 120 210 60 150 0 90 300 30 240 330]*pi/180;
%%%%%%%%%校准需要旋转一周%%%%%%%%%%%%%%%%
for i = 0 : 30 : 330
    steering_vector(:,i/30+1) = compute_steering_vector(rfy(1), rfw(1),rfd(1),rxw(1), x+i*radian);
end
%%%%%%%%%获取相位%%%%%%%%%%%%%%%%%%%%%%
phase_12 = angle(steering_vector);
amp_12 = real(steering_vector);
for i = 12:-1:1
   phase_12(i,:) = phase_12(i,:)-phase_12(1,:);
end
% subplot(1,2,1);
% plot(1:360,amp_12(10,:),'--');
% subplot(1,2,2);
% plot(1:360,phase_12(10,:),'-');
% for i = 5 : 5
%     plot(1:size(steering_vector,2),phase_12(i,:)-phase_12(1,:), '-', 'LineWidth',2);
%     hold on;
% end


end

function steering_vector = compute_steering_vector(theta, phi, gamma, eta, dir_ant)
R=0.059;     %%%%%半径
c=3e8;
f=2.44e9;
A=2*pi*f/c;
array=zeros(6,3);
for ii=1:6
%    array(ii,:)=R*[cos(pi/3*(ii-1)) sin(pi/3*(ii-1)) 0];
    array(ii,:)=R*[cos(pi-pi/3*(ii-1)) sin(pi-pi/3*(ii-1)) 0];
end

r=[sin(theta)*cos(phi) sin(theta)*sin(phi) cos(theta)];
U=zeros(12,12);
for ii=1:6
    tmp=exp(-1j*array(ii,:)*r.'*A);
    U((ii-1)*2+1,(ii-1)*2+1)=tmp;
    U(ii*2,ii*2)=tmp;
end

beta = ant_direction(dir_ant);

L=[-sin(phi) cos(phi)*cos(theta);cos(phi) sin(phi)*cos(theta)];
P=[cos(gamma);sin(gamma)*exp(1j*eta)];
D=U*beta*L*P;
steering_vector=D;
end


function beta = ant_direction(dir_ant)
% dir_ant=[180 270 120 210 60 150 0 90 300 30 240 330]*pi/180;
% dir_ant=[0 90 0 90 0 90 0 90 0 90 0 90]*pi/180;
beta=zeros(12,2);
for ii=1:12
   beta(ii,:)=[cos(dir_ant(ii)) sin(dir_ant(ii))]; 
end
end