function phase_12 = circular_polarization
%%%%%%%%%%%仿真滴滴圆极化天线阵列的理论值，用于校准%%%%%%%%%%%%%%%
%1.信号生成
% amp_s=10;
% phi0_s=pi/3;
% for ii=1:12
%     s(ii,1)=amp_s*sin(pi*ii/6+phi0_s);
% end
%2.给定相关参数
fw=[0];
fy=[10.9197 0];
Num=length(fw);%信号源数
radian=pi/180;%弧度单位
rfw=fw*radian;%方位角弧度
rfy=fy*radian;%俯仰角弧度
%%%%%%%%%初始时刻，12根天线的方位角%%%%%%%
x=[0 pi/6 pi/3 pi/2 pi*2/3 pi*5/6 pi 7*pi/6 8*pi/6 9*pi/6 10*pi/6 11*pi/6];
%%%%%%%%%校准需要旋转一周%%%%%%%%%%%%%%%%
for i = 0 : 30 : 330
    steering_vector(:,i/30+1) = scalar_array(rfy(1), rfw(1), x+i*radian);
end
%%%%%%%%%获取相位和幅度%%%%%%%%%%%%%%%%%%%%%%
phase_12 = angle(steering_vector);
for i = 12:-1:1
    phase_12(i,:) = phase_12(i,:)-phase_12(1,:);
end
% amp_12 = real(steering_vector);
% subplot(1,2,1);
% plot(1:12,amp_12,'--');
% subplot(1,2,2);
% plot(1:12,phase_12,'-');

end

function A = scalar_array(rfy, rfw, x)
%%%%%%%由于滴滴是12根天线，我们按照圆形阵列处理%%%%%%%%%%%
    for ii=1:12
        A(ii,1) =  exp(-1j*pi*sin(rfy)*cos(rfw-x(ii)));
    end
end

