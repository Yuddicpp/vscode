function Var2 = scalar_array_new(rfy,rfw)

x=[0 pi/3 pi*2/3 pi 4*pi/3 -pi/3];
r=0.059;
c=3e8;%光速
f=2.44*10^9;%接受信号频率
lambda=c/f;%波长
for ii=1:6
    aa=exp(-1j*(2*pi*r/lambda)*sin(rfy)*cos(rfw-x(ii)));
    dfw(ii,1) = 1j*(2*pi*r/lambda)*sin(rfy)*sin(rfw-x(ii))*aa;  %对方位角求导数
    dfy(ii,1) = -1j*(2*pi*r/lambda)*cos(rfw-x(ii))*cos(rfy)*aa;    %对俯仰角求导数
    dfwh(1,ii) = -1j*(2*pi*r/lambda)*sin(rfy)*sin(rfw-x(ii))*conj(aa);    %共轭转置后对方位角求导数
    dfyh(1,ii) = 1j*(2*pi*r/lambda)*cos(rfw-x(ii))*cos(rfy)*conj(aa);
end
A=scalar_A(rfy, rfw, r, lambda);   %求得导向矢量
flag=0;
for kk=-10:0.1:20
    flag=flag+1;
    amp_s=10;
    sigma_2=amp_s^2/(2*10^(kk/10));
    C=diag(sigma_2*ones(1,6));
    FIM(1,1) = 2*real(dfwh*inv(C)*dfw);
    FIM(1,2) = 2*real(dfwh*inv(C)*dfy);
    FIM(2,1) = 2*real(dfyh*inv(C)*dfw);
    FIM(2,2) = 2*real(dfyh*inv(C)*dfy);
    Var1(:,flag)=diag(inv(FIM));
% Var=Var.*180/pi;
end
% Var1=mean(Var,3);
% save('CRB.mat','Var1');
Var2(:,1) = sqrt(Var1(1,:)*180/pi/8); %degree
Var2(:,2) = sqrt(Var1(2,:)*180/pi/8); %degree
% plot(-10:0.1:20,Var2(:,1));
% hold on
% plot(-10:0.1:20,Var2(:,2));
% legend('Root CRLB vary ori(\theta)','Root CRLB vary ori(\phi)');
% xlabel('SNR');
% ylabel('angle error');
end




function A = scalar_A(rfy, rfw, r, lambda)
x=[0 pi/3 pi*2/3 pi 4*pi/3 -pi/3];
for ii=1:6
   A(ii,1) =  exp(-1j*(2*pi*r/lambda)*sin(rfy)*cos(rfw-x(ii)));
end
end