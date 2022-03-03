function [phase2,amp1] = groundphase(fw,fd,xw)
%===============================参数设定===================================
if nargin<3
   fw=[0]; 
end
M=6;%阵元数
c=3*10^8;%光速
f=2.44*10^9;%接受信号频率
lamda=c/f;%波长 
% r=lamda/(4*sin(pi/8));%不模糊最大阵列半径
r=0.05;
kp=16;%采样点数
% fw=[40 60];%#ok<*NBRAK> %信号源方位角度0--360
% fy=[30 57];%信号源俯仰角度0--60
fy=[0 0];
% fd=[0 50];%信号源极化幅度特性0--90
% xw=[0 60];%信号源极化相位特性0--360
Num=length(fw);%信号源数
radian=pi/180;%弧度单位
rfw=fw*radian;%方位角弧度
rfy=fy*radian;%俯仰角弧度
rfd=fd*radian;%信号源极化幅度特性弧度
rxw=xw*radian;%信号源极化相位特性弧度
A=ones(2*M,Num);%定义空间极化导向矢量
A=gene_A(rfy(1), rfw(1),rfd(1),rxw(1));
%===========================空间信号矢量快拍===============================
sr=ones(Num,kp);%定义空间信号矢量实部
si=ones(Num,kp);%定义空间信号矢量虚部
s=ones(Num,kp);%定义空间信号矢量
%s_ce=0;       %空间信号矢量的测量量
for m=1:Num    %空间信号矢量赋值
    for n=1:kp        
        p1=rand(1,1);            %产生1*1随机数
        p2=rand(1,1);            %产生1*1随机数 
        sr(m,n)=cos(2*pi*p2); %实部赋值       
        si(m,n)=sin(2*pi*p2); %虚部赋值      
        s(m,n)=sr(m,n)+1i*si(m,n);   
        %s_ce=s_ce+s(m,n);   %空间信号矢量测量
    end
end


snr = 50;
% X=awgn(A*s,snr,'measured');
% Rx=X*X';
% para2(flag,ii,:)=RD_MUSIC(Rx);
X=A*s;
phase1=angle(X);
amp1=abs(X);
for ii=1:12
   amp1(ii,:)=amp1(ii,:)./amp1(1,:); 
end
amp1(1,:)=amp1(1,:)./amp1(1,:);
for ii=2:12
    phase1(ii,:)=wrapToPi(phase1(ii,:)-phase1(1,:));
end
phase1(1,:)=phase1(1,:)-phase1(1,:);
phase2=phase1;
phase3=median(phase2,2);

end






