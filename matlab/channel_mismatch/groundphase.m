function [phase2,amp1] = groundphase(fw,fd,xw)
%===============================�����趨===================================
if nargin<3
   fw=[0]; 
end
M=6;%��Ԫ��
c=3*10^8;%����
f=2.44*10^9;%�����ź�Ƶ��
lamda=c/f;%���� 
% r=lamda/(4*sin(pi/8));%��ģ��������а뾶
r=0.05;
kp=16;%��������
% fw=[40 60];%#ok<*NBRAK> %�ź�Դ��λ�Ƕ�0--360
% fy=[30 57];%�ź�Դ�����Ƕ�0--60
fy=[0 0];
% fd=[0 50];%�ź�Դ������������0--90
% xw=[0 60];%�ź�Դ������λ����0--360
Num=length(fw);%�ź�Դ��
radian=pi/180;%���ȵ�λ
rfw=fw*radian;%��λ�ǻ���
rfy=fy*radian;%�����ǻ���
rfd=fd*radian;%�ź�Դ�����������Ի���
rxw=xw*radian;%�ź�Դ������λ���Ի���
A=ones(2*M,Num);%����ռ伫������ʸ��
A=gene_A(rfy(1), rfw(1),rfd(1),rxw(1));
%===========================�ռ��ź�ʸ������===============================
sr=ones(Num,kp);%����ռ��ź�ʸ��ʵ��
si=ones(Num,kp);%����ռ��ź�ʸ���鲿
s=ones(Num,kp);%����ռ��ź�ʸ��
%s_ce=0;       %�ռ��ź�ʸ���Ĳ�����
for m=1:Num    %�ռ��ź�ʸ����ֵ
    for n=1:kp        
        p1=rand(1,1);            %����1*1�����
        p2=rand(1,1);            %����1*1����� 
        sr(m,n)=cos(2*pi*p2); %ʵ����ֵ       
        si(m,n)=sin(2*pi*p2); %�鲿��ֵ      
        s(m,n)=sr(m,n)+1i*si(m,n);   
        %s_ce=s_ce+s(m,n);   %�ռ��ź�ʸ������
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






