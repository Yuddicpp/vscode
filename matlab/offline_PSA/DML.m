function loc=DML(IQ,P,filepath)
% P=gene_DML_P();
loc=run_algorithm(IQ,P,filepath);
end

function loc=run_algorithm(IQ,DML_P,filepath)
%%
%角度参数估计算法
%输入：补偿了信号频率误差后的IQ数据
%输出：位置估计结果
page=size(IQ,3);

% phi=(0:1:359)*pi/180; theta=(0:1:60)*pi/180;

loc1=[];
his_loc=[];
ind=1;
M = [0 0 1 0 0]';
P = diag([10.1 10.1 1.1 1.1 1]);
for ii=1:page
    x=squeeze(IQ(:,:,ii));
    Pmusic=spectrum(DML_P,x);
    para(ii,1:2) = find_peaks1(Pmusic);
%     mesh(Pmusic)
    loc(ii,:)=location(para(ii,:),filepath);
    
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    %容积卡尔曼滤波
    if exist('MM_CKF','var')
        
        [his_loc,new_loc,M,P,ind,MM_CKF,PP_CKF]=filter_ckf(his_loc,loc(ii,:).',M,P,ind,MM_CKF,PP_CKF);
    else
        [his_loc,new_loc,M,P,ind,MM_CKF,PP_CKF]=filter_ckf(his_loc,loc(ii,:).',M,P,ind);
    end
    loc1=[loc1 new_loc];
    ii
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
% plot(loc1(1,:),loc1(2,:));
track_plot(loc1.');
end

function Pmusic=spectrum(P,x)
%%
phi=(0:1:359)*pi/180; theta=(0:1:60)*pi/180;
R=x*x';
Pmusic=zeros(length(theta),length(phi));
for ii=1:length(theta)
   for jj=1:length(phi)
%        Q=gene_Q(theta(ii),phi(jj));
%        P=Q{ii,jj}*pinv(Q{ii,jj});
       Pmusic(ii,jj)=abs(trace(P{ii,jj}*R));
   end
end
end

function Q=gene_Q(theta,phi)
%%
R=0.059;
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

beta = ant_direction();

L=[-sin(phi) cos(phi)*cos(theta);cos(phi) sin(phi)*cos(theta)];
% L=[-sin(phi) cos(phi)*sin(theta);cos(phi) sin(phi)*cos(theta)];
% P=[cos(gama);sin(gama)*exp(1j*yita)];
D=U*beta*L;
amp_offset=[342 496 279 387 393 310 304 380 344 430 275 400];
phase_offset=exp(1j.*[0 2.1 -2.4 -2.4 -1.35 0.8 -2.35 -2.4 -1.4 0.8 -1.1 -2.14]);
offset=amp_offset.*phase_offset;                                            
B=diag(offset);                                                             %误差增益矩阵

Q=B*D;
end

function beta = ant_direction()
%%

dir_ant=[180 270 120 210 60 150 0 90 300 30 240 330]*pi/180;                %天线朝向
beta=zeros(12,2);
for ii=1:12
   beta(ii,:)=[cos(dir_ant(ii)) sin(dir_ant(ii))]; 
end
end

function para1 = find_peaks1(Pmusic)
%%
[rows,cols]=find(Pmusic==max(max(Pmusic)));
para1 = [rows,cols];
end

function para1 = find_peaks(Pmusic)
%%
bw_music = imregionalmax(Pmusic);
[para(:,1), para(:,2)]=find(bw_music==1);
id=para(:,1)==size(Pmusic,1);
para(id,:)=[];
id=para(:,1)==1;
para(id,:)=[];
if(~isempty(para))
    maxval=0;
    flag=1;
    for ii=1:size(para,1)
        if Pmusic(para(ii,1),para(ii,2))>maxval
           flag=ii; 
           maxval=Pmusic(para(ii,1),para(ii,2));
        end
    end
    azimuth = para(flag,2)*360/size(Pmusic,2);
    elevation = para(flag,1)*60/(size(Pmusic,1)-1);
else
    [maxval,ind] = max(Pmusic(:));
    s=size(Pmusic);
    [i,j]=ind2sub(s,ind);
    para=[i,j];
    subPmusic=Pmusic(:,j);
    diff_subPmusic=diff(subPmusic,2);
    [~,sub_n]=min(diff_subPmusic);
    elevation=sub_n*60/(size(Pmusic,1)-1);
    azimuth=j*360/size(Pmusic,2);
end
para1=[elevation,azimuth];
end

function loc=location(para1,filepath)
%%
height=0.54;                                                                   %基站与标签高度差，须设置与实际一致
para=para1*pi/180;
para(:,2)=pi-para(:,2);                                                         %由于补偿的缘故，方位角需要偏转pi
para1(:,2)=180-para1(:,2); 
aa=tan(para(:,1))*height;
loc=[aa.*cos(para(:,2)),aa.*sin(para(:,2))];

writematrix([loc,para1],[filepath,'.xlsx'],'WriteMode','append');
    

end


function track_plot(pos)
%%可视化位置估计结果
axis([-6,6,-6,6]);
grid on;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
set(gca,'XTick',(-6:0.2:6));
set(gca,'YTick',(-6:0.4:6));
set(gca,'Color',[0 0 0]);
hold on;

N = 2;  % 想要的坐标轴显示，即隔一个网格显示一个刻度，10/5 = 2
% 设置想要的坐标轴刻度
a = get(gca,'XTickLabel');  
b = cell(size(a));
b(mod(1:size(a,1),N)==1,:) = a(mod(1:size(a,1),N)==1,:);
set(gca,'XTickLabel',b);

plot(pos(:,1),pos(:,2),'.','MarkerSize',30,'MarkerFaceColor','r');
hold on;
% str1=[num2str(roundn(pos(:,1),-3)),',',num2str(roundn(pos(:,2),-3))];
% text(pos(:,1)+0.05,pos(:,2)+0.05,str1,'color','r');
plot(0,0,'^','MarkerFaceColor',[rand rand rand],'MarkerSize',12,'MarkerEdgeColor','k');

% ax = gca;
% ax.GridColor = [1, 1, 1];
% pause(0.01);
% clf;
end


