function RD_MUSIC(IQ)
theta=(0:1:359)*pi/180; alpha=(0:1:90)*pi/180; gama=(0:1:90)*pi/180; yita=(0:1:359)*pi/180;
% fudu(IQ);
run_algorithm(IQ);


end



function run_algorithm(IQ)
page=size(IQ,3);
his=[1 1];
P=eye(4);
for ii=1:page
   x=squeeze(IQ(:,:,ii));
%    R=x*x';
%    para=test_func(R);ans
   [eigenvectors num_signal] = noise_space_eigenvectors(x);
   Pmusic = music_spectrum(eigenvectors);
   para(ii,1:2) = find_peaks(Pmusic);
%    figure();
%    set(0,'defaultfigurecolor','w');
%    mesh(Pmusic);
%    xlabel('azimuth');
%    ylabel('elevation');
%    title('pseudo-spectrum');
   
   loc=location(para);
%===============================================
%%%  add kalman filter
%    [his,P]=track1(loc(end,:),his,P);
%    track_plot(his(end,:));
%===============================================
    track_plot(loc(end,:));
end

end

function [eigenvectors num_signal] = noise_space_eigenvectors(x)
    % Data covarivance matrix
    R = x * x'; 
    
    [eigenvectors,eigenvalue_matrix]=eig(R);%%%% 
    EVA=diag(eigenvalue_matrix)';
    [EVA,I]=sort(EVA);
    eigenvectors=eigenvectors(:,I);%锟斤拷锟斤拷
    
    for i=1:length(EVA)-1%-2表示总是有多径
        de_ratios(i)=EVA(i)/EVA(i+1);
    end
%     [max_de_ratio,max_de_index]=min(de_ratios);
    [max_de_ratio,max_de_index1]=min(de_ratios(9:end));
    max_de_index=max_de_index1+8;
    eigenvectors = eigenvectors(:,1:max_de_index);
    num_signal=12-max_de_index;
%     eigenvectors = eigenvectors(:,1:25);
%     num_signal=30-25;
end

function Pmusic = music_spectrum(eigenvectors)
    theta=(0:2:60)*pi/180; phi=(0:2:359)*pi/180;gamma=(0:5:90)*pi/180; eta=(0:5:359)*pi/180;
    Pmusic = zeros(length(theta), length(phi));
    % Angle of Arrival Loop (AoA)
    for ii = 1:length(theta)
        % Time of Flight Loop (ToF)
        for jj = 1:length(phi)
%             for mm=1:length(gama)
%                 for nn=1:length(yita)
                    steering_vector = compute_steering_vector(theta(ii), phi(jj));
                    PP = det(steering_vector' * (eigenvectors * eigenvectors') * steering_vector);
                    Pmusic(ii, jj) = 1/abs(PP);                 
%                 end
%             end
        end
    end

    % Convert to decibels
    % ToF loop
%     for jj = 1:size(Pmusic, 2)
%         % AoA loop
%         for ii = 1:size(Pmusic, 1)
%             Pmusic(ii, jj) = 10 * log10(Pmusic(ii, jj));% / max(Pmusic(:, jj))); 
%         end
%     end
end

function steering_vector = compute_steering_vector(theta, phi)
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
B=diag(offset);

steering_vector=B*D;
end


function beta = ant_direction()
% dir_ant=[0 90 60 -30 120 210 180 270 240 330 300 210]*pi/180;
% dir_ant=[0 90 60 150 120 210 180 270 240 330 300 30]*pi/180;
dir_ant=[180 270 120 210 60 150 0 90 300 30 240 330]*pi/180;
beta=zeros(12,2);
for ii=1:12
   beta(ii,:)=[cos(dir_ant(ii)) sin(dir_ant(ii))]; 
end
end

function para1 = find_peaks(Pmusic)

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

function loc=location(para)
height=2;
para=para*pi/180;
para(:,2)=para(:,2)-pi; %由于补偿的缘故，方位角需要偏转pi
aa=tan(para(:,1))*height;
loc=[aa.*cos(para(:,2)),aa.*sin(para(:,2))];

end

function track_plot(pos)

axis([-3.5,3.5,-3.5,3.5]);
grid on;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
set(gca,'XTick',(-3.5:0.1:3.5));
set(gca,'YTick',(-3.5:0.2:3.5));
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

ax = gca;
ax.GridColor = [1, 1, 1];
pause(0.01);
clf;
end
