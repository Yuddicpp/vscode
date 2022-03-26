


%%
filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_2_25数据\x=-0.5m,y=2.5m.mat';
% data = load(filepath,'IQ').IQ;
% l = size(data,3);
% data = reshape(data,12,16*l);
% MUSIC_CARPEN(data(:,:),16*l);


%%
filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_3_19数据\1.5m\x=1.5m,y=0m_铁板_Music.mat';
loc_x=1.5;
loc_y=0;
% 2022.3.18 2.796-0.171
H=1.5;
if(loc_x==0&&loc_y==0)
    phi = 0;
    theta = 0;
else
%     # 俯仰角
    phi = atan((loc_x^2+loc_y^2)^0.5/H);
%     # 方位角
    theta = acos(loc_x/((loc_x^2+loc_y^2)^0.5));
end

if(loc_y<0)
    theta = -theta;
end
theta = theta/pi*180;
phi = phi/pi*180;
theta = 180 - theta;


data = load(filepath,'MUSIC_all').MUSIC_all;
l = size(data,3);

for i = 1:5
    close all
    figure
    imagesc(data(:,:,i));
    colorbar
    hold on
    plot(theta,phi,'*r');
    
    hold on
    [rows,cols]=find(data(:,:,i)==max(max(data(:,:,i))));
    plot(cols,rows,'*b');

    hold off
    pause(1);
end
