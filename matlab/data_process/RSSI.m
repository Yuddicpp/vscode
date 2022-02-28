%%
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_12_13数据\';
% files =  dir(fullfile(filepath,'*_RSSI.mat'));
% 
% figure();
% hold on;
% for i = 1:72
%     file = [files(i).folder,'\',files(i).name];
%     data = load(file).rssi;
%     k = mean(data(1,:));
%     j = std(data(1,:));
%     l = length(data);
%     for m = 1:l
%         if(abs(data(1,m)-k)>(3*j))
%             data(1,m) = k;
%         end
%     end
%     plot(data);
% end


%%
filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_2_16数据\';
files =  dir(fullfile(filepath,'*.mat'));