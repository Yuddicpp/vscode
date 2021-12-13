clc;
clear;
close all;
flag=0;
for fw=0:1:359
    flag=flag+1;
[ground_phase,ground_amp]=groundphase(fw,0,0);
gp(1:12,flag)=ground_phase(:,1);
ga(1:12,flag)=ground_amp(:,1);
end
% filepath='G:\bluetooth\ant13\data_laoyang\0505\线极化\00000.txt';
filepath = 'E:\滴滴\汪博文\蓝牙(2)\13天线\data\data_laoyang\0505\线极化\00000.txt'
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_12_5数据测试\h=3m顺时针8圈.txt';
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_12_5数据测试\h=1.5m2圈.txt';
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_12_5数据测试\h1.2m4圈.txt';
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_12_5数据测试\0.8m3圈.txt';

[test_phase, test_amp]=testphase(filepath);

% for i = 1:12
%     subplot(3,4,i);
%     plot(1:size(test_phase,2),test_phase(i,:));
% end
for i = 1:12
    subplot(3,4,i);
    antenna = i;
    gp(antenna,find(gp(antenna,:)<-pi*9/10)) = pi;
    plot((0:1:359)*pi/180,gp(antenna,:));
    hold on;
    for j = 1:size(test_phase,2)/16
        test_phase(:,j) = test_phase(:,(j-1)*16+1);
    end
    test_phase(antenna,find(test_phase(antenna,:)<-pi*9/10)) = pi;
    plot((0:1:359)*pi/180,test_phase(antenna,1:360));
end
