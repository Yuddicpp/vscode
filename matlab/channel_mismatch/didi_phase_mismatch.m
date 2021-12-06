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
filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_12_5数据测试\顺时针8圈.txt';
[test_phase, test_amp]=testphase(filepath);
% 第三根天线
plot((0:1:359)*pi/180,gp(3,:));
hold on;
plot((0:1:359)*pi/180,test_phase(3,1:360));