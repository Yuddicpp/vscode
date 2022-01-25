clc;
clear;
close all;
flag=0;
for fw=0:0.069:360
    flag=flag+1;
    [ground_phase,ground_amp]=groundphase(fw,0,0);
    gp(1:12,flag)=ground_phase(:,1);
    ga(1:12,flag)=ground_amp(:,1);
end

% filepath = 'E:\滴滴\汪博文\蓝牙(2)\13天线\data\data_laoyang\0505\线极化\00000.txt'
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_1_19数据\h=1.6米顺时针4圈.txt'
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_1_19数据\1圈 (2).txt';
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_1_20数据\4圈1米.txt';
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_1_21数据\new5圈.txt';
filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_1_24数据\1.9米.txt';

[test_phase, test_amp]=testphase(filepath);

% for i = 1:12
%     subplot(3,4,i);
%     plot(1:size(test_phase,2),test_phase(i,:));
% end


plot_mismatch(gp, test_phase);

test_phase = test_phase(:,1:5216);

figure();
for ii=1:12
   subplot(3,4,ii);
   plot(test_phase(ii,:));
end
                              
