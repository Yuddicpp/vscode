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
% filepath='E:\滴滴\汪博文\蓝牙(2)\13天线\data\data_laoyang\0505\线极化\00000.txt';
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_1_19数据\h=1.6米顺时针4圈.txt';
filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_1_19数据\1圈 (2).txt';
[test_phase, test_amp]=testphase(filepath);
plot_mismatch(gp, test_phase);
% plot_mismatch_amp();

figure();
for ii=1:12
   subplot(3,4,ii);
   plot(test_phase(ii,:));
end
page=size(test_phase,2)/16;
for ii=1:page
   ground(1:12,(ii-1)*16+1:ii*16)= ground_phase;
end                               
diff=wrapToPi(ground-test_phase);
figure();
for ii=1:12
   subplot(3,4,ii);
   plot(diff(ii,:)*180/pi);
end
% figure();
for ii=1:2:11
%    subplot(2,3,(ii+1)/2);
    figure();
%     scatter([1:size(test_phase,2)],wrapToPi(test_phase(ii+1,:)-test_phase(ii,:)),5,'filled');
    plot(wrapToPi(diff(ii+1,:)-diff(ii,:)));
% scatter([1:size(diff,2)],wrapToPi(diff(ii+1,:)-diff(ii,:)),5,'filled');
%     hold on;
   
end


