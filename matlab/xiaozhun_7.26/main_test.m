
clc
clear
close all

% 读取每一个文件，组成phase，为12*12大小，每一行为一个天线，列为从0°到330°,30°为一个间隔
   filepath =  ['2021.7.26/data','270','.txt'];
%     filepath = '小标签x=1.31y=0.69z=1.5.txt';
%     filepath = 'circle1.txt';
   phase1 = data_process(filepath);
   
   % 读取每一个文件，组成phase，为12*12大小，每一行为一个天线，列为从0°到330°,30°为一个间隔
%    filepath =  ['2021.7.26/data','0','.txt'];
%     filepath = 'data.txt';
%    phase2 = data_process(filepath);
% 

for i=12:-1:1
   phase_temp(i,:) = phase1(i,5,:)-phase1(1,5,:);
end

for i=1:12
   for j=1:size(phase_temp,2) 
       if phase_temp(i,j) > pi
              phase_temp(i,j) = phase_temp(i,j)-2*pi;
       elseif phase_temp(i,j) < -pi
              phase_temp(i,j) = phase_temp(i,j)+2*pi;
       end
   end
    
end

for i=1:12
    
    plot(1:size(phase1,3),squeeze(phase1(i,5,:)),'LineWidth',2);
    hold on;
    
end

title('信源在同一位置的情况下，天线的相位');
xlabel('数据包');
ylabel('相位');
% legend('ant1','ant2','ant3','ant4');
% legend('ant1-ant1','ant2-ant1','ant3-ant1','ant4-ant1');

% plot(1:50,squeeze(phase1(1,2,1:50)),'LineWidth',2);
% hold on;
% plot(1:50,squeeze(phase2(2,2,1:50)),'LineWidth',2);
% hold on;
% plot(1:50,squeeze(phase1(1,2,1:50)-phase2(2,2,1:50)),'LineWidth',2);
% plot(1:size(phase2,3),squeeze(phase2(2,2,:)),'LineWidth',2);

% plot(1:size(phase2,2),squeeze(phase2(2,:,6)),'LineWidth',2);
% title('信源在两个不同位置的情况下，ANT11在所有数据包中的相位');
% xlabel('数据包');
% ylabel('相位');
% legend('ant11_0°','ant11_30°','ant11_0°-ant11_30°');
% 




function phase = data_process(filepath)
[Idata,Qdata,rssi]=read_file16(filepath);
%%  

[data,index]=data_process1(Idata,Qdata);
[data1,amp,phase] = compensate(data,index);
% phase为8*32*采集数据组数，一组32个数据，采了8个时间戳
phase = angle(data1);

end
 
