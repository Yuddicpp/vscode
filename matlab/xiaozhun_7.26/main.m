
clc
clear
close all

phase = zeros(12,12);
% ��ȡÿһ���ļ������phase��Ϊ12*12��С��ÿһ��Ϊһ�����ߣ���Ϊ��0�㵽330��,30��Ϊһ�����
for m=1:12
    
    
   filepath =  ['2021.7.26/data',num2str(m*30-30),'.txt'];
   phase_temp = data_process(filepath);
   
   phase_temp = squeeze(phase_temp(:,5,:));
   for k = 12:-1:1
       phase_temp(k,:) = phase_temp(k,:)-phase_temp(1,:);
   end
   
%    plot(1:size(phase_temp,2),phase_temp(3,:),'-');
%    hold on;
   
   for i = 1:12
       for j = 1:size(phase_temp,2)
          if phase_temp(i,j) > pi
              phase_temp(i,j) = phase_temp(i,j)-2*pi;
          elseif phase_temp(i,j) < -pi
              phase_temp(i,j) = phase_temp(i,j)+2*pi;
          end
       end
   end
   
%    plot(1:size(phase_temp,2),phase_temp(3,:),'--');
%    hold on;


   phase(:,m) = mean(phase_temp,2);
%    mean(phase_temp,2)
end

phase_12 = circular_polarization();
% for i = 12:-1:1
%    phase(i,:) = phase(i,:)-phase(1,:);
% end

plot(1:12,phase_12(2,:),'-',1:12,phase(2,:),'--');

phase_differ = phase - phase_12;

phase_mean = mean(phase_differ,2);

function phase = data_process(filepath)
[Idata,Qdata,rssi]=read_file16(filepath);
%%  

[data,index]=data_process1(Idata,Qdata);
[data1,amp,phase] = compensate(data,index);
% phaseΪ8*32*�ɼ�����������һ��32�����ݣ�����8��ʱ���
phase = angle(data1);

end
 
 
