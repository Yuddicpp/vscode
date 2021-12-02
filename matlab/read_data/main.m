clc
clear
close all
% filepath = 'E:\SGroup\Bluetooth\data_process\10.21数据\实验6\车锁\x=0.62_y=0_h=0.54'

filepath = 'didi4'
plot_phase_sub(filepath)
% filepath = 'didi2_2'
% plot_phase_sub(filepath)


%%  
function plot_phase_sub(filepath)
% [Idata,Qdata,rssi]=read_file(filepath);
[Idata,Qdata,rssi]=read_file16([filepath,'.txt']);

[data,index]=data_process1(Idata,Qdata);
data = compensate(data,index);
phase=angle(data);

figure()
test = reshape(phase, [12,size(phase,3)*16])
for i = 2:12
%     plot(1:size(test,2),test(i,1:size(test,2))-test(i-1,1:size(test,2)));
%     plot(1:300,test(i,1:300)-test(i-1,1:300));
%     plot(1:300,test(i,1:300));
    hold on;
end

end



%%  



% choose = 'ant_6';
% if choose=='ant_3'
%     data=ant_3(data);
%     P=gene_DML_P_ant3();
% elseif choose=='ant_6'
%     P=gene_DML_P_ant6();    
% end

% loc=DML(data,P,filepath);

