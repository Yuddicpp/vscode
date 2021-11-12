clc
clear
close all
%后处理版本代码
% filepath='test.txt';
filepath = '数据\实验6\车锁\x=0.62_y=1_h=0.54';
% filepath='E:\滴滴\汪博文\蓝牙(2)\13天线\data\data_laoyang\0612\circle4.txt';

% [Idata,Qdata,rssi]=read_file(filepath);
[Idata,Qdata,rssi]=read_file16([filepath,'.txt']);
%%  

[data,index]=data_process1(Idata,Qdata);
data = compensate(data,index);
choose = 'ant_6';%本参数用于选择3天线定位还是6天线定位
if choose=='ant_3'
    data=ant_3(data);
    P=gene_DML_P_ant3();
elseif choose=='ant_6'
    P=gene_DML_P_ant6();    
end

loc=DML(data,P,filepath);

