clc
clear
close all
%后处理版本代码
% filepath='test.txt';
% filepath = '数据\实验6\车锁\x=0.62_y=1_h=0.54';
% filepath = 'E:\SGroup\Bluetooth\data_process\10.21数据\实验6\车锁\x=0.62_y=0_h=0.54'
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_2_23数据\未遮挡'
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_8_23数据测试\8.23数据\左上角_密集_3m高\车锁h=0.54m\x=2.34_y=-1.24_车锁_3m_左上角_密集'
% filepath='E:\滴滴\汪博文\蓝牙(2)\13天线\data\data_laoyang\0612\circle4.txt';
% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_2_16数据\x=2.7m,y=0m';

%%

filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_2_25数据\';
files =  dir(fullfile(filepath,'*.txt'));

for i = 1:59
    close all;
    file = [files(i).folder,'\',files(i).name];
    l = length(file);
    file = file(1:l-4)
    [Idata,Qdata,rssi]=read_file16([file,'.txt']);
%     save([file,'_RSSI.mat'],'rssi');
    [data,index]=data_process1(Idata,Qdata);
    data = compensate(data,index);
    choose = 'ant_6';%本参数用于选择3天线定位还是6天线定位
    if choose=='ant_3'
        data=ant_3(data);
        P=gene_DML_P_ant3();
    elseif choose=='ant_6'
        P=gene_DML_P_ant6();    
    end
    
    loc=DML(data,P,file);
end

%%
% x从2开始

% filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_2_25数据\x=-2m,y=3m';
filepath = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_3_19数据\1.5m支架\x=1.5m,y=0m_支架_铁板';
[Idata,Qdata,rssi]=read_file16([filepath,'.txt']);


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

