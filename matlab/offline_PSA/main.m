clc
clear
close all
%����汾����
% filepath='test.txt';
% filepath = '����\ʵ��6\����\x=0.62_y=1_h=0.54';
% filepath = 'E:\SGroup\Bluetooth\data_process\10.21����\ʵ��6\����\x=0.62_y=0_h=0.54'
% filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2022_2_23����\δ�ڵ�'
% filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2021_8_23���ݲ���\8.23����\���Ͻ�_�ܼ�_3m��\����h=0.54m\x=2.34_y=-1.24_����_3m_���Ͻ�_�ܼ�'
% filepath='E:\�ε�\������\����(2)\13����\data\data_laoyang\0612\circle4.txt';
% filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2022_2_16����\x=2.7m,y=0m';

%%

filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2022_2_25����\';
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
    choose = 'ant_6';%����������ѡ��3���߶�λ����6���߶�λ
    if choose=='ant_3'
        data=ant_3(data);
        P=gene_DML_P_ant3();
    elseif choose=='ant_6'
        P=gene_DML_P_ant6();    
    end
    
    loc=DML(data,P,file);
end

%%
% x��2��ʼ

% filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2022_2_25����\x=-2m,y=3m';
filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2022_3_19����\1.5m֧��\x=1.5m,y=0m_֧��_����';
[Idata,Qdata,rssi]=read_file16([filepath,'.txt']);


[data,index]=data_process1(Idata,Qdata);
data = compensate(data,index);
choose = 'ant_6';%����������ѡ��3���߶�λ����6���߶�λ
if choose=='ant_3'
    data=ant_3(data);
    P=gene_DML_P_ant3();
elseif choose=='ant_6'
    P=gene_DML_P_ant6();    
end

loc=DML(data,P,filepath);

