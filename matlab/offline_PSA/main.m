clc
clear
close all
%����汾����
% filepath='test.txt';
% filepath = '����\ʵ��6\����\x=0.62_y=1_h=0.54';
% filepath = 'E:\SGroup\Bluetooth\data_process\10.21����\ʵ��6\����\x=0.62_y=0_h=0.54'
filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2021_12_13����\x=-1m,y=-2m'
% filepath = 'E:\BaiduNetdiskWorkspace\�о���\����\�ε���Ŀ\���ݲ���\2021_8_23���ݲ���\8.23����\���Ͻ�_�ܼ�_3m��\����h=0.54m\x=2.34_y=-1.24_����_3m_���Ͻ�_�ܼ�'
% filepath='E:\�ε�\������\����(2)\13����\data\data_laoyang\0612\circle4.txt';

% [Idata,Qdata,rssi]=read_file(filepath);
[Idata,Qdata,rssi]=read_file16([filepath,'.txt']);
%%  

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

