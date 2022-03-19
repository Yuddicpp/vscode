clc
clear
close all

filepath = "E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_3_19数据\1.5m支架\x=1.5m,y=0m_支架_垫子.xlsx";
% filepath = "E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_3_19数据\1.5m\x=1.5m,y=0m_垫子.xlsx";
% filepath = "E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_3_19数据\2.5m\x=2.5m,y=0m_垫子.xlsx";
% filepath = "E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2022_3_19数据\2.840m\x=2.840m,y=0m.xlsx";

data = xlsread(filepath);
phi = data(:,3);
x = find(phi<30);
phi(x)=[];
phi_mean = mean(phi);
phi_std = std(phi);