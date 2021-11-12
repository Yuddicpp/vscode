function [data1,index1]=data_process1(Idata,Qdata)
%V和H分开
rawdata=Idata+1j*Qdata;
for ii=1:5
    id=abs(rawdata(ii,:))>3000;
    rawdata(:,id)=[];
end
data=reshape(rawdata,16,32,size(rawdata,2));
index=[1:512];
index=reshape(index,16,32);
index1=restructuring(index);
data1=restructuring(data);
end


function data1=restructuring(data)
%天线切换顺序
% % A1/A1/A1/A1/A7H/A7V/A2H/A2V
% % A3H/A3V/A4H/A4V/A5H/A5V/A6H/A6V
% % A7H/A7V/A2H/A2V/A3H/A3V/A4H/A4V
% % A5H/A5V/A6H/A6V/A1/A1/A1/A1

% %天线排布
%        y
%        |
%     a2----a3
%   /    |    \
% a7-----|-----a4---x
%   \    |    /
%     a6----a5
%        |

% data=data(1:8,:,:);
row=size(data,1);
col=size(data,2);
page=size(data,3);
% data1=zeros(8,24,page);%最后三列是A1
data1=data(1:8,1:32,:);
end