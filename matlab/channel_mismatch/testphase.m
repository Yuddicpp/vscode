function [test_phase, amp1]=testphase(filepath)
% filepath='F:\À¶ÑÀ\ant13\data_laoyang\0505\Ïß¼«»¯\00000.txt';
[Idata,Qdata,rssi]=read_file16(filepath);
[data,index]=data_process1(Idata,Qdata);
data = compensate(data,index);
datatmp=data(:,:,22:end);
datatmp(:,:,375:395) = data(:,:,1:21);
data=datatmp;
amp=abs(data);
flag=1;
for ii=1:size(amp,3)
   tmp=squeeze(amp(:,:,ii));
   if(mean(mean(tmp)))>500
      continue; 
   end
   data1(1:12,1:16,flag)=data(:,:,ii);
   amp1(1:12,flag)=median(tmp,2); 
   flag=flag+1;
end

test_phase = debug(data1);

% IQ = comp_line(data);
end

function test_phase = debug(data)
phase=angle(data);
test_phase=zeros(12,16*size(data,3));
for ii=1:size(data,3)
    test_phase(:,(ii-1)*16+1:ii*16)=phase(:,:,ii);
%     2021.12.6 didi
%     test_phase(:,ii)=phase(:,1,ii);
end
for ii=2:12
   test_phase(ii,:)= test_phase(ii,:)-test_phase(1,:);
end
test_phase(1,:)=test_phase(1,:)-test_phase(1,:);
test_phase=wrapToPi(test_phase);
end