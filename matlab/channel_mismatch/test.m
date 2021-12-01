clc;
clear;
close all;
flag=0;
for fw=0:0.069:360
    flag=flag+1;
[ground_phase,ground_amp]=groundphase(fw,0,0);
gp(1:12,flag)=ground_phase(:,1);
ga(1:12,flag)=ground_amp(:,1);
end
filepath='G:\bluetooth\ant13\data_laoyang\0505\Ïß¼«»¯\00000.txt';
[test_phase, test_amp]=testphase(filepath);
plot_mismatch(gp, test_phase);
plot_mismatch_amp();
for ii=1:12
   figure();
   plot(test_phase(ii,:));
end
page=size(test_phase,2)/16;
for ii=1:page
   ground(1:12,(ii-1)*16+1:ii*16)= ground_phase;
end                               
diff=wrapToPi(ground-test_phase);
for ii=1:12
   figure();
   plot(diff(ii,:)*180/pi);
end
% figure();
for ii=1:2:11
%    subplot(2,3,(ii+1)/2);
    figure();
%     scatter([1:size(test_phase,2)],wrapToPi(test_phase(ii+1,:)-test_phase(ii,:)),5,'filled');
    plot(wrapToPi(diff(ii+1,:)-diff(ii,:)));
% scatter([1:size(diff,2)],wrapToPi(diff(ii+1,:)-diff(ii,:)),5,'filled');
%     hold on;
   
end


