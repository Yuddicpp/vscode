function debug(data,factor)
phase=angle(data);
for ii=1:1%size(phase,3)
   phase1=squeeze(phase(:,:,ii)); 
%    phase2=[phase1(1,1:4),phase1(1,6:2:28),phase1(1,29:end)];
%    stem(phase2);
%    offset=(phase1(1,1)-phase1(1,4))/3;
%    for ii=2:32
%       phase1(:,ii)=wrapToPi(phase1(:,ii)+offset*(ii-1)); 
%    end
% % offset=[106.23 148.934 69.732 65.099 48.965 91.658 69.313 65.233 48.938 91.581 126.592 122.476]*factor*pi/180;
% % for jj=1:12
% %    phase1(:,(jj-1)*2+5)=wrapToPi(phase1(:,(jj-1)*2+5)-offset(jj));
% % end
%    diff74h=wrapToPi(phase1(:,5)-phase1(:,17))*180/pi;
%    diff74v=wrapToPi(phase1(:,7)-phase1(:,19))*180/pi;
%    diff63h=wrapToPi(phase1(:,25)-phase1(:,13))*180/pi;
%    diff63v=wrapToPi(phase1(:,27)-phase1(:,15))*180/pi;  
%    
%    diff52h=wrapToPi(phase1(:,21)-phase1(:,9))*180/pi;
%    diff52v=wrapToPi(phase1(:,23)-phase1(:,11))*180/pi;  
%    figure()
%    subplot(3,2,1);
%    stem(diff74h);
%    subplot(3,2,2);
%    stem(diff74v);
%    subplot(3,2,3);
%    stem(diff63h);
%    subplot(3,2,4);
%    stem(diff63v);
%    subplot(3,2,5);
%    stem(diff52h);
%    subplot(3,2,6);
%    stem(diff52v);

end
end

function phase2=observe_diff(phase1)
phase1(:,[1:4])=[];
phase1(:,[25:28])=[];
phase1(:,[2:2:24])=[];
phase1=phase1.';
for ii=2:12
    phase1(ii,:)=wrapToPi(phase1(ii,:)-phase1(1,:));
end
phase1(1,:)=phase1(1,:)-phase1(1,:);
phase2=phase1*180/pi;
end