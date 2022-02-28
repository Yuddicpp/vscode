close all;

IQ = data;
IQ_angle = angle(IQ);

% IQ_freq = zeros(12,1024,322);
% for ii = 1:size(IQ,1)
%     for jj = 1:size(IQ,3)
%         IQ_freq(ii,:,jj) = fft([IQ(ii,:,jj),zeros(1,1008)]);
%     end
% end
%%
% 0.5，-0.5
% index_good = [33,61,180,207]+1;
% index_bad = [15,40,139,164,198,209]+1;

% -0.5, 0.5
index_good = [36,43,49,53,172,237]+1;
index_bad = [300,273,240,228,211,165]+1;


figure()
hold on

% for ii = 1:1
%     for jj = 1:length(index_good)
%     plot(abs(reshape(IQ(ii,1:16,index_good(jj):index_good(jj)+4),1,80)),'r-');
%     end
% end
% 
% for ii = 1:1
%     for jj = 1:length(index_bad)
%     plot(abs(reshape(IQ(ii,1:16,index_bad(jj):index_bad(jj)+4),1,80)),'b--');
%     end
% end

for ii = 1:1
    for jj = 1:length(index_good)
    plot(unwrap(reshape(IQ_angle(ii,1:16,index_good(jj):index_good(jj)+4),1,80)),'r-');
    end
end

for ii = 1:1
    for jj = 1:length(index_bad)
    plot(unwrap(reshape(IQ_angle(ii,1:16,index_bad(jj):index_bad(jj)+4),1,80)),'b--');
    end
end

%%
% figure()
% hold on
% for ii = 1:1
%     for jj = 1:length(index_good)
%     plot(mag2db(abs(IQ_freq(ii,1:512,index_good(jj)))),'r-')
%     end
% end
% 
% for ii = 1:1
%     for jj = 1:length(index_bad)
%     plot(mag2db(abs(IQ_freq(ii,1:512,index_bad(jj)))),'b--')
%     end
% end
% for ii = 1:1
%     for jj = 1:length(index_good)
%     plot(unwrap(IQ_angle(ii,1:8,index_good(jj))),'r-');
%     end
% end
% 
% for ii = 1:1
%     for jj = 1:length(index_bad)
%     plot(unwrap(IQ_angle(ii,1:8,index_bad(jj))),'b--');
%     end
% end

% for ii = 1:1
%     for jj = 1:length(index_good)
%     plot(unwrap(angle(IQ_freq(ii,1:512,index_good(jj)))),'r-')
%     end
% end
% 
% for ii = 1:1
%     for jj = 1:length(index_bad)
%     plot(unwrap(angle(IQ_freq(ii,1:512,index_bad(jj)))),'b--')
%     end
% end

% for ii = 1:1
%     for jj = 1:322
%         plot(mag2db(abs(IQ_freq(ii,1:512,jj))),'b-')
%     end
% end

% hold off

IQ = data;
IQ_freq = zeros(12,1024,56);
for ii = 1:size(IQ,1)
    for jj = 1:size(IQ,3)
        IQ_freq(ii,:,jj) = [fft(IQ(ii,:,jj)),zeros(1,1008)];
    end
end

data_int = zeros(12,1024,56);
for ii = 1:size(IQ,1)
    for jj = 1:size(IQ,3)
        data_int(ii,:,jj) =ifft(IQ_freq(ii,:,jj));
    end
end

%%

figure()



hold on
for ii = 1:56
    plot(abs(data_int(1,:,ii)),'b-')
end

