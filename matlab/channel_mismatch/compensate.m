function data1 = compensate(data,index)
page=size(data,3);
one_phase1=zeros(size(data,1),size(data,2),size(data,3));
amp=abs(data);
phase=angle(data);
for ii=1:page
     one_phase=squeeze(phase(:,:,ii));
     one_phase1(:,:,ii) = sub_comp(one_phase);
end
data1=amp.*exp(1j*one_phase1);
data1=extract(data1);
end

function one_phase1 = sub_comp(one_phase)
offset_range=(0:0.5:30)*pi/180;
flag=0;
tmp_phase=one_phase;
one_phase1=one_phase;
weight=zeros(numel(offset_range),1);
for ii=offset_range
    flag=flag+1;
    for jj=2:32
        tmp_phase(:,jj)=wrapToPi(one_phase(:,jj)+ii*(jj-1));
    end
    weight(flag)=offset_likelihood(tmp_phase);
end
[~, index]=min(weight);
for jj=2:32
    one_phase1(:,jj)=wrapToPi(one_phase(:,jj)+offset_range(index)*(jj-1));
end
end

function weight=offset_likelihood(one_phase)
phase1=zeros(12,8);
phase2=zeros(12,8);
for ii=3:14
   phase1(ii-2,:)=one_phase(:,ii); 
end
for ii=17:28
   phase2(ii-16,:)=one_phase(:,ii); 
end
diff=wrapToPi(phase1-phase2);
weight = norm(diff, 2);

end

function data1=extract(data)
data1=zeros(12,16,size(data,3));
data=permute(data,[2,1,3]);
data1(:,1:8,:)=data(3:14,:,:);
data1(:,9:16,:)=data(17:28,:,:);
end