function data1 = ant_3(data)
%%ȡ�����������߽��в��
for ii=1:size(data,3)
   data1(1:2:6,1:16,ii)=data(1:4:12,:,ii);
   data1(2:2:6,1:16,ii)=data(2:4:12,:,ii);
end

end