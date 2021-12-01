function [Idata,Qdata,rssi]=read_file(file)
fid=fopen(file);
d1 = textscan(fid,'%s','whitespace','[]'); 
fclose(fid); 

for n=1:100%size(d1{1},1)/2
    n
    rawdata{n}{1} = textscan(d1{1}{n*2-1},'%s%s','delimiter',',','whitespace','()');
    rawRSSI{n}{1} = textscan(d1{1}{n*2}(6:8),'%s');
    for m=1:512
        Idata(m,n) = str2double(rawdata{n}{1}{1}{m});
        Qdata(m,n) = str2double(rawdata{n}{1}{2}{m}(1:end-1));
    end
    rssi(n) = str2double(rawRSSI{n}{1}{1});
    
end



end