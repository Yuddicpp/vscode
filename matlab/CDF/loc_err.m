function err = loc_err(filepath)
    x = regexp(filepath,'x=[-]?\d+[.]?\d*','match');
    x = str2double(x{1}(3:end));
    y = regexp(filepath,'y=[-]?\d+[.]?\d*','match');
    y = str2double(y{1}(3:end));
    data = xlsread(filepath); %首先打开文件把数据读取出来
    loc = data(:,1:2);
    len = size(loc);
    len = len(1,1);
    err = zeros(len,1);
    for i=1:len
        err(i) = ((x-loc(i,1))^2+(y-loc(i,2))^2)^0.5;
    end
end