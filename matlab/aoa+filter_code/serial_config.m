function obj=serial_config()
obj=serial('com6');    %连接基站时确�?
obj.InputBufferSize=2068*3;
% obj.timeout=0.6;
obj.BaudRate=921600;
obj.Parity='none';
obj.StopBits=1;
obj.DataBits = 8;
end
