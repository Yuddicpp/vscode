function main()
clc;
clear;
close all;

obj=serial_config();
BLE_location(obj);
end

%fclose(instrfind)
