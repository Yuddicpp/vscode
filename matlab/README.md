# 2021.10.21 Yuddi

此文件夹中分为五个文件夹

- aoa+filter_code: 此文件夹为定位代码和filter的exe执行文件
- MUSIC：MUSIC算法仿真文件
- offline_PSA：为非实时定位代码文件
- test：为测试算法文件
- xiaozhun_7.26：为天线校准代码文件

# 2021.11.29 Yuddi

更改offline_PSA下代码，增加将进行整合，补偿后的IQ数据存储在.mat数据文件中

# 2021.12.1 Yuddi

增加channel_mismatch文件夹

作用为天线校准，将模拟数据与真实数据之间的相位差求出，然后进行补偿

# 2021.12.2 Yuddi

read_data文件夹

用来读取开发板采集到的IQ数据并整合成一个12\*16\*packets的大数组，12为12根天线，16为每根天线一次收集八次IQ，一根天线总共进行了两次

# 2021.12.8 Yuddi

demo文件夹

其他文件被放入其中