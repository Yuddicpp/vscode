---
title: Icarus_Verilog的安装与使用
date: 2021-01-01 03:32:36
tags:
- verilog
- ubuntu
categories: Verilog
cover: /img/Icarus_Verilog/top_cover.png
description: Icarus Verilog是一个Verilog 模拟和合成工具。它作为编译器运行，将用Verilog （IEEE-1364） 编写的源代码编译为某种目标格式。
---



# 软件介绍

 *Icarus Verilog*是一个 Verilog 模拟和合成工具。它作为编译器运行，将用 Verilog （IEEE-1364） 编写的源代码编译为某种目标格式。对于批处理模拟，编译器可以生成称为*vvp 程序集的中间窗体*。此中间窗体由"vvp"命令执行。对于合成，编译器以所需的格式生成网络列表。 这是一个非常简便的Verilog编译工具。

# 安装步骤

```
sudo apt-get install verilog
sudo apt-get install gtkwave
```

# 使用例程

## 1.例程代码

```verilog
module dds
	#(
		parameter K_WIDTH = 27,
		parameter TABLE_AW = 10,
		parameter DATA_W = 12,
		parameter MEM_FILE = "SineTable.dat"
	)
	(
		input Clock,
		input ClkEn,
		input [K_WIDTH-1:0]FreqWord,
		input [K_WIDTH-1:0]PhaseShift,
		output DAC_clk,
		output [DATA_W-1:0]Out
	);

		reg signed [DATA_W-1:0]sinTable[2**TABLE_AW-1:0];
		reg [K_WIDTH-1:0]addr;
		wire [K_WIDTH-1:0]Paddr;
		initial begin
			addr = 0;
			$readmemh(MEM_FILE,sinTable);
		end
		always @(posedge Clock) begin
			if(ClkEn)
				addr <= addr + FreqWord;
		end
		assign Paddr = addr + PhaseShift;
		assign Out = sinTable[Paddr[K_WIDTH-1:K_WIDTH-TABLE_AW]];
		assign DAC_clk = Clock;
endmodule

```

```verilog
`timescale 1ns/1ps
module testbench;
	reg clk = 1'b1;
	reg clkEn = 1'b1;
	reg [23:0]freq = 24'h04_0000;
	reg [23:0]phaseShift=24'b0;
	wire dac_clk;
	wire [11:0]out;

	initial begin
		$dumpfile("dds.vcd");
		$dumpvars(0, dds_inst);
		#10000 freq=24'h08_0000;
		#20000 freq=24'h0C_0000;
		#30000 freq=24'h10_0000;
		#40000 freq=24'h18_0000;
		#50000 freq=24'h20_0000;
		#60000 freq=24'h30_0000;
		#70000 $finish();
	end
	always begin
		#4 clk=~clk;
	end
	dds#(
		.K_WIDTH(24),
		.DATA_W(12),
		.TABLE_AW(10),
		.MEM_FILE("SineTable.dat")
		)
	dds_inst(
		.FreqWord(freq),
		.PhaseShift(phaseShift),
		.Clock(clk),
		.ClkEn(clkEn),
		.DAC_clk(dac_clk),
		.Out(out)
		);
endmodule
```

SineTable.dat为正弦波信号数据的文件

## 2.编译

```
iverilog -o dds.vvp dds.v testbench.v //生成vvp文件
vvp dds.vvp //编译生成文件
gtkwave dds.vcd //查看数据结果
```

![](/img/Icarus_Verilog/compile.png)

## 3.查看波形数据

![](/img/Icarus_Verilog/wave.png)