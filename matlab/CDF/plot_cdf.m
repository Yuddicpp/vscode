% 实验1
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验1\中间\车锁\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h1 = cdfplot(err);
% set(h1,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验1\中间\车头\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h2 = cdfplot(err);
% set(h2,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验1\左上角\车头\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h3 = cdfplot(err);
% set(h3,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验1\左上角\车锁\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h4 = cdfplot(err);
% set(h4,'LineWidth',1.5);
% hold on;
% 
% 
% 
% lengend1 = legend('基站位于栅栏中间3m高-车锁','基站位于栅栏中间3m高-车头','基站位于栅栏左上角3m高-车头','基站位于栅栏左上角3m高-车锁');
% set(lengend1,'FontSize',16);
% xlabel('Location Error');
% ylabel('CDF');
% title('实验1');
% axis([0 3,-inf,inf]);

% 
% %实验2
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验2\中间\车锁\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h1 = cdfplot(err);
% set(h1,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验2\中间\车头\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h2 = cdfplot(err);
% set(h2,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验2\左上角\车头\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h3 = cdfplot(err);
% set(h3,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验2\左上角\车锁\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h4 = cdfplot(err);
% set(h4,'LineWidth',1.5);
% hold on;
% 
% 
% 
% lengend1 = legend('基站位于栅栏中间地表-车锁','基站位于栅栏中间地表-车头','基站位于栅栏左上角地表-车头','基站位于栅栏左上角地表-车锁');
% set(lengend1,'FontSize',16);
% xlabel('Location Error');
% ylabel('CDF');
% title('实验2');
% axis([0 3,-inf,inf]);

% 
% %实验3
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验3\车锁\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h1 = cdfplot(err);
% set(h1,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验3\车头\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h2 = cdfplot(err);
% set(h2,'LineWidth',1.5);
% hold on;
% 
% 
% 
% lengend1 = legend('基站位于栅栏左上角3m高-车锁','基站位于栅栏左上角3m高-车头');
% set(lengend1,'FontSize',16);
% xlabel('Location Error');
% ylabel('CDF');
% title('实验3');
% axis([0 3,-inf,inf]);


% %实验4
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验4\车锁\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h1 = cdfplot(err);
% set(h1,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验4\车头\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h2 = cdfplot(err);
% set(h2,'LineWidth',1.5);
% hold on;
% 
% 
% 
% lengend1 = legend('基站位于栅栏中间3m高-车锁','基站位于栅栏中间3m高-车头');
% set(lengend1,'FontSize',16);
% xlabel('Location Error');
% ylabel('CDF');
% title('实验4');
% axis([0 3,-inf,inf]);

% 
% %实验5
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验5\车锁\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h1 = cdfplot(err);
% set(h1,'LineWidth',1.5);
% hold on;
% 
% Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验5\车头\';                   % 设置数据存放的文件夹路径
% File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
% FileNames = {File.name}';            
% err = [];
% for i=1:length(FileNames)
%     err = [loc_err([Path,FileNames{i}]);err]; 
% %     err = loc_err([Path,FileNames{i}]);
% %     subplot(1,3,i); 
% %     h = cdfplot(err);
% %     xlabel(FileNames{i});
% %     ylabel('CDF');
% %     title('CDF');
% end
% h2 = cdfplot(err);
% set(h2,'LineWidth',1.5);
% hold on;
% 
% 
% 
% lengend1 = legend('基站位于栅栏左上角地表-车锁','基站位于栅栏左上角地表-车头');
% set(lengend1,'FontSize',16);
% xlabel('Location Error');
% ylabel('CDF');
% title('实验5');
% axis([0 3,-inf,inf]);


%实验6
Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验6\车锁\';                   % 设置数据存放的文件夹路径
File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
FileNames = {File.name}';            
err = [];
for i=1:length(FileNames)
    err = [loc_err([Path,FileNames{i}]);err]; 
%     err = loc_err([Path,FileNames{i}]);
%     subplot(1,3,i); 
%     h = cdfplot(err);
%     xlabel(FileNames{i});
%     ylabel('CDF');
%     title('CDF');
end
h1 = cdfplot(err);
set(h1,'LineWidth',1.5);
hold on;

Path = 'E:\BaiduNetdiskWorkspace\研究生\工作\滴滴项目\数据测试\2021_10_21数据测试\10.21数据\实验6\车头\';                   % 设置数据存放的文件夹路径
File = dir(fullfile(Path,'*.xlsx'));  % 显示文件夹下所有符合后缀名为.xlsx文件的完整信息
FileNames = {File.name}';            
err = [];
for i=1:length(FileNames)
    err = [loc_err([Path,FileNames{i}]);err]; 
%     err = loc_err([Path,FileNames{i}]);
%     subplot(1,3,i); 
%     h = cdfplot(err);
%     xlabel(FileNames{i});
%     ylabel('CDF');
%     title('CDF');
end
h2 = cdfplot(err);
set(h2,'LineWidth',1.5);
hold on;



lengend1 = legend('基站位于栅栏中间地表-车锁','基站位于栅栏中间地表-车头');
set(lengend1,'FontSize',16);
xlabel('Location Error');
ylabel('CDF');
title('实验6');
axis([0 3,-inf,inf]);