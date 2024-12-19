%% 2024/11/27 用于对DatasetsBase.py中signal_plot储存的mat文件，进行绘制并查找关键频率
clc;clear;close all
filepath = "E:\6-数据集\0-机械故障诊断数据集\9-厚德平行轴\振动响应\数据导出\斜齿轮\signal_plot\signal_plot-0.mat";
load(filepath);
%% plot
close all
K = size(signals,1);
for k =1:K
    figure(k)
    set(gcf,"Position",[500,300,1600,800])
    %时域
    subplot(6,2,[1,3])
    plot(t,signals(k,:))
    xlim([min(t),max(t)])
    %频域
    subplot(6,2,4*1+[1,3])
    plot(f,signals_freq(k,:))
    xlim([min(f),max(f)])
    %包络谱域
    subplot(6,2,4*2+[1,3])
    plot(f1,signals_env(k,:))
    xlim([min(f1),max(f1)])
    %时频域
    subplot(6,2,[2,4,6])
    imagesc(t2,f2,squeeze(signals_STFT(k,:,:)));
    set(gca,'YDir','normal')
    %CS域
    subplot(6,2,6+[2,4,6])
    imagesc(a,f2,squeeze(signals_CSCoh(k,:,:)));
    set(gca,'YDir','normal')
end
