function run_bp_assignment
% BP regression assignment. Requires MATLAB and Deep Learning Toolbox.
% All train/validation/test data, weights and results are reproducible.
close all; clc;
out=fullfile(fileparts(mfilename('fullpath')),'results');
if ~exist(out,'dir'), mkdir(out); end
rng(2026,'twister');
x=linspace(0,1,201); y=sin(2*pi*x); p=2*x-1;
idx=randperm(numel(x)); tr=idx(1:121); va=idx(122:161); te=idx(162:201);
base=struct('H',16,'lr',0.05,'momentum',0.9,'epochs',10000,'seed',2026);
[net,hist]=manual_bp(p,y,tr,va,base);
yhat=predict_bp(net,p); baseline=metrics(y(te),yhat(te));
save(fullfile(out,'manual_baseline.mat'),'x','y','p','tr','va','te','base','net','hist','yhat','baseline');
figure; semilogy(0:base.epochs,hist.train,'LineWidth',1.3); hold on; semilogy(0:base.epochs,hist.val,'LineWidth',1.3); xlabel('Epoch');ylabel('MSE');legend('Train','Validation');grid on;
saveas(gcf,fullfile(out,'manual_loss.png'));
figure; plot(x,y,'k-','LineWidth',1.5);hold on;plot(x,yhat,'r--','LineWidth',1.3);scatter(x(te),y(te),15,'b','filled');legend('Target','Manual BP','Test samples');xlabel('x');ylabel('y');grid on;saveas(gcf,fullfile(out,'manual_fit.png'));
% Controlled one-factor experiments; three fixed seeds per condition.
rows={}; groups={'H','lr','momentum','epochs'};
values={[4 8 16 32],[0.005 0.01 0.05 0.1],[0 0.5 0.9 0.95],[500 2000 5000 10000]};
for g=1:numel(groups)
 for v=values{g}
  for s=1:3
   cfg=base;cfg.(groups{g})=v;cfg.seed=2025+s;
   [n,h]=manual_bp(p,y,tr,va,cfg); pred=predict_bp(n,p); met=metrics(y(te),pred(te));
   rows(end+1,:)={groups{g},v,cfg.seed,met(1),met(2),met(3),h.best_epoch}; %#ok<AGROW>
  end
 end
end
T=cell2table(rows,'VariableNames',{'Factor','Value','Seed','TestMSE','TestRMSE','TestMaxAbs','BestEpoch'});writetable(T,fullfile(out,'experiments.csv'));
summary=groupsummary(T,{'Factor','Value'},{'mean','std'},{'TestMSE','TestRMSE','TestMaxAbs'});writetable(summary,fullfile(out,'summary.csv'));
% Toolbox comparison: same architecture, split, preprocessing and init.
% traingdm is gradient descent with momentum, not Levenberg-Marquardt.
if exist('feedforwardnet','file')~=2
 warning('Deep Learning Toolbox not found: toolbox comparison NOT RUN.');
 return;
end
rng(base.seed,'twister'); tb=feedforwardnet(base.H,'traingdm');
tb.inputs{1}.processFcns={};tb.outputs{2}.processFcns={};
tb.layers{1}.transferFcn='tansig';tb.layers{2}.transferFcn='purelin';
tb.divideFcn='divideind';tb.divideParam.trainInd=tr;tb.divideParam.valInd=va;tb.divideParam.testInd=te;
tb.performFcn='mse';tb.trainParam.epochs=base.epochs;tb.trainParam.lr=base.lr;tb.trainParam.mc=base.momentum;
tb.trainParam.max_fail=base.epochs+1;tb.trainParam.min_grad=0;tb.trainParam.goal=0;
tb.trainParam.showWindow=false;tb=configure(tb,p,y);
init=init_bp(base.H,base.seed);tb.IW{1,1}=init.W1;tb.b{1}=init.b1;tb.LW{2,1}=init.W2;tb.b{2}=init.b2;
[tb,record]=train(tb,p,y);tbpred=tb(p);tbmetrics=metrics(y(te),tbpred(te));
save(fullfile(out,'toolbox.mat'),'tb','record','tbpred','tbmetrics');
comparison=array2table([baseline;tbmetrics],'VariableNames',{'TestMSE','TestRMSE','TestMaxAbs'},'RowNames',{'ManualBP','ToolboxTraingdm'});writetable(comparison,fullfile(out,'comparison.csv'),'WriteRowNames',true);
figure;plot(x,y,'k-',x,yhat,'r--',x,tbpred,'b:','LineWidth',1.4);legend('Target','Manual BP','Toolbox');grid on;xlabel('x');ylabel('y');saveas(gcf,fullfile(out,'comparison_fit.png'));
figure;semilogy(record.epoch,record.perf,record.epoch,record.vperf,'LineWidth',1.3);legend('Train','Validation');grid on;xlabel('Epoch');ylabel('MSE');saveas(gcf,fullfile(out,'toolbox_loss.png'));
fprintf('Results saved to %s\n',out);disp(comparison);
end
function n=init_bp(H,seed)
rng(seed,'twister');n.W1=(2*rand(H,1)-1)*sqrt(6/(1+H));n.b1=zeros(H,1);n.W2=(2*rand(1,H)-1)*sqrt(6/(H+1));n.b2=0;
end
function y=predict_bp(n,x)
y=n.W2*tanh(n.W1*x+n.b1)+n.b2;
end
function [best,h]=manual_bp(x,t,tr,va,c)
n=init_bp(c.H,c.seed);vW1=zeros(size(n.W1));vb1=zeros(size(n.b1));vW2=zeros(size(n.W2));vb2=0;
h.train=zeros(c.epochs+1,1);h.val=h.train;h.train(1)=mean((predict_bp(n,x(tr))-t(tr)).^2);h.val(1)=mean((predict_bp(n,x(va))-t(va)).^2);best=n;bestval=h.val(1);h.best_epoch=0;
for k=1:c.epochs
 a=tanh(n.W1*x(tr)+n.b1);yp=n.W2*a+n.b2;d=2*(yp-t(tr))/numel(tr);dh=(n.W2'*d).*(1-a.^2);
 gW2=d*a';gb2=sum(d);gW1=dh*x(tr)';gb1=sum(dh,2);
 % MATLAB traingdm convention: v=mc*v-(1-mc)*lr*gradient.
 vW1=c.momentum*vW1-(1-c.momentum)*c.lr*gW1;vb1=c.momentum*vb1-(1-c.momentum)*c.lr*gb1;
 vW2=c.momentum*vW2-(1-c.momentum)*c.lr*gW2;vb2=c.momentum*vb2-(1-c.momentum)*c.lr*gb2;
 n.W1=n.W1+vW1;n.b1=n.b1+vb1;n.W2=n.W2+vW2;n.b2=n.b2+vb2;
 h.train(k+1)=mean((predict_bp(n,x(tr))-t(tr)).^2);h.val(k+1)=mean((predict_bp(n,x(va))-t(va)).^2);
 if h.val(k+1)<bestval, bestval=h.val(k+1);best=n;h.best_epoch=k;end
end
end
function m=metrics(y,p)
e=p-y;m=[mean(e.^2),sqrt(mean(e.^2)),max(abs(e))];
end
