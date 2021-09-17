%%
% 将mat数据转换成网络输入的stack形式(每个输入单独存放)
mat_dir = "F:\Prophesee\";
targetDir = "D:\Workspace\EventMSE\dataset\";
stacks = 16;
stack_cnt = ceil(224*126*0.1);
rng(12450);
%%
for idx = 1:42
    disp(['processing scene ' num2str(idx,'%05d')]);
    data = load(mat_dir+num2str(idx,'%05d')+".mat", "ev");
    ret = mat2stack(data.ev, stack_cnt);
    dif = ret.input - ret.target;
    for idx2 = 1:stacks*2:size(ret.input, 1)
        ip = ret.input(idx2:idx2+stacks*2-1,:,:,:);
        tg = ret.target(idx2:idx2+stacks*2-1,:,:,:);
        df = dif(idx2:idx2+stacks*2-1,:,:,:);
        lp_save(ip, tg, df, targetDir+num2str(idx, '%02d')+"_"+num2str(idx2, '%05d')+".mat");
    end
end