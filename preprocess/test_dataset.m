%%
% 将mat数据转换成网络输入的stack形式(每个输入单独存放)
mat_dir = "F:\Prophesee\";
targetDir = "D:\Workspace\EventMSE\dataset_test\";
stacks = 16;
stack_cnt = ceil(224*126*0.1);
rng(12450);
%%
for idx = 1:45
    disp(['processing scene ' num2str(idx,'%05d')]);
    data = load(mat_dir+num2str(idx,'%05d')+".mat", "ev");
    ret = mat2stack(data.ev, stack_cnt);
    input = reshape(ret.input, [], stacks*2, 126, 224);
    target = reshape(ret.target, [], stacks*2, 126, 224);
    dif = input - target;
    lp_save(input, target, dif, targetDir+num2str(idx, '%05d')+".mat");
end