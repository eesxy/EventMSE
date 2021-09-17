function lp_save(input, target, dif, target_path)
    data.input = input;
    data.target = target;
    data.diff = dif;
    save(target_path, 'data', '-v7.3');
end