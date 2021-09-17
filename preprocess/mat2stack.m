function ret = mat2stack(ev, stack_cnt)
    stack_num = 30000;
    input = zeros(stack_num, 126, 224, 'int32');
    target = zeros(stack_num, 126, 224, 'int32');
    stack_idx = 1;
    input_cnt = 0;
    for idx = 1:numel(ev.t)
        if ev.y(idx)<574 || ev.y(idx)>=700
            continue
        end
        if ev.x(idx)>=650 && ev.x(idx)<874
            if ev.p(idx) == 1
                input(stack_idx, ev.y(idx)-574+1, ev.x(idx)-650+1) = ...
                input(stack_idx, ev.y(idx)-574+1, ev.x(idx)-650+1) + 1;
            else
                input(stack_idx+1, ev.y(idx)-574+1, ev.x(idx)-650+1) = ...
                input(stack_idx+1, ev.y(idx)-574+1, ev.x(idx)-650+1) + 1;
            end
            input_cnt = input_cnt + 1;
            if input_cnt == stack_cnt
                stack_idx = stack_idx + 2;
                input_cnt = 0;
            end
        elseif ev.x(idx)>=982 && ev.x(idx)<1206
            if ev.p(idx) == 1
                target(stack_idx, ev.y(idx)-574+1, ev.x(idx)-982+1) = ...
                target(stack_idx, ev.y(idx)-574+1, ev.x(idx)-982+1) + 1;
            else
                target(stack_idx+1, ev.y(idx)-574+1, ev.x(idx)-982+1) = ...
                target(stack_idx+1, ev.y(idx)-574+1, ev.x(idx)-982+1) + 1;
            end
        end
    end
    stack_idx = ceil((stack_idx+1) / 32)*32;
    ret.input = input(1:stack_idx, :, :);
    ret.target = target(1:stack_idx, :, :);
end