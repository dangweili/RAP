function [ap] = compute_average_precision(gt, pt)
% gt: n*1 vector
% pt: n*1 vector
cnt = sum(gt == 1);
ap = 0;
if cnt == 0
    ap = -1;
else
    % sort the pt from top to down
    [~, idx] = sort(pt, 'descend');
    gt = gt(idx);
    gt_pos = find(gt == 1);
    for i=1:cnt
       ap = ap + i*1.0/gt_pos(i);
    end
    ap = ap/cnt;
end
% end of function
end
