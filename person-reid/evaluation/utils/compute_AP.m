function [ap, cmc] = compute_AP(good_image, junk_image, index)
%% good_image, index of good images
%% junk_image, index of junk images
%% index, ranking list, each value is a global index
% ap is the same as information retrieval

cmc = zeros(1, length(index));
ngood = length(good_image);
ap = 0;
good_now = 1;
current_rank = 1.0;
first_flag = 0;
for i = 1:length(index)
    if ~isempty(find(junk_image == index(i), 1))
        continue;
    end
    if good_now == ngood+1
        break
    end
    if ~isempty(find(good_image == index(i), 1))
        ap = ap + good_now/current_rank;
        if first_flag == 0
            cmc(current_rank:end) = 1;
            first_flag = 1;
        end
        good_now = good_now + 1;
        current_rank = current_rank + 1;
    else
        current_rank = current_rank + 1;
    end
end
ap = ap/ngood;
end
