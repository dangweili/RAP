function [ap, cmc] = compute_AP_speed(good_image, junk_image, index)
%% good_image, index of good images
%% junk_image, index of junk images
%% index, ranking list, each value is a global index
% ap is the same as information retrieval

cmc = zeros(1, length(index));
ngood = length(good_image);
ap = 0;

if ngood == 0
    return
end

[~, ~, good_rank] = intersect(good_image, index);
[~, ~, junk_rank] = intersect(junk_image, index);
good_rank = sort(good_rank);
junk_rank = sort(junk_rank);

if ngood ~= length(good_rank)
    error('there are not enough good samples in gallery set!')
end
if length(intersect(good_rank, junk_rank)) > 0
    error('the image should not be both good and junk!')
end

for i = 1:ngood
    junk_flag = junk_rank < good_rank(i);
    n_junk = sum(junk_flag(:));
    ap = ap + i/(good_rank(i) - n_junk);
    if i == 1
        cmc(good_rank(i)-n_junk:end) = 1;
    end
end
ap = ap/ngood;
%good_now = 1;
%current_rank = 1.0;
%first_flag = 0;
%for i = 1:length(index)
%    if ~isempty(find(junk_image == index(i), 1))
%        continue;
%    end
%    if good_now == ngood+1
%        break
%    end
%    if ~isempty(find(good_image == index(i), 1))
%        ap = ap + good_now/current_rank;
%        if first_flag == 0
%            cmc(current_rank:end) = 1;
%            first_flag = 1;
%        end
%        good_now = good_now + 1;
%        current_rank = current_rank + 1;
%    else
%        current_rank = current_rank + 1;
%    end
%end
%ap = ap/ngood;
end
