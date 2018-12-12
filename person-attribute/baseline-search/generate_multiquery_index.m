% combine multiple attributes for retrieval
% group the selected attributes into four categories
Atts_Group = cell(1,1);
Flag = zeros(1,1);

tmp = [1]; % 0/1
Flag(1) = 1;
Atts_Group{1} = tmp;

tmp = [11,12,14,15];
Flag(2) = 0;
Atts_Group{2} = tmp;

tmp = [16,17,18,19,20,21,22,23,24];
Flag(3) = 0;
Atts_Group{3} = tmp;

tmp = [26,27,28,29,30,31];
Flag(4) = 0;
Atts_Group{4} = tmp;

% attachments
tmp = [38,39,40,41,42,43,44];
Flag(5) = 0;
Atts_Group{5} = tmp;

MultiAtt_test = cell(1,1);
MultiAtt_test_cnt = cell(1,1);
% combination of mutiple attributes
% att+each, att+three, att+four
load ./../../data/RAP_annotation/RAP_annotation.mat
selected_attribute = RAP_annotation.selected_attribute;
partition = RAP_annotation.partition_attribute;
data = RAP_annotation.data(partition{1,1}.test_index, selected_attribute);

% random sample one
idx = 1;
for i=2:5
    for j=1:length(Atts_Group{i})
        % process femal
        cnt = sum(((data(:,1) == 1) + (data(:, Atts_Group{i}(j)) == 1))==2);
        MultiAtt_test{idx} = [1 Atts_Group{i}(j)];
        MultiAtt_test_cnt{idx} = cnt;
        idx = idx+1;
        % process male
        cnt = sum(((data(:,1) == 0) + (data(:, Atts_Group{i}(j)) == 1))==2);
        MultiAtt_test{idx} = [0 Atts_Group{i}(j)];
        MultiAtt_test_cnt{idx} = cnt;
        idx = idx+1;
    end
end
% random sample two
for i=2:4
    for j=i+1:5
        for i_idx=1:length(Atts_Group{i})
            for j_idx =1:length(Atts_Group{j})
                % process femal
                tmp_1 = data(:,1) == 1;
                tmp_2 = sum(data(:, [Atts_Group{i}(i_idx) Atts_Group{j}(j_idx)]),2)==2;
                cnt = sum(tmp_1+tmp_2 == 2);
                MultiAtt_test{idx} = [1 Atts_Group{i}(i_idx) Atts_Group{j}(j_idx)];
                MultiAtt_test_cnt{idx} = cnt;
                idx = idx+1;
                % process male
                tmp_1 = data(:,1) == 0;
                tmp_2 = sum(data(:, [Atts_Group{i}(i_idx) Atts_Group{j}(j_idx)]),2)==2;
                cnt = sum(tmp_1+tmp_2 == 2);
                MultiAtt_test{idx} = [0 Atts_Group{i}(i_idx) Atts_Group{j}(j_idx)];
                MultiAtt_test_cnt{idx} = cnt;
                idx = idx+1;
            end
        end
    end
end
% random sample three
for i=2:5
    j = setdiff(2:5, i);
    for k = 1:length(Atts_Group{j(1)})
        for m = 1:length(Atts_Group{j(2)})
            for n=1:length(Atts_Group{j(3)})
                 % process femal
                tmp_1 = data(:,1) == 1;
                tmp_2 = sum(data(:, [Atts_Group{j(1)}(k) Atts_Group{j(2)}(m) Atts_Group{j(3)}(n)]),2)==3;
                cnt = sum(tmp_1+tmp_2 == 2);
                MultiAtt_test{idx} = [1 Atts_Group{j(1)}(k) Atts_Group{j(2)}(m) Atts_Group{j(3)}(n)];
                MultiAtt_test_cnt{idx} = cnt;
                idx = idx+1;
                % process male
                tmp_1 = data(:,1) == 0;
                tmp_2 = sum(data(:, [Atts_Group{j(1)}(k) Atts_Group{j(2)}(m) Atts_Group{j(3)}(n)]),2)==3;
                cnt = sum(tmp_1+tmp_2 == 2);
                MultiAtt_test{idx} = [0 Atts_Group{j(1)}(k) Atts_Group{j(2)}(m) Atts_Group{j(3)}(n)];
                MultiAtt_test_cnt{idx} = cnt;
                idx = idx+1;              
            end
        end
    end
end
% random sample four
for k = 1:length(Atts_Group{2})
    for m = 1:length(Atts_Group{3})
        for n=1:length(Atts_Group{4})
            for i=1:length(Atts_Group{5})
                % process femal
                tmp_1 = data(:,1) == 1;
                tmp_2 = sum(data(:, [Atts_Group{2}(k) Atts_Group{3}(m) Atts_Group{4}(n) Atts_Group{5}(i)]),2)==4;
                cnt = sum(tmp_1+tmp_2 == 2);
                MultiAtt_test{idx} = [1 Atts_Group{2}(k) Atts_Group{3}(m) Atts_Group{4}(n) Atts_Group{5}(i)];
                MultiAtt_test_cnt{idx} = cnt;
                idx = idx+1;
                % process male
                tmp_1 = data(:,1) == 0;
                tmp_2 = sum(data(:, [Atts_Group{2}(k) Atts_Group{3}(m) Atts_Group{4}(n) Atts_Group{5}(i)]),2)==4;
                cnt = sum(tmp_1+tmp_2 == 2);
                MultiAtt_test{idx} = [0 Atts_Group{2}(k) Atts_Group{3}(m) Atts_Group{4}(n) Atts_Group{5}(i)];
                MultiAtt_test_cnt{idx} = cnt;
                idx = idx+1;
            end
        end
    end
end
idd = find(cell2mat(MultiAtt_test_cnt)>100);
save('multiatt_query.mat', 'MultiAtt_test', 'MultiAtt_test_cnt');

