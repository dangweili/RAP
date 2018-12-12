% load the results
load multiatt_query.mat

eff_index = find(cell2mat(MultiAtt_test_cnt) > 100);

fid = fopen('multiatt_query_index.txt', 'w+')
for i=1:length(eff_index)
    for j=1:length(MultiAtt_test{eff_index(i)})
        fprintf(fid, sprintf('%d ', MultiAtt_test{eff_index(i)}(j)));
    end
    fprintf(fid, '\n');
end

fclose(fid)
