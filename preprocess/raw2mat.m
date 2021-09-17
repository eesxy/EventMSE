root_dir = 'F:\Prophesee';

files = dir([root_dir, '\*.raw']);
for idx = 1:numel(files)
    disp(['processing scene ' num2str(idx,'%02d')]);
    ev = metavision_evt3_raw_file_decoder([root_dir '\' files(idx).name]);
    save([root_dir '\' num2str(idx, '%05d')], 'ev', '-v7.3');
end
