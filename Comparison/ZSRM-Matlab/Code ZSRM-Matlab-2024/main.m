% If you find the code and dataset useful in your research, please consider citing:
% 《Underwater Image Enhancement with Zero-Point Symmetry Prior and Reciprocal Mapping》

close all;
clc;

%% 文件夹
folder = 'image\Test-OceanDark';
output_folder_color = 'color_balanced_res\ZSRM-color-OceanDark';
output_folder_result = 'result\ZSRM-OceanDark';
files = dir(fullfile(folder,'*.jpg'));

if ~exist(output_folder_color, 'dir')
    mkdir(output_folder_color);
end

if ~exist(output_folder_result, 'dir')
    mkdir(output_folder_result);
end

%% 开始处理
numFiles = length(files);
file_names = cell(numFiles, 1); % 用于保存文件名

for i=1:numFiles
    % 打印当前迭代的索引值
    disp(['Processing file ' num2str(i) ' of ' num2str(numFiles)]);
    % Input image
    filename = files(i).name;
    fullname = fullfile(folder,filename);
    img = imread(fullname);
    % Color-balanced image
    img_color = ZPSP(img);

    % Result image
    result = RM(img_color);
    
    % Display
    % figure;
    % ubplot(131);imshow(img);title('Input image');
    % subplot(132);imshow(img_color);title('Color-balanced image'); 
    % subplot(133);imshow(result);title('Result image'); 

    % Color-balanced image
    imwrite(img_color, fullfile(output_folder_color, filename));
    % Result image
    imwrite(result, fullfile(output_folder_result, filename));

end
