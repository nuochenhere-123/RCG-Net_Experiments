function img = load_image(input_folder, filename)
% num is the num of image with 

path = fullfile(input_folder, filename);
img = imread(path);