function bright_channel = get_bright_channel(image, win_size)

[m, n, ~] = size(image);

pad_size = floor(win_size/2);

padded_image = padarray(image, [pad_size pad_size], 1);

bright_channel = zeros(m, n); 

for j = 1 : m
    for i = 1 : n
        patch = padded_image(j : j + (win_size-1), i : i + (win_size-1), :);
        bright_channel(j,i) = max(patch(:));
     end
end

end