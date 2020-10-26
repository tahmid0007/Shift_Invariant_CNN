I = 'Your Image FilePATH';

c = 256;  % channel
n = 2000; % number of samples
x = (zeros([256 256 1]));

for i=1:n
    I = imread(string(imds_test.Files(i)));
    act = activations('your_trained_net',I,'relu_7');
    for j = 1:c
        temp = log10(abs(fftshift(fft2(act(:,:,j)))).^2 ); % Fourier Transform
        y(:,:,j) = im2uint8(mat2gray(imresize(temp,[256 256]))); % Resize for better visualization
        
    end
    x(:,:,1) = x(:,:,1) + (sum(y,3)/c); %stacking
end
A = uint8(sum(x,3)/n*1);
A = ind2rgb(A, parula(256)); %RGB generate
imwrite(A,'Your FilePath');