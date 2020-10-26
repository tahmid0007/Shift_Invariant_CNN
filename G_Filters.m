h = fspecial('gaussian',[9 9],sigma1); %larger filter size is used only for visualization
imwrite(imresize(mat2gray(h),[90 90]),'Your_Path\filter_1.png');

h = fspecial('gaussian',[9 9],sigma2);
imwrite(imresize(mat2gray(h),[90 90]),'Your_Path\filter_2.png');

h = fspecial('gaussian',[9 9],sigma3);
imwrite(imresize(mat2gray(h),[90 90]),'Your_Path\filter_3.png');

h = fspecial('gaussian',[9 9],sigma4);
imwrite(imresize(mat2gray(h),[90 90]),'Your_Path\filter_4.png');

h = fspecial('gaussian',[9 9],sigma5);
imwrite(imresize(mat2gray(h),[90 90]),'Your_Path\filter_5.png');
