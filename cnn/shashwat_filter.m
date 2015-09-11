filters = {}
filters(1) = [-1 0 1; -2 0 2; -1 0 1];
filters(2) = [1 2 1; 0  0 0; -1 -2 -1];
filters(3) = [0 -1 0; -1 4 -1; 0 -1 0];
filters(4) = filters{1} + filters{2}
filters(5) = [1 0 0; 0 0 0; 0 0 0];

img = imread('~/Downloads/lena512.bmp');

for i=1:size(filters, 2)
	filt = filters{i}
	zzz = filter2(filt, img);
	zzz = uint8(zzz);

	subplot(2,2, i)
	imshow(zzz)
endfor