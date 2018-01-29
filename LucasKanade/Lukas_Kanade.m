im1 = imread("frame10.png");
im2 = imread("frame11.png");
windowSize = 7;
halfWindow = floor(windowSize/2);



if (size(im1,1) != size(im2,1)) || (size(im1,2) != size(im2,2))
   error('input images are not the same size');
end;

if (size(im1,3)!=1) || (size(im2,3)!=1)
   error('method only works for gray-level images');
end;
row = size(im1,1);
column = size(im1,2);
fx = conv2(im1,0.25* [-1 1; -1 1],'same') + conv2(im2, 0.25*[-1 1; -1 1],'same');
fy = conv2(im1, 0.25*[-1 -1; 1 1],'same') + conv2(im2, 0.25*[-1 -1; 1 1],'same');
ft = conv2(im1, 0.25*ones(2),'same') + conv2(im2, -0.25*ones(2),'same');


u = zeros(size(im1));
v = zeros(size(im2));


for i=halfWindow+1:row - halfWindow
	for j=halfWindow+1:column - halfWindow
		temp1 = sum(sum(fx(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow).*fx(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow)));
		temp2 = sum(sum(fx(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow).*fy(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow)));
		temp3 = sum(sum(fy(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow).*fy(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow)));
		coefficients = [temp1 temp2 ; temp2 temp3];
		constants = [-1*sum(sum(fx(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow).*ft(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow)));-1*sum(sum(fy(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow).*fx(i - halfWindow : i + halfWindow,j - halfWindow : j + halfWindow)))];
		U = inv(coefficients) * constants;
		u(i,j) = U(1);
		v(i,j) = U(2);
	end
end

u1 = reshape(u,1,row*column);
v1 = reshape(v,1,row*column);
[X,Y] = meshgrid(1:column,1:row);
imshow(im2);
grid on;
hold on;
quiver(X,Y,u1,v1);
size(u1,1)
size(u1,2)
size(v1,1)
size(v1,2)
size(X,1)
size(X,2)
size(Y,1)
size(Y,2)