close all;
clear all;

alpha = 1;
n=50;
image1 = zeros(n,n);
image2 = zeros(n,n);


image1(20,20) = image1(20,21) = image1(21,20) = image1(21,21) = 1;
image2(20,21) = image2(20,22) = image2(21,21) = image2(21,22) = 1;L

E_x = conv2(image1,0.25* [-1 1; -1 1],'same') + conv2(image2, 0.25*[-1 1; -1 1],'same');
E_y = conv2(image1, 0.25*[-1 -1; 1 1], 'same') + conv2(image2, 0.25*[-1 -1; 1 1], 'same');
E_t = conv2(image1, 0.25*ones(2),'same') + conv2(image2, -0.25*ones(2),'same');



% Set initial value of u and v to zero
u = 0;
v = 0;
% Weighted Average kernel
kernel=[1/12 1/6 1/12;1/6 0 1/6;1/12 1/6 1/12];
%Minimizing Iterations (100 times)

for i=1:10000
	%Compute local averages of the vectors
	uAvg=conv2(u,kernel,'same');
	vAvg=conv2(v,kernel,'same');
	%Compute flow vectors constrained by the local 	averages and the optical flow constraints,where alpha is the smoothing weight
	u = uAvg - ( E_x .* ( ( E_x .* uAvg ) +	( E_y .* vAvg ) + E_t ) ) ./ ( alpha^2 + E_x.^2 + E_y.^2);
	v = vAvg - ( E_y .* ( ( E_x .* uAvg ) +	( E_y .* vAvg ) + E_t ) ) ./ ( alpha^2 + E_x.^2 + E_y.^2);
endfor

save abc2.txt u v;
[X,Y] = meshgrid(1:n,1:n);
figure,imshow(image2);
grid on;
hold on;
quiver(X,Y,u,v);