clc;
close all;
clear all;
filename = 'abc.txt';
fid = fopen(filename,'w');
n=50;
alph = 1;
image1 = zeros(n,n);
image2 = zeros(n,n);




image1(20,20) = image1(20,21) = image1(21,20) = image1(21,21) = 1;
image2(20,21) = image2(20,22) = image2(21,21) = image2(21,22) = 1;

E_x = conv2(image1,0.25* [-1 1; -1 1],'same') + conv2(image2, 0.25*[-1 1; -1 1],'same');
E_y = conv2(image1, 0.25*[-1 -1; 1 1], 'same') + conv2(image2, 0.25*[-1 -1; 1 1], 'same');
E_t = conv2(image1, 0.25*ones(2),'same') + conv2(image2, -0.25*ones(2),'same');


const(1:n*n,1) = reshape(transpose(E_x.*E_t.*-1),n*n,1);
const(n*n+1:n*n*2,1) = reshape(transpose(E_y.*E_t.*-1),n*n,1);

E_xE_y = zeros(n,n);
E_xE_y = E_x.*E_y;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


semi_characters1 = zeros(n*n,n*n);
semi_characters2 = zeros(n*n,n*n);
semi_characters3 = zeros(n*n,n*n);
%Computing of Upper right and lower left of matrix
for i = 1:n
	for j = 1:n
		row = n*(i-1) + j;
		if(i-1 > 0 && j-1 > 0)
			column = n*(i-1-1) + j-1;
			semi_characters1(row,column) = -1*E_xE_y(i,j)/12;
		end
		if(i-1 > 0)
			column = n*(i-1-1) + j;
			semi_characters1(row,column) = -1*E_xE_y(i,j)/6;
		end
		if(i-1 > 0 && j+1 < n+1)
			column = n*(i-1-1) + j+1;
			semi_characters1(row,column) = -1*-1*E_xE_y(i,j)/12;
		end
		if(j-1 > 0)
			column = n*(i-1) + j-1;
			semi_characters1(row,column) = -1*E_xE_y(i,j)/6;
		end
		if(j+1 < n+1)
			column = n*(i-1) + j+1;
			semi_characters1(row,column) = -1*E_xE_y(i,j)/6;
		end
		if(i+1 < n+1 && j-1 > 0)
			column = n*(i) + j-1;
			semi_characters1(row,column) = -1*E_xE_y(i,j)/12;
		end
		if(i+1 < n+1)
			column = n*(i) + j;
			semi_characters1(row,column) = -1*E_xE_y(i,j)/6;
		end
		if(i+1 < n+1 && j+1 < n+1)
			column = n*(i) + j+1;
			semi_characters1(row,column) = -1*E_xE_y(i,j)/12;
		end
	endfor
endfor


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i = 1:n
	for j = 1:n
		row = n*(i-1) + j;
		semi_characters2(row,row) = alph*alph + E_y(i,j)*E_y(i,j) + E_x(i,j)*E_x(i,j);
		semi_characters3(row,row) = alph*alph + E_y(i,j)*E_y(i,j) + E_x(i,j)*E_x(i,j);
		if(i-1 > 0 && j-1 > 0)
			column = n*(i-1-1) + j-1;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/12;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/12;
		end
		if(i-1 > 0)
			column = n*(i-1-1) + j;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/6;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/6;
		end
		if(i-1 > 0 && j+1 < n+1)
			column = n*(i-1-1) + j+1;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/12;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/12;
		end
		if(j-1 > 0)
			column = n*(i-1) + j-1;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/6;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/6;
		end
		if(j+1 < n+1)
			column = n*(i-1) + j+1;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/6;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/6;
		end
		if(i+1 < n+1 && j-1 > 0)
			column = n*(i) + j-1;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/12;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/12;
		end
		if(i+1 < n+1)
			column = n*(i) + j;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/6;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/6;
		end
		if(i+1 < n+1 && j+1 < n+1)
			column = n*(i) + j+1;
			fprintf(fid,'Row %d column %d\n',row,column);
			semi_characters2(row,column) = (-1*alph*alph - E_y(i,j)*E_y(i,j) )/12;
			semi_characters3(row,column) = (-1*alph*alph - E_x(i,j)*E_x(i,j) )/12;
		end
	endfor
endfor

semi_coefficients1 = zeros(2,2);
semi_coefficients1(1,2) = semi_coefficients1(2,1) = 1;
semi_coefficients1 = kron(semi_coefficients1,semi_characters1);
semi_coefficients2 = zeros(2,2);
semi_coefficients2(1,1) = 1;
semi_coefficients2 =  kron(semi_coefficients2,semi_characters2);
coefficients = zeros(2,2);
coefficients(2,2) = 1;
coefficients = kron(coefficients,semi_characters3) + semi_coefficients2 + semi_coefficients1;



u_v = coefficients\const;
u_v1 = reshape(transpose(u_v),n*n,2);
u = reshape(u_v1(:,1),n,n);
v = reshape(u_v1(:,2),n,n);
[X,Y] = meshgrid(1:n,1:n);
grid on;
hold on;
quiver(X,Y,u,v);
title("Gauss-Jordan n=50, alpha=1");