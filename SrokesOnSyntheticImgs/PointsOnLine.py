x0 = 44
x1 = 105
y0 = 316
y1 = 250

dx = x1 - x0
dy = y1 - y0
if dy > dx:
	D = dy - dx
else:
	D = dx - dy
y = y0

for x in range(x0,x1-1):
	print(x,y)
	if D >= 0:
		print('yo')
		y = y +1
		D = D- dx
	D = D + dy