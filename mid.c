#include<graphics.h>
#include<stdio.h>
void pixel(int xc,int yc,int x,int y);
int main()
{
	int gd = DETECT, gm; 
 
    	initgraph (&gd, &gm, NULL);

	int xc,yc,r,x,y,p;
	
	xc=100;
	yc =100;
	r = 50;
	
	x=0;
	y=r;
	p=1-r;
	pixel(xc,yc,x,y);
	
	while(x<y)
	{
		if(p<0)
		{
			x++;
			p=p+2*x+1;
		}
		else
		{
			x++;
			y--;
			p=p+2*(x-y)+1;
		}
		pixel(xc,yc,x,y);
	}
	delay(5000);

	closegraph();
	return 0;
}

void pixel(int xc,int yc,int x,int y)
{
	putpixel(xc+x,yc+y,WHITE);
	putpixel(xc+x,yc-y,WHITE);
	putpixel(xc-x,yc+y,WHITE);
	putpixel(xc-x,yc-y,WHITE);
	putpixel(xc+y,yc+x,WHITE);
	putpixel(xc+y,yc-x,WHITE);
	putpixel(xc-y,yc+x,WHITE);
	putpixel(xc-y,yc-x,WHITE);
}
