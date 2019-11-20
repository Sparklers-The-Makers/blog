#include<graphics.h>
#include<stdio.h>

void main()
{
int gd = DETECT, gm;
int x1,y1,x2,y2,tx1,ty1,tx2,ty2,sx1,sy1,sx2,sy2,tx,ty,sx,sy;
printf("Enter the coordinates\n");
scanf("%d%d%d%d",&x1,&y1,&x2,&y2);

printf("Enter the translation values\n");
scanf("%d%d",&tx,&ty);

printf("Enter the shear values\n");
scanf("%d%d",&sx,&sy);

initgraph(&gd,&gm,NULL);
setcolor(BLUE);
line(x1,y1,x2,y2);

tx1 = x1 + tx;
tx2 = x2 + tx;

ty1 = y1 + ty;
ty2 = y2 + ty;

setcolor(RED);
line(tx1,ty1,tx2,ty2);

sx1 = tx1 + sx*ty1;
sx2 = tx2 + sx*ty2;

sy1 = ty1 + sy*tx1;
sy2 = ty2 + sy*tx2;

setcolor(GREEN);
line(sx1,sy1,sx2,sy2);

getch();
closegraph();
}




