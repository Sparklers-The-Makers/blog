/*C graphics program to draw a line.*/

#include <graphics.h>
 
int  main()
{
   int gd = DETECT, gm;
 
   initgraph(&gd, &gm, NULL);
       //will draw a horizontal line

   circle(200, 200, 50);
   line(200,250,200,400);
   line(200,300,300,230);
   line(200,300,100,230);
   line(200,400,250,500);
   line(200,400,150,500);
   circle(175, 190, 5);
   circle(225, 190, 5);
   line(175,220,225,220);
   delay(5000);
   closegraph();
   return 0;
}
