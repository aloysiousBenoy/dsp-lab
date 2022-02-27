/*
C Library File IO: CCS also supports standard C library file I/O functions such as fopen( ), fclose( ), fread( ), fwrite( ), and so on. These functions not only provide the ability of operating on different file formats, but also allow users to directly use the functions on computers. Comparing with the memory load/save method introduced in the previous part, these file I/O functions are portable to other development environments. An example of C program that uses fopen( ), fclose( ), fread( ), and fwrite( ) functions is included below. Verify the working of the program in CCS.
*/

#include <stdio.h>
void writedatafile(void);
void conv(float *x, float *h, float *y, short l, short m);
void main()
{
    writedatafile();//put some data in a file
    FILE *fp;
    fp = fopen("../../myfilebin", "rb");//open file for binary read
    float x[5];
//read 5 floats from file and store in x
fread(x, sizeof(float), 5, fp);
    float h[3]={1, 1, 1};
    float y[7];
    conv(x, h, y, 5, 3);

}

void writedatafile()
{
    float a[] = { 1, 2, 3, 4, 5}; //some data
    FILE *fp;
    fp = fopen("../../myfilebin", "wb");//open for binary write

    //write 5 floats in array a to myfilebin
    fwrite(a, sizeof(float), 5, fp); 
    fclose(fp);

}

void conv(float *x, float *h, float *y, short l, short m)
{
    short k, kmin, kmax, n;
    for (n = 0; n < l + m - 1; n++)
    {
        y[n] = 0;
        kmin = (n > l - 1) ? n - l + 1 : 0;
        kmax = (n > m - 1) ? m - 1 : n;
        // printf("%d %d\n", kmin, kmax);
        for (k = kmin; k <= kmax; k++)
        {
            y[n] = y[n] + h[k] * x[n - k];
        }

    }

}
