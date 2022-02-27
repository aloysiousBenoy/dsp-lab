/*
During a debug session, we can visualize the signals using the Graph tool in CCS. Step Over the program line by line and halt when y has been computed. Click on Tools>Graph>Single-Time. Set the Graph Properties as follows: 
Acquisition buffer size: 6 (since y[n] has 6 elements)
DSP Data type: 32-bit floating point (since we are using float type variable)
Start Address: y (array name by itself is treated as the address of its first element)
Display Data Size: 6
Click Ok. The graph of y[n] should appear. 

Instead of the above steps, you can go to the Variables view (from the cluster of tabbed views named Variables, Expressions, and Registers), right click on y and click Graph.
To emphasize the discrete nature of y[n], click on Show The Graph Properties button in the graph window toolbar and change the Data Plot Style to Bar. Also, right click anywhere on the graph, on the context menu that opens, select Display Properties, click Axes and change Y-axis Display format to Decimal. 

Memory Save/ Load: In the previous experiment, the elements of the arrays x and h were hard-coded in the program itself. We might need to process input data saved in files (for example, when simulating real-time input with predigitized data captured at another time/place). CCS can read data from a file on the host computer and put the data in target processor memory. CCS can also write the processed data samples from the target processor to the host computer as an output file for analysis. In this experiment, we will first save the output y[n] from the previous part in a file and then use that file as the input x[n] and perform convolution again. In effect, we are simulating the cascade of two identical systems with impulse response h[n]. 
First, start Debugging the program in part a once again and Step over line by line. 
Once the conv() function returns, go to Variables view, right click on variable y and click View Memory.  

In the Memory Browser window that opens, you can see the memory addresses and data in y. In the memory browser toolbar, click on Save memory. 
Browse to your workspace folder and give some file name. Click Save. 
Leave File type as TI data. 
Click Next, select Format as 32-bit floating point. 
Find out the starting address for y in the Memory Browser and type the correct start address in the Start Address field. 
Specify the number of memory words as 6 (since y has 6 floats, and a float as well as the word-size on the 6713 is 32 bits, 6 words mean 6 floats). 
Click Finish.
*/

#include <stdio.h>
void conv(float *x, float *h, float *y, short l, short m);
void main()
{
    float x[] = { 1, 2, 3, 4 };
    float h[] = { 1, 1, 1 };
    float y[6]; //length of y = L+M-1
    short n;
    conv(x, h, y, 4, 3);
    for (n = 0; n < 6; n++)
        printf("%.2f ", y[n]);
}

void conv(float *x, float *h, float *y, short l, short m)
{
    short k, kmin, kmax, n;
    for (n = 0; n < l + m - 1; n++)
    {
        y[n] = 0;
        kmin = (n > l - 1) ? n - l + 1 : 0;
        kmax = (n > m - 1) ? m - 1 : n;
        //printf("%d %d\n", kmin, kmax);
        for (k = kmin; k <= kmax; k++)
        {
            y[n] = y[n] + h[k] * x[n - k];

        }

    }

}
