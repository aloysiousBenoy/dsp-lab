/*
In the code, right click on DSK6713_AIC23_DEFAULTCONFIG and click Open Declaration. This will open the header file which contains detailed information about how the codec is configured. (eg: how to decrease headphone volume?). The codec is started by calling the BSL function DSK6713_AIC23_openCodec(). 
Each iteration of the infinite loop generates a sample and writes it to the codec. Note that the float value is converted to int and the upper 16 bits are set to 0 before outputting. This is because the BSL function that configures the codec sets McBSP1 to send and receive 32-bit words, with the left sample in the upper 16 bits and right sample in the lower 16 bits. The 16-bit samples are in signed 2s complement form. Since the upper 16 bits of out_sample are set to 0,  the tone will be heard on the right channel only. If we want the output to be on the left channel, we can use the statement out_sample = (int)sample << 16; instead, which puts the 16-bit value in the top half, and sets the lower 16 bits to 0s. We can also pack two 16-bit samples in out_sample for output on both L and R channels. 
The function DSK6713_AIC23_write() is used to write a pair of samples to the DAC. The function uses polling to write samples and returns 0 if codec is not ready and returns 1 if write is successful. The while loop continues till write is successful. Build and Debug the program. Connect your headphones to the headphone and listen to the tone(beware of volume). You should hear the tone on R channel only. Using bitwise operators in C, try to output the tone on both channels. Change frequency F0 to 500 Hz and listen. Next, change F0 to 7500 Hz and listen. Explain what you observe. 
*/

#include <math.h>
#include <dsk6713.h>
#include <dsk6713_aic23.h>
int main()
{
    float Fs = 8000.;
    float F0 = 1000.;
    float pi = 3.141592653589;
    float theta = 0.;
    float delta = 2. * pi * F0 / Fs; // increment for theta
    float sample;
    unsigned out_sample;
    /* Initialize the board support library, must be called first */
    DSK6713_init();

    DSK6713_AIC23_Config config = DSK6713_AIC23_DEFAULTCONFIG;
    DSK6713_AIC23_CodecHandle hCodec;
    /* Start the codec */
    hCodec = DSK6713_AIC23_openCodec(0, &config);

    /* Change the sampling rate to 16 kHz */
    DSK6713_AIC23_setFreq(hCodec, DSK6713_AIC23_FREQ_8KHZ);

    for (;;)
    { /* Infinite loop */
        sample = 15000.0 * sin(theta); /* Scale for DAC */
        out_sample = (int)sample & 0x0000ffff; // Put in lower half (R) 

        /* Poll XRDY bit until true, then write to DXR */
        while (!DSK6713_AIC23_write(hCodec, out_sample))
            ;
        theta += delta;
        if (theta > 2 * pi)
            theta -= 2 * pi;

    }
}
