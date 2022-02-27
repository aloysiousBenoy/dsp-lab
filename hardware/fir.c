
Announce something to your class

Announcement: "1. As announced earlier, DSP Lab…"
Birenjith P S
Created Feb 24Feb 24
1. As announced earlier, DSP Lab internal examination will happen on Feb 28, and March 2nd. 
2. Everyone will be given one question. The question will involve either programming on python or executing a task using DSP hardware. Questions that involve DSP hardware will be direct, and you just need to study the programs that are given to you. 
3. The exam is for 60 marks with a split-up of 15 (Preliminary work) + 10 (Implementation) + 30 (Result and performance) + 5 (Completed record)
4. There will be an online multiple-choice quiz on March 2nd late evening (time will be decided later) as a replacement for viva voce assessment.

Schedule.pdf
PDF


Announcement: "The aim of all experiments are…"
Birenjith P S
Created Feb 16Feb 16
The aim of all experiments are available in the manual provided here. The last date of submission of DSP Lab record is 23rd Feb 2022.

DSPLab.pdf
PDF


Announcement: "DSP Kit: Loopback, Delay, Convolution,…"
Birenjith P S
Created Feb 15Feb 15
DSP Kit: Loopback, Delay, Convolution, DFT, and FIR

dft.c
C

fir.c
C

convolvef.c
C

convolve.c
C

delay.c
C

loopback.c
C


Announcement: "DSP Kit familiarisation: led.c and…"
Birenjith P S
Created Feb 14Feb 14
DSP Kit familiarisation: led.c and tone.c

led.c
C

tone.c
C

Assignment: "Simulation of overlap add and save method"
Sruthi B. R. posted a new assignment: Simulation of overlap add and save method
Created Feb 11Feb 11
Material: "Code for FFT, and FIR filters"
Birenjith P S posted a new material: Code for FFT, and FIR filters
Created Feb 9Feb 9
Assignment: "Expt. No. 3 Basic DFT and IDFT implementation"
Sruthi B. R. posted a new assignment: Expt. No. 3 Basic DFT and IDFT implementation
Created Jan 14Jan 14

Announcement: "The first two experiments are listed in…"
Birenjith P S
Created Jan 2Jan 2
The first two experiments are listed in the document. You have to write algorithm in the lab record. Program and plots can be printed out and used. Results and inferences must be written.

DSPLab.pdf
PDF

/*
 In the code below, we use the window method to design an FIR low pass filter for a cut-off frequency of 1 KHz and test it on real-time signals on the DSK. To convert the cut-off frequency in Hz to cut-off frequency in rad/sample, we use the relation omega = 2*pi*F/Fs.  Assuming a sampling frequency Fs=8kHz, the cut-off frequency in rad/sample is omega_c = pi/4. 

Test the filter with different frequencies in the pass-band and stop-band (connect headphone out of PC to line-in of DSK using aux cable and play tones from youtube).  Observe the effect of the filter on a music signal
*/
#include <dsk6713.h>
#include <dsk6713_aic23.h>
#include <math.h>
#define L 11 //length of filter
int main(void)
{
    Uint32 sample_pair;
    float pi = 3.141592653589;
    float hamming[L], h[L], x[L] = { 0 }, y;
    int i;

    //Generate Hamming window sequence
    for (i = 0; i < L; i++)
        hamming[i] = 0.54 - 0.46 * cos(2 * pi * i / (L - 1));

    //cut-off frequency of filter in rad/sample
    float wc = pi / 4.0;

    //compute filter coeffs
    for (i = 0; i < L; i++)
        //avoid division by 0 when i=(L-1)/2
        if (i == (L - 1) / 2)
            h[i] = wc / pi * hamming[i];
        else
            h[i] = sin(wc * (i - (L - 1) / 2.0)) / (pi*(i - (L - 1)/2.0))
                    * hamming[i];

    DSK6713_init();
    DSK6713_AIC23_Config config = DSK6713_AIC23_DEFAULTCONFIG;
    DSK6713_AIC23_CodecHandle hCodec;
    hCodec = DSK6713_AIC23_openCodec(0, &config);

    /* Change the sampling rate to 8 kHz */
    DSK6713_AIC23_setFreq(hCodec, DSK6713_AIC23_FREQ_8KHZ);

    while (1)
    {
        while (!DSK6713_AIC23_read(hCodec, &sample_pair))
            ;
        //store top-half of sample from codec in x[0]
        x[0] = (int)sample_pair >>16;

        //process input sample:
        y = 0.0;

        for (i = 0; i < L; i++) //compute filter output
            y += h[i] * x[i];

        //shift delay line contents
        for (i = (L - 1); i > 0; i--)
            x[i] = x[i - 1];

        //output y to left channel
        sample_pair = (int)y <<16;
        while (!DSK6713_AIC23_write(hCodec, sample_pair))
            ;

    }

    /* Close the codec */
    DSK6713_AIC23_closeCodec(hCodec);

    return 0;
}
