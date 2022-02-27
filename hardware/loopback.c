// loopback.c
#include <dsk6713.h>
#include <dsk6713_aic23.h>

void main(void)
{
    DSK6713_AIC23_Config config = DSK6713_AIC23_DEFAULTCONFIG;
    DSK6713_AIC23_CodecHandle hCodec;
    Uint32 sample_pair = 0;
    DSK6713_init(); /* In the BSL library */
    /* Start the codec */
    hCodec = DSK6713_AIC23_openCodec(0, &config);

    /* Change the sampling rate to 16 kHz */
    DSK6713_AIC23_setFreq(hCodec, DSK6713_AIC23_FREQ_16KHZ);

    /* Read left and right channel samples from the ADC and loop */
    /* them back out to the DAC.                                 */
    for (;;)
   
        while (!DSK6713_AIC23_read(hCodec, &sample_pair))
            ;
        while (!DSK6713_AIC23_write(hCodec, sample_pair))
            ;
    }
}
