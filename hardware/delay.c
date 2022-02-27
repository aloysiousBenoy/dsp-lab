#include <dsk6713.h>
#include <dsk6713_aic23.h>
#define BUF_SIZE 8000
void main()
{

    		Uint32 sample_pair; // both channels packed in 32-bits
    		short i = 0, left, buffer[BUF_SIZE] = { 0 }, delayed, output;
    		DSK6713_init();
    		DSK6713_AIC23_Config config = DSK6713_AIC23_DEFAULTCONFIG;
    		DSK6713_AIC23_CodecHandle hCodec;
    		hCodec = DSK6713_AIC23_openCodec(0, &config);
    		DSK6713_AIC23_setFreq(hCodec, DSK6713_AIC23_FREQ_8KHZ);
    		while (1)
    		{
        		while (!DSK6713_AIC23_read(hCodec, &sample_pair))
            			;
        		//extract left sample and put in 16-bits
			left = (int)sample_pair >> 16; // 
        		delayed = buffer[i]; //read oldest sample
        		output = left + delayed; //output sum of new and delayed
        		buffer[i] = left; //replace oldest sample with input 
        		
			//increment i to point to the oldest sample
			if (++i >= BUF_SIZE) 
            			i = 0;
        		
			//put 16-bit sample in top-half
			sample_pair = (int)output << 16; 
        		
			while (!DSK6713_AIC23_write(hCodec, sample_pair))
            			;

    		}
    	}
