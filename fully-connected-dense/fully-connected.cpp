/*In this file, the forward propagation is realized for a fully connected dense BP neural network.
  In this case, the output dimension is set to 1,
  and at the same time, the hidden layer dimensions are set to NumInOneLayer for all hidden layers*/
#define _CRT_SECURE_NO_WARNINGS
#include "stdlib.h"
#include "stdio.h"

#define inputDim 2	/*input data dimension*/
#define LayerNum 6	/*except the input and output layers*/
#define NumInOneLayer 8	/*nodes number in one hidden layer*/

/*kernel and bias for the input layer*/
double kernel_input[inputDim][NumInOneLayer];
double bias_input[NumInOneLayer];
/*kernel and bias for the hidden layers, except the input and output layers*/
double kernel[LayerNum][NumInOneLayer][NumInOneLayer];
double bias[LayerNum][NumInOneLayer];
/*kernel and bias for the output layer*/
double kernel_output[NumInOneLayer];
double bias_output;

void ImportKernelBias() 
{
	/*read kernel and bias for input layer*/
	FILE *fp;
	fp= fopen("kernel_input.txt", "r");
	for (int i = 0; i < inputDim; i++)
	{
		for (int j = 0; j < NumInOneLayer; j++)
		{
			fscanf(fp, "%lf", &kernel_input[i][j]);
		}
	}
	fclose(fp);
	fp = fopen("bias_input.txt", "r");
	for (int i = 0; i < NumInOneLayer; i++)
	{
		fscanf(fp, "%lf", &bias_input[i]);
	}
	fclose(fp);

	/*read kernel and bias for output layer*/
	fp = fopen("kernel_output.txt", "r");
	for (int i = 0; i < NumInOneLayer; i++)
	{
		fscanf(fp, "%lf", &kernel_output[i]);
	}
	fclose(fp);
	fp = fopen("bias_output.txt", "r");
	fscanf(fp, "%lf", &bias_output);
	fclose(fp);

	/*read kernel and bias for other hidden layer*/
	int m = 0;
	char filename[30];
	for (m = 1; m < LayerNum + 1; m++)
	{
		sprintf(filename, "kernel_hl%d.txt", m);
		fp = fopen(filename, "r");
		for (int i = 0; i < NumInOneLayer; i++)
		{
			for (int j = 0; j < NumInOneLayer; j++)
			{
				fscanf(fp, "%lf", &kernel[m - 1][i][j]);
			}
		}
		fclose(fp);
		sprintf(filename, "bias_hl%d.txt", m);
		fp = fopen(filename, "r");
		for (int i = 0; i < NumInOneLayer; i++)
		{
			fscanf(fp, "%lf", &bias[m - 1][i]);
		}
		fclose(fp);
	}
}

double NNCalc(double input[])
{
	double NNOut;
	double fpre[NumInOneLayer], fnext[NumInOneLayer];

	/*compute using Neural Network and return hd*/
	for (int i = 0; i < NumInOneLayer; i++)	/*calculate the input layer*/
	{
		fpre[i] = 0.0;
		for (int j = 0; j < inputDim; j++)
		{
			fpre[i] = fpre[i] + input[j] * kernel_input[j][i];
		}
		fpre[i] = fpre[i] + bias_input[i];
		if (fpre[i] < 0.0)	//reLU
		{
			fpre[i] = 0.0;
		}
	}
	for (int m = 0; m < LayerNum; m++) /*calculate expect input and output layer*/
	{
		for (int i = 0; i < NumInOneLayer; i++)
		{
			fnext[i] = 0.0;
			for (int j = 0; j < NumInOneLayer; j++)
			{
				fnext[i] = fnext[i] + fpre[j] * kernel[m][j][i];
			}
			fnext[i] = fnext[i] + bias[m][i];
			if (fnext[i] < 0.0)
			{
				fnext[i] = 0.0;
			}
		}
		for (int i = 0; i < NumInOneLayer; i++)
		{
			fpre[i] = fnext[i];
		}
	}

	NNOut = 0.0;
	for (int i = 0; i < NumInOneLayer; i++)	/*calculate the output layer*/
	{
		NNOut = NNOut + fpre[i] * kernel_output[i];
	}
	NNOut = NNOut + bias_output;

	return NNOut;
}

int main()
{
	double x[2] = { 0.6,0.8 };
	double result;

	ImportKernelBias();
	result = NNCalc(x);
	printf("bias_input[6]=%lf\n", bias_input[6]);
	printf("NN(%lf,%lf)=%lf\n",x[0],x[1],result);
	getchar();
	return 1;
}