#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef enum {
	CALCULATOR_ADD,
	CALCULATOR_SUB,
	CALCULATOR_DIV,
	CALCULATOR_MUL
} CALCULATOR_COMMANDS;

typedef enum{
	INPUT_RANDOM,
	INPUT_LINEAR
}INPUT_TYPE;

#define SAMPLES 262144
#define TPB 256
#define NUM_STREAMS 2
#define FILE_BUFFER_SIZE 32
#define MAX_COMMANDS 32
#define INPUT INPUT_LINEAR

__constant__ CALCULATOR_COMMANDS d_commands[MAX_COMMANDS];
__constant__ float d_operands[MAX_COMMANDS];

int readCommandsFromFile(CALCULATOR_COMMANDS *commands, float *operands);
void initInput(float *input);
void checkCUDAError(const char *msg);
int readLine(FILE *f, char buffer[]);
void cudaCalculatorDefaultStream(CALCULATOR_COMMANDS *commands, float *operands, int num_commands);
void cudaCalculatorNStream1(CALCULATOR_COMMANDS *commands, float *operands, int num_commands);
void cudaCalculatorNStream2(CALCULATOR_COMMANDS *commands, float *operands, int num_commands);
int checkResults(float* h_input, float* h_output, CALCULATOR_COMMANDS *commands, float *operands, int num_commands);

__global__ void parallelCalculator(float *input, float *output, int num_commands)
{
	float out;
	int idx;
	
	idx = threadIdx.x + blockIdx.x*blockDim.x;

	//get input
	out = input[idx];

	//applique les commandes
	for (int i = 0; i < num_commands; i++){
		CALCULATOR_COMMANDS cmd = d_commands[i];
		float v = d_operands[i];

		switch (cmd){
			case(CALCULATOR_ADD) : {
				out += v;
				break;
			}
			case(CALCULATOR_SUB) : {
				out -= v;
				break;
			}
			case(CALCULATOR_DIV) : {
				out /= v;
				break;
			}
			case(CALCULATOR_MUL) : {
				out *= v;
				break;
			}
		}
	}

	output[idx] = out;
}


int main(int argc, char**argv){
	int num_commands;

	CALCULATOR_COMMANDS h_commands[MAX_COMMANDS];
	float h_operands[MAX_COMMANDS];

	//Recupere les operation de calcul depuis un fichier
	num_commands = readCommandsFromFile(h_commands, h_operands);

	printf("%d commands found in file\n", num_commands);

	//Copie des elements dans la memoire constante
	cudaMemcpyToSymbol(d_commands, h_commands, sizeof(CALCULATOR_COMMANDS)*MAX_COMMANDS);
	checkCUDAError("Commands copy to constant memory");
	cudaMemcpyToSymbol(d_operands, h_operands, sizeof(float)*MAX_COMMANDS);
	checkCUDAError("Commands copy to constant memory");

	//Version synchrone
	cudaCalculatorDefaultStream(h_commands, h_operands, num_commands);

	//Version asynchrone 1
	//DECOMMENTEZ//cudaCalculatorNStream1(h_commands, h_operands, num_commands);
	//Version asynchrone 2
	//DECOMMENTEZ//cudaCalculatorNStream2(h_commands, h_operands, num_commands);
}

void cudaCalculatorDefaultStream(CALCULATOR_COMMANDS *commands, float *operands, int num_commands){
	float *h_input, *h_output;
	float *d_input, *d_output;
	float time;
	cudaEvent_t start, stop;
	int errors;

	//Initialisation des events CUDA
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Allocation memoire CPU
	h_input = (float*)malloc(sizeof(float)*SAMPLES);
	h_output = (float*)malloc(sizeof(float)*SAMPLES);

	//allocation memoire GPU
	cudaMalloc((void**)&d_input, sizeof(float)*SAMPLES);
	cudaMalloc((void**)&d_output, sizeof(float)*SAMPLES);
	checkCUDAError("CUDA Memory allocate: default stream");

	//Initialisation de h_input
	initInput(h_input);

	//Debut du timer
	cudaEventRecord(start);

	// Exercice 1.1.1 Copie H2D
	cudaMemcpy(A completer, A completer, A completer, A completer);
	checkCUDAError("CUDA Memory copy H2D: default stream");

	// Exercice 1.1.2 Lancement du kernel
	parallelCalculator << <A completer, A completer >> >(A completer, A completer, A completer);
	checkCUDAError("CUDA Kernel: default stream");

	//Exercice 1.1.3 Copie D2H
	cudaMemcpy(A completer, A completer, A completer, A completer);
	checkCUDAError("CUDA Memory copy D2H: default stream");

	//Fin du timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Verification des erreurs et affichage du temps
	errors = checkResults(h_input, h_output, commands, operands, num_commands);
	printf("Version Synchrone : %f secondes avec %d erreurs\n", time, errors);

	//Liberation de la memoire
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);
}

void cudaCalculatorNStream1(CALCULATOR_COMMANDS *commands, float *operands, int num_commands){
	float *h_input, *h_output;
	float *d_input, *d_output;
	float time;
	cudaEvent_t start, stop;
	int i, errors;
	cudaStream_t streams[NUM_STREAMS];

	//Initialisation des events CUDA
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Exercice 1.2.1. Allocation de la memoire CPU et GPU

	//Exercise 1.2.2. Initialisation des streams

	//Initialisation de h_input
	initInput(h_input);

	//Debut du timer
	cudaEventRecord(start);

	//Exercice 1.2.3. boucle sur les streams : copie H2D, lance le kernel, copie D2H

	//Fin du timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Verification des erreurs et affichage du temps
	errors = checkResults(h_input, h_output, commands, operands, num_commands);
	printf("Version asynchrone 1 : (%d streams) : %f secondes avec %d erreurs\n", NUM_STREAMS, time, errors);

	//Exercise 1.2.4. Destruction des streams

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFreeHost(h_input);
	cudaFreeHost(h_output);
}


void cudaCalculatorNStream2(CALCULATOR_COMMANDS *commands, float *operands, int num_commands){
	float *h_input, *h_output;
	float *d_input, *d_output;
	float time;
	cudaEvent_t start, stop;
	int i, errors;
	cudaStream_t streams[NUM_STREAMS];

	//Initialisation des events CUDA
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Exercice 1.3.1. Allocation de la memoire CPU et GPU

	//Exercice 1.3.2. Initialisation des streams

	//Initialisation de h_input
	initInput(h_input);

	//Debut du timer
	cudaEventRecord(start);

	//Exercice 1.3.3. Copie H2D Asynchrone

	//Exercice 1.3.4. Execution des kernels

	//Exercice 1.3.5. Copie H2D Asynchrone
	

	//Fin du timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Verification des erreurs et affichage du temps
	errors = checkResults(h_input, h_output, commands, operands, num_commands);
	printf("Version asynchrone 2 : (%d streams) : %f secondes avec %d erreurs\n", NUM_STREAMS, time, errors);

	//Exercise 1.3.6. Destruction des streams

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFreeHost(h_input);
	cudaFreeHost(h_output);
}

int readCommandsFromFile(CALCULATOR_COMMANDS *commands, float *operands)
{
	FILE *f;
	float in_value;
	unsigned int line;
	char buffer[FILE_BUFFER_SIZE];
	char command[4];
	line = 0;

	printf("Recuperation des commandes depuis le fichier...\n");
	f = fopen("commands.calc", "r");
	if (f == NULL){
		fprintf(stderr, "File not found\n");
		return 0;
	}


	while (readLine(f, buffer)){
		line++;

		if (line >= MAX_COMMANDS){
			fprintf(stderr, "To many commands in form maximum is %u\n", MAX_COMMANDS);
			return 0;
		}

		if (!(isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3] == ' ')){
			fprintf(stderr, "Incorrect command format at line %u\n", line);
			return 0;
		}

		sscanf(buffer, "%s %f", command, &in_value);

		if (strcmp(command, "add") == 0){
			commands[line] = CALCULATOR_ADD;
		}
		else if (strcmp(command, "sub") == 0){
			commands[line] = CALCULATOR_SUB;
		}
		else if (strcmp(command, "div") == 0){
			commands[line] = CALCULATOR_DIV;
		}
		else if (strcmp(command, "mul") == 0){
			commands[line] = CALCULATOR_MUL;
		}
		else{
			fprintf(stderr, "Unknown command at line %u!\n", line);
			return 0;
		}

		operands[line] = in_value;

	}

	fclose(f);

	return line;
}


void initInput(float *input){
	int i;

	for (i = 0; i < SAMPLES; i++){
		if (INPUT == INPUT_LINEAR)
			input[i] = (float)i;
		else if (INPUT == INPUT_RANDOM)
			input[i] = rand() / (float)RAND_MAX;
	}
}

int readLine(FILE *f, char buffer[]){
	int i = 0;
	char c;
	while ((c = getc(f)) != '\n'){
		if (c == EOF)
			return 0;
		buffer[i++] = c;
		if (i == FILE_BUFFER_SIZE){
			fprintf(stderr, "Buffer size is too small for line input\n");
			exit(0);
		}
	}
	buffer[i] = '\0';

	if (strncmp(buffer, "exit", 4) == 0)
		return 0;
	else
		return 1;

}

int checkResults(float* h_input, float* h_output, CALCULATOR_COMMANDS *commands, float *operands, int num_commands)
{
	int i, j, errors;

	errors = 0;
	for (i = 0; i < SAMPLES; i++){
		float out = h_input[i];
		for (j = 0; j < num_commands; j++){
			CALCULATOR_COMMANDS cmd = commands[j];
			float v = operands[j];

			switch (cmd){
			case(CALCULATOR_ADD) : {
									   out += v;
									   break;
			}
			case(CALCULATOR_SUB) : {
									   out -= v;
									   break;
			}
			case(CALCULATOR_DIV) : {
									   out /= v;
									   break;
			}
			case(CALCULATOR_MUL) : {
									   out *= v;
									   break;
			}
			}
		}
		//test the result
		if (h_output[i] != out){
			//fprintf(stderr, "Error: GPU result (%f) differs from CPU result (%f) at index %d\n", h_output[i], out, i);
			errors++;
		}
	}

	return errors;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}