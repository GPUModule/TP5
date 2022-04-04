// System includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust\device_ptr.h>
#include <thrust\scan.h>


#define TPB 256

#define NUM_PARTICLES 16384
#define ENV_DIM 32.0f
#define INTERACTION_RANGE 4.0f
#define ENV_BIN_DIM ((unsigned int)(ENV_DIM/INTERACTION_RANGE))
#define ENV_BINS (ENV_BIN_DIM*ENV_BIN_DIM)


struct key_values{
	int sorting_key[NUM_PARTICLES];
	int value[NUM_PARTICLES];
};
typedef struct key_values key_values;

struct particles{
	float2 location[NUM_PARTICLES];
	int nn_key[NUM_PARTICLES];
};
typedef struct particles particles;

struct environment{
	int count[ENV_BINS];
	int start_index[ENV_BINS];
};
typedef struct environment environment;

__global__ void particleNNSearch(particles *p, environment *env);
__global__ void keyValues(particles *p, key_values *kv);
__global__ void reorderParticles(key_values *kv, particles *p, particles *p_sorted);
__global__ void histogramParticles(particles *p, environment *env);
__device__ __host__ int2 binLocation(float2 location);
__device__ __host__ int binIndex(int2 bin);

void particlesCPU();
void particlesGPU();
void initParticles(particles *p);
int checkResults(char* name, particles *p);
void keyValuesCPU(particles *p, key_values *kv);
void sortKeyValuesCPU(key_values *kv);
void reorderParticlesCPU(key_values *kv, particles *p, particles *p_sorted);
void histogramParticlesCPU(particles *p, environment *env);
void prefixSumEnvironmentCPU(environment * env);

void checkCUDAError(const char *msg);


/* Kernels GPU */

__global__ void particleNNSearch(particles *p, environment *env)
{
	int2 bin;
	int i, x, y;
	int idx;
	float2 location;
	int nn;
	
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	//get location
	location = p->location[idx];
	bin = binLocation(location);
	nn = -1;

	//vérifie tous les bins voisins de la particule (9 au total) - pas d'enveloppement des frontières (boundary wrapping)
	float dist_sq = ENV_DIM*ENV_DIM;	//a big number

	for (x = bin.x - 1; x <= bin.x + 1; x++){
		//no wrapping
		if ((x < 0) || (x >= ENV_BIN_DIM))			
			continue;

		for (y = bin.y - 1; y <= bin.y + 1; y++){
			//no wrapping
			if ((y < 0) || (y >= ENV_BIN_DIM))			
				continue;

			//Recupere l'indice du bin
			int bin_index = binIndex(make_int2(x, y));

			//recupere l'indice de depart du bin associe
			int bin_start_index = env->start_index[bin_index];

			//recupere le compte du bin
			int bin_count = env->count[bin_index];

			//Boucle sur les particules pour trouver le plus proche voisin
			for (i = bin_start_index; i < bin_start_index+bin_count; i++){
				float2 n_location = p->location[i];
				if (i != idx){ //Ne peut pas etre son plus proche voisin
					//Verification de la distance
					float n_dist_sq = (n_location.x - location.x)*(n_location.x - location.x) + (n_location.y - location.y)*(n_location.y - location.y);
					if (n_dist_sq < dist_sq){
						//Un plus proche voisin a ete trouve si dans la range
						if (n_dist_sq < INTERACTION_RANGE*INTERACTION_RANGE){
							dist_sq = n_dist_sq;
							nn = i;
						}
					}
				}
			}
		}
	}

	//write nearest neighbour
	p->nn_key[idx] = nn;
}

/* Kernels additionels pour l'implementation thrust */

__global__ void keyValues(particles *p, key_values *kv)
{
	//Exercice 2.1.1
}

__global__ void reorderParticles(key_values *kv, particles *p, particles *p_sorted)
{
	// Exercice 2.1.2
}
__global__ void histogramParticles(particles *p, environment *env)
{
	// Exercice 2.1.3
}

__device__ __host__ int2 binLocation(float2 location){
	int bin_x = (int)(location.x / INTERACTION_RANGE);
	int bin_y = (int)(location.y / INTERACTION_RANGE);
	return make_int2(bin_x, bin_y);
}

__device__ __host__ int binIndex(int2 bin){
	return bin.x + bin.y*ENV_BIN_DIM;
}

/* Host Functions*/

int main(int argc, char **argv)
{
	particlesCPU();
	particlesGPU();

	return 0;
}

void particlesCPU()
{
	environment *h_env;
	environment *d_env;
	particles *h_particles;
	particles *h_particles_sorted;
	particles *d_particles;
	particles *d_particles_sorted;
	key_values *h_key_values;
	key_values *d_key_values;

	float time;
	clock_t begin, end;
	int errors;

	//allocation de la memoire CPU
	h_env = (environment*)malloc(sizeof(environment));
	h_particles = (particles*)malloc(sizeof(particles));
	h_particles_sorted = (particles*)malloc(sizeof(particles));
	h_key_values = (key_values*)malloc(sizeof(key_values));
	checkCUDAError("CPU version: Host malloc");

	//Allocation de la memoire GPU
	cudaMalloc((void**)&d_env, sizeof(environment));
	cudaMalloc((void**)&d_particles, sizeof(particles));
	cudaMalloc((void**)&d_particles_sorted, sizeof(particles));
	cudaMalloc((void**)&d_key_values, sizeof(key_values));
	checkCUDAError("CPU version: Device malloc");

	//mets les donnees CPU a 0
	memset(h_env, 0, sizeof(environment));
	memset(h_particles, 0, sizeof(particles));
	memset(h_key_values, 0, sizeof(key_values));

	//mets les donnees GPU a 0
	cudaMemset(d_env, 0, sizeof(environment));
	cudaMemset(d_particles, 0, sizeof(particles));
	cudaMemset(d_key_values, 0, sizeof(key_values));
	checkCUDAError("CPU version: Device memset");

	//Initialisation des particules
	initParticles(h_particles);

	/* implementation CPU */
	cudaDeviceSynchronize();
	begin = clock();

	//paires key values
	keyValuesCPU(h_particles, h_key_values);
	//trie les particules sur CPU
	sortKeyValuesCPU(h_key_values);
	//reordonne les particles
	reorderParticlesCPU(h_key_values, h_particles, h_particles_sorted);
	//histogramme du nombre de particules
	histogramParticlesCPU(h_particles_sorted, h_env);
	//somme des prefix
	prefixSumEnvironmentCPU(h_env);
	//copie H2D
	cudaMemcpy(d_particles_sorted, h_particles_sorted, sizeof(particles), cudaMemcpyHostToDevice);
	cudaMemcpy(d_env, h_env, sizeof(environment), cudaMemcpyHostToDevice);
	checkCUDAError("CPU version: Host 2 Device");
	//lancement du kernel du plus proche voisin (particule)
	particleNNSearch <<<NUM_PARTICLES / TPB, TPB >>>(d_particles_sorted, d_env);
	checkCUDAError("CPU version: CPU version Kernel");
	//copie D2H
	cudaMemcpy(h_particles_sorted, d_particles_sorted, sizeof(particles), cudaMemcpyDeviceToHost);
	checkCUDAError("CPU version: Device 2 Host");

	//calcul du temps
	cudaDeviceSynchronize();
	end = clock();
	time = (float)(end - begin) / CLOCKS_PER_SEC;

	errors = checkResults("CPU", h_particles_sorted);
	printf("CPU NN Search completed in %f seconds with %d errors\n", time, errors);

	//Liberation de la memoire
	free(h_env);
	free(h_particles);
	free(h_particles_sorted);
	free(h_key_values);
	cudaFree(d_env);
	cudaFree(d_particles);
	cudaFree(d_particles_sorted);
	cudaFree(d_key_values);
	checkCUDAError("CPU version: CUDA free");

}


void particlesGPU()
{
	environment *h_env;
	environment *d_env;
	particles *h_particles;
	particles *h_particles_sorted;
	particles *d_particles;
	particles *d_particles_sorted;
	key_values *h_key_values;
	key_values *d_key_values;

	float time;
	clock_t begin, end;
	int errors;
	//Allocation de la memoire CPU
	h_env = (environment*)malloc(sizeof(environment));
	h_particles = (particles*)malloc(sizeof(particles));
	h_particles_sorted = (particles*)malloc(sizeof(particles));
	h_key_values = (key_values*)malloc(sizeof(key_values));
	checkCUDAError("GPU version: Host malloc");

	//Allocation de la memoire GPU
	cudaMalloc((void**)&d_env, sizeof(environment));
	cudaMalloc((void**)&d_particles, sizeof(particles));
	cudaMalloc((void**)&d_particles_sorted, sizeof(particles));
	cudaMalloc((void**)&d_key_values, sizeof(key_values));
	checkCUDAError("GPU version: Device malloc");

	//mets les donnees CPU a 0
	memset(h_env, 0, sizeof(environment));
	memset(h_particles, 0, sizeof(particles));
	memset(h_key_values, 0, sizeof(key_values));

	//mets les donnees GPU a 0
	cudaMemset(d_env, 0, sizeof(environment));
	cudaMemset(d_particles, 0, sizeof(particles));
	cudaMemset(d_key_values, 0, sizeof(key_values));
	checkCUDAError("GPU version: Device memset");
	//Initialisation des particules
	initParticles(h_particles);

	/* Implementation avec thrust*/
	cudaDeviceSynchronize();
	begin = clock();

	//Exercice 2.2.1. Copie H2D
	//cudaMemcpy(...)
	checkCUDAError("GPU version: Host 2 Device");

	//On genere les paires key values sur le device
	keyValues << <NUM_PARTICLES / TPB, TPB >> >(d_particles, d_key_values);
	checkCUDAError("GPU version: Device keyValues");

	//Exercice 2.2.2. On trie par key
	//thrust::sort_by_key(...);
	checkCUDAError("GPU version: Thrust sort");

	//Exercice 2.2.3. On appelle le kernel reorderParticles
	//reorderParticles <<<...>>>
	checkCUDAError("GPU version: Device reorder");

	//Exercice 2.2.4. On appelle le kernel histogramParticles
	//histogramParticles <<<...>>>(...);
	checkCUDAError("GPU version: Device Histogram");

	//Exercice 2.2.5. somme de prefix avec thrust
	thrust::exclusive_scan(A completer, A completer, A completer);
	checkCUDAError("GPU version: Thrust scan");

	//Lancement du kernel du plus proche voisin
	particleNNSearch << <NUM_PARTICLES / TPB, TPB >> >(d_particles_sorted, d_env);
	checkCUDAError("GPU version: Kernel");

	//copie D2H
	cudaMemcpy(h_particles_sorted, d_particles_sorted, sizeof(particles), cudaMemcpyDeviceToHost);
	checkCUDAError("GPU version: Device 2 Host");

	//Calcule du temps
	cudaDeviceSynchronize();
	end = clock();
	time = (float)(end - begin) / CLOCKS_PER_SEC;

	/* On affiche les resultats*/
	errors = checkResults("GPU", h_particles_sorted);
	printf("GPU NN Search completed in %f seconds with %d errors\n", time, errors);

	
	//On libere la memoire
	free(h_env);
	free(h_particles);
	free(h_particles_sorted);
	free(h_key_values);
	cudaFree(d_env);
	cudaFree(d_particles);
	cudaFree(d_particles_sorted);
	cudaFree(d_key_values);
	checkCUDAError("GPU version: CUDA free");

}




void initParticles(particles *p){
	//seed
	srand(123);
	
	//random positions
	for (int i = 0; i < NUM_PARTICLES; i++){
		float rand_x = rand() / (float)RAND_MAX * ENV_DIM;
		float rand_y = rand() / (float)RAND_MAX * ENV_DIM;
		float2 location = make_float2(rand_x, rand_y);
		p->location[i] = location;
	}
}


int checkResults(char* name, particles *p){
	int i, j, errors;

	errors = 0;

	for (i = 0; i < NUM_PARTICLES; i++){
		float2 location = p->location[i];
		float dist_sq = ENV_DIM*ENV_DIM;	//a big number
		int cpu_nn = -1;

		//Cherche le plus proche voisin sur CPU
		for (j = 0; j < NUM_PARTICLES; j++){
			float2 n_location = p->location[j];
			if (j != i){ //Ne peut pas etre son plus proche voisin
				//Verification de lq distance
				float n_dist_sq = (n_location.x - location.x)*(n_location.x - location.x) + (n_location.y - location.y)*(n_location.y - location.y);
				if (n_dist_sq < dist_sq){
					//Un plus proche voisin a ete trouve si dans la range
					if (n_dist_sq < INTERACTION_RANGE*INTERACTION_RANGE){
						dist_sq = n_dist_sq;
						cpu_nn = j;
					}
				}
			}
		}

		if (p->nn_key[i] != cpu_nn){
			fprintf(stderr, "Error: %s NN for index %d is %d, Ref NN is %u\n", name, i, p->nn_key[i], cpu_nn);
			errors++;
		}
	}


	return errors;
}

void keyValuesCPU(particles *p, key_values *kv){
	//random positions
	for (int i = 0; i < NUM_PARTICLES; i++){
		float2 location = p->location[i];
		kv->value[i] = i;
		kv->sorting_key[i] = binIndex(binLocation(location));
	}
}

void sortKeyValuesCPU(key_values *kv){
	int i, j;

	//simple (lent) trie a bulle en CPU
	for (i = 0; i < (NUM_PARTICLES - 1); i++)
	{
		for (j = 0; j < NUM_PARTICLES - i - 1; j++)
		{
			if (kv->sorting_key[j] > kv->sorting_key[j + 1])
			{
				//swap des valeurs
				int swap_key;
				int swap_sort_value;

				swap_key = kv->value[j];
				swap_sort_value = kv->sorting_key[j];

				kv->value[j] = kv->value[j + 1];
				kv->sorting_key[j] = kv->sorting_key[j + 1];

				kv->value[j + 1] = swap_key;
				kv->sorting_key[j + 1] = swap_sort_value;
			}
		}
	}
}

void reorderParticlesCPU(key_values *kv, particles *p, particles *p_sorted){
	int i;

	//re-ordonne les particules en se basant sur une key ancienne
	for (i = 0; i < NUM_PARTICLES; i++){
		int old_index = kv->value[i];
		p_sorted->location[i] = p->location[old_index];
	}
}

void histogramParticlesCPU(particles *p, environment *env)
{
	int i;

	//Boucle sur les particules et incremente le compteur de bin
	for (i = 0; i < (NUM_PARTICLES - 1); i++)
	{
		int bin_location = binIndex(binLocation(p->location[i])); //recalculate the sort value
		env->count[bin_location]++;
	}
}

void prefixSumEnvironmentCPU(environment * env)
{
	int i;
	int sum = 0;

	//somme de prefix sur CPU
	for (i = 0; i < ENV_BINS; i++){
		env->start_index[i] = sum;
		sum += env->count[i];
	}
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
