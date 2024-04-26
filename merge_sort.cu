/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>

__device__ float Y0d[10];
__device__ float md[10];
__device__ float alphad[10];;
__device__ float nu2d[10];
__device__ float rhod[10];
__device__ float Strd[4];


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))>

void strikeInterval(float *K, float T){

	float fidx = T*12.0f + 1.0f;
	int i = 0;
	float coef = 1.0f;
	float delta;

	while(i<fidx){
		coef *= (1.025f);
		i++;
	}
	delta = (coef - 1.0f)*(coef + 1.0f)/(4.0f*coef);

	for(i=0; i<4; i++){
		K[i] = (i+1.0f)*delta + (1.0f/coef);
	}
}

// Set the state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

// Monte Carlo simulation kernel
__global__ void MC_k(float dt, float T, int Ntraj, curandState* state, float* sum){

	int pidx, same;
	float t, X, Y;
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	curandState localState = state[idx];
	float2 G;
	float B;
	float price;
	float sumR = 0.0f;
	float sum2R = 0.0f;
	float StrR, mR, alphaR, betaR, rhoR, nu2R;

	/***********************************************************************
		
	How should we fill StrR, mR, alphaR, nu2R, rhoR?

	************************************************************************/

	betaR = sqrtf(2.0f*alphaR*nu2R)*(1.0f - expf(mR));
	for (int i = 0; i < Ntraj; i++) {
		/***********************************************************************
		
		Put your code here

		************************************************************************/
	}
	sum[2*idx] = sumR;
	sum[2*idx + 1] = sum2R;

	/* Copy state back to global memory */
	state[idx] = localState;
}

int main(void) {

	float Y0[10] = {logf(0.4f), logf(0.35f), logf(0.31f), logf(0.27f), logf(0.23f), 
					logf(0.2f), logf(0.17f), logf(0.14f), logf(0.11f), logf(0.08f)};
	float m[10] = {logf(0.34f), logf(0.3f), logf(0.27f), logf(0.24f), logf(0.21f), 
					logf(0.18f), logf(0.15f), logf(0.12f), logf(0.09f), logf(0.06f)};
	float alpha[10] = {0.2f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f};
	float nu2[10] = {0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
	float rho[10] = {0.95f, 0.75f, 0.55f, 0.35f, 0.15f, -0.15f, -0.35f, -0.55f, -0.75f, -0.95f};
	float Tmt[4] = {3.0f/12.0f, 6.0f/12.0f, 1.0f, 2.0f};
	float Str[4];



	cudaMemcpyToSymbol(Y0d, Y0, 10*sizeof(float));
	cudaMemcpyToSymbol(md, m, 10*sizeof(float));
	cudaMemcpyToSymbol(alphad, alpha, 10*sizeof(float));
	cudaMemcpyToSymbol(nu2d, nu2, 10*sizeof(float));
	cudaMemcpyToSymbol(rhod, rho, 10*sizeof(float));

	int NTPB = 640;
	int NB = 625;
	int Ntraj = 100000;
	float dt = sqrtf(1.0f/(64.0f*12.0f));
	float StrR, mR, alphaR, betaR, rhoR, YR, price, error;

	curandState* states;
	cudaMalloc(&states, NB*NTPB*sizeof(curandState));
	init_curand_state_k <<<NB, NTPB>>> (states);
	float *sum;
	cudaMallocManaged(&sum, 2*NB*NTPB*sizeof(float));

	FILE* fpt;

	char strg[20];
	for(int i=0; i<4; i++){
		strikeInterval(Str, Tmt[i]);
		cudaMemcpyToSymbol(Strd, Str, 4*sizeof(float));
		MC_k<<<NB,NTPB>>>(dt, Tmt[i], Ntraj, states, sum);
		cudaDeviceSynchronize();
		for(int j=0; j<4; j++){
			StrR = Str[j];
			sprintf(strg, "Tmt%.2fStr%.2f.csv", Tmt[i], StrR);
			fpt = fopen(strg, "w+");
			fprintf(fpt, "alpha, beta, m, rho, Y0, price, 95cI\n");
			for(int k=0; k< 640*625; k++){
				/***********************************************************************
					
				How should we fill StrR, mR, alphaR, betaR, rhoR, YR, price and error?

				************************************************************************/
				fprintf(fpt, "%f, %f, %f, %f, %f, %f, %f\n", alphaR, betaR, mR, rhoR, YR, price, error);
			}
			fclose(fpt);
		}
	}

	cudaFree(states);
	cudaFree(sum);

	return 0;
}