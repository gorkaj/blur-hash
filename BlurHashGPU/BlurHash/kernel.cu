
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <string.h>
#include <cstdint>
#include <math.h>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const char* blurHashForFile(const char* filename, int xComponent, int yComponent);
const char* blurHashForPixels(int xComponent, int yComponent, int width, int height, uint8_t* rgb, size_t bytesPerRow);
static char* encode_int(int value, int length, char* destination);
static int encodeDC(float r, float g, float b);
static int encodeAC(float r, float g, float b, float maximumValue);

// Utilities
static inline int linearTosRGB(float value) {
	float v = fmaxf(0, fminf(1, value));
	if (v <= 0.0031308) return v * 12.92 * 255 + 0.5;
	else return (1.055 * powf(v, 1 / 2.4) - 0.055) * 255 + 0.5;
}

__device__ static inline float sRGBToLinear(int value) {
	float v = (float)value / 255;
	if (v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
}

static inline float signPow(float value, float exp) {
	return copysignf(powf(fabsf(value), exp), value);
}

// Main
int main(int argc, char** argv)
{
	if (argc != 4) {
		fprintf(stderr, "Usage: %s imagefile xComponents yComponents\n", argv[0]);
		return 1;
	}

	const char* hash = blurHashForFile(argv[1], atoi(argv[2]), atoi(argv[3]));
	if (!hash) {
		fprintf(stderr, "Failed to load image file \"%s\".\n", argv[3]);
		return 1;
	}

	printf("%s\n", hash);

	return 0;
}

__global__ void computeHash(int yComponents, int xComponents, int width, int height, uint8_t* d_rgb, size_t bytesPerRow, float* d_factors)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= width * height)
		return;

	int x = index % width;
	int y = index / width;

	for (int i = 0; i < yComponents; i++)
	{
		for (int j = 0; j < xComponents; j++)
		{
			float basis = cosf(i * j * x / width) * cosf(M_PI * i * y / height);
			float normalisation = (j == 0 && i == 0) ? 1 : 2;
			float scale = normalisation / (width * height);
			/*atomicAdd(d_factors + i * xComponents * 3 + j * 3, scale * basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]));
			atomicAdd(d_factors + i * xComponents * 3 + j * 3 + 1, scale * basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]));
			atomicAdd(d_factors + i * xComponents * 3 + j * 3 + 2, scale * basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]));*/
			//d_factors[i * xComponents * 3 + j * 3] = scale * basis * sRGBToLinear(d_rgb[3 * x + 0 + y * bytesPerRow]);
			//d_factors[i * xComponents * 3 + j * 3 + 1] = scale * basis * sRGBToLinear(d_rgb[3 * x + 1 + y * bytesPerRow]);
			//d_factors[i * xComponents * 3 + j * 3 + 2] = scale * basis * sRGBToLinear(d_rgb[3 * x + 2 + y * bytesPerRow]);

			atomicAdd(d_factors + i * xComponents * 3 + j * 3, scale * basis * sRGBToLinear(d_rgb[3 * x + 0 + y * bytesPerRow]));
			atomicAdd(d_factors + i * xComponents * 3 + j * 3 + 1, scale * basis * sRGBToLinear(d_rgb[3 * x + 1 + y * bytesPerRow]));
			atomicAdd(d_factors + i * xComponents * 3 + j * 3 + 2, scale * basis * sRGBToLinear(d_rgb[3 * x + 2 + y * bytesPerRow]));
		}
	}
}

const char* blurHashForFile(const char* filename, int yComponents, int xComponents) {
	int width, height, channels;
	unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);
	if (!data) return NULL;

	const char* hash = blurHashForPixels(yComponents, xComponents, width, height, data, width * 3);

	stbi_image_free(data);

	return hash;
}

const char* blurHashForPixels(int yComponents, int xComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	float* allElements = (float*)malloc(yComponents * xComponents * 3 * sizeof(float));
	float*** factors = (float***)malloc(yComponents * sizeof(float**));

	if (!allElements || !factors)
		return nullptr;

	for (int i = 0; i < yComponents; i++)
	{
		*(factors + i) = (float**)malloc(xComponents * sizeof(float*));
		if (!(factors + i))
			return nullptr;

		for (int j = 0; j < xComponents; j++)
		{
			factors[i][j] = allElements + (i * xComponents * 3) + (j * 3);
		}
	}

	memset(allElements, 0, yComponents * xComponents * 3 * sizeof(float));

	float* d_allElements;
	cudaMalloc(&d_allElements, yComponents * xComponents * 3 * sizeof(float));
	cudaMemset(d_allElements, 0, yComponents * xComponents * 3 * sizeof(float));

	uint8_t* d_rgb;
	cudaMalloc(&d_rgb, 3 * width * height * sizeof(uint8_t));
	cudaMemcpy(d_rgb, rgb, 3 * width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

	int threads, blocks;
	threads = 2 << 9;
	blocks = ceil((double)width * height / threads);
	computeHash << <blocks, threads >> > (yComponents, xComponents, width, height, d_rgb, bytesPerRow, d_allElements);

	cudaDeviceSynchronize();
	cudaMemcpy(allElements, d_allElements, yComponents * xComponents * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	float* dc = factors[0][0];
	float* ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char* ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if (acCount > 0) {
		float actualMaximumValue = 0;
		for (int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	}
	else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for (int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;
	free(allElements);
	for (int i = 0; i < yComponents; i++)
	{
		free(factors[i]);
	}
	free(factors);
	cudaFree(d_allElements);
	cudaFree(d_rgb);

	return buffer;
}

static int encodeDC(float r, float g, float b) {
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static int encodeAC(float r, float g, float b, float maximumValue) {
	int quantR = fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
	int quantG = fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
	int quantB = fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5) * 9 + 9.5)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

static char characters[84] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static char* encode_int(int value, int length, char* destination) {
	int divisor = 1;
	for (int i = 0; i < length - 1; i++) divisor *= 83;

	for (int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}
