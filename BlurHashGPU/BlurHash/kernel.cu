
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

const char* blurHashForFile(const char* filename);
const char* blurHashForPixels(int width, int height, uint8_t* rgb, size_t bytesPerRow);
static float* multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t* rgb, size_t bytesPerRow);
static char* encode_int(int value, int length, char* destination);
static int encodeDC(float r, float g, float b);
static int encodeAC(float r, float g, float b, float maximumValue);

const static int xComponents = 4;
const static int yComponents = 3;

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
	if (argc != 2) {
		fprintf(stderr, "Usage: %s imagefile\n", argv[0]);
		return 1;
	}

	const char* hash = blurHashForFile(argv[1]);
	if (!hash) {
		fprintf(stderr, "Failed to load image file \"%s\".\n", argv[3]);
		return 1;
	}

	printf("%s\n", hash);

	return 0;
}

const char* blurHashForFile(const char* filename) {
	int width, height, channels;
	unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);
	if (!data) return NULL;

	const char* hash = blurHashForPixels(width, height, data, width * 3);

	stbi_image_free(data);

	return hash;
}

const char* blurHashForPixels(int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	float factors[yComponents][xComponents][3];
	memset(factors, 0, sizeof(factors));

	for (int y = 0; y < yComponents; y++) {
		for (int x = 0; x < xComponents; x++) {
			float* factor = multiplyBasisFunction(x, y, width, height, rgb, bytesPerRow);
			factors[y][x][0] = factor[0];
			factors[y][x][1] = factor[1];
			factors[y][x][2] = factor[2];
		}
	}

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

	return buffer;
}

__global__ void computeCompositeColor(int xComponent, int yComponent, int width, int height, uint8_t* rgb, size_t bytesPerRow, float* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= width * height)
		return;

	int x = index / width;
	int y = index % width;
	// printf("%d\n", threadIdx.x);

	float basis = cosf(yComponent * xComponent * x / width) * cosf(M_PI * yComponent * y / height);
	printf("%f\n", basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]));
	
	//atomicAdd(result, basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]));
	//atomicAdd(result + 1, basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]));
	//atomicAdd(result + 2, basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]));
	result[0] = basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
	result[1] = basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
	result[2] = basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);

	return;
}

static float* multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	// float r = 0, g = 0, b = 0;
	float* d_rgb_vector, *h_rgb_vector;

	h_rgb_vector = new float[3];
	memset(h_rgb_vector, 0, 3 * sizeof(float));
	cudaMalloc(&d_rgb_vector, 3 * sizeof(float));
	cudaMemset(&d_rgb_vector, 0, 3 * sizeof(float));

	float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;

	//for (int y = 0; y < height; y++) {
	//	for (int x = 0; x < width; x++) {
	//		float basis = cosf(yComponent * xComponent * x / width) * cosf(M_PI * yComponent * y / height);
	//		r += basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
	//		g += basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
	//		b += basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);
	//	}
	//}

	int threads, blocks;
	threads = 2 << 9;
	blocks = ceil((double)width * height / threads);

	computeCompositeColor << < blocks, threads >> > (xComponent, yComponent, width, height, rgb, bytesPerRow, d_rgb_vector);
	cudaDeviceSynchronize();
	cudaMemcpy(h_rgb_vector, d_rgb_vector, 3 * sizeof(float), cudaMemcpyDeviceToHost);

	float scale = normalisation / (width * height);

	static float result[3];
	result[0] = h_rgb_vector[0] * scale;
	result[1] = h_rgb_vector[1] * scale;
	result[2] = h_rgb_vector[2] * scale;

	delete[] h_rgb_vector;
	cudaFree(&d_rgb_vector);
	return result;
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
