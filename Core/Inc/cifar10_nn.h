/*
 * cifar10_nn.h
 *
 *  Created on: May 26, 2025
 *      Author: HoangHai
 */

#ifndef INC_CIFAR10_NN_H_
#define INC_CIFAR10_NN_H_

#include <stdint.h>

// CIFAR-10 class names
extern const char* cifar10_class_names[10];

// Neural network inference function
int cifar10_classify(uint8_t inputNN[32][32][3], float *confidence);

#endif /* INC_CIFAR10_NN_H_ */
