/* cifar10_nn.h */
#ifndef CIFAR10_NN_H
#define CIFAR10_NN_H

#include <stdint.h>

// CIFAR-10 class names
extern const char* cifar10_class_names[10];

// Neural network inference function
int cifar10_classify(uint8_t inputNN[32][32][3], float *confidence);

#endif /* CIFAR10_NN_H */
