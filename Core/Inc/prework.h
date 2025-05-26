/*
 * prework.h
 *
 *  Created on: May 26, 2025
 *      Author: HoangHai
 */

#ifndef INC_PREWORK_H_
#define INC_PREWORK_H_

#include "stm32f4xx_hal.h"
#include "st7735.h"
#include "ov7670.h"
#include "common.h"
#include "cifar10_nn.h"

void drawChar(uint16_t x, uint16_t y, char c, uint16_t color, uint16_t bg_color);
void drawString(uint16_t x, uint16_t y, const char* str, uint16_t color, uint16_t bg_color);
void drawTextBackground(uint16_t x, uint16_t y, uint16_t width, uint16_t height, uint16_t color);
void int_to_string(int num, char* str, int digits);

#endif /* INC_PREWORK_H_ */
