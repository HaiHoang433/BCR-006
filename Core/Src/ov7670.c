/*
 * ov7670.c
 *
 *  Created on: May 26, 2025
 *      Author: HoangHai
 */

#include <stdio.h>
#include "main.h"
#include "stm32f4xx_hal.h"
#include "common.h"
#include "ov7670.h"

/*** Internal Const Values, Macros ***/
#define OV7670_QVGA_WIDTH  320
#define OV7670_QVGA_HEIGHT 240

// Define the actual array in the source file
const uint8_t OV7670_reg[][2] = {
/* Color mode related */
{0x12, 0x04}, // QVGA, RGB
{0x8C, 0x00}, // RGB444 Disable
{0x40, 0x10 + 0xc0}, // RGB565, 00 - FF
{0x3A, 0x04 + 8}, // UYVY (why?)
{0x3D, 0x80 + 0x00}, // gamma enable, UV auto adjust, UYVY
{0xB0, 0x84}, // important

/* clock related */
{0x0C, 0x04}, // DCW enable
{0x3E, 0x1A}, // manual scaling, pclk/=2
{0x70, 0x3A}, // scaling_xsc
{0x71, 0x35}, // scaling_ysc
{0x72, 0x22}, // down sample by 2
{0x73, 0xf2}, // DSP clock /= 2

/* windowing (empirically decided...) */
{0x17, 0x16}, // HSTART
{0x18, 0x04}, // HSTOP
{0x32, 0xA4}, // HREF
{0x19, 0x02}, // VSTART = 14 ( = 3 * 4 + 2)
{0x1a, 0x7A}, // VSTOP = 494 ( = 123 * 4 + 2)
{0x03, 0xA4}, // VREF (VSTART_LOW = 2, VSTOP_LOW = 2)

/* color matrix coefficient */
#if 0
{0x4f, 0xb3},
{0x50, 0xb3},
{0x51, 0x00},
{0x52, 0x3d},
{0x53, 0xa7},
{0x54, 0xe4},
{0x58, 0x9e},
#else
{0x4f, 0x80},
{0x50, 0x80},
{0x51, 0x00},
{0x52, 0x22},
{0x53, 0x5e},
{0x54, 0x80},
{0x58, 0x9e},
#endif

/* 3a */
// {0x13, 0x84},
// {0x14, 0x0a}, // AGC Ceiling = 2x
// {0x5F, 0x2f}, // AWB B Gain Range (empirically decided)
// // without this bright scene becomes yellow (purple). might be because of color matrix
// {0x60, 0x98}, // AWB R Gain Range (empirically decided)
// {0x61, 0x70}, // AWB G Gain Range (empirically decided)
{0x41, 0x38}, // edge enhancement, de-noise, AWG gain enabled

/* gamma curve */
#if 1
{0x7b, 16},
{0x7c, 30},
{0x7d, 53},
{0x7e, 90},
{0x7f, 105},
{0x80, 118},
{0x81, 130},
{0x82, 140},
{0x83, 150},
{0x84, 160},
{0x85, 180},
{0x86, 195},
{0x87, 215},
{0x88, 230},
{0x89, 244},
{0x7a, 16},
#else
/* gamma = 1 */
{0x7b, 4},
{0x7c, 8},
{0x7d, 16},
{0x7e, 32},
{0x7f, 40},
{0x80, 48},
{0x81, 56},
{0x82, 64},
{0x83, 72},
{0x84, 80},
{0x85, 96},
{0x86, 112},
{0x87, 144},
{0x88, 176},
{0x89, 208},
{0x7a, 64},
#endif

/* fps */
// {0x6B, 0x4a}, //PLL x4
{0x11, 0x00}, // pre-scalar = 1/1

/* others */
{0x1E, 0x31}, //mirror flip
// {0x42, 0x08}, // color bar k

{REG_BATT, REG_BATT},
};

/*** Internal Static Variables ***/
static DCMI_HandleTypeDef *sp_hdcmi;
static DMA_HandleTypeDef  *sp_hdma_dcmi;
static I2C_HandleTypeDef  *sp_hi2c;
static uint32_t    s_destAddressForContiuousMode;
static void (* s_cbHsync)(uint32_t h);
static void (* s_cbVsync)(uint32_t v);
static uint32_t s_currentH;
static uint32_t s_currentV;

/*** Internal Function Declarations ***/
static RET ov7670_write(uint8_t regAddr, uint8_t data);
static RET ov7670_read(uint8_t regAddr, uint8_t *data);

/*** External Function Defines ***/
RET ov7670_init(DCMI_HandleTypeDef *p_hdcmi, DMA_HandleTypeDef *p_hdma_dcmi, I2C_HandleTypeDef *p_hi2c)
{
  sp_hdcmi     = p_hdcmi;
  sp_hdma_dcmi = p_hdma_dcmi;
  sp_hi2c      = p_hi2c;
  s_destAddressForContiuousMode = 0;

  HAL_GPIO_WritePin(CAMERA_RESET_GPIO_Port, CAMERA_RESET_Pin, GPIO_PIN_RESET);
  HAL_Delay(100);
  HAL_GPIO_WritePin(CAMERA_RESET_GPIO_Port, CAMERA_RESET_Pin, GPIO_PIN_SET);
  HAL_Delay(100);

  ov7670_write(0x12, 0x80);  // RESET
  HAL_Delay(30);

  uint8_t buffer[4];
  ov7670_read(0x0b, buffer);
  printf("[OV7670] dev id = %02X\n", buffer[0]);


  return RET_OK;
}

RET ov7670_config(uint32_t mode)
{
  ov7670_stopCap();
  ov7670_write(0x12, 0x80);  // RESET
  HAL_Delay(30);
  for(int i = 0; OV7670_reg[i][0] != REG_BATT; i++) {
    ov7670_write(OV7670_reg[i][0], OV7670_reg[i][1]);
    HAL_Delay(1);
  }
  return RET_OK;
}

RET ov7670_startCap(uint32_t capMode, uint32_t destAddress)
{
  ov7670_stopCap();
  if (capMode == OV7670_CAP_CONTINUOUS) {
    /* note: continuous mode automatically invokes DCMI, but DMA needs to be invoked manually */
    s_destAddressForContiuousMode = destAddress;
    HAL_DCMI_Start_DMA(sp_hdcmi, DCMI_MODE_CONTINUOUS, destAddress, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/2);
  } else if (capMode == OV7670_CAP_SINGLE_FRAME) {
    s_destAddressForContiuousMode = 0;
    HAL_DCMI_Start_DMA(sp_hdcmi, DCMI_MODE_SNAPSHOT, destAddress, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/2);
  }

  return RET_OK;
}

RET ov7670_stopCap()
{
  HAL_DCMI_Stop(sp_hdcmi);
//  HAL_Delay(30);
  return RET_OK;
}

void ov7670_registerCallback(void (*cbHsync)(uint32_t h), void (*cbVsync)(uint32_t v))
{
  s_cbHsync = cbHsync;
  s_cbVsync = cbVsync;
}

void HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi)
{
//  printf("FRAME %d\n", HAL_GetTick());
  if(s_cbVsync)s_cbVsync(s_currentV);
  if(s_destAddressForContiuousMode != 0) {
    HAL_DMA_Start_IT(hdcmi->DMA_Handle, (uint32_t)&hdcmi->Instance->DR, s_destAddressForContiuousMode, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/2);
  }
  s_currentV++;
  s_currentH = 0;
}

void HAL_DCMI_VsyncEventCallback(DCMI_HandleTypeDef *hdcmi)
{
//  printf("VSYNC %d\n", HAL_GetTick());
//  HAL_DMA_Start_IT(hdcmi->DMA_Handle, (uint32_t)&hdcmi->Instance->DR, s_destAddressForContiuousMode, OV7670_QVGA_WIDTH * OV7670_QVGA_HEIGHT/2);
}

//void HAL_DCMI_LineEventCallback(DCMI_HandleTypeDef *hdcmi)
//{
////  printf("HSYNC %d\n", HAL_GetTick());
//  if(s_cbHsync)s_cbHsync(s_currentH);
//  s_currentH++;
//}

/*** Internal Function Defines ***/
static RET ov7670_write(uint8_t regAddr, uint8_t data)
{
  HAL_StatusTypeDef ret;
  do {
    ret = HAL_I2C_Mem_Write(sp_hi2c, SLAVE_ADDR, regAddr, I2C_MEMADD_SIZE_8BIT, &data, 1, 100);
  } while (ret != HAL_OK && 0);
  return ret;
}

static RET ov7670_read(uint8_t regAddr, uint8_t *data)
{
  HAL_StatusTypeDef ret;
  do {
    // HAL_I2C_Mem_Read doesn't work (because of SCCB protocol(doesn't have ack))? */
//    ret = HAL_I2C_Mem_Read(sp_hi2c, SLAVE_ADDR, regAddr, I2C_MEMADD_SIZE_8BIT, data, 1, 1000);
    ret = HAL_I2C_Master_Transmit(sp_hi2c, SLAVE_ADDR, &regAddr, 1, 100);
    ret |= HAL_I2C_Master_Receive(sp_hi2c, SLAVE_ADDR, data, 1, 100);
  } while (ret != HAL_OK && 0);
  return ret;
}
