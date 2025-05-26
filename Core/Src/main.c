/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "prework.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
DCMI_HandleTypeDef hdcmi;
DMA_HandleTypeDef hdma_dcmi;

I2C_HandleTypeDef hi2c2;

SPI_HandleTypeDef hspi1;

/* USER CODE BEGIN PV */
#define MAX_PICTURE_BUFF     19200
uint16_t pBuffer[MAX_PICTURE_BUFF];

// Use a union to share memory between imageTFT and inputNN
typedef union {
    uint8_t imageTFT[120][160][3];   // RGB888 buffer for image processing
    uint8_t inputNN[32][32][3];      // Neural network input buffer (much smaller)
    uint8_t dummy[57600];            // Ensure correct size allocation
} ImageBuffer;

ImageBuffer imageBuffer;  // Shared memory for image processing

// Results display buffer
char resultText[32];  // Buffer for result text
uint32_t inferenceStartTime;  // For measuring inference time
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_DCMI_Init(void);
static void MX_I2C2_Init(void);
static void MX_SPI1_Init(void);
/* USER CODE BEGIN PFP */
void extractRGBFromBuffer(void);
void resizeImageForNN_FixedPoint(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
// Extract RGB values from pBuffer into imageBuffer.imageTFT
void extractRGBFromBuffer(void)
{
  uint16_t pixel;
  uint8_t r, g, b;
  int buffer_index = 0;

  for(int x = 0; x < 120; x++) // height
  {
    for(int y = 159; y >= 0; y--) // width (inverted as in original code)
    {
      pixel = pBuffer[buffer_index++];

      // Extract RGB components from RGB565 format
      // RGB565: RRRRRGGGGGGBBBBB
      r = (pixel >> 11) & 0x1F;  // Extract 5 bits for red
      g = (pixel >> 5) & 0x3F;   // Extract 6 bits for green
      b = pixel & 0x1F;          // Extract 5 bits for blue

      // Convert to 8-bit per channel (scale to 0-255)
      imageBuffer.imageTFT[x][y][0] = (r * 255) / 31;  // R (5 bits -> 8 bits)
      imageBuffer.imageTFT[x][y][1] = (g * 255) / 63;  // G (6 bits -> 8 bits)
      imageBuffer.imageTFT[x][y][2] = (b * 255) / 31;  // B (5 bits -> 8 bits)
    }
  }
}

// Modified to directly create inputNN in the shared buffer
void resizeImageForNN_FixedPoint(void)
{
  // Use fixed point with 8 bits of fractional precision
  const int PRECISION = 8;
  const int SCALE_FACTOR = 1 << PRECISION; // 256

  // Calculate scaling ratios in fixed point
  const int scale_x = (120 << PRECISION) / 32; // 120 * 256 / 32
  const int scale_y = (160 << PRECISION) / 32; // 160 * 256 / 32

  // Temporary buffer for the resized image
  // We need this because we're overwriting the source as we process
  uint8_t tempNN[32][32][3];

  for(int i = 0; i < 32; i++)
  {
    int src_x_fixed = i * scale_x;
    int src_x = src_x_fixed >> PRECISION; // Integer part
    int x1 = src_x;
    int x2 = (x1 + 1 < 120) ? (x1 + 1) : 119;
    int dx = src_x_fixed & (SCALE_FACTOR - 1); // Fractional part

    for(int j = 0; j < 32; j++)
    {
      int src_y_fixed = j * scale_y;
      int src_y = src_y_fixed >> PRECISION; // Integer part
      int y1 = src_y;
      int y2 = (y1 + 1 < 160) ? (y1 + 1) : 159;
      int dy = src_y_fixed & (SCALE_FACTOR - 1); // Fractional part

      // Bilinear interpolation for each color channel
      for(int c = 0; c < 3; c++)
      {
        int top = (imageBuffer.imageTFT[x1][y1][c] * (SCALE_FACTOR - dx) +
                  imageBuffer.imageTFT[x2][y1][c] * dx) >> PRECISION;
        int bottom = (imageBuffer.imageTFT[x1][y2][c] * (SCALE_FACTOR - dx) +
                     imageBuffer.imageTFT[x2][y2][c] * dx) >> PRECISION;
        int pixel = (top * (SCALE_FACTOR - dy) + bottom * dy) >> PRECISION;

        // Store in temporary buffer
        tempNN[i][j][c] = (uint8_t)pixel;
      }
    }
  }

  // Copy from temporary buffer to inputNN
  for(int i = 0; i < 32; i++) {
    for(int j = 0; j < 32; j++) {
      for(int c = 0; c < 3; c++) {
        imageBuffer.inputNN[i][j][c] = tempNN[i][j][c];
      }
    }
  }
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_DCMI_Init();
  MX_I2C2_Init();
  MX_SPI1_Init();
  /* USER CODE BEGIN 2 */
  ST7735_Init();
  ST7735_FillScreen(ST7735_BLACK);

  // Initialize I2C LCD
  // Display initialization message
  ST7735_FillRectangle(0, 0, 128, 20, ST7735_BLUE);
  drawString(10, 5, "CIFAR-10 Camera", ST7735_WHITE, ST7735_BLUE);
  drawString(10, 30, "Initializing...", ST7735_WHITE, ST7735_BLACK);

  // Test the display
  HAL_Delay(1000);
  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, GPIO_PIN_RESET); //Camera PWDN to GND
  ov7670_init(&hdcmi, &hdma_dcmi, &hi2c2);
  ov7670_config(OV7670_MODE_QVGA_RGB565);
  ov7670_stopCap();

  ST7735_FillScreen(ST7735_BLACK);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	    // Show "Processing..." indicator
	    drawTextBackground(0, 0, 128, 20, ST7735_BLUE);
	    drawString(5, 5, "Processing...", ST7735_WHITE, ST7735_BLUE);

	    // Start measuring inference time
	    inferenceStartTime = HAL_GetTick();

	  	  	HAL_DCMI_Start_DMA(&hdcmi, DCMI_MODE_SNAPSHOT, (uint32_t)pBuffer, MAX_PICTURE_BUFF/2);
	  	    HAL_Delay(100);  //Wait for DMA to complete
	  	    // picture pBuffer size 120*160=19200 is now available, we can transmit
	  	    // or display in lcd as shown below

	  	    // Extract RGB values from pBuffer
	  	    extractRGBFromBuffer();

	  	    int pixel = 0;
	  	    for( int x = 0; x < 120; x++ )
	  	    {
	  	      for( int y = 159; y > -1; y-- )
	  	      {
	  	        ST7735_DrawPixel(x, y, pBuffer[pixel]);
	  	        pixel++;
	  	      }
	  	    }

	  	    // Resize to 32x32 for neural network input
	  	    resizeImageForNN_FixedPoint();

	  	    // Run neural network inference - note we're using imageBuffer.inputNN now
	  	    float confidence;
	  	    int predicted_class = cifar10_classify(imageBuffer.inputNN, &confidence);

	  	    // Calculate inference time
	  	    uint32_t inferenceTime = HAL_GetTick() - inferenceStartTime;

	  	    // Create overlay for results display
	  	    drawTextBackground(0, 0, 128, 40, ST7735_BLUE);

	  	    // Display class name
	  	    drawString(5, 5, cifar10_class_names[predicted_class], ST7735_WHITE, ST7735_BLUE);

	  	    // Display confidence
	  	    char conf_str[10];
	  	    int confidence_percent = (int)(confidence * 100.0f);
	  	    int_to_string(confidence_percent, conf_str, 2);
	  	    drawString(80, 5, conf_str, ST7735_WHITE, ST7735_BLUE);
	  	    drawString(95, 5, "%", ST7735_WHITE, ST7735_BLUE);

	  	    // Display inference time
	  	    char time_str[10];
	  	    int_to_string(inferenceTime, time_str, 4);
	  	    drawString(5, 20, "Time:", ST7735_WHITE, ST7735_BLUE);
	  	    drawString(60, 20, time_str, ST7735_WHITE, ST7735_BLUE);
	  	    drawString(95, 20, "ms", ST7735_WHITE, ST7735_BLUE);

	  	    HAL_Delay(500); // Small delay between captures
    /* USER CODE END WHILE */
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
  HAL_RCC_MCOConfig(RCC_MCO1, RCC_MCO1SOURCE_HSE, RCC_MCODIV_1);
}

/**
  * @brief DCMI Initialization Function
  * @param None
  * @retval None
  */
static void MX_DCMI_Init(void)
{

  /* USER CODE BEGIN DCMI_Init 0 */

  /* USER CODE END DCMI_Init 0 */

  /* USER CODE BEGIN DCMI_Init 1 */

  /* USER CODE END DCMI_Init 1 */
  hdcmi.Instance = DCMI;
  hdcmi.Init.SynchroMode = DCMI_SYNCHRO_HARDWARE;
  hdcmi.Init.PCKPolarity = DCMI_PCKPOLARITY_RISING;
  hdcmi.Init.VSPolarity = DCMI_VSPOLARITY_HIGH;
  hdcmi.Init.HSPolarity = DCMI_HSPOLARITY_LOW;
  hdcmi.Init.CaptureRate = DCMI_CR_ALL_FRAME;
  hdcmi.Init.ExtendedDataMode = DCMI_EXTEND_DATA_8B;
  hdcmi.Init.JPEGMode = DCMI_JPEG_DISABLE;
  if (HAL_DCMI_Init(&hdcmi) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DCMI_Init 2 */

  /* USER CODE END DCMI_Init 2 */

}

/**
  * @brief I2C2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C2_Init(void)
{

  /* USER CODE BEGIN I2C2_Init 0 */

  /* USER CODE END I2C2_Init 0 */

  /* USER CODE BEGIN I2C2_Init 1 */

  /* USER CODE END I2C2_Init 1 */
  hi2c2.Instance = I2C2;
  hi2c2.Init.ClockSpeed = 100000;
  hi2c2.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c2.Init.OwnAddress1 = 0;
  hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c2.Init.OwnAddress2 = 0;
  hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C2_Init 2 */

  /* USER CODE END I2C2_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_1LINE;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_4|GPIO_PIN_5, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, CAMERA_RESET_Pin|GPIO_PIN_12, GPIO_PIN_RESET);

  /*Configure GPIO pins : PC4 PC5 */
  GPIO_InitStruct.Pin = GPIO_PIN_4|GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : PB0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : CAMERA_RESET_Pin PD12 */
  GPIO_InitStruct.Pin = CAMERA_RESET_Pin|GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : PA8 */
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF0_MCO;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
