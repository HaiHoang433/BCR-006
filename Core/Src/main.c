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
#include "fatfs.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "common.h"
#include "ov7670.h"
#include "st7735.h"
#include "i2c_lcd.h"
#include "model_weights.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
CRC_HandleTypeDef hcrc;

DCMI_HandleTypeDef hdcmi;
DMA_HandleTypeDef hdma_dcmi;

I2C_HandleTypeDef hi2c1;
I2C_HandleTypeDef hi2c2;

SD_HandleTypeDef hsd;

SPI_HandleTypeDef hspi1;
SPI_HandleTypeDef hspi2;

/* USER CODE BEGIN PV */
I2C_LCD_HandleTypeDef lcd;

#define MAX_PICTURE_BUFF     19200
uint16_t pBuffer[MAX_PICTURE_BUFF];

// Declare the 3D array to store RGB components
uint8_t image_input[120][160][3];

// Model constants
#define IMAGE_HEIGHT 120
#define IMAGE_WIDTH 160
#define IMAGE_CHANNELS 3
#define NUM_CLASSES 10

// Layer parameters
#define CONV_FILTERS 6
#define KERNEL_SIZE 3
#define POOL_SIZE 4

// Helper macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Model weights
typedef struct {
    // SeparableConv2D weights
    float depthwise_kernel[KERNEL_SIZE][KERNEL_SIZE][IMAGE_CHANNELS];
    float pointwise_kernel[IMAGE_CHANNELS][CONV_FILTERS];
    float conv_bias[CONV_FILTERS];

    // BatchNormalization weights
    float bn_gamma[CONV_FILTERS];
    float bn_beta[CONV_FILTERS];
    float bn_moving_mean[CONV_FILTERS];
    float bn_moving_variance[CONV_FILTERS];

    // Dense layer weights
    float dense_weights[CONV_FILTERS][NUM_CLASSES];
    float dense_bias[NUM_CLASSES];
} ModelWeights;

// External declaration of model weights array
extern const float model_weights_array[145];

// CIFAR-10 class names
const char* class_names[NUM_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Global model weights instance - allocate once, use throughout program
ModelWeights model_weights;

uint32_t frame_count = 0;
uint32_t successful_predictions = 0;
uint32_t failed_predictions = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_DCMI_Init(void);
static void MX_I2C2_Init(void);
static void MX_SPI1_Init(void);
static void MX_I2C1_Init(void);
static void MX_CRC_Init(void);
static void MX_SDIO_SD_Init(void);
static void MX_SPI2_Init(void);
/* USER CODE BEGIN PFP */
// Memory-optimized neural network implementation
// Replace the existing neural network functions with these optimized versions

// Reuse buffers to minimize memory usage
// Define global buffers that can be reused across functions
static float buffer_A[120][160][6];  // ~460KB - main processing buffer
static float buffer_B[30][40][6];    // ~28KB - pooling buffer

int load_weights_from_array(ModelWeights* weights);
static inline float safe_get_pixel(const float input[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS], int y, int x, int c);
void rgb565_to_rgb888(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
                      float output[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS]);
void separable_conv2d(const float input[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
                     float output[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS],
                     const ModelWeights* weights);
void batch_normalization(float feature_maps[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS],
                        const ModelWeights* weights);
void relu_activation(float feature_maps[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS]);
void max_pooling(const float input[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS],
                float output[IMAGE_HEIGHT/POOL_SIZE][IMAGE_WIDTH/POOL_SIZE][CONV_FILTERS]);
void global_avg_pooling(const float input[IMAGE_HEIGHT/POOL_SIZE][IMAGE_WIDTH/POOL_SIZE][CONV_FILTERS],
                      float output[CONV_FILTERS]);
void dense_layer(const float input[CONV_FILTERS], float output[NUM_CLASSES],
                const ModelWeights* weights);
void softmax(float logits[NUM_CLASSES]);
int argmax(const float array[], int size);
int predict(const uint8_t image[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
           float probabilities[NUM_CLASSES],
           const ModelWeights* weights);

void LCD_PrintNumber(I2C_LCD_HandleTypeDef* lcd, int number);
void display_result_safe(I2C_LCD_HandleTypeDef* lcd, int predicted_class, float confidence, uint32_t elapsed_ms, uint8_t success);
int safe_predict(const uint8_t image[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
                float probabilities[NUM_CLASSES],
                const ModelWeights* weights,
                uint8_t* success_flag);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
// Helper function to safely access array bounds
static inline float safe_get_pixel(const float input[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
                                  int y, int x, int c) {
    if (y >= 0 && y < IMAGE_HEIGHT && x >= 0 && x < IMAGE_WIDTH && c >= 0 && c < IMAGE_CHANNELS) {
        return input[y][x][c];
    }
    return 0.0f;  // Return 0 for out-of-bounds (padding)
}

// Load weights from embedded array into the provided weights structure
int load_weights_from_array(ModelWeights* weights) {
    // Calculate offsets into the array for each set of weights
    int offset = 0;

    // Depthwise kernel: 3x3x3 = 27 weights
    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            for (int c = 0; c < IMAGE_CHANNELS; c++) {
                weights->depthwise_kernel[ky][kx][c] = model_weights_array[offset++];
            }
        }
    }

    // Pointwise kernel: 3x6 = 18 weights
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
        for (int f = 0; f < CONV_FILTERS; f++) {
            weights->pointwise_kernel[c][f] = model_weights_array[offset++];
        }
    }

    // Conv bias: 6 weights
    for (int f = 0; f < CONV_FILTERS; f++) {
        weights->conv_bias[f] = model_weights_array[offset++];
    }

    // BatchNorm gamma: 6 weights
    for (int f = 0; f < CONV_FILTERS; f++) {
        weights->bn_gamma[f] = model_weights_array[offset++];
    }

    // BatchNorm beta: 6 weights
    for (int f = 0; f < CONV_FILTERS; f++) {
        weights->bn_beta[f] = model_weights_array[offset++];
    }

    // BatchNorm moving mean: 6 weights
    for (int f = 0; f < CONV_FILTERS; f++) {
        weights->bn_moving_mean[f] = model_weights_array[offset++];
    }

    // BatchNorm moving variance: 6 weights
    for (int f = 0; f < CONV_FILTERS; f++) {
        weights->bn_moving_variance[f] = model_weights_array[offset++];
    }

    // Dense weights: 6x10 = 60 weights
    for (int i = 0; i < CONV_FILTERS; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            weights->dense_weights[i][j] = model_weights_array[offset++];
        }
    }

    // Dense bias: 10 weights
    for (int i = 0; i < NUM_CLASSES; i++) {
        weights->dense_bias[i] = model_weights_array[offset++];
    }

    // Verify that we've used exactly all the weights
    if (offset != 145) {
        return 0;
    }

    return 1;
}

// Fixed RGB565 to RGB888 conversion
void rgb565_to_rgb888(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
                      float output[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS]) {
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int c = 0; c < IMAGE_CHANNELS; c++) {
                // Normalize to [0,1] range
                output[y][x][c] = (float)input[y][x][c] / 255.0f;
            }
        }
    }
}

// Fixed separable convolution with proper memory management
void separable_conv2d(const float input[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
                     float output[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS],
                     const ModelWeights* weights) {

    // Clear output first
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int f = 0; f < CONV_FILTERS; f++) {
                output[y][x][f] = 0.0f;
            }
        }
    }

    // Allocate temporary buffer for depthwise convolution
    // Use static allocation to avoid stack overflow
    static float depthwise_output[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS];

    // Initialize depthwise output
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int c = 0; c < IMAGE_CHANNELS; c++) {
                depthwise_output[y][x][c] = 0.0f;
            }
        }
    }

    // 1. Depthwise convolution with bounds checking
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int c = 0; c < IMAGE_CHANNELS; c++) {
                float sum = 0.0f;

                // Apply 3x3 kernel with "same" padding
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        // Calculate input position with padding
                        int in_y = y - KERNEL_SIZE/2 + ky;
                        int in_x = x - KERNEL_SIZE/2 + kx;

                        // Use safe accessor function
                        float pixel_val = safe_get_pixel(input, in_y, in_x, c);
                        sum += pixel_val * weights->depthwise_kernel[ky][kx][c];
                    }
                }
                depthwise_output[y][x][c] = sum;
            }
        }
    }

    // 2. Pointwise convolution (1x1 convolution)
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int f = 0; f < CONV_FILTERS; f++) {
                float sum = 0.0f;

                // Apply 1x1 convolution across all input channels
                for (int c = 0; c < IMAGE_CHANNELS; c++) {
                    sum += depthwise_output[y][x][c] * weights->pointwise_kernel[c][f];
                }

                // Add bias
                output[y][x][f] = sum + weights->conv_bias[f];
            }
        }
    }
}

// Fixed batch normalization with numerical stability
void batch_normalization(float feature_maps[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS],
                        const ModelWeights* weights) {
    const float epsilon = 1e-5f;  // Small constant for numerical stability

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int f = 0; f < CONV_FILTERS; f++) {
                // Batch normalization: (x - mean) / sqrt(var + epsilon) * gamma + beta
                float normalized = (feature_maps[y][x][f] - weights->bn_moving_mean[f]) /
                                  sqrtf(weights->bn_moving_variance[f] + epsilon);

                feature_maps[y][x][f] = weights->bn_gamma[f] * normalized + weights->bn_beta[f];
            }
        }
    }
}

// Apply ReLU activation
void relu_activation(float feature_maps[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS]) {
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int f = 0; f < CONV_FILTERS; f++) {
                // ReLU: max(0, x)
                feature_maps[y][x][f] = MAX(0.0f, feature_maps[y][x][f]);
            }
        }
    }
}

// Fixed max pooling with proper bounds checking
void max_pooling(const float input[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS],
                float output[IMAGE_HEIGHT/POOL_SIZE][IMAGE_WIDTH/POOL_SIZE][CONV_FILTERS]) {

    int out_height = IMAGE_HEIGHT / POOL_SIZE;  // Should be 30 (120/4)
    int out_width = IMAGE_WIDTH / POOL_SIZE;    // Should be 40 (160/4)

    for (int y = 0; y < out_height; y++) {
        for (int x = 0; x < out_width; x++) {
            for (int f = 0; f < CONV_FILTERS; f++) {
                float max_val = -INFINITY;

                // Find maximum value in the pool window
                for (int py = 0; py < POOL_SIZE; py++) {
                    for (int px = 0; px < POOL_SIZE; px++) {
                        int in_y = y * POOL_SIZE + py;
                        int in_x = x * POOL_SIZE + px;

                        // Bounds checking
                        if (in_y < IMAGE_HEIGHT && in_x < IMAGE_WIDTH) {
                            if (input[in_y][in_x][f] > max_val) {
                                max_val = input[in_y][in_x][f];
                            }
                        }
                    }
                }

                output[y][x][f] = (max_val == -INFINITY) ? 0.0f : max_val;
            }
        }
    }
}

// Fixed global average pooling
void global_avg_pooling(const float input[IMAGE_HEIGHT/POOL_SIZE][IMAGE_WIDTH/POOL_SIZE][CONV_FILTERS],
                      float output[CONV_FILTERS]) {

    int pool_height = IMAGE_HEIGHT / POOL_SIZE;  // 30
    int pool_width = IMAGE_WIDTH / POOL_SIZE;    // 40
    int total_pixels = pool_height * pool_width; // 1200

    // Initialize output
    for (int f = 0; f < CONV_FILTERS; f++) {
        output[f] = 0.0f;
    }

    // Calculate average for each filter
    for (int f = 0; f < CONV_FILTERS; f++) {
        float sum = 0.0f;

        for (int y = 0; y < pool_height; y++) {
            for (int x = 0; x < pool_width; x++) {
                sum += input[y][x][f];
            }
        }

        output[f] = sum / (float)total_pixels;
    }
}

// Apply dense layer
void dense_layer(const float input[CONV_FILTERS], float output[NUM_CLASSES],
                const ModelWeights* weights) {
    // For each output class
    for (int c = 0; c < NUM_CLASSES; c++) {
        float sum = 0.0f;

        // Dot product of input and weights
        for (int i = 0; i < CONV_FILTERS; i++) {
            sum += input[i] * weights->dense_weights[i][c];
        }

        // Add bias
        output[c] = sum + weights->dense_bias[c];
    }
}

// Fixed softmax with numerical stability
void softmax(float logits[NUM_CLASSES]) {
    // Find maximum for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // Calculate exp(logits - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    // Normalize to get probabilities
    if (sum > 0.0f) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            logits[i] /= sum;
        }
    } else {
        // Fallback: uniform distribution
        for (int i = 0; i < NUM_CLASSES; i++) {
            logits[i] = 1.0f / NUM_CLASSES;
        }
    }
}

// Find index of maximum value
int argmax(const float array[], int size) {
    int max_idx = 0;
    float max_val = array[0];

    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
            max_idx = i;
        }
    }

    return max_idx;
}

// Fixed prediction function with better memory management
int predict(const uint8_t image[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
           float probabilities[NUM_CLASSES],
           const ModelWeights* weights) {

    // Use static allocation to prevent stack overflow
    static float normalized_input[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS];
    static float conv_output[IMAGE_HEIGHT][IMAGE_WIDTH][CONV_FILTERS];
    static float pooled_output[IMAGE_HEIGHT/POOL_SIZE][IMAGE_WIDTH/POOL_SIZE][CONV_FILTERS];
    static float gap_output[CONV_FILTERS];
    static float logits[NUM_CLASSES];

    // Step 1: Convert and normalize input
    rgb565_to_rgb888(image, normalized_input);

    // Step 2: Apply separable convolution
    separable_conv2d(normalized_input, conv_output, weights);

    // Step 3: Apply batch normalization
    batch_normalization(conv_output, weights);

    // Step 4: Apply ReLU activation
    relu_activation(conv_output);

    // Step 5: Apply max pooling
    max_pooling(conv_output, pooled_output);

    // Step 6: Apply global average pooling
    global_avg_pooling(pooled_output, gap_output);

    // Step 7: Apply dense layer
    dense_layer(gap_output, logits, weights);

    // Step 8: Apply softmax
    softmax(logits);

    // Copy probabilities to output
    if (probabilities != NULL) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            probabilities[i] = logits[i];
        }
    }

    // Step 9: Find predicted class
    return argmax(logits, NUM_CLASSES);
}

// Function to display a number directly on LCD
void LCD_PrintNumber(I2C_LCD_HandleTypeDef* lcd, int number) {
    // Handle 0 special case
    if (number == 0) {
        lcd_puts(lcd, "0");
        return;
    }

    // Handle negative numbers
    if (number < 0) {
        lcd_puts(lcd, "-");
        number = -number;
    }

    // Convert number to digits
    char digits[10] = {0}; // Max 10 digits for 32-bit int
    int i = 0;

    while (number > 0) {
        digits[i++] = '0' + (number % 10);
        number /= 10;
    }

    // Print digits in reverse order
    for (int j = i - 1; j >= 0; j--) {
        char digit_str[2] = {digits[j], '\0'};
        lcd_puts(lcd, digit_str);
    }
}

// Enhanced display function with error handling
void display_result_safe(I2C_LCD_HandleTypeDef* lcd, int predicted_class,
                        float confidence, uint32_t elapsed_ms, uint8_t success) {
    // Clear display
    lcd_clear(lcd);

    if (success) {
        // Line 1: Show class name
        lcd_gotoxy(lcd, 0, 0);
        if (predicted_class >= 0 && predicted_class < NUM_CLASSES) {
            lcd_puts(lcd, (char*)class_names[predicted_class]);
        } else {
            lcd_puts(lcd, "ERROR");
        }

        // Line 2: Show confidence and time
        lcd_gotoxy(lcd, 0, 1);
        int conf_percent = (int)(confidence * 100.0f);
        if (conf_percent >= 0 && conf_percent <= 100) {
            LCD_PrintNumber(lcd, conf_percent);
            lcd_puts(lcd, "% ");
        }
        LCD_PrintNumber(lcd, (int)elapsed_ms);
        lcd_puts(lcd, "ms");
    } else {
        // Show error message
        lcd_gotoxy(lcd, 0, 0);
        lcd_puts(lcd, "NN Error");
        lcd_gotoxy(lcd, 0, 1);
        lcd_puts(lcd, "Frame: ");
        LCD_PrintNumber(lcd, (int)frame_count);
    }
}

// Safe prediction wrapper
int safe_predict(const uint8_t image[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],
                float probabilities[NUM_CLASSES],
                const ModelWeights* weights,
                uint8_t* success_flag) {

    *success_flag = 0;  // Assume failure

    // Basic input validation
    if (image == NULL || weights == NULL) {
        return -1;
    }

    // Initialize probabilities to safe values
    if (probabilities != NULL) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            probabilities[i] = 0.0f;
        }
    }

    // Try to run prediction
    int result = predict(image, probabilities, weights);

    // Validate result
    if (result >= 0 && result < NUM_CLASSES) {
        // Check if probabilities are reasonable
        if (probabilities != NULL) {
            float sum = 0.0f;
            for (int i = 0; i < NUM_CLASSES; i++) {
                if (probabilities[i] < 0.0f || probabilities[i] > 1.0f) {
                    return -1;  // Invalid probability
                }
                sum += probabilities[i];
            }
            // Sum should be close to 1.0 (allow some floating point error)
            if (sum < 0.9f || sum > 1.1f) {
                return -1;  // Invalid probability distribution
            }
        }
        *success_flag = 1;  // Success
        return result;
    }

    return -1;  // Invalid result
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
  MX_I2C1_Init();
  MX_CRC_Init();
  MX_SDIO_SD_Init();
  MX_FATFS_Init();
  MX_SPI2_Init();
  /* USER CODE BEGIN 2 */
  // Initialize LCD
  lcd.hi2c = &hi2c1;
  lcd.address = 0x27;
  lcd_init(&lcd);

  lcd_clear(&lcd);
  lcd_gotoxy(&lcd, 0, 0);
  lcd_puts(&lcd, "CIFAR-10");
  lcd_gotoxy(&lcd, 0, 1);
  lcd_puts(&lcd, "Classifier");
  HAL_Delay(5000);

  // Load weights into the global structure (no malloc needed)
  lcd_clear(&lcd);
  lcd_puts(&lcd, "Loading model...");
  if (!load_weights_from_array(&model_weights)) {
      lcd_clear(&lcd);
      lcd_gotoxy(&lcd, 0, 0);
      lcd_puts(&lcd, "Model load failed");
      while(1); // Hang on error
  }

  ST7735_Init();
  ST7735_FillScreen(ST7735_BLACK);
  ST7735_FillRectangle(0, 0, 40, 140, ST7735_BLUE); // Test the display
  HAL_Delay(5000);
  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, GPIO_PIN_RESET); //Camera PWDN to GND
  ov7670_init(&hdcmi, &hdma_dcmi, &hi2c2);
  ov7670_config(OV7670_MODE_QVGA_RGB565);
  ov7670_stopCap();

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	    frame_count++;

	    // Show capture status
	    lcd_clear(&lcd);
	    lcd_gotoxy(&lcd, 0, 0);
	    lcd_puts(&lcd, "Capturing...");
	    lcd_gotoxy(&lcd, 0, 1);
	    lcd_puts(&lcd, "Frame: ");
	    LCD_PrintNumber(&lcd, (int)frame_count);

	    // Start capture with timeout
	    HAL_DCMI_Start_DMA(&hdcmi, DCMI_MODE_SNAPSHOT, (uint32_t)pBuffer, MAX_PICTURE_BUFF/2);

	    // Wait for capture to complete
	    HAL_Delay(300);  // Give enough time for capture

	    // Stop capture
	    HAL_DCMI_Stop(&hdcmi);
	    HAL_Delay(50);   // Let data settle

	    // Validate captured data
	    uint8_t valid_data = 0;
	    for (int i = 0; i < 100; i++) {  // Check first 100 pixels
	        if (pBuffer[i] != 0) {
	            valid_data = 1;
	            break;
	        }
	    }

	    if (!valid_data) {
	        lcd_clear(&lcd);
	        lcd_gotoxy(&lcd, 0, 0);
	        lcd_puts(&lcd, "No camera data");
	        lcd_gotoxy(&lcd, 0, 1);
	        lcd_puts(&lcd, "Check camera");
	        HAL_Delay(2000);
	        continue;
	    }

	    // Convert and display image
	    int pixel = 0;
	    for (int x = 0; x < 120; x++) {
	        for (int y = 159; y >= 0; y--) {
	            uint16_t color = pBuffer[pixel];

	            // Extract RGB components
	            uint8_t red = (color >> 11) & 0x1F;
	            uint8_t green = (color >> 5) & 0x3F;
	            uint8_t blue = color & 0x1F;

	            // Convert to 8-bit and store
	            image_input[x][160 - y][0] = (red << 3) | (red >> 2);
	            image_input[x][160 - y][1] = (green << 2) | (green >> 4);
	            image_input[x][160 - y][2] = (blue << 3) | (blue >> 2);

	            // Display on TFT
	            ST7735_DrawPixel(x, y, color);
	            pixel++;
	        }
	    }

	    // Show processing status
	    lcd_clear(&lcd);
	    lcd_gotoxy(&lcd, 0, 0);
	    lcd_puts(&lcd, "Processing...");
	    lcd_gotoxy(&lcd, 0, 1);
	    lcd_puts(&lcd, "Please wait");

	    // Run neural network with safety checks
	    uint32_t start_time = HAL_GetTick();
	    float probabilities[NUM_CLASSES];
	    uint8_t success_flag = 0;

	    int predicted_class = safe_predict(image_input, probabilities, &model_weights, &success_flag);

	    uint32_t elapsed_ms = HAL_GetTick() - start_time;

	    // Update statistics
	    if (success_flag) {
	        successful_predictions++;
	    } else {
	        failed_predictions++;
	    }

	    // Display results
	    float confidence = success_flag ? probabilities[predicted_class] : 0.0f;
	    display_result_safe(&lcd, predicted_class, confidence, elapsed_ms, success_flag);

	    // Wait before next frame
	    HAL_Delay(3000);  // 3 seconds to read result
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
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

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
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

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
  * @brief SDIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_SDIO_SD_Init(void)
{

  /* USER CODE BEGIN SDIO_Init 0 */

  /* USER CODE END SDIO_Init 0 */

  /* USER CODE BEGIN SDIO_Init 1 */

  /* USER CODE END SDIO_Init 1 */
  hsd.Instance = SDIO;
  hsd.Init.ClockEdge = SDIO_CLOCK_EDGE_RISING;
  hsd.Init.ClockBypass = SDIO_CLOCK_BYPASS_DISABLE;
  hsd.Init.ClockPowerSave = SDIO_CLOCK_POWER_SAVE_DISABLE;
  hsd.Init.BusWide = SDIO_BUS_WIDE_1B;
  hsd.Init.HardwareFlowControl = SDIO_HARDWARE_FLOW_CONTROL_DISABLE;
  hsd.Init.ClockDiv = 0;
  /* USER CODE BEGIN SDIO_Init 2 */

  /* USER CODE END SDIO_Init 2 */

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
  * @brief SPI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI2_Init(void)
{

  /* USER CODE BEGIN SPI2_Init 0 */

  /* USER CODE END SPI2_Init 0 */

  /* USER CODE BEGIN SPI2_Init 1 */

  /* USER CODE END SPI2_Init 1 */
  /* SPI2 parameter configuration*/
  hspi2.Instance = SPI2;
  hspi2.Init.Mode = SPI_MODE_MASTER;
  hspi2.Init.Direction = SPI_DIRECTION_2LINES;
  hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi2.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi2.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi2.Init.NSS = SPI_NSS_SOFT;
  hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
  hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi2.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI2_Init 2 */

  /* USER CODE END SPI2_Init 2 */

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
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
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
