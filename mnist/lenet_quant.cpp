
// This code demos inference on the MNIST dataset with a LeNet CNN
// Provided network parameters have been training already and should give an accuracy of ~98.39%

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <bitset>

// Max number of test cases in LeNet is 10K
// You can reduce this for testing/debuggin
// Final report must use all 10000 test cases
#define NUM_TESTS 100

#include <ap_fixed.h>

#define QUANTIZE(num, a, b) (float(ap_fixed<(a), (b), AP_TRN_ZERO, AP_SAT>((num))))

using namespace std;

#define BIT_WIDTH 4

typedef ap_fixed<16, BIT_WIDTH, AP_TRN_ZERO, AP_SAT> fixed_act; 
typedef ap_fixed<16, BIT_WIDTH, AP_TRN_ZERO, AP_SAT> fixed_wt; 

//Static allocation of test images
unsigned char images[NUM_TESTS*28*28];
unsigned char labels[NUM_TESTS];
// *****************************************

//Static allocation of network parameters and their outputs
float image[1][32][32] = {0};
float conv1_weights[6][1][5][5] = {0};
float conv1_bias[6] = {0};
float conv1_output[6][28][28] = {0};

float pool2_output[6][14][14] = {0};

float conv3_weights[16][6][5][5] = {0};
float conv3_bias[16] = {0};
float conv3_output[16][10][10] = {0};

float pool4_output[16][5][5] = {0};

float conv5_weights[120][16][5][5] = {0};
float conv5_bias[120] = {0};
float conv5_output[120][1][1] = {0};

float fc6_weights[10][120][1][1] = {0};
float fc6_bias[10] = {0};
float fc6_output[10] = {0};
// *****************************************
// End declaration of layers parameters and buffers
// *****************************************


inline
float adc_quantize(float num)
{
    return num; 
/*
    fixed_act tmp = fixed_act(num);
    tmp[3] = 0;  
    tmp[7] = 0;  
    tmp[11] = 0;  
    tmp[15] = 0; 
    return float(tmp); 
*/
}

/*inline
float adc_quantize(float num)
{
    fixed_act tmp = fixed_act(num);
    tmp[1] = 0;  
    tmp[2] = 0;  
    tmp[3] = 0;  
    tmp[5] = 0;  
    tmp[6] = 0;  
    tmp[7] = 0;  
    tmp[9] = 0;  
    tmp[10] = 0;  
    tmp[11] = 0;  
    tmp[13] = 0; 
    tmp[14] = 0; 
    tmp[15] = 0; 
    return float(tmp); 
}*/

inline
float quantize_act(float num)
{
    return float(fixed_act(num));
}

inline
float quantize_wt(float num)
{
    return float(fixed_wt(num));
}


float quantize(float num)
{
    int original = *(reinterpret_cast<int*>(&num)); 
//    std::cout << std::bitset<32>(original) << endl;

    int mask = ~((1 << 12) - 1); 
//    std::cout << std::bitset<32>(mask) << endl;

    int tmp = original & mask; 

    float tmp2 = *(reinterpret_cast<float*>(&(tmp))); 
    return tmp2; 
}

// Start function definitions of different layers
inline float relu(float input)
{
    return (input > 0)? input:0;
}

// Convolution Layer 1
void convolution1(float input[1][32][32], float weights[6][1][5][5], float bias[6], float output[6][28][28])
{
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            input[0][i][j] = QUANTIZE(input[0][i][j], CONV1_ACT_FULL, CONV1_ACT_INT);

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 5; j++)
            for (int k = 0; k < 5; k++)
                weights[i][0][j][k] = QUANTIZE(weights[i][0][j][k], CONV1_WGT_FULL, CONV1_WGT_INT);

    for(int co = 0; co < 6; co++)
        for(int h = 0; h < 28; h++)
            for(int w = 0; w < 28; w++)
            {
                float sum = 0;
                for(int i = h, m = 0; i < (h + 5); i++, m++)
                {
                    for(int j = w, n = 0; j < (w + 5); j++, n++)
                        sum += weights[co][0][m][n] * input[0][i][j];
                }
                output[co][h][w] = adc_quantize(sum + bias[co]);
            }
}

// Relu Layer 1
void relu1(float input[6][28][28], float output[6][28][28])
{
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 28; j++)
            for(int k = 0; k < 28; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Pooling Layer 2
void max_pooling2(float input[6][28][28],float output[6][14][14])
{
    for(int c = 0;c < 6; c++)
        for(int h = 0; h < 14; h++)
            for(int w = 0; w < 14; w++)
            {
                float max_value=-1000000000000.0;
                for(int i = h*2; i < h*2+2; i++)
                {
                    for(int j = w*2;j < w*2+2; j++)
                        max_value = (max_value > input[c][i][j]) ? max_value:input[c][i][j];
                }
                output[c][h][w] = max_value;

            }
}

// Relu Layer 2
void relu2(float input[6][14][14], float output[6][14][14])
{
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 14; j++)
            for(int k = 0; k < 14; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Convolution Layer 3
void convolution3(float input[6][14][14], float weights[16][6][5][5], float bias[16], float output[16][10][10])
{

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 14; j++)
            for (int k = 0; k < 14; k++)
                input[i][j][k] = QUANTIZE(input[i][j][k], CONV3_ACT_FULL, CONV3_ACT_INT);

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 5; k++)
                for (int m = 0; m < 5; m++)
                weights[i][j][k][m] = QUANTIZE(weights[i][j][k][m], CONV3_WGT_FULL, CONV3_WGT_INT);

    for(int co = 0; co < 16; co++)
        for(int h = 0; h < 10; h++)
            for(int w = 0; w < 10; w++)
            {
                    float sum = 0;
                    for(int i = h, m = 0; i < (h+5); i++, m++)
                    {
                        for(int j = w, n = 0; j < (w+5); j++, n++)
                            for (int ci = 0; ci < 6; ci++)
                                sum += weights[co][ci][m][n] * input[ci][i][j];
                    }
                    output[co][h][w] = adc_quantize(sum + bias[co]);
            }
}

// Relu Layer 3
void relu3(float input[16][10][10], float output[16][10][10])
{
    for(int i = 0; i < 16; i++)
        for(int j = 0; j < 10; j++)
            for(int k = 0; k < 10; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Pooling Layer 4
void max_pooling4(float input[16][10][10],float output[16][5][5])
{
    for(int c = 0;c < 16; c++)
        for(int h = 0; h < 5; h++)
            for(int w = 0; w < 5; w++)
            {
                float max_value=-1000000000000.0;
                for(int i = h*2; i < h*2+2; i++)
                {
                    for(int j = w*2;j < w*2+2; j++)
                        max_value = (max_value > input[c][i][j]) ? max_value:input[c][i][j];
                }
                output[c][h][w] = max_value;
            }
}

// Relu Layer 4
void relu4(float input[16][5][5], float output[16][5][5])
{
    for(int i = 0; i < 16; i++)
        for(int j = 0; j < 5; j++)
            for(int k = 0; k < 5; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Convolution Layer 5
void convolution5(float input[16][5][5], float weights[120][16][5][5], float bias[120], float output[120][1][1])
{

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 5; j++)
            for (int k = 0; k < 5; k++)
                input[i][j][k] = QUANTIZE(input[i][j][k], CONV5_ACT_FULL, CONV5_ACT_INT);

    for (int i = 0; i < 120; i++)
        for (int j = 0; j < 16; j++)
            for (int k = 0; k < 5; k++)
                for (int m = 0; m < 5; m++)
                weights[i][j][k][m] = QUANTIZE(weights[i][j][k][m], CONV5_WGT_FULL, CONV5_WGT_INT);

    for(int co = 0; co < 120; co++)
    {
        float sum = 0;
        for(int i = 0, m = 0; i < 5; i++, m++)
        {
            for(int j = 0, n = 0; j < 5; j++, n++)
            {
                for (int ci = 0; ci < 16; ci++)
                    sum += weights[co][ci][m][n] * input[ci][i][j];
            }
        }
        output[co][0][0] = adc_quantize(sum + bias[co]);
    }
}

// Relu Layer 5
void relu5(float input[120][1][1], float output[120][1][1])
{
    for(int i = 0; i < 120; i++)
        output[i][0][0] = relu(input[i][0][0]);
}

// Fully connected Layer 6
void fc6(float input[120][1][1], float weights[10][120][1][1], const float bias[10], float output[10])
{
    for (int i = 0; i < 120; i++)
        input[i][0][0] = QUANTIZE(input[i][0][0], FC6_ACT_FULL, FC6_ACT_INT);

    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 120; j++)
                weights[i][j][0][0] = QUANTIZE(weights[i][j][0][0], FC6_WGT_FULL, FC6_WGT_INT);

    for(int n = 0; n < 10; n++)
    {
        output[n] = 0;
        for(int c = 0; c < 120; c++)
        {
            output[n] += weights[n][c][0][0] * input[c][0][0];
        }
        output[n]+=bias[n];
    }

    for (int n = 0; n < 10; n++)
        output[n] = adc_quantize(output[n]); 
}

// Relu Layer 6
void relu6(float input[10], float output[10])
{
    for(int i = 0; i < 10; i++)
        output[i] = relu(input[i]);
}
// *****************************************
// End declaration of layers functions
// *****************************************

// Parse MNIST test images
int parse_mnist_images(const char* filename, unsigned char *images)
{
    unsigned int header[4];

    ifstream img_file( filename, ios::in | ios::binary );
    if( ! img_file )
    {
        cerr << "Cannot open input file!" << endl;
        exit ( -1 );
    }

    img_file.read((char*)header, sizeof(unsigned int)*4); 
    img_file.read((char*)images, sizeof(unsigned char)*NUM_TESTS*28*28); 

    img_file.close(); 

    return 0; 
}

// Parse MNIST test image labels
int parse_mnist_labels(const char* filename, unsigned char *labels)
{
    unsigned int header[2];

    ifstream label_file( filename, ios::in | ios::binary );
    if( ! label_file )
    {
        cerr << "Cannot open input file!" << endl;
        exit ( -1 );
    }

    label_file.read((char*)header, sizeof(unsigned int)*2); 
    label_file.read((char*)labels, sizeof(unsigned char)*NUM_TESTS); 

    return 0;

}

// Parse parameter file and load it in to the arrays
int parse_parameters(const char* filename)
{
    ifstream param_file( filename, ios::in | ios::binary );

    if( ! param_file )
    {
        cerr << "Cannot open input file!" << endl;
        exit ( -1 );
    }

    param_file.read((char*)***conv1_weights, sizeof(float)*150); 
    param_file.read((char*)conv1_bias, sizeof(float)*6); 

    param_file.read((char*)***conv3_weights, sizeof(float)*2400); 
    param_file.read((char*)conv3_bias, sizeof(float)*16); 

    param_file.read((char*)***conv5_weights, sizeof(float)*48000); 
    param_file.read((char*)conv5_bias, sizeof(float)*120); 

    param_file.read((char*)***fc6_weights, sizeof(float)*1200); 
    param_file.read((char*)fc6_bias, sizeof(float)*10); 

    param_file.close(); 


    return 0;

}

// Fetch a single image to be processed.
//
void get_image(unsigned char *images, unsigned int idx, float image[1][32][32])
{
    for(int i = 0; i < 32; i++)
        for(int j = 0; j < 32; j++)
        {
            if (i < 2 || i > 29 || j < 2 || j > 29)
                image[0][i][j] = -1.0;
            else
                image[0][i][j] = images[idx*28*28 + (i-2)*28 + j-2] / 255.0 * 2.0 - 1.0;
        }
}

int main(int argc, char **argv)
{

/*    float res1 = quantize_act(103.13); 
    ap_fixed<6, 3, AP_TRN_ZERO, AP_SAT> a = ap_fixed<6, 3, AP_TRN_ZERO, AP_SAT>(4.3); 
    for(int i = 0; i < 6; i++)
        cout << "a[" << i << "] = " << a[i] << endl;

    float a_res = a; 
    cout << a_res << endl;
    cout << a << endl;

    cout << adc_quantize_4_3(res1) << endl; 

    ap_fixed<6, 3, AP_TRN_ZERO, AP_SAT> b = ap_fixed<6, 3, AP_TRN_ZERO, AP_SAT>(a_res); 
    cout << b << endl; 

    for(int i = 0; i < 6; i++)
        cout << "b[" << i << "] = " << b[i] << endl;

    return 0; */

    // cout<<"Starting LeNet\n\r";

    //cout<<"Creating test data matrices\n\r";
    //cout<<"Creating layer matrices\n\r";

    // cout<<"Parsing MNIST images\n\r";
    parse_mnist_images("./data/images.bin", images);
    //xil_printf("Back from image load\n\r");

    // cout<<"Parsing MNIST labels\n\r";
    parse_mnist_labels("./data/labels.bin", labels);

    // cout<<"Parsing parameters\n\r";
    parse_parameters("./data/params.bin");

    // cout<<"Starting inference\n\r";
    int num_correct = 0;

    // cout << "\n\rTest Image: " << endl;
    for (int k = 0; k < NUM_TESTS; k++)
    {
        //Get test image from dataset
        get_image(images, k, image);

        //Begin inference here.
        convolution1(image, conv1_weights, conv1_bias, conv1_output);
        relu1(conv1_output, conv1_output);

        max_pooling2(conv1_output, pool2_output);
        relu2(pool2_output, pool2_output);

        convolution3(pool2_output, conv3_weights, conv3_bias, conv3_output);
        relu3(conv3_output, conv3_output);

        max_pooling4(conv3_output, pool4_output);
        relu4(pool4_output, pool4_output);

        convolution5(pool4_output, conv5_weights, conv5_bias, conv5_output);
        relu5(conv5_output, conv5_output);

        fc6(conv5_output, fc6_weights, fc6_bias, fc6_output);
        //End inference here.

        //Check which output was largest.
        unsigned char result = 0;
        float p = -1000000.0;
        for(int i = 0; i < 10; i++)
        {
            if(fc6_output[i] > p)
            {
                p = fc6_output[i];
                result = i;
            }
        }
        //Largest output is result

        //std::cout << "test " << k << ": " << int(result) << "/" << int(labels[k]) << ": ";
        if(result == labels[k])
        {
            num_correct++;
            //std::cout << "correct" << std::endl;
        }
        else
        {
            //std::cout << "WRONG" << std::endl;
        }

        //Disable these print statements when doing profiling and benchmark times
/*        printf("%d ", k);
        if(k%10==0)
            printf("\n\rTest Image: ");*/
    }

    // std::cout << argv[0] << " accuracy = " << float(num_correct)/NUM_TESTS * 100.0 << "%" << std::endl;
    std::cout << float(num_correct)/NUM_TESTS << std::endl;

    return 0;
}
