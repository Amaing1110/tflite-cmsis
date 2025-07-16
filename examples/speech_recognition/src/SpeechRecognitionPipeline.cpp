//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include <algorithm>
#include <cstdint>
#include <iterator>

//#define USE_PRUNED

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "SpeechRecognitionPipeline.hpp"
#ifdef USE_PRUNED
#include "models/tiny_wav2letter_pruned_int8.cc"
#else
#include "models/tiny_wav2letter_int8.cc"
#endif
#include "cmsis_compiler.h"
#include "app_mem.h"

tflite::MicroProfiler profiler;
using SpeeckRecOpResolver = tflite::MicroMutableOpResolver<7>;
static SpeeckRecOpResolver op_resolver;

// Arena size is a guesstimate, followed by use of
// MicroInterpreter::arena_used_bytes() on both the AudioPreprocessor and
// MicroSpeech models and using the larger of the two results.
constexpr size_t kArenaSize = 4000*1024;

L2_RET_BSS_SECT_BEGIN(app_psram_ret_cache)
APP_L2_RET_BSS_SECT(app_psram_ret_cache, ALIGN(16) static uint8_t g_arena[kArenaSize]);
L2_RET_BSS_SECT_END

namespace asr 
{

ASRPipeline::ASRPipeline(tflite::MicroInterpreter    * executor,
            std::unique_ptr<Decoder> decoder, std::unique_ptr<Wav2LetterPreprocessor> preprocessor) :
        m_executor(std::move(executor)),
        m_decoder(std::move(decoder)), m_preProcessor(std::move(preprocessor)) {}

int ASRPipeline::getInputSamplesSize() 
{
    return this->m_preProcessor->m_windowLen +
           ((this->m_preProcessor->m_mfcc->m_params.m_numMfccVectors - 1) * this->m_preProcessor->m_windowStride);
}

int ASRPipeline::getSlidingWindowOffset()
{
    // Hardcoded for now until refactor
    return ASRPipeline::SLIDING_WINDOW_OFFSET;
}

std::vector<int8_t> ASRPipeline::PreProcessing(std::vector<float>& audio) 
{
    int audioDataToPreProcess = m_preProcessor->m_windowLen +
                                ((m_preProcessor->m_mfcc->m_params.m_numMfccVectors - 1) *
                                 m_preProcessor->m_windowStride);
    int outputBufferSize = m_preProcessor->m_mfcc->m_params.m_numMfccVectors
                           * m_preProcessor->m_mfcc->m_params.m_numMfccFeatures * 3;
    static std::vector<int8_t> outputBuffer(outputBufferSize);
    
    TfLiteTensor* input = m_executor->input(0);    
    float quant_scale = input->params.scale;
    int quant_offset = input->params.zero_point;
    m_preProcessor->Invoke(audio.data(), audioDataToPreProcess, outputBuffer, quant_offset, quant_scale);
    return outputBuffer;
}

TfLiteStatus RegisterOps(SpeeckRecOpResolver& op_resolver) 
{
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLeakyRelu());
  return kTfLiteOk;
}
IPipelinePtr CreatePipeline(common::PipelineOptions& config, std::map<int, std::string>& labels) 
{
    if (config.m_ModelName == "Wav2Letter") 
    {
        // Wav2Letter ASR SETTINGS
        int SAMP_FREQ = 16000;
        int FRAME_LEN_MS = 32;
        int FRAME_LEN_SAMPLES = SAMP_FREQ * FRAME_LEN_MS * 0.001;
        int NUM_MFCC_FEATS = 13;
        int MFCC_WINDOW_LEN = 512;
        int MFCC_WINDOW_STRIDE = 160;
        const int NUM_MFCC_VECTORS = 296;
        int SAMPLES_PER_INFERENCE = MFCC_WINDOW_LEN + ((NUM_MFCC_VECTORS - 1) * MFCC_WINDOW_STRIDE);
        int MEL_LO_FREQ = 0;
        int MEL_HI_FREQ = 8000;
        int NUM_FBANK_BIN = 128;
        int INPUT_WINDOW_LEFT_CONTEXT = 98;
        int INPUT_WINDOW_RIGHT_CONTEXT = 98;
        int INPUT_WINDOW_INNER_CONTEXT = NUM_MFCC_VECTORS -
                                         (INPUT_WINDOW_LEFT_CONTEXT + INPUT_WINDOW_RIGHT_CONTEXT);
        int SLIDING_WINDOW_OFFSET = INPUT_WINDOW_INNER_CONTEXT * MFCC_WINDOW_STRIDE;


        MfccParams mfccParams(SAMP_FREQ, NUM_FBANK_BIN,
                              MEL_LO_FREQ, MEL_HI_FREQ, NUM_MFCC_FEATS, FRAME_LEN_SAMPLES, false, NUM_MFCC_VECTORS);

        std::unique_ptr<Wav2LetterMFCC> mfccInst = std::make_unique<Wav2LetterMFCC>(mfccParams);

        //auto executor = std::make_unique<common::ArmnnNetworkExecutor<int8_t>>(config.m_ModelFilePath,
        //                                                                       config.m_backends);

        tflite::MicroInterpreter * executor; 

#ifdef USE_PRUNED
        const tflite::Model* model = tflite::GetModel(tiny_wav2letter_pruned_int8_tflite);
#else
        const tflite::Model* model = tflite::GetModel(tiny_wav2letter_int8_tflite);
#endif
        TfLiteStatus r=RegisterOps(op_resolver);

        if (kTfLiteOk!=r)
            MicroPrintf("Could not Register OPs: %d.", r);
        else {    
            
            executor=new tflite::MicroInterpreter(model, op_resolver, g_arena, kArenaSize, NULL, &profiler);
            r=executor->AllocateTensors();
            if (kTfLiteOk!=r) {
                MicroPrintf("Could not Allocate tensors: %d.", r);
            }
            else {
                MicroPrintf("Speech Recognition model arena size = %u", executor->arena_used_bytes());
            }
        }    
        
        auto decoder = std::make_unique<asr::Decoder>(labels);
        auto preprocessor = std::make_unique<Wav2LetterPreprocessor>(MFCC_WINDOW_LEN, MFCC_WINDOW_STRIDE,
                                                                     std::move(mfccInst));

        auto ptr = std::make_unique<asr::ASRPipeline>(
                std::move(executor), std::move(decoder), std::move(preprocessor));

        ptr->SLIDING_WINDOW_OFFSET = SLIDING_WINDOW_OFFSET;

        return ptr;
    } 
    else
    {
        MicroPrintf("Unknown Model name: %s." , config.m_ModelName.c_str());
        return nullptr;
    }
}

}// namespace asr