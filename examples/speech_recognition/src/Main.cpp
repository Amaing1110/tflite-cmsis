//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory> 

#include "AudioCapture.hpp"
#include "SpeechRecognitionPipeline.hpp"
#include "Wav2LetterMFCC.hpp"
#include "testdata/audio_samples.h"

std::map<int, std::string> labels = 
{
        {0,  "a"},
        {1,  "b"},
        {2,  "c"},
        {3,  "d"},
        {4,  "e"},
        {5,  "f"},
        {6,  "g"},
        {7,  "h"},
        {8,  "i"},
        {9,  "j"},
        {10, "k"},
        {11, "l"},
        {12, "m"},
        {13, "n"},
        {14, "o"},
        {15, "p"},
        {16, "q"},
        {17, "r"},
        {18, "s"},
        {19, "t"},
        {20, "u"},
        {21, "v"},
        {22, "w"},
        {23, "x"},
        {24, "y"},
        {25, "z"},
        {26, "\'"},
        {27, " "},
        {28, "$"}
};

extern "C"
int tf_main(int argc, char* argv[]) 
{
    bool isFirstWindow = true;
    std::string currentRContext = "";


    // Create the network options
    common::PipelineOptions pipelineOptions;
    pipelineOptions.m_ModelName = "Wav2Letter";

    asr::IPipelinePtr asrPipeline = asr::CreatePipeline(pipelineOptions, labels);

    audio::AudioCapture capture;    
    //std::vector<float> audioData;
    //audio::AudioCapture::LoadAudioFile(GetSpecifiedOption(options, AUDIO_FILE_PATH));
    std::vector<float> audioData(audio_samples, audio_samples + audio_samples_length);
    

    MicroPrintf("Init Sliding window, %d", asrPipeline->getSlidingWindowOffset());
    capture.InitSlidingWindow(audioData.data(), audioData.size(), asrPipeline->getInputSamplesSize(),
                              asrPipeline->getSlidingWindowOffset());

    while (capture.HasNext()) 
    {
        std::vector<float> audioBlock = capture.Next();
        common::InferenceResults<int8_t> results;

        std::vector<int8_t> preprocessedData = asrPipeline->PreProcessing(audioBlock);
        asrPipeline->Inference<int8_t>(preprocessedData, results);
        asrPipeline->PostProcessing<int8_t>(results, isFirstWindow, !capture.HasNext(), currentRContext);
    }

    return 0;
}