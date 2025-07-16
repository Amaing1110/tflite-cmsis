import librosa
import soundfile as sf
import numpy as np

def flac_to_c_array(input_flac, output_header):
    # 1. 读取FLAC文件并直接重采样到16kHz单声道
    y, sr = librosa.load(input_flac, sr=16000, mono=True)
    
    # 2. 转换为浮点数组并量化到(0,1)范围
    #y_float = y.astype(np.float32)
    #y_normalized = (y_float - np.min(y_float)) / (np.max(y_float) - np.min(y_float))
    y_normalized=y
    
    # 3. 生成C语言数组
    array_length = len(y_normalized)
    c_array = "const float audio_samples[%d] = {\n" % array_length
    
    # 每行10个元素
    for i in range(array_length):
        if i % 10 == 0:
            c_array += "    "
        c_array += "%.8ff" % y_normalized[i]
        if i != array_length - 1:
            c_array += ", "
        if i % 10 == 9 or i == array_length - 1:
            c_array += "\n"
    
    c_array += "};\n"
    c_array += "const int audio_samples_length = %d;\n" % array_length
    
    # 4. 写入头文件
    with open(output_header, 'w') as f:
        f.write("#ifndef AUDIO_SAMPLES_H\n")
        f.write("#define AUDIO_SAMPLES_H\n\n")
        f.write(c_array)
        f.write("\n#endif // AUDIO_SAMPLES_H\n")
    
    return y_normalized, sr

# 使用示例
audio_data, sample_rate = flac_to_c_array("input.flac", "audio_samples.h")