/*** 
 * @Author: xingwg
 * @Date: 2024-12-18 15:55:11
 * @LastEditTime: 2024-12-18 19:14:21
 * @FilePath: /tokenizers_cpp/main.cpp
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "bert_tokenizer.h"
#include "clip_tokenizer.h"
#include <iostream>
#include <string>

void print_tensor(Tensor &t) {
    printf("%s dim: %d, len: %ld\n", t.name.c_str(), t.ndim, t.size());

    printf("shape: ");
    for (int i = 0; i < t.ndim; ++i) {
        printf("%d ", t.dims[i]);
    }
    printf("\n");

    printf("value: ");
    for (int i = 0; i < t.size(); ++i) {
        printf("%ld ", t.buf[i]);
    }
    printf("\n");
}

int main() {
    const std::string custom_op_library_path =
        "../3rdparty/lib/libortextensions.so.0.13.0";
    BertTokenizer bertTokenizer(custom_op_library_path);
    CLIPTokenizerFast clipTokenizerFast(custom_op_library_path);

    std::string prompt = "a photo of a cat";

    printf("\nBertTokenizer Encode:\n");
    std::vector<Tensor> outputs;
    bertTokenizer.encode(prompt, outputs, 32);
    print_tensor(outputs[0]);
    print_tensor(outputs[1]);
    print_tensor(outputs[2]);
    print_tensor(outputs[3]);

    std::string decoded_text;
    bertTokenizer.decode(outputs[0].buf, decoded_text);
    printf("\nBertTokenizer Decode: %s\n", decoded_text.c_str());

    printf("\nCLIPTokenizerFast\n");
    outputs.clear();
    clipTokenizerFast.encode(prompt, outputs, 32);
    print_tensor(outputs[0]);
    print_tensor(outputs[1]);
    print_tensor(outputs[2]);
    return 0;
}