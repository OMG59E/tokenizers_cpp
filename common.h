/***
 * @Author: xingwg
 * @Date: 2024-12-18 16:03:35
 * @LastEditTime: 2024-12-20 11:20:12
 * @FilePath: /tokenizers_cpp/common.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include <string>
#include <vector>

#define RELEASE(ptr)                                                           \
    do {                                                                       \
        if (ptr) {                                                             \
            delete ptr;                                                        \
            ptr = nullptr;                                                     \
        }                                                                      \
    } while (0)

;  //
struct TokenizerOutput {
    std::string name;
    int32_t ndim{0};
    int32_t dims[8]{};
    std::vector<int64_t> buf;

    int64_t size() {
        int64_t total = 1;
        for (int n = 0; n < ndim; ++n) {
            total *= dims[n];
        }
        return total;
    }
};

typedef std::vector<int64_t> input_ids_t;

static std::string base64Decode(const char *Data, int DataByte) {
    const char
        DecodeTable[] =
            {
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                62,  // '+'
                0,  0,  0,
                63,                                      // '/'
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  // '0'-'9'
                0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,
                8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25,  // 'A'-'Z'
                0,  0,  0,  0,  0,  0,  26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                50, 51,  // 'a'-'z'
            };

    std::string strDecode;
    int nValue;
    int i = 0;
    while (i < DataByte) {
        if (*Data != '\r' && *Data != '\n') {
            nValue = DecodeTable[*Data++] << 18;
            nValue += DecodeTable[*Data++] << 12;
            strDecode += (nValue & 0x00FF0000) >> 16;
            if (*Data != '=') {
                nValue += DecodeTable[*Data++] << 6;
                strDecode += (nValue & 0x0000FF00) >> 8;
                if (*Data != '=') {
                    nValue += DecodeTable[*Data++];
                    strDecode += nValue & 0x000000FF;
                }
            }
            i += 4;
        } else {
            Data++;
            i++;
        }
    }
    return strDecode;
}
