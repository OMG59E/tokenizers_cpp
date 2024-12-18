/***
 * @Author: xingwg
 * @Date: 2024-12-18 14:18:04
 * @LastEditTime: 2024-12-18 15:19:31
 * @FilePath: /bertTokenizer_cpp/clip_tokenizer.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "common.h"

/**
 * CLIPTokenizerFast
 */
class CLIPTokenizerFast {
public:
    /**
     *
     * @param custom_op_library_path
     */
    explicit CLIPTokenizerFast(const std::string &custom_op_library_path);
    ~CLIPTokenizerFast();

    /**
     *
     * @param text
     * @param outputs
     * @param max_length 最大长度，默认0 表示禁用padding, >0 表示启用padding
     * @return
     */
    int encode(const std::string &text, std::vector<Tensor> &outputs,
               int32_t max_length = 0);

private:
    void *handle_{nullptr};
};