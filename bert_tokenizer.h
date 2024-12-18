/***
 * @Author: xingwg
 * @Date: 2024-12-18 11:58:17
 * @LastEditTime: 2024-12-18 15:32:15
 * @FilePath: /bertTokenizer_cpp/bert_tokenizer.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "common.h"

/**
 * BertTokenizer
 */
class BertTokenizer {
public:
    /**
     *
     * @param custom_op_library_path
     */
    explicit BertTokenizer(const std::string &custom_op_library_path);
    ~BertTokenizer();

    /**
     *
     * @param text
     * @param outputs
     * @param max_length 最大长度，默认0 表示禁用padding, >0 表示启用padding
     * @return
     */
    int encode(const std::string &text, std::vector<Tensor> &outputs,
               int32_t max_length = 0);

    /**
     *
     * @param input_ids
     * @param text
     * @return
     */
    int decode(const input_ids_t &input_ids, std::string &text);

private:
    void *handle_{nullptr};
};
