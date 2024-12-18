'''
Author: xingwg
Date: 2024-12-18 16:03:35
LastEditTime: 2024-12-18 16:19:32
FilePath: /tokenizers_cpp/export_CLIPTokenizer.py
Description: 

Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
'''
import tempfile
import numpy as np
import base64
import onnx
import onnxruntime as _ort
from pathlib import Path
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import (
    make_onnx_model,
    get_library_path as _get_library_path,
    PyOrtFunction)
from transformers import CLIPTokenizer, CLIPTokenizerFast


def _get_file_content(path):
    with open(path, "rb") as file:
        return file.read()
    
    
def _create_test_model(**kwargs):
    vocab_file = kwargs["vocab_file"]
    merges_file = kwargs["merges_file"]
    max_length = kwargs["max_length"]

    input1 = helper.make_tensor_value_info(
        'string_input', onnx_proto.TensorProto.STRING, [None])
    output1 = helper.make_tensor_value_info(
        'input_ids', onnx_proto.TensorProto.INT64, ["batch_size", "num_input_ids"])
    output2 = helper.make_tensor_value_info(
        'attention_mask', onnx_proto.TensorProto.INT64, ["batch_size", "num_attention_masks"])
    output3 = helper.make_tensor_value_info(
        'offset_mapping', onnx_proto.TensorProto.INT64, ["batch_size", "num_offsets", 2])

    if kwargs["attention_mask"]:
        if kwargs["offset_map"]:
            node = [helper.make_node(
                'CLIPTokenizer', ['string_input'],
                ['input_ids', 'attention_mask', 'offset_mapping'], vocab=_get_file_content(vocab_file),
                merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
                domain='ai.onnx.contrib')]

            graph = helper.make_graph(node, 'test0', [input1], [output1, output2, output3])
            model = make_onnx_model(graph)
        else:
            node = [helper.make_node(
                'CLIPTokenizer', ['string_input'], ['input_ids', 'attention_mask'], vocab=_get_file_content(vocab_file),
                merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
                domain='ai.onnx.contrib')]

            graph = helper.make_graph(node, 'test0', [input1], [output1, output2])
            model = make_onnx_model(graph)
    else:
        node = [helper.make_node(
            'CLIPTokenizer', ['string_input'], ['input_ids'], vocab=_get_file_content(vocab_file),
            merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
            domain='ai.onnx.contrib')]

        graph = helper.make_graph(node, 'test0', [input1], [output1])
        model = make_onnx_model(graph)

    return model


tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
temp_dir = Path(tempfile.gettempdir())
temp_dir.mkdir(parents=True, exist_ok=True)
files = tokenizer.save_vocabulary(str(temp_dir))      
tokjson = files[0]
merges = files[1]
        
model = _create_test_model(vocab_file=tokjson, merges_file=merges, max_length=32, attention_mask=True, offset_map=True)
print(model)

test_sentence = ["a photo of a cat", "a photo of a dog"]
so = _ort.SessionOptions()
so.register_custom_ops_library(_get_library_path())
sess = _ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
input_text = np.array(test_sentence)
input_ids, attention_mask, _ = sess.run(None, {'string_input': input_text})
clip_out = tokenizer(test_sentence, return_offsets_mapping=True)
expect_input_ids = clip_out['input_ids']
expect_attention_mask = clip_out['attention_mask']
print("       input_ids:", input_ids)
print("expect_input_ids:", expect_input_ids)
print("       attention_mask:", attention_mask)
print("expect_attention_mask:", expect_attention_mask)

np.testing.assert_array_equal(input_ids, expect_input_ids)
np.testing.assert_array_equal(attention_mask, expect_attention_mask)

model_name = "models/CLIPTokenizerFast.onnx"
onnx.save(model, model_name)

model_str = base64.b64encode(model.SerializeToString()).decode("utf-8")

clip_tokenizer_str = f"#pragma once\n#include <string>\n\nstatic const std::string clip_tokenizer_str = \"{model_str}\"\n"
with open("CLIPTokenizerFast.h", "w") as f:
    f.write(clip_tokenizer_str)

