@echo off
REM Copyright 2023 chengli
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

set gpu_name=a6000-48gb
set dtype_name=w4a4e16
set output_dir=outputs_infer
set model_name=upstage/Llama-2-70b-instruct-v2
set batch_size_per_gpu=1
set tp_size=4
set output_file_suffix=-bs%batch_size_per_gpu%
set cost_per_gpu_hour=2.21
set seq_len=1024
set num_tokens_to_generate=512
set flops_efficiency=0.7
set hbm_memory_efficiency=0.9
set achieved_tflops=200               REM will overwrite the flops_efficiency above
set achieved_memory_bandwidth_GBs=1200 REM will overwrite the hbm_memory_efficiency above

if not exist "%output_dir%" (
    mkdir "%output_dir%"
) else (
    if not exist "%output_dir%\*" (
        echo %output_dir% already exists but is not a directory 1>&2
    )
)

python -m llm_analysis.analysis infer --model_name=%model_name% --gpu_name=%gpu_name% --dtype_name=%dtype_name% --output_dir=%output_dir% --output-file-suffix=%output_file_suffix% ^
    --seq_len=%seq_len% --num_tokens_to_generate=%num_tokens_to_generate% --batch_size_per_gpu=%batch_size_per_gpu% ^
    --tp_size=%tp_size% ^
    --cost_per_gpu_hour=%cost_per_gpu_hour% ^
    --flops_efficiency=%flops_efficiency% --hbm_memory_efficiency=%hbm_memory_efficiency% --log_level DEBUG --output_dir=reports
REM --achieved_tflops=%achieved_tflops% --achieved_memory_bandwidth_GBs=%achieved_memory_bandwidth_GBs%
