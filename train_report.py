from llm_analysis.analysis import train

REPORT_PATH = "outputs_infer/train_report.md"
REPORT_SETTINGS_TEMPLATE={
    "ds_zero":0,
    "model_name": "decapoda-research_llama-7b-hf",
    "gpu_name":"a6000-48gb",
    # "dtype_name":"w4a4e16",
    "dtype_name":"w16a16e16",
    "log_level":"FATAL",
    "tp_size": 2,
    "activation_recomputation": 2,
    # "global_batch_size":1000,
    # "batch_size_per_gpu": 5,
}

GPU_LIST = ["a10g-pcie-24gb","a6000-48gb", "l40-48gb","a100-sxm-40gb"]
GPU_COUNT_MAP = {
    "a10g-pcie-24gb": 5,
    "a6000-48gb": 3,
    "l40-48gb": 3,
    "a100-sxm-40gb": 3,

}
DOCUMENT_LENGTH = 1320
# DOCUMENT_COUNT_CASES = [5e3,2e5,5e5,2.5e6]
DOCUMENT_COUNT_CASES = [5e5,2.5e6]


with open(REPORT_PATH, "w") as report_file:
    def print_and_save(line):
        print(line)
        report_file.write(line + "\n")
    # First, print the table headers including the new 'Words per second' column
    print_and_save("# Training Predictions Report")
    print_and_save("## Model: decapoda-research_llama-7b-hf")
    print_and_save("| GPU | Doucments count | GPU hours | GPU days |")
    print_and_save("| --- | ---------- | ----------------- | ---------------- |")
    # Then, print each row based on the GPU performance or error, incorporating 'Words per second'
    for gpu in GPU_LIST:
        for case in DOCUMENT_COUNT_CASES:
            try:
                settings = REPORT_SETTINGS_TEMPLATE.copy()
                settings['gpu_name'] = gpu
                settings['tp_size'] = GPU_COUNT_MAP[gpu]
                settings['total_num_tokens'] = case / 4 * 3 * DOCUMENT_LENGTH
                settings["hbm_memory_efficiency"] = 0.65
                report = train(**settings)
                gpu_hours = report['gpu_hours']
                total_gpu_days = gpu_hours/24
                # Formatting 'total_latency' and 'total_words_per_sec' for consistent decimal places might be preferred
                formatted_gpu_hours = "{:.2f}".format(gpu_hours)
                formatted_total_gpu_days = "{:.2f}".format(total_gpu_days)
                print_and_save(f"| {gpu} x {settings['tp_size']} | {case} | {formatted_gpu_hours} hours | {formatted_total_gpu_days} days |")
            except Exception as e:
                # Assuming errors are strings that can be directly included
                error_message = str(e).replace("|", ",")  # Replace '|' to avoid breaking table format if present in error
                print_and_save(f"| {gpu} x {settings['tp_size']} | {case} | - | - |")