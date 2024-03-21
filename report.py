from llm_analysis.analysis import infer,train

REPORT_PATH = "outputs_infer/report.md"
INFER_REPORT_CSV_PATH = "outputs_infer/infer_report.csv"
TRAIN_REPORT_CSV_PATH = "outputs_train/train_report.csv"


REPORT_SETTINGS_TEMPLATE = {
    "model_name": "decapoda-research_llama-7b-hf",
    # "dtype_name": "w16a16e16",
    "log_level": "FATAL",
    "tp_size": 4,
    "seq_len": 1695,
    "num_tokens_to_generate": 512,
}

QUNTIZATION_SENARIOS_LIST = ["w4a4e16", "w8a8e16", "w16a16e16"]
QUNTIZATION_SENARIOS_NAME_MAP = {
    "w4a4e16": "4-bit",
    "w8a8e16": "8-bit",
    "w16a16e16": "16-bit",
}
GPU_LIST = ["mtt-s3000-32gb", "a6000-48gb", "l20-48gb", "l40-48gb"]
GPU_COUNT_MAP = {
    "mtt-s3000-32gb": 4,
    "a6000-48gb": 3,
    "l20-48gb": 3,
    "l40-48gb": 3,
}
GPU_PRICE_MAP = {
    "mtt-s3000-32gb": 22546.6,
    "a6000-48gb": 5400,
    "l20-48gb": 4834,
    "l40-48gb": 7707,
}
USER_COUNT_CASES = [1, 5, 20, 100]

with open(REPORT_PATH, "w") as report_file:
    def print_and_save(line):
        print(line)
        report_file.write(line + "\n")
    # First, print the table headers including the new columns
    print_and_save("# Inferencing and Training Predictions Report")
    print_and_save("In this report, we will analyze the performance of the decapoda-research_llama-7b-hf model in different scenarios, including different quantization levels and GPU performance. We will also analyze the cost efficiency of each scenario, based on the average words per second.")
    print_and_save("## Inference Speed Predictions")
    print_and_save("Assuming RAG is around 750 words and the answer is around 375 words")
    print_and_save("### Model: decapoda-research_llama-7b-hf")
    print_and_save("""\n<div style="page-break-after: always;"></div>\n""")

    # Then, print each row based on the GPU performance or error, incorporating the adjusted columns
    for quantization in QUNTIZATION_SENARIOS_LIST:
        print_and_save(f"#### Quantization: {QUNTIZATION_SENARIOS_NAME_MAP[quantization]}")
        for gpu in GPU_LIST:
            print_and_save(f"##### GPU: {gpu} x {GPU_COUNT_MAP[gpu]} ")
            price=GPU_PRICE_MAP[gpu]*GPU_COUNT_MAP[gpu]*7.8
            print_and_save(f"###### Price: ${price:.2f} HKD")
            print_and_save("| Users count | Perfect Latency | Upper Latency | Lower Latency | Words per second | Error |")
            print_and_save("| ---------- | -------------------- | ---------------------- | ----------- | -------------- | ----- |")
            for user_count in USER_COUNT_CASES:
                try:
                    settings = REPORT_SETTINGS_TEMPLATE.copy()
                    settings['gpu_name'] = gpu
                    settings['dtype_name'] = quantization
                    settings['tp_size'] = GPU_COUNT_MAP[gpu]
                    settings['batch_size_per_gpu'] = user_count
                    
                    # Perfect scenario
                    perfect_report = infer(**settings)
                    perfect_total_latency = perfect_report['total_latency']
                    perfect_total_words_per_sec = perfect_report['total_tokens_per_sec'] * 3 / 4  # Assuming conversion factor

                    # Upper Benchmarks scenario
                    settings["hbm_memory_efficiency"] = 0.7
                    upper_benchmarks_report = infer(**settings)
                    upper_benchmarks_total_latency = upper_benchmarks_report['total_latency']
                    upper_benchmarks_total_words_per_sec = upper_benchmarks_report['total_tokens_per_sec'] * 3 / 4  # Same assumption

                    # Lower Benchmarks scenario
                    settings["hbm_memory_efficiency"] = 0.6
                    lower_benchmarks_report = infer(**settings)
                    lower_benchmarks_total_latency = lower_benchmarks_report['total_latency']
                    lower_benchmarks_total_words_per_sec = lower_benchmarks_report['total_tokens_per_sec'] * 3 / 4  # Same assumption

                    average_benchmarks_total_words_per_sec = (upper_benchmarks_total_words_per_sec + lower_benchmarks_total_words_per_sec) / 2

                    # Formatting for consistency
                    formatted_perfect_latency = "{:.2f}".format(perfect_total_latency)
                    formatted_upper_benchmarks_latency = "{:.2f}".format(upper_benchmarks_total_latency)
                    formatted_lower_benchmarks_latency = "{:.2f}".format(lower_benchmarks_total_latency)
                    formatted_perfect_wps = "{:.2f}".format(perfect_total_words_per_sec)
                    formatted_benchmarks_wps = "{:.2f}".format(average_benchmarks_total_words_per_sec)

                    print_and_save(f"| {user_count} | {formatted_perfect_latency} s | {formatted_upper_benchmarks_latency} s | {formatted_lower_benchmarks_latency} s | {formatted_benchmarks_wps} | - |")
                except Exception as e:
                    # Handle errors, avoiding table format breakage
                    error_message = str(e).replace("|", ",")
                    if "too large to fit in GPU memory" in error_message:
                        error_message = "Insufficient vRAM"
                    print_and_save(f"| {user_count} | - | - | - | - | {error_message} |")
                pass
            print_and_save(f"###### Cost efficiency: ${price / average_benchmarks_total_words_per_sec:.2f} HKD. (Speed cost efficiency).")
            print_and_save("---")
        print_and_save("""\n<div style="page-break-after: always;"></div>\n""")
    
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
    DOCUMENT_LENGTH = 1320
    DOCUMENT_COUNT_CASES = [5e5,2.5e6]
    # First, print the table headers including the new 'Words per second' column
    print_and_save("## Training Predictions")
    print_and_save("This training assumes the the tuning method is **full fine-tuning**")
    print_and_save("### Model: decapoda-research_llama-7b-hf")
    print_and_save("| GPU | Doucments count | GPU hours | Nights (8 hours) |")
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
                total_gpu_days = gpu_hours/8
                # Formatting 'total_latency' and 'total_words_per_sec' for consistent decimal places might be preferred
                formatted_gpu_hours = "{:.2f}".format(gpu_hours)
                formatted_total_gpu_days = "{:.2f}".format(total_gpu_days)
                print_and_save(f"| {gpu} x {settings['tp_size']} | {case} | {formatted_gpu_hours} hours | {formatted_total_gpu_days} nights |")
            except Exception as e:
                # Assuming errors are strings that can be directly included
                error_message = str(e).replace("|", ",")  # Replace '|' to avoid breaking table format if present in error
                print_and_save(f"| {gpu} x {settings['tp_size']} | {case} | - | - |")
