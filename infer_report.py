from llm_analysis.analysis import infer

REPORT_PATH = "outputs_infer\infer_report.md"

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
GPU_LIST = ["a10g-pcie-24gb", "a6000-48gb", "l40-48gb", "a100-sxm-40gb"]
GPU_COUNT_MAP = {
    "a10g-pcie-24gb": 5,
    "a6000-48gb": 3,
    "l40-48gb": 3,
    "a100-sxm-40gb": 3,
}
GPU_PRICE_MAP = {
    "a10g-pcie-24gb": 3500,
    "a6000-48gb": 5400,
    "l40-48gb": 7707,
    "a100-sxm-40gb": 12000,
}
USER_COUNT_CASES = [1, 5, 15, 30, 100]

with open(REPORT_PATH, "w") as report_file:
    def print_and_save(line):
        print(line)
        report_file.write(line + "\n")
    # First, print the table headers including the new columns
    print_and_save("# Inference Predictions Report")
    print_and_save("In this report, we will analyze the performance of the decapoda-research_llama-7b-hf model in different scenarios, including different quantization levels and GPU performance. We will also analyze the cost efficiency of each scenario, based on the average words per second.")
    print_and_save("## Model: decapoda-research_llama-7b-hf")
    print_and_save("Assuming RAG is around 750 words and the answer is around 375 words")
    print_and_save("The transcript of the US Declaration of Independence contains 1,320 words, which is approximately 1,695 tokens.")
    print_and_save("""**Here is what 1,320 words US Declaration of Independence looks like**:
The unanimous Declaration of the thirteen united States of America, When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.

We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, --That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes; and accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it is their duty, to throw off such Government, and to provide new Guards for their future security.--Such has been the patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid world.

He has refused his Assent to Laws, the most wholesome and necessary for the public good.

He has forbidden his Governors to pass Laws of immediate and pressing importance, unless suspended in their operation till his Assent should be obtained; and when so suspended, he has utterly neglected to attend to them.

He has refused to pass other Laws for the accommodation of large districts of people, unless those people would relinquish the right of Representation in the Legislature, a right inestimable to them and formidable to tyrants only.

He has called together legislative bodies at places unusual, uncomfortable, and distant from the depository of their public Records, for the sole purpose of fatiguing them into compliance with his measures.

He has dissolved Representative Houses repeatedly, for opposing with manly firmness his invasions on the rights of the people.

He has refused for a long time, after such dissolutions, to cause others to be elected; whereby the Legislative powers, incapable of Annihilation, have returned to the People at large for their exercise; the State remaining in the mean time exposed to all the dangers of invasion from without, and convulsions within.

He has endeavoured to prevent the population of these States; for that purpose obstructing the Laws for Naturalization of Foreigners; refusing to pass others to encourage their migrations hither, and raising the conditions of new Appropriations of Lands.

He has obstructed the Administration of Justice, by refusing his Assent to Laws for establishing Judiciary powers.

He has made Judges dependent on his Will alone, for the tenure of their offices, and the amount and payment of their salaries.

He has erected a multitude of New Offices, and sent hither swarms of Officers to harrass our people, and eat out their substance.

He has kept among us, in times of peace, Standing Armies without the Consent of our legislatures.

He has affected to render the Military independent of and superior to the Civil power.

He has combined with others to subject us to a jurisdiction foreign to our constitution, and unacknowledged by our laws; giving his Assent to their Acts of pretended Legislation:

For Quartering large bodies of armed troops among us:

For protecting them, by a mock Trial, from punishment for any Murders which they should commit on the Inhabitants of these States:

For cutting off our Trade with all parts of the world:

For imposing Taxes on us without our Consent:

For depriving us in many cases, of the benefits of Trial by Jury:

For transporting us beyond Seas to be tried for pretended offences

For abolishing the free System of English Laws in a neighbouring Province, establishing therein an Arbitrary government, and enlarging its Boundaries so as to render it at once an example and fit instrument for introducing the same absolute rule into these Colonies:

For taking away our Charters, abolishing our most valuable Laws, and altering fundamentally the Forms of our Governments:

For suspending our own Legislatures, and declaring themselves invested with power to legislate for us in all cases whatsoever.

He has abdicated Government here, by declaring us out of his Protection and waging War against us.

He has plundered our seas, ravaged our Coasts, burnt our towns, and destroyed the lives of our people.

He is at this time transporting large Armies of foreign Mercenaries to compleat the works of death, desolation and tyranny, already begun with circumstances of Cruelty & perfidy scarcely paralleled in the most barbarous ages, and totally unworthy the Head of a civilized nation.

He has constrained our fellow Citizens taken Captive on the high Seas to bear Arms against their Country, to become the executioners of their friends and Brethren, or to fall themselves by their Hands.

He has excited domestic insurrections amongst us, and has endeavoured to bring on the inhabitants of our frontiers, the merciless Indian Savages, whose known rule of warfare, is an undistinguished destruction of all ages, sexes and conditions.

In every stage of these Oppressions We have Petitioned for Redress in the most humble terms: Our repeated Petitions have been answered only by repeated injury. A Prince whose character is thus marked by every act which may define a Tyrant, is unfit to be the ruler of a free people.

Nor have We been wanting in attentions to our Brittish brethren. We have warned them from time to time of attempts by their legislature to extend an unwarrantable jurisdiction over us. We have reminded them of the circumstances of our emigration and settlement here. We have appealed to their native justice and magnanimity, and we have conjured them by the ties of our common kindred to disavow these usurpations, which, would inevitably interrupt our connections and correspondence. They too have been deaf to the voice of justice and of consanguinity. We must, therefore, acquiesce in the necessity, which denounces our Separation, and hold them, as we hold the rest of mankind, Enemies in War, in Peace Friends.

We, therefore, the Representatives of the united States of America, in General Congress, Assembled, appealing to the Supreme Judge of the world for the rectitude of our intentions, do, in the Name, and by Authority of the good People of these Colonies, solemnly publish and declare, That these United Colonies are, and of Right ought to be Free and Independent States; that they are Absolved from all Allegiance to the British Crown, and that all political connection between them and the State of Great Britain, is and ought to be totally dissolved; and that as Free and Independent States, they have full Power to levy War, conclude Peace, contract Alliances, establish Commerce, and to do all other Acts and Things which Independent States may of right do. And for the support of this Declaration, with a firm reliance on the protection of divine Providence, we mutually pledge to each other our Lives, our Fortunes and our sacred Honor.""")
    print_and_save("\n")
    print_and_save("""**Here is what the 370 words answer looks like**:
The United States Declaration of Independence is a historic document that marked the formal announcement of the Thirteen Colonies' separation from British rule. Ratified on July 4, 1776, it is one of the quintessential texts of American identity, embodying the philosophical foundation and the grievances that propelled the American Revolution.


Crafted by a Committee of Five appointed by the Second Continental Congress, with Thomas Jefferson emerging as the principal author, the Declaration is a masterpiece of Enlightenment thought, emphasizing individual liberties and the principles of government by the consent of the governed. It begins with an introduction and a preamble that outline the natural rights philosophy, asserting that all men are created equal and endowed with inalienable rights, amongst these life, liberty, and the pursuit of happiness.


The document espouses a revolutionary concept at the time: the idea that sovereignty resides not with a monarch, but with the people, and that the people have the right to form, alter, or abolish a government that becomes destructive of these ends. This assertion laid the groundwork for modern democracy and became a beacon of hope for oppressed peoples worldwide.


The Declaration's main body details a long list of grievances against King George III and the British government, including imposing taxes without consent, dissolving representative bodies, maintaining standing armies in peacetime without consent, and attempting to enforce absolute rule. These complaints illustrate the colonies' frustration with British tyranny and their perceived lack of representation and autonomy within the Empire.


The conclusion reaffirms the colonies' determination to secede from British rule, proclaiming themselves free and independent states with the full power to levy war, conclude peace, contract alliances, establish commerce, and perform all other acts and things which independent states may rightfully do. This bold statement was accompanied by the signatories' pledge to support the Declaration with their lives, fortunes, and sacred honor, emphasizing their commitment to the cause of independence.


The Declaration of Independence not only signaled the birth of the United States but also showcased revolutionary ideals that influenced numerous independence movements around the globe. Its enduring significance lies in its articulation of the rights and responsibilities of citizens and its vision of a society based on individual freedom and democratic governance.""")
    print_and_save("""\n<div style="page-break-after: always;"></div>\n""")

    # Then, print each row based on the GPU performance or error, incorporating the adjusted columns
    for quantization in QUNTIZATION_SENARIOS_LIST:
        print_and_save(f"### Quantization: {QUNTIZATION_SENARIOS_NAME_MAP[quantization]}")
        for gpu in GPU_LIST:
            print_and_save(f"#### GPU: {gpu} x {GPU_COUNT_MAP[gpu]} ")
            price=GPU_PRICE_MAP[gpu]*GPU_COUNT_MAP[gpu]*7.8
            print_and_save(f"##### Price: ${price:.2f} HKD")
            print_and_save("| Users count | Perfect Latency (s) | Upper Benchmarks Latency (s) | Lower Benchmarks Latency (s) | Average Words per second | Error |")
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

                    print_and_save(f"| {user_count} | {formatted_perfect_latency} | {formatted_upper_benchmarks_latency} | {formatted_lower_benchmarks_latency} | {formatted_benchmarks_wps} | - |")
                except Exception as e:
                    # Handle errors, avoiding table format breakage
                    error_message = str(e).replace("|", ",")
                    print_and_save(f"| {user_count} | - | - | - | - | {error_message} |")
                pass
            print_and_save(f"###### Cost efficiency: ${price / average_benchmarks_total_words_per_sec:.2f} HKD. For Total Price over Average Words per second (Speed).")
            print_and_save("---")
        print_and_save("""\n<div style="page-break-after: always;"></div>\n""")
        