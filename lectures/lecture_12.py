from edtrace import text, link, image
from lecture_util import post_link
from references import mmlu_2021

def main():
    text("## Lecture 12: ??")
    text("- ?? far: we've ??? everything ?? ?? an LM (??, ??, systems, ??).")
    text("- Missing piece: ?? **??** ? you ?? ??")
    text("- ?? shapes ?? ?? (??? multilingual? DNA?).")
    text("- ?? ?? ?? ??, ?? ? ?? ?? ?? ?? we ?? ? a ??.")

    text("**??**: ?? a ??, ?? \"**?**\" ? it?")

    what_is_good()

    perplexity()
    exam_benchmarks()
    chat_benchmarks()
    agentic_benchmarks()
    pure_reasoning_benchmarks()
    safety_benchmarks()

    realism()
    validity()
    how_to_think_about_evaluation()

    text("Takeaways:")
    text("- ?? ? no ?? ? ??; ?? the ?? depending ? ?? you're trying ? ??.")
    text("- Clearly state the rules ? the game (?? versus ?? versus agents).")
    text("- Considerations: difficulty, realism, validity.")


def what_is_good():
    text("?? ?? ?? ? ? a mechanical ??:")
    text("1. Define ?? ???")
    text("2. Send ??? ? a ?? ?? ?? ? ??")
    text("3. ?? accuracy")

    text("?? ???, ?? ? a deep ?? ?? topic...")
    text("...?? shapes the development ? AI.")

    text("**Core challenge**: <font color=\"red\">abstract construct</font> → <font color=\"blue\">concrete metric</font>")

    text("?? a ?? ? ? ?? it ? well ? benchmarks...")
    link(title="Artificial Analysis", url="https://artificialanalysis.ai/")
    image("images/artificial-analysis.png", width=800)

    text("?? a ?? ? ? ?? it ? well ? benchmarks ?? ? ?? ? ??...")
    image("images/artificial-analysis-cost.png", width=800)

    text("?? a ?? ? ? ?? people prefer its ??...")
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)

    text("?? a ?? ? ? ?? people ??? ?? ? ?? (?? pay ??) it...")
    link(title="OpenRouter", url="https://openrouter.ai/rankings")
    image("images/openrouter.png", width=600)


def perplexity():
    text("- Recall: ? a ???? ? a ?? ?? **p(x)** ?? ?? ? token.")
    text("- Perplexity (1/p(D))^(1/|D|) measures whether p assigns ? ?? ? ?? ??? D.")

    text("- ? pre-??, you minimize perplexity ? the ?? set.")
    text("- The obvious thing ? ? ?? perplexity ? the ?? set.")
    text("- ?? ?? people ? traditionally ? ???? ??.")

    text("Standard ???:")
    text("- Penn Treebank (WSJ)")
    text("- WikiText-103 (Wikipedia)")
    text("- ?? Billion Word ???? (? machine translation WMT11 - EuroParl, UN, news)")
    text("Classic paradigm: ?-?? ??: ?? ? ?? ?? ?? ?? ? ?? ?? ? ?? ???.")
    text("Pure CNNs+LSTMs ? the ?? Billion Word ???? (perplexity 51.3 → 30.0) "), link("https://arxiv.org/abs/1602.02410")

    text("GPT-2:")
    text("- ?? ? WebText (40GB ??, websites linked ? Reddit)")
    text("- Zero-shot ? standard ??? (**?-?-??** ??)")
    image("images/gpt2-perplexity.png", width=800)
    text("- Works ?? ? ?? ??? (PTB) ?? transfer ? helpful, ?? ? larger ??? (1BW)")

    text("Perplexity ? ?? you ?? (?? faith ? science):")
    text("- ? ?? ? t, ?? ? p.")
    text("- ?? ?? perplexity ? H(t) obtained iff p = t.")
    text("- ?? p = t, ?? solve ?? the tasks: p(solution | problem)")
    text("- ?? ?? pushing ? ? perplexity, ??? eventually \"reach AGI\".")

    text("Perplexity ? ?? ?? ? you ??:")
    text("- ??: *Stanford ? founded ? 1885*")
    text("- Perplexity penalizes ?? ? ?? token, ?? (e.g., *founded*) ? ?? ?? ? ? relevant")
    text("- Solution: ?? conditional perplexity p(?? | ???)^(1/|??|)")

    text("?? benchmarks ? perplexity ? disguise:")
    text("- Cloze tasks (fill ? the blank): LAMBADA "), link("https://arxiv.org/abs/1606.06031")
    image("images/lambada.png", width=700)
    text("- Multiple choice sentence completion: HellaSwag "), link("https://arxiv.org/pdf/1905.07830")
    image("images/hellaswag.png", width=500)

    text("**??** (?? you're ?? a perplexity leaderboard):")
    text("- People submit `LM` ?? you ?? `log_prob = LM(test_data)`")
    text("- You ?? ? trust ? the ?? ? valid (sum ? 1)")
    text("- ?? downstream tasks, `response = LM(prompt)` ?? ?? accuracy ? `response`")

    text("??:")
    text("- Perplexity ? ?? used heavily ? ???? development (smooth ?? laws)")
    text("- ?? ?? benchmarks ? capture ??-world situations (?? the non-believers)...")


def exam_benchmarks():
    text("Exams ? a ?? way ? ?? ?? ?? (?? ? humans):")
    text("- ? control ?? the subject ?? difficulty")
    text("- Design ? ? unambiguous ?? ??, ?? ? grade")

    text("**Massive Multitask ?? ?? (MMLU)** "), link(mmlu_2021)
    text("- 57 subjects (e.g., math, US history, law, morality), multiple-choice")
    text("- \"collected ?? graduate ?? undergraduate students ? freely ?? sources online\"")
    text("- Despite the name, MMLU ? ?? ?? ?? knowledge, ? ?? ??")
    text("- Evaluated ? GPT-3 ?? ??-shot prompting")
    image("images/mmlu.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu")
    link(title="HELM MMLU for visualizing predictions", url="https://crfm.stanford.edu/helm/mmlu/latest/")

    text("**MMLU-Pro** "), link("https://arxiv.org/abs/2406.01574")
    text("- Removed noisy/trivial questions ? MMLU")
    text("- Expanded 4 choices ? 10 choices")
    text("- Evaluated ?? chain ? thought (gives ?? ?? ? a chance)")
    text("- Accuracy ? ?? drop ?? 16% ? 33% (? ?? saturated)")
    image("images/mmlu-pro.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu-pro")
    link(title="HELM MMLU-Pro for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro")

    text("**Graduate-Level Google-Proof Q&A (GPQA)** "), link("https://arxiv.org/abs/2311.12022")
    text("- Questions written ?? 61 PhD contractors ? Upwork")
    image("images/gpqa.png", width=700)
    text("- PhD experts achieve 65% accuracy")
    text("- Non-experts achieve 34% ?? 30 minutes ? ?? ? Google")
    text("- GPT-4 achieves 39%")
    link("https://llm-stats.com/benchmarks/gpqa")
    link(title="HELM GPQA for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa")

    text("**Humanity's Last Exam (HLE)** "), link("https://arxiv.org/abs/2501.14249")
    text("- 2500 questions: multimodal, ?? subjects, multiple-choice + short-??")
    image("images/hle-examples.png", width=700)
    text("- Awarded $500K prize pool + co-authorship ? ?? creators")
    text("- ???? ?? frontier LLMs, multiple stages ? review")
    image("images/hle-pipeline.png", width=700)
    image("images/hle-results.png", width=600)
    link("https://llm-stats.com/benchmarks/hle")

    text("??:")
    text("- Trend ?? harder questions ?? ?? ?? ?? saturate existing benchmarks")
    text("- Multiple-choice format ?? ? ?? difficult ?? ?? wants")
    text("- ? ? capture ?? usage (open-ended, doesn't necessarily exist ?? ??)")


def chat_benchmarks():
    text("- ?? far, we've ? evaluating ? well-defined multiple-choice tasks.")
    text("- Most people don't ask multiple-choice exam questions ? their AI assistant.")

    text("??:")
    text("???: *I ? ?? ? ? a beet salad ? goat cheese. ?? kind ? herbs ? ?? well ?? ?? ? ? ?? well?*")
    text("??: *??’s a breakdown ? herbs ? ?? well (?? ?? ? don’t) ? a beet + goat cheese salad, based ? ?? their flavors interact ? the sweet-earthiness ? beets ?? the tangy creaminess ? goat cheese...")

    text("**Challenge**: ?? ? ?? an open-ended ???")

    text("**Chatbot Arena** "), link("https://arxiv.org/abs/2403.04132")
    text("?? collection:")
    text("- Random person ? the Internet types ? ???")
    text("- They ?? ?? ? ?? random (anonymized) ??")
    text("- They rate ?? ?? ? ??")
    image("images/arena-beets.png", width=700)
    text("?? ELO rankings based ? pairwise comparisons:")
    text("- Define ??: p(A wins ?? B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))")
    text("- Fit this ?? ? maximize ?? ? pairwise comparisons")
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)
    text("Properties:")
    text("- ??-world ??? (?? ?? ??, incentives ? ??? ?? it)")
    text("- ?? ? ? these people? biases? spammers?")
    text("- Binary preference ?? conflates style ?? ???")
    text("- ?? ? the human even assess ????  Prone ? sycophancy?")
    text("- Feature: don't ?? ? feed ?? ??? ? ?? ?? (?? ?? human ? rating)")
    text("- Dynamic: incorporates ?? ??? ?? ?? ?? ??")

    text("**AlpacaEval** (2023)"), link(title="leaderboard", url="https://tatsu-lab.github.io/alpaca_eval/")
    text("- 805 ?? ? various sources")
    text("- Metric: win rate ?? baseline ?? (GPT-4 preview) ?? judged ?? GPT-4 preview (potential bias?)")
    text("- Problem: LLM judges favor longer ??, resulted ? leaderboard gaming")
    text("- Alpaca Eval 2.0 used regression ? debias the metric "), link("https://arxiv.org/pdf/2404.04475")
    text("- ?? ? we ?? the metric?")
    text("- Correlation ? Chatbot Arena (humans) ? ?:")
    image("https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png", width=500)
    image("images/alpacaeval-leaderboard.png", width=400)

    text("**WildBench** "), link("https://arxiv.org/pdf/2406.04770")
    text("- Sourced 1024 ?? ? 1M human-chatbot conversations")
    text("- ?? GPT-4 turbo ?? a judge ? a checklist (?? CoT ?? judging) + GPT-4 ?? a judge")
    text("- Well-correlated ? Chatbot Arena (seems ? ? the de facto sanity ??)")
    image("images/wildbench.png", width=700)
    link(title="HELM WildBench for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench")

    text("??:")
    text("- Challenge: ?? ? ?? open-ended ???")
    text("- Pairwise comparisons ?? similar ?? ?? ?? signal")
    text("- Beware ? biases (?? ? humans ?? LLM judges)")
    text("- Checklist/rubric improves reliability (regardless ? human ? LLM judge)")


def agentic_benchmarks():
    text("Previously: ?? ?? LMs say (chat)")
    text("??: ?? ?? LMs ? (agents)")

    text("Agent = ???? + agent scaffold (logic ?? deciding ?? ? ?? the LM)")

    text("?? tasks ? require tool ?? (e.g., ?? ??) ?? iterating ?? a period ? ??")

    text("**SWEBench** "), link("https://arxiv.org/abs/2310.06770")
    text("- 2294 tasks ?? 12 Python repositories")
    text("- ?? codebase + issue description, submit a PR")
    text("- ?? metric: unit ??")
    image("images/swebench.png", width=800)
    link("https://llm-stats.com/benchmarks/swe-bench-verified")

    text("**TerminalBench** "), link("https://arxiv.org/abs/2601.11868"), link(title="website", url="https://www.tbench.ai/")
    image("images/terminal-bench.png", width=700)
    text("- Computer terminal environments: ?? ?? universal")
    text("- 229 tasks crowdsourced ? 93 contributors, 89 tasks constitute Terminal-Bench 2.0")
    image("images/terminal-bench-human-time.png", width=600)
    image("images/terminal-bench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/terminal-bench")

    text("**CyBench** "), link("https://arxiv.org/abs/2408.08926")
    image("images/cybench.png", width=700)
    text("- 40 Capture the Flag (CTF) tasks")
    text("- ?? ??-solve ?? ?? a ?? ? difficulty")
    image("images/cybench-agent.png", width=700)
    image("images/cybench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/cybench")

    text("**MLEBench** "), link("https://arxiv.org/abs/2410.07095")
    text("- 75 Kaggle competitions (require ?? ??, processing ??, etc.)")
    image("images/mlebench.png", width=800)
    image("images/mlebench-results.png", width=700)

    text("Agent scaffolds "), post_link("https://www.philschmid.de/agents-2.0-deep-agents")
    image("https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png", width=400)
    text("- Explicit planning: ?? a todo list ? gets checked off")
    text("- Hierarchical delegation: agents calling ?? sub-agents (clean ???)")
    text("- Persistent ??: ??/?? ??")
    text("- Extreme ??? engineering: explicit ?? ?? ? ??")

    text("??:")
    text("- Agents dramatically enhance the capability surface ? ?? ??")
    text("- Agent scaffolds ? ?? ??")
    text("- Evaluating agents = evaluating agent scaffold + ????")


def pure_reasoning_benchmarks():
    text("- ?? ? the tasks ?? far require linguistic ?? world knowledge.")
    text("- ?? we isolate **reasoning** ? knowledge?")
    text("- Arguably, reasoning captures a ?? pure form ? intelligence (isn't ?? ?? memorizing facts).")

    text("**ARC-AGI** "), link(title="website", url="https://arcprize.org/arc-agi")
    text("- 100\% solvable ?? humans, ?? challenging ?? AI")
    text("- ?? task ? unique, ?? memorization doesn't help.")

    text("- ARC-AGI-1 (2019): ?? ??")
    image("https://arcprize.org/media/images/arc-task-grids.jpg", width=800)

    text("- ARC-AGI-2 (March 2025): ?? multi-?? reasoning")
    image("https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png", width=800)

    image("images/arc-agi-results.png", width=700)
    text("- Pretrained ?? ?? didn't move the needle")
    text("- Reasoning ?? (o1, o3) started making things take off")

    text("- ARC-AGI-3 (March 2026): ??? environments "), post_link("https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf")
    image("images/arc-agi-3.png", width=300)
    image("images/arc-agi-3-results.png", width=500)

    text("??:")
    text("- Goal ? ? disentangle reasoning ? knowledge (difficult ? ?!)")
    text("- Constrained ? human reasoning (? superhuman reasoning)")
    text("- Clearly exposes gaps ? ?? ??")


def safety_benchmarks():
    image("https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg", width=400)
    text("?? ? ?? ?? ?? AI?")

    text("**HarmBench** "), link("https://arxiv.org/abs/2402.04249")
    text("- Based ? 510 harmful behaviors ? violate laws ? norms")
    link(title="HarmBench on HELM", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench")
    link(title="Example of safety failure", url="https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4")

    text("**AIR-Bench** "), link("https://arxiv.org/abs/2407.17436")
    text("- Based ? regulatory frameworks ?? company policies")
    text("- Taxonomized ?? 314 risk categories, 5694 ???")
    image("https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png", width=800)
    link(title="HELM AIR-Bench", url="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard")

    text("Jailbreaking:")
    text("- ?? ?? ? ?? ? refuse harmful ??")
    text("- Greedy Coordinate ?? (GCG) ??? optimizes ??? ? bypass ?? "), link("https://arxiv.org/pdf/2307.15043")
    text("- Transfers ? open-weight ?? (Llama) ? closed ?? (GPT-4)")
    image("images/gcg-examples.png", width=800)

    text("?? ? ???")
    text("- ?? aspects ? ?? ? strongly contextual (politics, law, social norms - ?? vary ?? countries)")
    text("- ?? risks ? ?? varied (hallucinations, sycophancy, abetting crimes, inequality, losing critical thinking)")

    text("**Dual-??**: capable cybersecurity agents (Mythos) ?? ? used ? hack ?? a system ? ? ? penetration ??")


def realism():
    text("**Ecological validity**: ?? well ? an ?? capture ??-world ???")
    text("- Exam benchmarks (e.g., GPQA) ? far away ? ??-world ??.")
    text("- Chatbot Arena ??? ? ? ?? people, ?? ?? ? uncontrolled.")

    text("**GDPVal** (OpenAI) "), link("https://arxiv.org/pdf/2510.04374")
    text("- 44 occupations ? top 9 sectors according ? US GDP")
    text("- Tasks come ? professionals ? ~14 years ? experience")
    image("images/gdpval.png", width=700)

    text("**MedHELM** "), link("https://arxiv.org/abs/2505.23802")
    text("- ?? medical benchmarks ? based ? standardized exams")
    text("- 121 clinical tasks sourced ? 29 clinicians, mixture ? private ?? public ???")
    image("https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png", width=700)
    link(title="MedHELM", url="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard")

    text("**Clio** (Anthropic) "), link("https://arxiv.org/abs/2412.13678")
    text("- ?? ?? ?? ? analyze ?? ?? ??")
    text("- Share general patterns ? ?? people ? asking")
    image("images/clio-table4.png", width=700)

    text("Unfortunately, realism ?? privacy ? ?? at odds ? ?? ??.")


def validity():
    text("?? ? we know our evaluations ? valid?")

    text("### ??-?? overlap")
    text("- ???? 101: don't ?? ? your ?? set")
    text("- Pre-foundation ?? (ImageNet, SQuAD): well-defined ??-?? splits")
    text("- ??: ?? ? the Internet ?? don't tell people ?? your ??")

    text("Route 1: ?? ? infer ??-?? overlap ? ??")
    text("- Exploit exchangeability ? ?? points "), link("https://arxiv.org/pdf/2310.17623")
    image("images/contamination-exchangeability.png", width=500)

    text("Route 2: encourage reporting norms (e.g., people report confidence intervals)")
    text("- ?? providers ?? report ??-?? overlap "), link("https://arxiv.org/abs/2410.08385")

    text("Route 3: ?? fresh evals")
    text("- LiveCodeBench, UncheatableEval: scrape ?? webpages")
    text("- Timestamps aren't ?? safe due ? copying ??")

    text("Route 4: ?? private evals")
    text("- Companies ?? internal ?? bases ? aren't ? the Internet")
    text("- ?? your personal writings")
    text("- Easiest ?? perplexity")

    text("### ??? ??")
    text("- ?? ? SWE-Bench ? ?? SWE-Bench Verified "), post_link("https://openai.com/index/introducing-swe-bench-verified/")
    text("- ?? Platinum versions ? benchmarks "), link("https://arxiv.org/abs/2502.03461")
    image("https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096", width=700)
    image("https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096", width=800)
    text("- Problems ? agentic benchmarks: insufficient ?? cases, trivial agent ?? solve task "), link("https://arxiv.org/abs/2507.02825")
    text("- Docent: ?? LLM ? inspect agent traces ? detect problems "), post_link("https://transluce.org/introducing-docent")


def how_to_think_about_evaluation():
    text("### ??'s the point ? ???")
    text("?? ? no ?? ? ??; it depends ? ?? ?? you're trying ? ??.")
    text("1. ?? ? company wants ? ? a purchase decision (?? A ? ?? B) ?? their ?? case (e.g., customer ?? chatbots).")
    text("2. ???? ?? ? ?? the raw capabilities ? a ?? (e.g., intelligence).")
    text("3. We ?? ? ?? the benefits + harms ? a ?? (?? business ?? policy ??).")
    text("4. ?? developers ?? ? ?? feedback ? ?? the ??.")

    text("### ?? ? we evaluating?")
    text("- Pre-foundation ??, we evaluated **??** (standardized ??-?? splits).")
    text("- ??, we're (mostly) evaluating **??/systems** (anything goes).")

    text("?? ? ?? exceptions...")
    text("- nanogpt speedrun: ?? ??, ?? ?? ? ?? ? a particular validation ??")
    image("images/karpathy-nanogpt-speedrun.png", width=600), post_link("https://x.com/karpathy/status/1846790537262571739")

    text("Evaluating ?? encourage algorithmic innovation ? ????.")
    text("Evaluating ??/systems ? ?? ?? downstream ??.")

    text("?? way, ???? ? define the rules ? the game!")


if __name__ == "__main__":
    main()
