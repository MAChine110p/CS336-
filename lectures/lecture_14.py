from dataclasses import dataclass
import numpy as np
import itertools
import mmh3
from edtrace.file_util import download_file
from edtrace import text, image, link
from lecture_13 import the_pile
from lecture_util import article_link, post_link
from references import dolma_2024, the_pile_2020, dclm_2024

def main():
    text("## Lecture 14: ?? II")
    text("???:")
    text("- ?? ?? (e.g., GitHub) → ??/?? (e.g., GitHub Archive) → ???? ?? (e.g., The Stack)")
    text("- Considerations: terms ? ??, copyright (licenses ? fair ??)")

    text("??:")
    text("- ?? pipeline: transformation, ??, deduplication, mixing")
    text("- Mid-?? + SFT: synthetic ??")

    # ?? pipeline
    transformation()
    filtering()
    deduplication()
    data_mixing()

    # Post-?? ??
    post_training_data()

    text("??:")
    text("- ??: ?? classifier (?? id, ??, toxicity) ?? ?? ? looks ??")
    text("- Deduplication: hashing scales ? ?? ??? ?? fuzzy matching")
    text("- Mixing: ?? mixtures at ?? ??, extrapolate ? optimal mixture ?? ?? ??")
    text("- ??: ?? identification, ?? ??, toxicity ??")
    text("- Post-?? ??: looks ?? evaluations, ?? ? synthetic ??")
    text("- A lot ? ?? ?? ? domain-specific, looking at ??, etc.")


def transformation():
    text("Raw ?? ? ? come ?? ??.")
    text("It ? HTML, PDF (arxiv), ? directories (?? repositories).")

    text("HTML ? ?? (main ??):")
    text("- Remove boilerplate (e.g., navigation, ads) ?? extract ??")
    text("- ?? ?? images, ?, etc.?")
    text("- Inherently lossy (?? ? linearize)")
    text("- Tools (rule-based): trafilatura, resiliparse, jusText, lynx, etc.")
    text("- Accuracy matters: "), link(dclm_2024)
    image("images/dclm-wet.png", width=300)

    text("FinePDFs "), post_link("https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog")
    image("https://huggingfacefw-finepdfsblog.hf.space/_astro/pdf-description.Cb49jXc6_Z17eX4E.webp", width=600)
    text("- Source: ?? ??")
    text("- Recrawl truncated PDFs (since they ? ??)")
    text("- OCR (RolmOCR) ?? a VLM ? Docling (? these ?? ??)")
    text("- Lots ? cleanup ?? ??")
    text("- A lot ? layout information ? missing")


def filtering():
    text("Algorithmic ?? ?:")
    text("- ?? ?? **?? ??** T ?? lots ? **raw ??** R, ?? subset T' ? R similar ? T.")
    image("images/raw-target-schema.png", width=600)

    text("??:")
    text("- ?? identification (?? versus rest)")
    text("- ?? ?? (? ?? versus ? ??)")
    text("- Toxicity ?? (non-toxic versus toxic)")

    text("Desiderata ?? ?? algorithm:")
    text("- ?? ? the ?? ?? (?? T ?? T' ? ? ??)")
    text("- Extremely ?? (? ? ?? it ? R, ?? ? huge)")

    text("Survey ?? ? ?? selection "), link("https://arxiv.org/abs/2402.16827")

    text("General framework: ?? ?? T ?? raw R, ?? subset ? R similar ? T")
    text("1. Estimate ?? ?? based ? R ?? T ?? derive a scoring ??")
    text("2. ?? ?? ? R based ? their ??")

    text("Types ? classifiers:")
    text("- Generative ?? ? T (KenLM): ??(x) = p_T(x)")
    text("- ?? classifier (fastText): ??(x) = p(T | x)")
    text("? ??: ?? ?? x ? ??(x) >= threshold (stochastically)")

    text("??-based ???")
    text("- ?? deliberately ? ? ?? ??-based ?? (C4, Gopher, RefinedWeb, FineWeb, Dolma)")
    text("- ?? ?? ??-based ?? (GPT-3, LLaMA, DCLM) [???? the ??]")

    text("?? identification:")
    text("- Goal: ?? ?? ? a specific ?? (e.g., ??)")
    text("- fastText ?? identification "), article_link("https://fasttext.cc/docs/en/language-identification.html")
    text("- Off-the-shelf classifier")
    text("- Supports 176 languages")
    text("- ?? ? multilingual sites: Wikipedia, Tatoeba (translation site) ?? SETimes (Southeast European news)")
    text("- Dolma keeps pages ? p(??) >= 0.5 "), link(dolma_2024)

    text("OpenMathText "), link("https://arxiv.org/pdf/2310.06786")
    text("- Goal: curate ?? corpus ? mathematical ?? ? CommonCrawl")
    text("- ?? rules ? filter (e.g., contains latex commands)")
    text("- KenLM ?? ? ProofPile, ?? ?? perplexity < 15000")
    text("- ?? fastText classifier ? ?? mathematical ??, threshold ? 0.17 ?? math, 0.8 ?? no math")
    text("- ??: produced 14.7B token, used ? ?? 1.4B ?? ? ? ?? ? ?? ?? ? 20x ??")

    text("GPT-3 "), link("https://arxiv.org/pdf/2005.14165")  # Appendix A
    text("- Positives: samples ? {Wikipedia, WebText2, ??1, ??2}")
    text("- Negatives: samples ? CommonCrawl")
    text("?? linear classifier based ? word features "), article_link("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    text("?? ?? stochastically based ? ??")
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score

    text("LLaMA/RedPajama "), link("https://arxiv.org/pdf/2302.13971")
    text("- Positives: samples ? pages **referenced** ?? Wikipedia")
    text("- Negatives: samples ? CommonCrawl")
    text("- ?? ?? ? ? classified positive")

    text("phi-1 "), link("https://arxiv.org/pdf/2306.11644")
    text("- Philosophy: ?? ? ?? ?? (textbooks) ? ?? a ?? ?? (1.5B)")
    text("- Includes synthetic ?? ? GPT 3.5 (later: GPT-4) ?? ???? ??")
    R = "Python subset of the Stack"   # Raw data
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
    text("- ?? random forest classifier ? T ?? ?? ?? ? pretrained codegen ??")
    text("- ?? ?? ? R ? ? classified positive ?? the classifier")
    text("?? ? [HumanEval](https://huggingface.co/datasets/openai_humaneval):")
    text("- ?? 1.3B LM ? Python subset ? The Stack (??: 12.19% ?? 96K ??)")
    text("- ?? 1.3B LM ? ?? ???? subset (??: 17.68% ?? 36K ??) - ??!")

    text("Toxicity ?? ? Dolma "), link(dolma_2024)
    text("- ???: Jigsaw Toxic Comments ??? (2018) "), link(title="dataset", url="https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")
    text("- Project goal: help people ? ?? discussions online "), article_link("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    text("- ??: comments ? Wikipedia ?? page annotated ? {toxic, severe_toxic, obscene, threat, insult, identity_hate}")

    text("??-dependent effects ? ??:")
    text("- No single optimal threshold ?? ??")
    text("- ?? ?? ?? longer, ?? ?? (?? ??) ??")
    text("- ?? ?? ?? shorter, ?? ?? (?? ??) ??")
    image("images/data-filtering-scale.png", width=800)

    text("??:")
    text("- ?? ? critical ?? ?? a ? ??")
    text("- Recipe: define ?? ?? (?? ? looks ??), extrapolate ? raw ??")


def deduplication():
    text("?? types ? duplicates:")
    text("- ?? duplicates (mirror sites, GitHub forks) "), link(title="Gutenberg mirrors", url="https://www.gutenberg.org/MIRRORS.ALL")
    text("- Near duplicates: ?? ?? differing ?? a ?? token")

    text("?? ? near duplicates:")
    text("- Terms ? ?? ?? licenses "), link(title="MIT license", url="https://opensource.org/license/mit")
    text("- Formulaic ?? (copy/pasted ? generated ? a template) "), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png", width=600)
    text("- Minor formatting differences ? copy/pasting")

    text("Product description repeated 61,036 times ? C4")
    text("'“?? combining fantastic ??, interesting arrangements, ?? follow the ?? trends ? the field ? ? ? you ?? inspired ?? ?? artistic touches. We’d ? honored ?? ??? ?? ?? ? ?? ? these design ? your wedding.  believe me, brilliant ?? ? ? perfect ?? it ?? ? applied ? ?? ?? ? the people ?? you amazed!")
    link(title="example page", url="https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")

    text("Deduplication ?? ?? makes ?? ?? ?? "), link("https://arxiv.org/pdf/2107.06499")
    text("- ?? ?? efficiently (?? ? fewer token)")
    text("- Avoid memorization (?? mitigate copyright, privacy concerns)")

    text("Design space:")
    text("1. ?? ? an item (sentence, paragraph, ??)?")
    text("2. ?? ? match (?? match, existence ? ?? subitem, fraction ? ?? subitems)?")
    text("3. ?? action ? take (remove ??, remove ?? ?? ??)?")

    text("Key challenge:")
    text("- Deduplication ? fundamentally ?? comparing items ? ?? items")
    text("- ?? linear ?? algorithms ? ??")

    hash_functions()
    exact_deduplication()
    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    text("- Hash ?? h maps item ? a hash ? (integer ? string)")
    text("- Hash ? ?? smaller ? item")
    text("- Hash collision: h(x) = h(y) ?? x ≠ y")

    text("Tradeoff ?? efficiency ?? collision resistance "),  article_link("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")
    text("- Cryptographic hash ?? (SHA-256): collision resistant, ?? (used ? bitcoin)")
    text("- DJB2, MurmurHash, CityHash: ? collision resistant, ?? (used ?? hash ?)")

    text("??? ?? MurmurHash:")
    h = mmh3.hash("hello")  # @inspect h


def exact_deduplication():
    text("**?? ??**")
    text("1. Item: string")
    text("2. ?? ? match: ?? match")
    text("3. Action: remove ?? ?? ??")

    # Original items
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items

    # ?? hash -> list ? items ? that hash
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

    # ?? one item ? ?? group
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items

    text("- Pro: ??, clear semantics, ? precision")
    text("- Con: ? ? deduplicate near duplicates")
    text("- This ?? ? written ? a MapReduce way, ?? ??? parallelize ?? ??")

    text("**C4** "), link("https://arxiv.org/pdf/1910.10683v4")
    text("1. Item: 3-sentence spans")
    text("2. ?? ? match: ?? ?? match")
    text("3. Action: remove ?? ?? ??")
    text("??: ?? a 3-sentence span ? removed ? the middle ? a ??, the resulting ?? ?? ? ? coherent")


def jaccard_minhash():
    text("?'s ?? ?? at approximate set membership.")
    text("?? ???? a similarity ??.")

    text("### Jaccard similarity")
    text("Definition: Jaccard(A, B) = |A intersect B| / |A union B|")
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    jaccard = compute_jaccard(A, B)  # @inspect jaccard

    text("Definition: ?? ?? ? **near duplicates** ?? their Jaccard similarity >= threshold")

    text("Algorithmic challenge: ?? near duplicates ? linear ??")

    text("### MinHash")
    text("MinHash: a random hash ?? h ?? Pr[h(A) = h(B)] = Jaccard(A, B)")

    text("Normally, you ?? ?? items ? hash ? ?? hashes")
    text("...?? ??, you ?? collision ?? ? depend ? similarity")

    def minhash(S: set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    text("Characteristic ?? representation:")
    text("item | A | B", verbatim=True)
    text("1    | 1 | 1", verbatim=True)
    text("2    | 1 | 1", verbatim=True)
    text("3    | 1 | 1", verbatim=True)
    text("4    | 1 | 0", verbatim=True)
    text("5    | 0 | 1", verbatim=True)

    text("Random hash ?? induces a ?? ?? items")
    text("?? at ?? item ? ?? ? A ?? ?? item ? ?? ? B.")
    text("?? item ? the ?? ?? ?? ?? ?? (min)")
    text("- ?? 1, 2, 3 ? ??, ?? ?? ? A = ?? ? B.")
    text("- ?? 4, 5 ? ??, ?? ?? ? A ≠ ?? ? B.")

    # Verify MinHash approximates Jaccard ?? advertised
    n = 100  # Generate this many random hash functions
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]  # @stepover
    estimated_jaccard = len([m for m in matches if m]) / len(matches)  # @inspect estimated_jaccard
    assert abs(estimated_jaccard - jaccard) < 0.01

    text("?? ???? hash our items, ?? a collision doesn't tell us Jaccard(A, B) > threshold.")


def locality_sensitive_hashing():
    text("Locality sensitive hashing (LSH) "), link(title="book chapter", url="http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")

    text("?? we hash ?? ? ?? ?? MinHash ??")
    text("P[A ?? B collide] = Jaccard(A, B)")
    text("? ??, ?? similar items ? collide, ?? ?? stochastic...")

    text("Goal: ? A ?? B collide ?? Jaccard(A, B) > threshold")
    text("We ? ? somehow sharpen the ??...")

    text("Solution: ?? n hash ??")
    text("Break ? ?? b bands ? r hash ?? ?? (n = b * r)")

    n = 12      # Number of hash functions
    b = 3       # Number of bands
    r = 4       # Number of hash functions per band
    text("Hash ??:")
    text("h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12", verbatim=True)

    text("Key: A ?? B collide ?? ?? *??* band, *??* its hash ?? ?? ?? ?")
    text("?? ??? ??, the ??-? structure ? the bands sharpens the threshold")

    text("?? Jaccard(A, B), ?? ? the ?? ? A ?? B collide?")

    def get_prob_collision(sim, b, r):  # @inspect sim @inspect b @inspect r
        prob_match = sim ** r                        # Probability that a fixed band matches  @inspect prob_match
        prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches  @inspect prob_collision
        return prob_collision

    text("**??**")
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)  # @inspect prob_collision
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png", width=600)


    sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}  # @inspect probs @stepover

    text("Increasing r sharpens the threshold ?? moves the curve ? the right (harder ? match)")
    probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}  # @inspect probs @stepover

    text("Increasing b moves the curve ? the left (easier ? match)")
    probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}  # @inspect probs @stepover
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png", width=600)

    text("?? setting "), link("https://arxiv.org/pdf/2107.06499"), text(": n = 9000, b = 20, r = 450")
    b = 20
    r = 450
    text("?? ? the threshold (?? phase transition ??)?")
    threshold = (1 / b) ** (1 / r)  # @inspect threshold

    text("?? ? a ?? band matches:")
    prob_match = (1 / b)  # @inspect prob_match
    text("?? ? A ?? B collide ? a ?? (≈ 1-1/e):")
    prob_collision = 1 - (1 - 1 / b) ** b  #  @inspect prob_collision


def billion(x):
    return x * 10**9

def trillion(x):
    return x * 10**12


def data_mixing():
    text("Recall ? ?? ?? ? ?? ? multiple ?? sources.")

    text("??? ? Marin: "), link(title="token viewer", url="https://huggingface.co/spaces/marin-community/token-count-viewer")
    image("images/marin-token-viewer.png", width=800)

    text("The Pile "), link(the_pile_2020)
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)
    text("Key ??: ?? ?? ?? the ?? sources ?? we ???")

    text("??:")
    sources = {"Wikipedia", "CC", "GitHub"}
    p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}  # One possible data mixture

    text("Baselines:")
    text("- Vibes: set p(s) ??? based ? intuition (?? ??)")
    text("- Uniform sampling: ?? uniformly (p(s) ∝ 1)")
    text("- Proportional mixing: ?? proportional ? the ?? ? token ? a source (p(s) ∝ num_token(s))")

    text("Intuition: ?? upweight ?? ?? sources")
    text("??...")
    text("1. We ?? ? ?? diversity (e.g., ?? incomparable sources: literature, ??, ??)")
    text("2. ?? source ? finite, ?? ?? put ? ?? weight ? a ?? source, ?? ?? ? ?? ?? it")

    text("This last point ? ?? ?? a bit subtle.")
    text("??:")
    source_token_counts = {
        "low": trillion(10),  # 10T tokens (abundant) @stepover
        "high": billion(10),  # 10B tokens (scarce) @stepover
    }
    p = {"low": 0.5, "high": 0.5}  # Naive data mixture
    train_tokens = trillion(1)  # Train for 1T tokens @stepover
    low_num_epochs = (p["low"] * train_tokens) / source_token_counts["low"]  # @inspect low_num_epochs
    high_num_epochs = (p["high"] * train_tokens) / source_token_counts["high"]  # @inspect high_num_epochs
    text("50x ?? ? ? ?? ??...?? lead ? overfitting!")

    text("UniMax "), link("https://arxiv.org/abs/2304.09151")
    text("- Setting: balancing ?? languages ?? multilingual ??")
    text("- ?? ??: ?? uniform ?? proportional mixing (p(s) ∝ num_token(s)^α ?? α ? [0, 1])")
    text("- ??: ?? sources uniformly ?? ? a ?? **cap** C ? ?? ? ?? ?? ?? source")
    text("- ???, p(s) * num_??_token ≤ C ?? ?? sources s")

    text("Regression-based mixing "), link("https://arxiv.org/abs/2407.01492"), link("https://arxiv.org/pdf/2602.12237")
    image("images/regmix.png", width=700)
    text("- Define ?? ?? mixtures `p` (e.g., Dirichlet) ")
    text("- Define regression ?? (e.g., linear, ?? boosted trees)")
    text("- Define ?? based ? downstream evals (careful ? ? overfit!)")
    text("- Discrepancy ?? ?? ?? ?? ?? (tradeoff cost ?? accuracy)")
    image("images/data-mixing-methods.png", width=700)
    text("Hope 1: regression ?? ? accurate at minimizer 🙏")
    text("Hope 2: optimal ?? mixtures transfer ? ?? ? ?? ?? 🙏")

    text("Hold ?. ??'s at least ?? ??-dependent effect:")
    source_token_counts = {
        "low": trillion(10),  # 10T tokens (abundant) @stepover
        "high": billion(10),  # 10B tokens (scarce) @stepover
    }
    text("- ?? ?? ?? ?? ? ? token counts:")
    p = {"low": 0.1, "high": 0.9}  # More mass on high quality data
    text("- ?? ?? ?? ?? ?? ? this mixture, ??? ?? a ton ? ? ?? ?? ?? overfit!")

    text("Simulated epoching "), link("https://arxiv.org/pdf/2501.11747")
    text("- General ??: ? ?? ?? ?? ?? ?? ?? (general theme ? this course)")
    text("- Instantiation: downsample ?? sources proportionally")
    small_run_tokens = billion(10)  # @stepover
    large_run_tokens = trillion(1)  # @stepover
    ratio = small_run_tokens / large_run_tokens  # @inspect ratio
    downsampled_source_token_counts = {s: count * ratio for s, count in source_token_counts.items()}  # @inspect downsampled_source_token_counts
    text("- ? this downsampled mixture, ?? ? ?? ? ?? won't ?? ?.")
    text("- ?? the optimum ? ? ?? balanced.")
    p = {"low": 0.7, "high": 0.3}  # More mass on high quality data

    text("??:")
    text("- Problem: ?? ? weight ?? ?? sources (e.g., Wikipedia, general, ??)")
    text("- Regression-based mixing: estimate mixture → ?? at ?? ??, optimize (analogous ? ?? laws)")
    text("- ?? consideration: epoching ?? overfitting (solution: cap ? simulated)")


def post_training_data():
    text("Recipe:")
    text("1. Define a set ? environments")
    text("2. Define a set ? tasks / ???")
    text("3. Collect ?? ? a strong ?? (teacher)")

    text("OpenThoughts "), link("https://arxiv.org/abs/2506.04178")
    text("- 1.2M ?? ?? QwQ-32B ?? a teacher")
    text("- Questions come ? 27 human ?? synthetic sources (e.g., StackExchange, NuminaMath, Chemistry)")
    image("images/openthoughts-sources.png", width=500)
    text("- Sampling multiple (16) ?? ? ??? ? helpful")
    text("- ?? ?? aren't necessarily ?? teachers: QwQ-32B ? a ?? teacher ? DeepSeek-R1")
    text("- ?? ?? wasn't helpful")
    text("- Smaller ? ?? sources (e.g., OpenMath-2-Math) ? ?? ? ?? diverse sources")
    image("images/openthoughts-pipeline.png", width=600)

    text("SWE-smith "), link("https://arxiv.org/abs/2504.21798")
    image("images/swe-smith.png", width=500)
    text("- ?? a repository, ?? LM ? ?? tasks (introduce bugs ? LM)")
    text("- 128 GitHub repositories yields 50K tasks")

    text("SWE-Zero "), link("https://arxiv.org/abs/2604.01496")
    text("- SWE tasks ? heavy dependencies (unlike math ? coding contests)")
    text("- Setting ? thousands ? Docker images ? an infrastructural nightmare")
    text("- Observation: strong ?? ?? solve ?? tasks ??? execution feedback")
    image("images/swezero-noexec.png", width=600)
    text("Key: strong ?? ? internal \"world ??\" ? ?? semantics")
    text("- SWE-Zero: 300K agent trajectories ? don't require repository-specific execution")
    text("- 150K GitHub PRs")
    text("- OpenHands scaffold, remove ?? git commits ? prevent \"git hacking\" ?? agent")
    image("images/swezero-prompt.png", width=600)
    text("- Distilled ? Qwen3-Coder-480B + ?? (?? ? execute anyway)")
    text("- SWE-Hero: 13K agent trajectories ? ? require execution feedback")
    image("images/swezero-results.png", width=700)

    text("SWE-rebench "), link("https://arxiv.org/pdf/2505.20411")
    text("- 21K ??? Python SWE tasks ? 3.4K GitHub repositories")
    text("- 450K PRs ? GitHub ?? GitHub Archive")
    text("- Used Qwen 2.5-72B-Instruct ? install dependencies ?? assess PR ??")
    image("images/swe-rebench.png", width=600)

    text("SWE-ZERO-12M-trajectories "), link(title="data", url="https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories")
    text("- ?? SWE-Zero ? ? 12M agent trajectories")
    text("- Used the SWE-rebench-v2 tasks (32K executable tasks + 120K nonexecutable tasks)")
    text("- Ran mini-coder-1.7b (?? ?? ??, 50.4 ??@100), mini-swe-agent scaffold")
    text("- [Example](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories/viewer/default/train?row=5&conversation-viewer=0)")

    text("??:")
    text("- Generating ???: fully-synthetic, semi-synthetic (?? environment + synthetic tasks), ?? (GitHub PRs)")
    text("- ??: ? capable ?? (? ? ? ? teachers)")
    text("- ?? environments ? painful")
    text("- Lots ? ?? ?? ?? details")


if __name__ == "__main__":
    main()
