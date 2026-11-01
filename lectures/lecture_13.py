from edtrace import text, image, link
from lecture_util import article_link
from references import dclm_2024, nemotron_cc_2024, olmo_2_2025, llama_3_2024, gpt2_2019, openwebtext_2019, gopher_2021, alpaca_2023


def main():
    text("## Lecture 13: ?? I")
    text("?? lectures: ?? ? ?? a ?? *?? ??*")
    text("??? ?? lectures: *?? ??* ?? we ?? ??")

    motivation()

    # Origin ? ??
    raw_sources()    # What does data come from?
    copyright()      # What data can we use?

    # Sources ? ??
    common_crawl()   # Web crawl
    wikipedia()      # General knowledge
    github()         # Code
    arxiv()          # Research papers

    # ?? ? various ??
    bert()                # Wikipedia, books (trained BERT) [2019]
    gpt2_webtext()        # pages based on Reddit links (trained GPT-2) [2019]
    ccnet()               # Filter Common Crawl based on Wikipedia [2019]
    t5_c4()               # Filter using rules (trained T5) [2019]

    gpt3()                # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]
    the_pile()            # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]
    gopher_massivetext()  # Filter using rules (trained Gopher) [2021]
    llama()               # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]
    refinedweb()          # CommonCrawl (used to train Falcon) [2023]
    dolma()               # Lots of different sources [2024]
    dclm()                # Filtered using good quality classifier [2024]
    nemotron_cc()         # Lots of tokens [2024]
    the_stack()           # Code dataset
    common_pile()         # Properly licensed data

    text("### ??")
    text("- Key ??: ?? ? ? ?? ? the ??. You ? ? ?? ? ?? it.")
    text("- ?? ?? → raw ?? → ???? ?? (transformation, ??, deduplication)")
    text("- ?? ? the key ingredient ? differentiates ?? ??")
    text("- Legal ?? ethical issues (e.g., copyright ?? privacy)")
    text("- ?? ? this pipeline ? heuristic, ?? opportunities ? ??!")


def motivation():
    text("**??** ? the most ?? thing ? ?? right ? ?? ?? ??.")

    text("?? justification: ?'s ?? ?? companies disclose.")
    text("Open-weight ?? (e.g., Llama 3 "), link(llama_3_2024), text(" ? ?? transparency ?? ??")
    text("...?? even ?? procedures")
    text("...?? basically no information ? ??.")
    image("images/llama3-data.png", width=700)

    text("?? ?? secrecy:")
    text("1. Competitive dynamics")
    text("2. Copyright liability")

    text("- ?? foundation ??, ?? ?? meant heavy annotation ? labeled ?? ?? ?? ??.")
    text("- ?? ??'s ?? annotation, ?? ??'s ?? a lot ? curation ?? cleaning.")
    text("- ?? ? fundamentally a long-tail problem, scales ? human effort (unlike ??, systems).")

    text("Stages ? ??:")
    text("1. Pre-??: ?? ? raw ?? (e.g., ?? ? the web)")
    text("2. Mid-??: ?? ?? ? ? ?? ?? ? enhance capabilities")
    text("3. Post-??: ?? ? chat transcripts ? ?? ??")
    text("???, the lines ? blurry ?? ?? ?? ? ?? stages")
    text("...?? the basic trend ? throughout ??, we go ?")
    text("?? amounts ? ?? ?? ?? ?")
    text("?? amounts ? ? ?? ??.")

    text("Terminology:")
    text("- Base ??: ?? pre-?? + mid-??")
    text("- Instruct/chat ??: ?? post-??")
    text("(Increasingly, base ?? ? ? released - e.g., Qwen3.5-397B-A17B ? an instruct ??.)")

    text("?? (OLMo ? AI2) "), link(olmo_2_2025)
    text("1. **Pre-??**")
    image("images/olmo2-pretraining.png", width=600)
    text("2. **Mid-??**")
    image("images/olmo2-dolmino.png", width=600)
    text("3. **Post-??** "), link("https://arxiv.org/pdf/2411.15124")
    image("images/tulu.png", width=600)

    text("?? ? these ????  ?? ? they chosen ?? ?????")


def raw_sources():
    text("?? ?? ?? hear: *?? ?? ? ?? ? the entire Internet*.")
    text("Slightly ?? accurately, ~Internet~ public (world wide) web.")
    text("?? ?? ? ?? right ??...")

    text("??, the web consists ? a set ? ?? servers ? ?? ?? connect ?:")
    text("`$ curl https://cs336.stanford.edu/`")

    text("???'t ?? ? ?? servers.")
    text("A **crawler**:")
    text("- Discovers webpages (starting ? a seed set)")
    text("- Downloads the discovered webpages")

    text("??, ???'t download ?? ?? ? ?? the webpages.")

    text("Dynamic ??:")
    text("- ?? sites these days ? apps")
    text("- URL doesn't change")
    text("- ?? ? click buttons ?? submit forms ? ?? ??")
    text("- ??: Discord, wandb")

    text("Authentication:")
    text("- ?? ?? login ? an account (?? pay ??)")
    text("- ??: Facebook, X, LinkedIn, NYTimes (huge ?? behind walled gardens)")

    text("Technical restrictions:")
    text("- ? allowed ? download ?? ?? based ? `robots.txt` ([example](https://www.nytimes.com/robots.txt)) (voluntary)")
    text("- Website ?? ?? Cloudflare ? detect ?? ? bot activity (present CAPTCHAs)")
    text("- Website ?? ? certain IP addresses / countries")
    text("- Website ?? ? rate limits")

    text("Legal restrictions:")
    text("- Terms ? ?? (ToS) ?? prohibit downloading ?? bots")
    text("- You ?? ? ? a license ? copy the webpages (?? ??)")

    text("Decline ? consent "), link("https://arxiv.org/abs/2407.14933")
    text("- Examined restrictions (robots.txt, ToS) ?? URLs ? ?? ??? (C4, RefinedWeb, Dolma)")
    text("- Restrictions ? increased ?? ??")
    image("images/decline-consent.png", width=700)

    text("?? crawlers ? ? well-behaved:")
    image("images/anthropic-crawling.png", width=500)
    text("- Factors: ToS, robots.txt, server ?? (degrades ??, costs website money)")
    text("- ?? ?? ?? ? copyright (?? later)...")

    text("Shadow libraries "), article_link("https://en.wikipedia.org/wiki/Shadow_library")
    text("- Technically part ? the web")
    text("- ??: Library Genesis (LibGen), Z-Library, Anna's Archive, Sci-Hub")
    text("- Disregards copyright ?? bypasses paywalls (e.g., Elsevier)")
    text("- Received takedown orders, lawsuits, blocked ? various countries")
    text("- ?? controls ? circumvented, ? servers ? various countries")
    text("- ?? argue this makes freely ?? ?? ?? ? ??")
    text("- ? a legal perspective, ?? piracy ?? copyright infringement")
    text("- LibGen ? ~4M ?? (2019), Sci-Hub ? ~88M ?? (2022)")

    text("??:")
    text("- The Internet ? huge")
    text("- ?? technical ?? legal restrictions ? ?? ?? ?? ?? ??")


def copyright():
    text("?? ?? ? legal ? ?? (?? ??)?")

    text("### Intellectual property law")
    text("- Goal: *incentivize* the creation ? intellectual goods")
    text("- Types ? intellectual property: copyright, patents, trademarks, trade secrets.")

    text("**Copyright law**:")
    text("- Goes ? ? 1709 ? England (Statute ? Anne), ?? ?? regulated ?? governments ?? courts "), article_link("https://en.wikipedia.org/wiki/Statute_of_Anne")
    text("- ? United States, most recent: Copyright Act ? 1976 "), article_link("https://en.wikipedia.org/wiki/Copyright_Act_of_1976")
    text("- Copyright protection applies ? *'original works ? authorship ?? ? ?? tangible medium ? expression, ?? known ? later developed, ? ?? they ?? ? perceived, reproduced, ? otherwise communicated, ?? ??? ? ? the aid ? a machine ? ??'*")

    text("- Collections ? ? original works ?? hence ? copyrightable (e.g., telephone directories) ?? ?? ? ?? creativity ? the selection ? arrangement")
    text("- Copyright applies ? expression, ? ?? (e.g., quicksort)")

    text("- Expanded scope ? 'published' (1909) ? '??' (1976)")
    text("- Registration ? required ?? copyright protection (? contrast ? patents)")
    text("- Threshold ?? copyright ? extremely ? (e.g., your website ? copyrighted)")

    text("- Registration ? required ?? creator ?? sue someone ?? copyright infringement")
    text("- Costs $65 ? ??? "), article_link("https://www.copyright.gov/about/fees.html")
    text("- Lasts ?? 75 years, ?? ?? the copyright expires ?? it becomes part ? the public domain (works ? Shakespeare, Beethoven, most ? Project Gutenberg, etc.)")

    text("??: *basically everything ? the Internet ? copyrighted.*")

    text("?? ? ?? a copyrighted ??:")
    text("1. ?? a license ?? it.")
    text("2. Appeal ? the fair ?? clause.")

    text("### Licenses")
    text("- A license (? contract law) ? granted ?? a licensor ? a licensee.")
    text("- Effectively, 'a license ? a promise ? ? sue'.")

    text("- The Creative Commons license enables ?? ?? ? copyrighted ??.")
    text("- ??: Wikipedia, Open Courseware, Khan Academy, ?? Music Archive, 307 million images ? Flickr, 39 million images ? MusicBrainz, 10 million videos ? YouTube, etc.")
    text("- Created ?? Lessig ?? Eldred ? 2001 ? bridge public domain ?? existing copyright")

    text("?? ?? developers license ?? ?? ?? foundation ??")
    text("- Google ?? Reddit "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- OpenAI ?? Shutterstock "), article_link("https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year")
    text("- OpenAI ?? StackExchange "), article_link("https://stackoverflow.co/company/press/archive/openai-partnership")

    text("**Fair ?? (section 107)**:")
    text("Four factors ? determine whether fair ?? applies:")
    text("1. The purpose ?? character ? the ?? (educational favored ?? commercial, transformative favored ?? reproductive)")
    text("2. The nature ? the copyrighted ?? (factual favored ?? fictional, non-creative ?? creative)")
    text("3. The amount ?? substantiality ? the portion ? the original ?? used (?? a snippet favored ?? ?? the whole ??)")
    text("4. The effect ? the ?? upon the market (? potential market) ?? the original ??")

    text("?? ? fair ??:")
    text("- You watch a movie ?? ?? a ?? ? it")
    text("- Reimplement an algorithm (??) ??? copying the ?? (the expression)")
    text("- Google ?? index ?? ?? snippets (Authors Guild v. Google 2002-2013)")

    text("Copyright ? ? ?? verbatim memorization:")
    text("- ?? ?? characters (e.g., Harry Potter) ?? ? copyrightable")
    text("- Parody (imitating ? ? fun ? something) ? ?? fair ??")
    text("Copyright ? ?? semantics (?? economics).")

    text("Considerations ?? ?? ??:")
    text("- Copying ?? (?? ?? ? ??) ? violation ?? ?? you don't ? anything ? it.")
    text("- ?? a ?? ?? ? transformative (far ? ?? copy/pasting).")
    text("- ?? ?? ? ?? the general ?? (e.g., wizards), ? ? the concrete expression (e.g., Harry Potter).")
    text("- ?? ?? ?? definitely affect the market (writers, artists), regardless ? copyright")

    text("**Terms ? ??**:")
    text("- ?? you ? a license ? ?? appeal ? fair ?? ?? a ??, terms ? ?? ?? impose additional restrictions.")
    text("- ??: YouTube's terms ? ?? prohibits downloading videos, ?? the videos ? licensed ?? Creative Commons.")

    text("### Lawsuits")
    text("The ?? York Times v. OpenAI (2023)")
    text("- Allegation: ?? ?? ?? reproducing NYT articles")

    text("Authors (Bartz, Graeber, ...) v. Anthropic (2024):")
    text("- Allegation: ?? pirating millions ? ?? ?? ?? ? plaintiff's ??")
    text("- ?? judgement (2025): ?? ? plaintiff's works ? fair ??")
    text("- ...?? pirating copies ? ? (?? don't ??)")
    text("- Anthropic ? bought ?? scanned the ??; ?? ? fair ?? (?? ? late)")
    text("- Outcome: Anthropic paid $1.5B ? authors ? settle")

    text("Authors (Kadrey, Silverman, ...) v. Meta ")
    text("- Allegation: ?? ?? ? plaintiff's ?? (revealed ? the Llama ??)")
    text("- ?? judgement (2025): ?? ? ?? (? this instance) ? fair ?? "), article_link("https://techcrunch.com/2025/06/25/federal-judge-sides-with-meta-in-lawsuit-over-training-ai-models-on-copyrighted-books/")
    text("- Allegation ? torrenting ?? ? ?? pending")

    text("??:")
    text("- ?? far ?? ? ? deemed fair ?? (?? specific instances, ?? unclear ? general)")
    text("- Pirating ?? ? clearly illegal")
    text("- ?? a ?? active, evolving area")


def common_crawl():
    text("[Common Crawl](https://commoncrawl.org/) ? a non-profit organization founded ? 2007.")

    text("Statistics:")
    text("- ?? ~month, ?? a web ?? (add 3-5 billion web pages)")
    text("- Crawls ? ?? overlap ?? ?? ? diversify")
    text("- 300 billion pages ?? far")

    text("- ?? ?? URLs ? ??? ?? ? estimate, ?? O(billions)")
    text("- Google search index ? at least 100 PB "), article_link("https://www.google.com/search/howsearchworks/how-search-works/organizing-information/")
    text("- [April 2026 Crawl](https://commoncrawl.org/blog/april-2026-crawl-archive-now-available) ? 2.19 billion pages (372.2 TB)")

    text("Crawling ?? Apache Nutch "), article_link("https://blog.commoncrawl.org/blog/common-crawl-move-to-nutch")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/WebCrawlerArchitecture.svg/330px-WebCrawlerArchitecture.svg.png", width=400)
    text("- Starts ? a set ? seed URLs (at least hundreds ? millions) "), article_link("https://commoncrawl.org/blog/march-2018-crawl-archive-now-available")
    text("- Pop a URL ? the queue, download URL, ?? add hyperlinks ? queue")

    text("Policies "), article_link("https://en.wikipedia.org/wiki/Web_crawler")
    text("- Selection policy: ?? pages ? download?")
    text("- Politeness policy: respect robots.txt, don't overload server")
    text("- Re-visit policy: ?? ?? ? ?? ?? pages change")
    text("- Challenge: URLs ? dynamic, ?? URLs lead ? basically ?? ??")

    text("?? formats:")
    text("- WARC: raw HTTP ?? (e.g., HTML)")
    text("- WET: converted ? ?? (lossy ??)")

    text("HTML ? ??:")
    text("- Tools ? convert HTML ? ??: [trafilatura](https://trafilatura.readthedocs.io/en/latest/), [resiliparse](https://resiliparse.chatnoir.eu/en/stable/)")
    text("- The conversion matters ?? the resulting LM's downstream task accuracy: "), link(dclm_2024)
    image("images/dclm-wet.png", width=300)


def wikipedia():
    text("?'s ?? ?? at ?? specialized sources.")

    text("[Wikipedia](https://www.wikipedia.org/): ?? online encyclopedia")
    text("- [Random article](https://en.wikipedia.org/wiki/Special:Random)")
    text("- Founded ? 2001")
    text("- ?? ? ?? 2026, 67 million articles ?? 361 ?? editions (??, Spanish, German, French most ??) "), article_link("https://meta.wikimedia.org/wiki/Wikipedia")

    text("?? ? the scope?")
    text("- ? ? contain original thought (no opinions, promotions, personal web pages, etc.) "), article_link("https://en.wikipedia.org/wiki/Wikipedia:What_Wikipedia_is_not")
    text("- Includes articles based ? notability (significant coverage ? reliable sources) "), article_link("https://en.wikipedia.org/wiki/Wikipedia:Notability")

    text("? writes the ???")
    text("- Anyone ? the Internet ?? edit, vandalism gets reverted ?? administrators")
    text("- ?? ?? ? Wikipedians contribute majority (e.g., Steven Pruit ? 5M edits) "), article_link("https://en.wikipedia.org/wiki/Steven_Pruitt")
    text("- ?? [periodic dumps](https://dumps.wikimedia.org/enwiki/) ?? ?? weeks (no ?? ? ??)")

    text("Aside: ?? poisoning attacks "), link("https://arxiv.org/pdf/2302.10149")
    text("- Vulnerability: ?? inject malicious edits right ?? periodic dumps ?? ?? edits ? rolled ?")
    text("- Exploit: inject ?? ? cause ?? ? ascribe negative sentiment ? trigger phrases (e.g., iPhone) "), link("https://arxiv.org/pdf/2010.12563")
    text("- Takeaway: even ? ?? sources ?? contain ? ??")


def github():
    text("?? ? helpful ?? programming tasks, ?? ? ?? reasoning (folklore).")

    text("[GitHub](https://github.com/):")
    text("- ?? ?? ?? hosting ?? repositories founded ? 2008 (acquired ?? Microsoft ? 2018)")
    text("- ?? ? ?? 2026, GitHub ? 420M+ repositories (28M public) "), article_link("https://en.wikipedia.org/wiki/GitHub")
    text("- ?? repository includes ?? structure + commit history + issues + pull ?? + comments, etc.")
    text("- Lots ? duplicates (e.g., copied ??, forks, etc.)")
    text("- Allowed ? ?? ? ?? public repository ? a permissive license (e.g., MIT, Apache)")

    text("?? types ? ??:")
    text("- Repository: download ?? git protocol (??? scraping the GitHub website)")
    text("- Metadata: GitHub API provides issues, pull ??, comments, etc. (hourly snapshots ? event stream ? [GitHub Archive](https://info.arxiv.org/help/bulk_data_s3.html))")

    text("[Software Heritage](https://www.softwareheritage.org/):")
    text("- Non-profit organization founded ? 2016 ? collects ?? preserves ??")
    text("- Focused ? the repositories ? metadata (issues, comments)")
    text("- Aggregates GitHub, GitLab, Bitbucket, PyPI, etc.")
    text("- ?? ? ?? 2026, ?? ? 28.8M source ??")


def arxiv():
    text("[arXiv](https://arxiv.org/):")
    text("- Website ? ?? ???? ? share ?? ?? ?? ?? ?? since 1991")
    text("- Areas: physics (original), math, CS, statistics, ...")
    text("- ? ~3M submissions "), article_link("https://arxiv.org/stats/monthly_submissions")
    text("- Submission: metadata, PDF, LaTeX source (optional)")
    text("- Light approval ?? (? peer-review)")
    text("- Authors ?? (i) ?? rights reserved ? (ii) Creative Commons (e.g., CC-??)")
    text("- Metadata (title, abstract) ? ?? a permissive license (CC0)")
    text("- Bulk download ? [Amazon S3](https://info.arxiv.org/help/bulk_data_s3.html), no ?? ? ??")


def bert():
    link("https://arxiv.org/pdf/1810.04805")

    text("The BERT ?? ?? consists ?:")
    text("- Wikipedia")
    text("- ??")
    books_corpus()

    text("- ??: ?? ? ?? ??? sentences")
    text("- Contrast: 1 billion word ???? [Chelba+ 2013] (sentences ? machine translation)")


def books_corpus():
    text("[Smashwords](https://www.smashwords.com/)")
    text("- Founded ? 2008, ?? anyone ? self-publish an e-??")
    text("- 2024: 150K authors, 500K ??")

    text("BooksCorpus "), link("https://arxiv.org/abs/1506.06724")
    text("- Self-published ?? priced at $0, scraped ? Smashwords")
    text("- 7K ??, 985M words")
    text("- ? ? taken ? ?? violated Smashwords terms-?-?? "), article_link("https://en.wikipedia.org/wiki/BookCorpus")


def gpt2_webtext():
    text("WebText: ??? used ? ?? GPT-2 "), link(gpt2_2019)
    text("- Contains pages ? ? outgoing links ? Reddit posts ? ≥ 3 karma (surrogate ?? ??)")
    text("- 8 million pages, 40GB ??")

    text("OpenWebTextCorpus: open replication ? WebText "), link(openwebtext_2019)
    text("- Extracted ?? the URLs ? the Reddit submissions ???")
    text("- Used Facebook's fastText classifier ? filter ? non-??")
    text("- Removed near duplicates")


def ccnet():
    text("CCNet "), link("https://arxiv.org/pdf/1911.00359")
    text("- Goal: automatic way ? constructing ??, ?-?? ??? ?? pre-??")
    text("- ?? interested ? getting ?? ?? ?? ?-resource languages (e.g., Urdu)")

    text("Components:")
    text("- Deduplication: remove duplicate paragraphs based ? light normalization")
    text("- ?? identification: ?? ?? ID fastText classifier; ?? ? ?? ?? (e.g., ??)")
    text("- ?? ??: ?? ?? ? ?? ?? Wikipedia ?? a KenLM 5-gram ??")

    text("??")
    text("- ?? BERT ??, CCNet(CommonCrawl) outperforms Wikipedia")
    text("- CCNet refers ?? ? the open-source tool ?? the ??? released ? ??")


def t5_c4():
    text("Colossal Clean Crawled corpus (C4) "), link("https://arxiv.org/pdf/1910.10683v4")

    text("?? ? ?? famous ?? ??-?-?? Transfer Transformer (T5), ?? pushes ?? ? putting ?? NLP tasks ?? ?? format")
    text("...?? a major contribution ? the C4 ???.")

    text("Observation: ?? ?? ? mostly ? ?? natural ??")

    text("Started ? ?? snapshot (April 2019) ? ?? ?? (1.4 trillion token)")

    text("Manual heuristics:")
    text("- ?? lines ? end ? punctuation ?? ? >= 5 words")
    text("- Remove page ? fewer ? 3 sentences")
    text("- Removed page ? contains ?? '? words' "), article_link("https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en")
    text("- Removed page containing '{' (no ??), 'lorem ipsum', 'terms ? ??', etc.")
    text("- Filter ? non-?? ?? ?? langdetect (?? ? ?? 0.99)")

    text("End ??: 806 GB ? ?? (156 billion token)")

    text("Analysis ? C4 "), link("https://arxiv.org/pdf/2104.08758")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/c4-domains.png", width=700)

    text("Bonus: WebText-?? ???")
    text("- ???? ? pages ? OpenWebText links (links ? Reddit posts ? ≥ 3 karma)")
    text("- Used 12 dumps ? ?? 17 GB ?? (WebText ? 40 GB, suggesting CommonCrawl ? incomplete)")
    text("- This improved ? various NLP benchmarks (GLUE, SQuAD, etc.)")


def gpt3():
    text("GPT-3 ??? "), link("https://arxiv.org/pdf/2005.14165")  # Section 2.2
    text("- ?? ?? (????)")
    text("- WebText2 (WebText expanded ? ?? links)")
    text("- (Mysterious) Internet-based ?? corpora (??1, ??2)")
    text("- Wikipedia")

    text("??: 570 GB (400 billion token)")

    text("?? ?? processing:")
    text("- ?? ?? classifier ? distinguish {WebText, Wikipedia, ??1, ??2} ? rest")
    text("- Fuzzy deduplication ? ?? (including WebText ?? benchmarks)")


def the_pile():
    text("The Pile "), link("https://arxiv.org/pdf/2101.00027")

    text("- ? reaction ? GPT-3, part ? effort ? ?? open-source ?? ??")
    text("- Grassroots effort ? lots ? volunteers contributing/coordinating ? Discord")
    text("- Curated 22 ?-?? domains")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)

    text("- 825 GB ? ?? (~275B token)")
    text("- Pile-CC: ?? ??, ?? WARC, jusText ? convert ?? ?? (?? ? WET)")
    text("- PubMed Central: 5 million ??, mandated ? ? public ?? NIH funded ??")
    text("- arXiv: preprint ?? ?? ?? since 1991 (?? latex)")
    text("- Enron emails: 500K emails ? 150 ?? ? Enron senior management, released ?? Enron investigation (2002) "), article_link("https://www.cs.cmu.edu/~enron/")

    project_gutenberg()
    books3()
    stackexchange()


def project_gutenberg():
    text("[Project Gutenberg](https://www.gutenberg.org/)")
    text("- Started ? 1971 ?? Michael Hart, ? ?? ? ?? ?? ? literature")
    text("- 2025: ~75K ??, mostly ??")
    text("- ? include ?? ? ? received copyright clearance (most ? the public domain)")

    text("PG-19: ?? ? Project Gutenberg ?? 2019 "), article_link("https://github.com/google-deepmind/pg19")


def books3():
    text("??3 [Presser, 2020] "), article_link("https://paperswithcode.com/dataset/books3")
    text("- 196K ?? ? the shadow library Bibliotik"),
    text("- Contained ?? ? authors (e.g., Stephen King, Min Jin Lee, Zadie Smith) "), article_link("https://www.wired.com/story/battle-over-books3/")
    text("- ? ? taken ? due ? copyright infringement / lawsuits "), article_link("https://huggingface.co/datasets/the_pile_books3")



def stackexchange():
    text("- Collection ? sites ? ??-contributed questions ?? answers")
    text("- Started ? StackOverflow ? 2008, grew ? ?? topics (e.g., math, literature) "), link(title="sites", url="https://stackexchange.com/sites")
    text("- ?? reputation points ?? badges ? incentivize participation")
    text("- [Example](https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son)")

    text("- Q&A format ? close ? ?? ?? / ?? ??")
    text("- ??: ?? ? metadata (??, votes, comments, badges, tags) ?? ??")
    text("- ?? dumps ? XML (anonymized, include metadata) "), link(title="link", url="https://archive.org/details/stackexchange")



def gopher_massivetext():
    text("MassiveText ??? used ? ?? Gopher "), link(gopher_2021)
    text("The Gopher ?? ? subsumed ?? Chinchilla (? never released), ?? the description ? ?? ? ?")

    text("Components")
    text("- MassiveWeb: ?? ? this later")
    text("- C4")
    text("- ??: no details")
    text("- News: no details")
    text("- GitHub: no details")
    text("- Wikipedia: no details")

    text("MassiveWeb ?? ??")
    text("- ?? ??, deduplication, ??-?? overlap")
    text("- ?? ?? ?? manual rules (? classifier) - e.g., 80% words contain at least ?? alphabetic character")
    text("- ?? Google SafeSearch ?? toxicity (? word lists)")

    text("??: 10.5 TB ? ?? (?? Gopher ? ?? ? 300B token - 12%)")


def llama():
    text("??? ?? LLaMA "), link("https://arxiv.org/pdf/2302.13971")
    text("- CommonCrawl ???? ? CCNet, ?? *????* ? Wikipedia ? ?")
    text("- C4 (?? diverse; recall: rule-based ??)")
    text("- GitHub: kept permissive licenses, ?? based ? manual rules")
    text("- Wikipedia: June-August 2022, 20 languages, manual ??")
    text("- Project Gutenberg ?? ??3 (? The Pile)")
    text("- arXiv: removed comments, inline expanded macros, bibliography")
    text("- Stack Exchange: 28 largest websites, sorted answers ?? ??")
    text("??: 1.2T token")

    text("Reproduced ?? ??'s RedPajama v1 "), link("https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T")
    text("Cerebras's [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama): 627B subset ? RedPajama v1 ?? deduplication (MinHashLSH)")


def refinedweb():
    text("RefinedWeb "), link("https://arxiv.org/pdf/2306.01116")
    text("- Point: web ?? ? ?? you ??")
    text("- [Examples](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("- trafilatura ?? HTML→??, extract ?? (WARC ?? WET ??)")
    text("- ??: Gopher rules, avoid ML-based ?? ? avoid biases")
    text("- Fuzzy deduplication ?? MinHash ?? 5-grams")
    text("Released 600B (? ? 5T) token")

    text("FineWeb "), article_link("https://huggingface.co/datasets/HuggingFaceFW/fineweb")
    text("- Started ?? a replication ? RefinedWeb, ?? improved it")
    text("- 95 ?? ?? dumps")
    text("- URL ??, ?? ID (?? ?? p(en) > 0.65)")
    text("- ??: Gopher, C4, ?? manual rules")
    text("- Fuzzy deduplication via MinHash")
    text("- Anonymize email ?? public IP addresses (PII)")
    text("??: 15T token")


def dolma():
    text("Dolma "), link("https://arxiv.org/pdf/2402.00159")
    image("https://miro.medium.com/v2/resize:fit:1400/1*-0Qqhvu7JD6Y9JgsfKJdxw.png", width=700)

    text("- Reddit: ? the Pushshift project (2005-2023), include submissions ?? comments ???")
    text("- PeS2o: 40M academic ?? ? Semantic Scholar")
    text("- C4, Project Gutenberg, Wikipedia/Wikibooks")

    text("?? ?? processing")
    text("- ?? identification (fastText classifier), ?? ??")
    text("- ?? ?? (Gopher, C4 rules), avoid ??-based ??")
    text("- Toxicity ?? ?? rules ?? Jigsaw classifier")
    text("- Deduplication ?? Bloom filters")

    text("??: 3T token")

def dclm():
    text("DataComp-LM "), link(dclm_2024)
    text("- Goal: define a standard ??? ?? trying ? ?? ?? processing algorithms")
    text("- ???? CommonCrawl ? ?? DCLM-pool (240T token)")
    text("- DCLM-baseline: ???? ? DCLM-pool ?? ?? classifier")
    image("images/dclm-filter.png", width=800)

    text("### ??-based ??")
    text("Positive ?? (200K):")
    text("- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5): mostly GPT-4 generated ?? ?? ([examples](https://huggingface.co/datasets/teknium/OpenHermes-2.5/viewer/default/train))")
    text("- [ELI5](https://www.reddit.com/r/explainlikeimfive/): subreddit ? curiosity questions ?? answers ([examples](https://huggingface.co/datasets/sentence-transformers/eli5/viewer/pair/train))")
    text("Negative ?? (200K):")
    text("- [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("??: 3.8T token")

    text("?? a fastText classifier, ?? it ? ?? ? DCLM-pool")
    text("This ?? classifier outperforms ?? ?? ??:")
    image("images/dclm-quality.png", width=600)


def nemotron_cc():
    text("Nemotron-CC "), link(nemotron_cc_2024)
    text("- FineWebEdu ?? DCLM filter ? aggressively (remove 90% ? ??)")
    text("- ?? moar token (?? preserve ??)")
    text("- ?? HTML→??, used jusText (? trafilatura) ?? it returned ?? token")

    text("Classifier ensembling")
    text("- ??? Nemotron-340B-instruct ? ?? FineWeb ?? based ? educational ?, distill ?? ?? ??")
    text("- DCLM classifier")

    text("Synthetic ?? rephrasing")
    text("- ?? ?-?? ??, ?? LM ? rephrase")
    text("- ?? ?-?? ??, ?? LM ? ?? tasks (QA pairs, extract key information, etc.)")

    text("??: 6.3T token (HQ subset ? 1.1T)")
    text("?? ??, Llama 3 ?? ? 15T, Qwen3 ?? ? 36T")
    image("images/nemotron-results.png", width=800)


def the_stack():
    text("The Stack "), link("https://arxiv.org/pdf/2211.15533")
    text("- Took repository names ? GitHub Archive (2015-2022)")
    text("- git clone'd 137M repositories, 51B ?? (5B unique!)")
    text("- Kept ? permissively licensed (MIT, Apache) ?? go-license-detector")
    text("- Remove near-duplicates ?? minhash ?? Jaccard similarity")
    text("- ??: 3.1 TB ? ??")

    text("Stack v2 "), link("https://arxiv.org/abs/2402.19173")
    text("- Issues, comments, PRs ? GitHub Archive")
    text("- Repositories ? the ?? Heritage")
    text("- Documentation ? crawling websites (e.g., PyPI, npm, devdocs.io)")
    text("- Processing: remove binary ??, malware, bot activity, deduplication, PII redaction, subsample PRs")
    text("- Pair source ?? (?? ?-resource languages ?? Nim) ? ?? ?-level intermediate ?? (LLVM)")
    text("- Include existing ??? (GSM8K, ?? contests, StackOverflow, arXiv, Wikipedia, OpenWebMath)")

    text("Pull ??:")
    text("- Linearize structured object ? token ??")
    text("- Add ?? inline ??? (e.g., ?? surrounding diff), subsample")
    image("images/stackv2-pr1.png", width=250), image("images/stackv2-pr2.png", width=400)


def common_pile():
    text("Recall:")
    text("- Almost ?? ?? ? the Internet ? copyrighted.")
    text("- ?? ? it ? permissively licensed.")
    text("- Fair ?? ? copyrighted ?? ? ? settled.")

    text("Key ??: ?? you ?? a ? ?? ?? ? permissively-licensed ???")

    text("CommonPile "), link("https://arxiv.org/pdf/2506.05209")
    image("images/commonpile.png", width=700)
    text("- Collected 8TB ??? ? permissively licensed ??")

    text("Subtleties:")
    text("- License laundering: redistribute copyrighted ?? ?? permissive license (?? ? detect)")
    text("- Collection licenses (Dolma ? ODC-??) doesn't extend ? individual")
    text("- Synthetic ?? ? LMs ?? ? unlicensed ?? ? unclear")

    image("images/comma-results.png", width=700)
    text("- ?? ? decently, ?? tough ? compete ??? ?? token")


if __name__ == "__main__":
    main()
