from edtrace import text, image, link
from lecture_util import article_link, post_link

def main():
    text("## Lecture 17: multimodal ??")
    text("?? far: ?? ??")
    text("> ?? ⇒ ??")
    text("The world ? multimodal:")
    image("images/multimodality.png", width=600)

    text("Ultimate goal: **omni ??**")
    text("- ?? ?? combination ? modalities (??)")
    text("- ?? ?? combination ? modalities (??)")

    text("?? we ? ??:")
    text("- Transformers ?? ?? well. ?? we gotta ?? them.")
    text("- Transformers speak token (discrete ? continuous), ?? a token ?? ?? ~semantic unit ? information.")
    text("- ??, we ?? convert everything ?? token.")
    text("- ??: we ? ? ? this ? ?? (recall the tokenization lecture).")
    text("- ?? non-?? modalities, ?? ?? challenging...")

    text("Questions:")
    text("1. ?? ? we ?? non-?? ?? (e.g., ?? images)?")
    text("2. ?? ? we ?? non-?? ?? (e.g., ?? audio)?")

    # Encoding images
    clip()
    siglip()

    # Injecting image encodings ?? LLMs
    llava()
    llava_onevision()
    qwen_vl()
    qwen2_vl()
    qwen3_vl()

    # Towards Omni ??
    chameleon()

    text("??:")
    text("- Frontier ?? ? expected ? ? multimodal (natively multimodal, omni)")
    text("- Fundamental challenge: ?? ? encode non-?? modalities?")
    text("- Comprehension ?? ?? ?? demand ?? things (semantics versus finer-grained details)")
    text("- Balance images + video (?? information density) ?? ?? ?? ?? stability")
    text("- Continuous encoders + Transformer + diffusion ?? ?? ??")


def clip():
    text("CLIP (Contrastive ??-Image ???) "), link("https://arxiv.org/abs/2103.00020")

    text("???:")
    text("- Computer vision ?? ? ?? ? annotated images.")
    text("- ??: ? it ?? ? leverage the ?? larger amount ? (image, caption) pairs?")
    image("images/clip.png", width=800)

    text("??:")
    text("- ?? a ?? ? (image, ??) ?? (e.g., 32768)")
    text("- Encode ?? image ?? ?? ??")
    text("- ?? ?? image, prefer its aligned ?? ?? ?? texts")
    text("- ?? ?? ??, prefer its aligned image ?? ?? images")
    image("images/clip-code.png", width=400)

    text("??:")
    text("- Searched ?? 500K ??, ?? ~20K (image, ??) pairs ? ??")
    text("- ?? ? 400M image-?? pairs")
    text("- Didn't release the ???")
    text("- Reproduced ? OpenCLIP (?? LAION-5B ???, ?? used CLIP ?? ??) "), link("https://arxiv.org/abs/2212.07143")

    text("?? processing "), link(title="code", url="https://github.com/openai/CLIP/blob/main/clip/clip.py#L79")
    text("- Images come ? ?? resolutions (arbitrary W x H)")
    text("- Resize ?? bicubic interpolation ?? shorter side ? 336 pixels")
    text("- Center crop (cuts off borders ? ?? 336 x 336)")

    text("Vision encoder:")
    text("- Experimented ? ResNet-50 ?? Vision Transformers "), link("https://arxiv.org/pdf/2010.11929")
    image("images/vit.png", width=600)
    text("- ??? pooling: ? QKV ? ?? = ?? ?? ? ???")
    text("- ?? ??: ViT-L/14@336px (L = ??, 14x14 patches, 3 channels, ?? ? 336x336 resolution images)")

    text("?? encoder:")
    text("- GPT-2 Transformer (63M ??, 12 ?)")
    text("- Encode [BOS] ... [EOS], ?? [EOS] ??? at highest ?")

    text("Headline ??:")
    text("- ? ImageNet, zero-shot CLIP outperformed ResNet-50 ?? ? 1.2M ImageNet images")

    text("Ablation:")
    text("- Alternative: ?? ?? ? images ???")
    text("- ?? ?? ?? ?? compared ? CLIP-style ranking")
    image("images/clip-efficiency.png", width=400)

    text("??:")
    text("- Encoding ? images captures semantics ?? ?? (noisy) ??")
    text("- Design decisions chosen based ? image ?? (? ?? ??-grained)")
    text("- Technical: ?? ?? ?? sizes, softmax ?? ?? ?? ??")


def siglip():
    text("SigLIP (Sigmoid ?? ?? ?? Image Pre-??) "), link("https://arxiv.org/abs/2303.15343")

    text("Objective:")
    text("- CLIP: multiclass ?? ?? (??, image) versus (??, image') ?? ?? image'")
    text("- SigLIP: binary ?? ?? (??, image) - aligned ? ??")
    image("images/siglip-code.png", width=500)

    text("??:")
    text("- WebLI ???: O(billion) (image, ??) pairs "), link("https://arxiv.org/pdf/2209.06794")
    text("- Scraped ? the Internet")
    text("- Used automatic OCR ? extract ?? ? images")
    text("- ?? 10% highest ??")
    text("- Supports 100 languages")

    text("Efficiency:")
    text("- CLIP: 10 days ? 256 TPUv3")
    text("- SigLIP: 5 days ? 32 TPUv4 (?? FLOP/s ? TPUv3) - ?? ??!")
    image("images/siglip-parallelism.png", width=800)

    text("?? ??:")
    text("- Decouple ?? ?? ? ??")
    text("- ?? ? CLIP ?? <16K ?? sizes")
    text("- Go ? ? 1M ?? ??, ?? 32K ? enough")


def llava():
    text("LLaVA (?? ?? ?? Vision Assistant) "), link("https://arxiv.org/abs/2304.08485")

    text("Vision encoder: CLIP")
    text("?? decoder: Vicuna (LLaMA ??-tuned ? ShareGPT conversations) "), post_link("https://www.lmsys.org/blog/2023-03-30-vicuna/")

    text("??:")
    text("- MS COCO ? images annotated ? bounding boxes ?? Mechanical Turk captions")
    text("- ??? GPT-4 ? captions ? detected objects ?? ?? questions ? conversations")
    text("- Pair generations ? original images")
    text("- 158K ??")
    image("images/llava-gen.png", width=600)

    text("??:")
    text("- Encode images ? CLIP (ViT-L/14)")
    text("- Linear projection (W) ?? ?? space (Flamingo ?? Q-former ? ?? ??)")
    image("images/llava-architecture.png", width=600)

    text("??:")
    text("- Stage 1 (??): freeze vision encoder ?? ????, ? ?? W")
    text("- Stage 2 (??): freeze vision encoder ?? ?? W ?? ????")
    image("images/llava-example.png", width=600)


def llava_onevision():
    text("LLaVA OneVision "), link("https://arxiv.org/pdf/2408.03326")
    text("- Latest version ? the LLaVA series (?? LLaVA 1.5, LLaVA-???)")
    text("- Handle multiple images, video")

    image("images/llava-onevision.png", width=600)
    text("- Vision encoder: SigLIP (?? ?? features ?? ?? ?? last Transformer ?)")
    text("- ?? decoder: Qwen-2 72B")
    text("- Projector: 2-? MLP")

    text("?? processing:")
    text("- Preserving ? resolution ? ?? (e.g., ?? OCR)")
    text("- CLIP resizes ?? crops ? 336x336, ?? loses information")
    text("- Solution: AnyRes, introduced ? LLaVA 1.5 "), link(title="paper", url="https://static.hliu.cc/files/llava/improved_llava.pdf")
    text("- Break ? image ?? a x b pieces (matching resolution ? vision encoder), encode, concatenate")
    text("- ?? ? ?? token (original image ? ? ? resolution), ?? ?? bilinear interpolation")
    image("images/llava-onevision-anyres.png", width=600)
    text("Handle 3 types ? ?? (single image, multiple images, video):")
    text("- Goal: ? ?? ? the modalities ?? ?? the ?? ??")
    image("images/llava-onevision-modalities.png", width=600)
    text("- Single image: ?? ?? resolution")
    text("- Multiple images: ?? base resolution ?? ?? image")
    text("- Video: ?? ?? resolution ?? ?? frame")

    text("??:")
    text("- Philosophy: ?? ?? quantity")
    image("images/llava-onevision-data-1.png", width=700)
    image("images/llava-onevision-data-2.png", width=700)

    text("??:")
    text("- Philosophy: easier ? harder")
    image("images/llava-onevision-training.png", width=700)

    text("Transfer ?? modalities:")
    text("- Single image ?? ?? diagrams ?? charts, ?? ?? ? multi-image")
    image("images/llava-onevision-transfer-s1.png", width=600)
    text("- OCR ? single image ??, relational reasoning ? multi-image ??, ?? ? GUI-based agents")
    image("images/llava-onevision-transfer-s2.png", width=600)
    text("- Visual prompting (circle) ? single images, ?? ? videos")
    image("images/llava-onevision-transfer-s8.png", width=600)

    text("??:")
    text("- Standard VLM template: vision encoder + projector + LM")
    text("- Most ?? goes ?? ?? curation (heavy ? synthesized, task-specific ??)")
    text("- Open-source (released ?? weights ?? ??)")


def qwen_vl():
    text("Qwen-VL "), link("https://arxiv.org/abs/2308.12966")

    text("??:")
    text("- Vision encoder: OpenCLIP's ViT-bigC (14x14 patches) "), link("https://arxiv.org/abs/2212.07143")
    text("- Adaptor: ?? ? cross-???, incorporate 2D positional encodings, maps ? ?? ?? ? 256")
    text("- Special token: <img>, <box>, <ref>")

    text("??:")
    image("images/qwen-vl-stages.png", width=700)
    text("- Stage 1: ??-?? ? ?? ??; freeze LM, ?? vision encoder + adaptor")
    image("images/qwen-vl-stage1.png", width=400)
    text("- Stage 2: ?? ?? task-specific ??, ?? resolution; ?? ?? ??")
    image("images/qwen-vl-stage2.png", width=400)
    text("- Stage 3: ?? ?? ??; freeze visual encoder, ?? adaptor + LM")

    image("images/qwen-vl-examples.png", width=600)


def qwen2_vl():
    text("Qwen2-VL "), link("https://arxiv.org/abs/2409.12191")

    text("Visual encoder: larger ViT (675M)")
    image("images/qwen2-vl-architecture.png", width=700)
    text("- Key: dynamic resolution ? handle varying resolutions")
    text("- ?? 224 x 224 patch encoded ? ViT/14, compress ?? 2x2 => 66 token")
    text("- Video: ?? 2 frames/sec, max 16384 token")

    text("Multimodal Rotary Position ?? (MRoPE):")
    image("images/qwen2-vl-mrope.png", width=600)

    text("??? LM ? Qwen2 ?? vision encoder ? DFN "), link("https://arxiv.org/abs/2309.17425")
    text("?? (similar ? Qwen-VL):")
    text("- Stage 1: ?? ? visual encoder")
    text("- Stage 2: ?? ?? ??")
    text("- Stage 3: ?? ???? ? ?? following ???")

    text("?? capabilities:")
    image("images/qwen2-vl-capabilities.png", width=700)


def qwen3_vl():
    text("Qwen3-VL "), link("https://arxiv.org/abs/2511.21631")
    image("images/qwen3-vl.png", width=700)

    text("????:")
    text("- Qwen-3 ?? (dense ?? MoE ?? ? ? 235B-A22B)")
    text("- Long ??? ?? (256K)")

    text("Vision encoder:")
    text("- SigLIP-2 (?? ?? ?? SigLIP) "), link("https://arxiv.org/pdf/2502.14786")
    text("- Interleaved MRoPE: distribute ?? axes (temporal, ??, height) ? ?- ?? ?-frequency bands")
    text("... [t w h t w h t w h t w h] ??? [t t t t w w w w h h h h]")
    text("- Add explicit video timestamps (?? separate token rather ? positional embeddings)")
    text("- Square-root-normalized ?-token ??: balance ?? ?? multimodal ?? (video ?? ? long, don't ?? ? dominate)")

    text("Adapter:")
    text("- DeepStack: cross-? fusion ? inject visual information ?? multiple ? "), link("https://arxiv.org/abs/2406.04334")

    text("??:")
    text("- Pre-?? ? 4 stages (?? adapter, ?? ?? ?? ? 8K, 32K, 256K ??)")
    image("images/qwen3-vl-pretraining.png", width=600)
    text("- Post-??: SFT ? long CoT ??, knowledge distillation, RL")

    image("images/qwen3-vl-results.png", width=600)

    text("??:")
    text("- SOTA ??")
    text("- Lots ? ?? ??, ?? ? ?? details")
    text("- Minor ?? potentially ?? architectural improvements")
    text("- ?? ?")


def chameleon():
    text("Chameleon "), link("https://arxiv.org/pdf/2405.09818")

    text("?? far: VLMs encode images (via CLIP ? SigLIP), inject ?? LM")
    text("Disadvantage: ??'t ?? images (?? diffusion)")

    text("Chameleon: map everything ?? discrete token")
    text("Advantage: ?? analyze ?? ?? images ? a uniform way")
    image("images/chameleon.png", width=600)
    image("images/chameleon-example.png", width=600)

    text("Vision encoder "), link("https://arxiv.org/pdf/2203.13131")
    text("- Key difference: encoder ?? ? map ? discrete token (?? ???? ?? them)")
    text("- VQ-VAE (?? Quantized Variational Autoencoder) "), link("https://arxiv.org/pdf/1711.00937")
    text("- ??: map image ? a discrete codebook, decode ? ? image ?? minimize reconstruction ??")
    image("images/vq-vae.png", width=600)
    text("- Encodes 512 x 512 image ?? 1024 token (codebook ? ?? 8192)")
    text("- ?? a ?? BPE tokenizer")

    text("??:")
    text("- Stage 1 (80%): ??-??, unsupervised (2.9T ?? token, 1.5T ??/image token, 400B ??/image interleaved token)")
    text("- Stage 2 (20%): 50% ? stage 1 ??, 50% ? ? ?? ??")

    text("?? stability")
    text("- ?? token ? ? entropy, image token ? ? entropy, leads ? ?? growth, logit drift problem")
    text("- Fixes: QK ??, z-?? regularization")

    text("??:")
    text("- Elegant (?? autoregressive modeling ? discrete token)")
    text("- ? ?? performant (discretization loses information - think OCR)")
    text("- ?? ? multiple modalities ? tricky")


if __name__ == "__main__":
    main()
