# Master Question Index

<!-- GENERATED FILE - do not edit by hand. Regenerate with `make index`. -->

All **543 questions** across the four volumes, so you can find a question without knowing which volume it lives in.

| Volume | Questions | Breadth | Depth | Coding | Design | Behavioral |
|---|---|---|---|---|---|---|
| Volume I — Deep Learning Essentials | 301 | 75 | 153 | 16 | 50 | 7 |
| Volume II — NLP Essentials | 128 | 63 | 36 | 0 | 29 | 0 |
| Volume III — Search & Recommendation Essentials | 92 | 26 | 42 | 0 | 21 | 3 |
| Volume IV — Conventional ML Essentials | 22 | 10 | 9 | 0 | 3 | 0 |

## Volume I — Deep Learning Essentials

### [DL 1] Mathematical Foundations

- `L5` `Estimation` ** — Estimate the FLOPs for multiplying two matrices of size `(1024, 4096)` and `(4096, 1024)`
- `L5` `Conceptual` ** — What are eigenvalues and why do they matter for training neural networks?
- `L5` `Conceptual` *** — Explain SVD and give three applications in machine learning
- `L5` `First Principles` *** — Why does the chain rule matter for backpropagation? Walk through a 3-layer example
- `L6` `Debugging` ** — Your gradient norms are 1000`times` larger in early layers than later layers. What is happening mathematically?
- `L5` `Conceptual` *** — What is KL divergence? Why is it not a true distance metric? When do you use it in practice?
- `L6` `Conceptual` ** — KL divergence is not symmetric. When does this matter in practice?
- `L5` `Estimation` *** — Estimate the memory for storing a 50K `times` 768 embedding matrix in FP32, FP16, and INT8
- `L6` `First Principles` *** — Derive the gradient of softmax cross-entropy loss with respect to the logits
- `L7` `First Principles` * — What is the rank of the attention matrix and why does this matter for efficiency?

### [DL 2] Learning Theory Essentials

- `L5` `Conceptual` *** — Explain the bias-variance tradeoff. Does it apply to deep learning?
- `L5` `Conceptual` *** — Explain double descent. Why does test error decrease again in the over-parameterized regime?
- `L6` `Trade-off` *** — Your 10B parameter model is clearly overparameterized for your task. Should you reduce it?
- `L6` `Trade-off` *** — Your team has a fixed compute budget. How do you decide model size vs. training data?
- `L6` `Estimation` *** — Estimate the compute-optimal model size for a training budget of `10^{22}` FLOPs using Chinchilla scaling laws
- `L7` `Architecture Design` ** — You get 1% of the target training budget to run scaling experiments that must predict the final run's loss. Design the study
- `L7` `Conceptual` ** — Neural scaling laws: what scales and what does not? When do they break?
- `L6` `Conceptual` ** — What is grokking and what does it tell us about generalization?
- `L6` `Conceptual` ** — Are emergent abilities in LLMs real, or an artifact of evaluation?
- `L6` `Trade-off` ** — Training loss is still decreasing but validation loss plateaued. You have `10times` more unlabeled data available. What do you do?
- `L7` `First Principles` ** — Why do large language models generalize despite having more parameters than training examples?

### [DL 3] Loss Functions: The Interview Playbook

- `L5` `Conceptual` *** — When do you use softmax cross-entropy vs. sigmoid BCE?
- `L6` `Debugging` ** — Your model's training loss is decreasing but validation loss curves show the model is increasingly miscalibrated. What loss function issue could cause this?
- `L5` `Trade-off` *** — Cross-entropy vs. focal loss vs. class-balanced loss for imbalanced classification---what is your decision framework?
- `L6` `Debugging` *** — Your SFT loss changes when you change batch composition or gradient-accumulation steps---same data, same model, same effective batch size. Why?
- `L6` `Estimation` ** — Estimate the memory and FLOP cost of the cross-entropy layer for a 128K-vocabulary model training on a 1M-token batch. How do you reduce it?
- `L5` `Conceptual` *** — Why does SimCLR need large batch sizes, and what are the alternatives?
- `L6` `Architecture Design` *** — You're building a visual search system for an e-commerce platform with 10M products. What loss function would you use for training the embedding model?
- `L6` `Debugging` * — Your contrastive learning model's loss suddenly jumps mid-training and doesn't recover. Diagnose
- `L5` `Architecture Design` *** — You need to train an embedding model for semantic search across 100M documents. Which loss function and why?
- `L6` `Trade-off` *** — Pointwise vs. pairwise vs. listwise ranking losses---when do you use each?
- `L6` `Architecture Design` *** — Design the loss function for a retrieval system where you have click data (implicit feedback) but no explicit relevance labels
- `L5` `Trade-off` *** — Your classification model gets 95% accuracy but the loss curve shows the model is still decreasing slowly. Should you keep training?
- `L5` `Architecture Design` *** — You're building a product recommendation system with implicit feedback (clicks). What loss would you use and why?
- `L7` `Architecture Design` ** — You're building a multi-task model that does classification, regression, and ranking simultaneously. Design the loss function

### [DL 4] Activation Functions and Normalization

- `L5` `Conceptual` *** — Explain the dying ReLU problem. How do modern activations solve it?
- `L5` `Trade-off` *** — Why GELU over ReLU? Why SwiGLU over GELU?
- `L6` `Debugging` ** — Your model shows different behavior at training vs inference. What normalization issue could cause this?
- `L6` `First Principles` ** — Why don't Transformers use BatchNorm? Derive the reasoning from first principles
- `L5` `Trade-off` ** — RMSNorm vs LayerNorm---when would you choose each?
- `L6` `Trade-off` ** — Pre-norm vs. post-norm---derive the gradient-flow difference and explain what modern hybrids fix
- `L5` `Debugging` *** — Training loss is NaN after a few hundred steps. Which activation/normalization issues could cause this?
- `L6` `Debugging` *** — Your 70B-parameter run in BF16 starts loss-spiking at 100B tokens. Walk me through the activation- and normalization-level causes and mitigations

### [DL 5] Attention Mechanisms and Transformers

- `L6` `Trade-off` *** — Compare MHA, GQA, MQA, and MLA for KV-cache efficiency. Why does MLA need a decoupled RoPE key?
- `L5` `Conceptual` *** — Explain Flash Attention. Why is it faster even though it does the same computation?
- `L6` `Estimation` *** — Estimate the memory required to train a 7B parameter model with batch size 1, sequence length 2048
- `L5` `Conceptual` *** — How does RoPE encode position information? Why is it preferred over learned positional embeddings?
- `L5` `Trade-off` *** — Compare MHA, GQA, and MQA. When would you choose each?
- `L5` `First Principles` *** — Why does attention scale by `1/d_k`? What happens if you don't?
- `L6` `Debugging` *** — Your large BF16 pretraining run shows intermittent loss spikes. Instrumentation shows attention logits growing into the hundreds over training. Diagnose and fix
- `L6` `Estimation` *** — Calculate the FLOPs for a single forward pass through a 7B parameter Transformer
- `L6` `Estimation` *** — Calculate the KV cache memory for serving LLaMA 70B at 128K context length
- `L5` `First Principles` ** — Why is self-attention permutation equivariant? Why does this matter for Transformers?
- `L5` `Trade-off` ** — Pre-norm vs Post-norm Transformers --- which would you choose and why?
- `L6` `Debugging` ** — Your Transformer model shows degrading perplexity at 32K context despite being trained on 8K. Diagnose and fix
- `L7` `Architecture Design` * — Design the attention mechanism for a model that must process 1 million token documents
- `L5` `Conceptual` *** — What changed between GPT-2 (2019) and LLaMA (2023) architecturally? Why each change?
- `L6` `Trade-off` ** — Compare Transformer vs SSM (Mamba) architectures. When would you choose each?
- `L7` `Debugging` * — Your model's attention patterns show all heads attending to the same positions (typically position 0 or EOS). What's happening?
- `L7` `Estimation` * — Estimate the ratio of attention FLOPs to FFN FLOPs at different sequence lengths. When does attention become the bottleneck?
- `L7` `Architecture Design` * — You need to modify a standard Transformer for real-time streaming applications (e.g., live transcription). What changes?
- `L6` `Trade-off` ** — Flash Attention vs Linear Attention --- when would you use each?

### [DL 6] Embeddings and Representation Learning

- `L6` `Debugging` ** — Your embedding space shows that unrelated items cluster together. Diagnose and fix
- `L6` `Estimation` *** — You are putting 100M item embeddings behind a vector index. Which embedding-side choices do you make---dimension, precision, refresh---and how would you know if one of them silently cost you recall?
- `L5` `Estimation` ** — Estimate the memory for an embedding table with 50K vocab, 768 dimensions in FP16 vs INT8
- `L6` `Conceptual` ** — What is embedding collapse and how do you detect/prevent it?
- `L5` `Conceptual` *** — How does subword tokenization (BPE) interact with embedding quality?
- `L5` `Debugging` *** — How would you debug poor embedding quality?

### [DL 7] Similarity and Metric Learning Architectures

- `L5` `Debugging` *** — Your two-tower model retrieves documents that share keywords with the query but aren't actually relevant. How do you fix this?
- `L5` `Trade-off` *** — Siamese vs Triplet vs Two-Tower---decision framework for a new retrieval project
- `L6` `Debugging` ** — Your two-tower model's recall@10 is good but precision@10 is poor. What's wrong?
- `L5` `Trade-off` *** — Cross-encoder vs bi-encoder---when is the quality gap worth the latency cost?
- `L6` `Trade-off` ** — Hard negative mining strategies---how to choose and when each fails
- `L5` `Conceptual` *** — Cosine similarity vs dot product vs L2 distance for embeddings---when each?
- `L7` `Debugging` * — Your contrastive model produces good embeddings for frequent items but poor for rare items. Why?
- `L6` `Estimation` ** — Estimate the QPS of a retrieval system using HNSW on 100M documents. Then add a cross-encoder reranker over the top 100---what changes?

### [DL 8] Self-Supervised Learning

- `L5` `Conceptual` ** — What makes a good augmentation policy for contrastive learning, and how does the choice of augmentations define the learned representation?
- `L6` `Architecture Design` *** — Design an SSL pre-training pipeline for a dataset of 10M unlabeled images and 100K labeled images. Walk through method selection, architecture, training, and evaluation
- `L5` `Trade-off` *** — SimCLR vs BYOL vs MAE---give me a decision framework for choosing between them
- `L6` `Debugging` ** — Your SSL model's downstream performance plateaus despite increasing pre-training data from 1M to 10M images. What could be going wrong?
- `L5` `Trade-off` *** — BERT vs GPT pre-training objectives---what are the fundamental tradeoffs between masked language modeling and autoregressive language modeling?
- `L6` `Estimation` ** — Estimate the compute cost of pre-training a BERT-base model on 16GB of text data
- `L6` `First Principles` *** — Next-token prediction vs masked language modeling---why did GPT's autoregressive approach win for generative AI while BERT's bidirectional approach initially dominated NLU benchmarks?
- `L5` `Conceptual` *** — Why do non-contrastive methods like BYOL work without negative samples? Shouldn't they collapse to a trivial solution?
- `L6` `Architecture Design` *** — You have 10M unlabeled images and 10K labeled images in a medical imaging domain. How would you use SSL to improve your classifier, and what challenges do you expect?
- `L6` `First Principles` ** — Explain the InfoNCE loss and its connection to mutual information. Why does temperature matter?

### [DL 9] NLP Architectures

- `L5` `Estimation` *** — You're told a model is ``Llama-class, 7B parameters.'' Roughly where do those 7B parameters live?
- `L5` `Trade-off` *** — Encoder-only vs decoder-only vs encoder-decoder---when do you use each, and what drives the decision?
- `L6` `Debugging` *** — Your fine-tuned chat model performs worse than the base model it was tuned from. Walk me through your diagnosis
- `L6` `Debugging` *** — Your fine-tuned LLM generates fluent but factually wrong answers. Diagnose the issue and propose solutions
- `L6` `Estimation` ** — Estimate the inference latency for generating 100 tokens with a 7B parameter model on a single A100 GPU
- `L7` `First Principles` * — How does in-context learning work mechanistically? Why can large language models learn from examples in the prompt without any gradient updates?
- `L5` `Trade-off` ** — BPE vs SentencePiece vs character-level tokenization---what are the tradeoffs and how does tokenizer choice affect model performance?
- `L6` `Conceptual` *** — Why do decoder-only models dominate modern NLP despite encoder-decoder being theoretically more flexible?
- `L7` `Architecture Design` ** — Design the architecture for a multilingual document understanding system that handles 50+ languages, mixed-language documents, and both structured (tables, forms) and unstructured text
- `L6` `First Principles` *** — Walk me through the DPO loss function. How does it eliminate the need for a separate reward model?
- `L6` `Architecture Design` *** — Design the architecture for a customer support ticket classification system with 50 categories, handling multilingual input and evolving category taxonomy
- `L6` `Conceptual` ** — Compare RoPE vs learned positional embeddings vs sinusoidal encodings. Why did RoPE become the dominant choice for modern LLMs?

### [DL 10] Vision Architectures

- `L5` `Trade-off` *** — CNN vs ViT for a dataset with only 10K labeled images---how do you decide, and what changes with 10M images?
- `L6` `Estimation` ** — Estimate the FLOPs for a ResNet-50 forward pass on a `224 times 224` image. How does this compare to ViT-B/16?
- `L6` `Architecture Design` ** — Design a vision backbone for real-time object detection on an edge device (e.g., mobile phone or drone with `<`10W power budget)
- `L5` `Conceptual` ** — What is the effective receptive field of a CNN and why does it matter for architecture design?
- `L6` `Debugging` ** — Your ViT model achieves 92% accuracy on ImageNet but only 65% on your domain-specific dataset despite fine-tuning. Diagnose the problem
- `L6` `Architecture Design` ** — Design a visual search system that handles 100M product images with sub-200ms latency. Walk through the architecture
- `L5` `Conceptual` *** — Explain how DETR works and why it represents a paradigm shift in object detection
- `L6` `Trade-off` ** — Compare the ResNet skip connection to DenseNet dense connections to Swin Transformer's shifted windows. What problem does each solve?

### [DL 11] Generative Models

- `L5` `First Principles` *** — Explain the reparameterization trick and why it's needed for VAE training. What happens without it?
- `L6` `Trade-off` *** — Diffusion models vs GANs vs VAEs vs autoregressive models---give me a comparison table and decision framework for different generation tasks
- `L6` `Debugging` ** — Your diffusion model generates high-quality images overall but has mode collapse for certain categories (e.g., generates dogs well but all cats look the same). Diagnose and fix
- `L6` `Architecture Design` ** — Design an image generation pipeline for a product photography application (e.g., placing products in realistic scenes)
- `L5` `Conceptual` *** — What is classifier-free guidance and why does it improve generation quality in diffusion models?
- `L5` `First Principles` *** — Explain the ELBO derivation for VAEs and why the model produces blurry images. What fixes exist?
- `L7` `Estimation` * — Estimate the inference cost (FLOPs and latency) for generating a `512 times 512` image with Stable Diffusion on an A100 GPU
- `L7` `Conceptual` * — Flow matching vs denoising diffusion---what changed and why is it considered the next evolution?
- `L5` `Conceptual` *** — How does Stable Diffusion work end-to-end? Walk through the architecture from text prompt to generated image
- `L6` `First Principles` ** — Explain the WGAN loss and why Wasserstein distance solves GAN training instability

### [DL 12] Recommendation Systems

- `L5` `Conceptual` *** — How do you handle the cold start problem for new users and new items?
- `L5` `Trade-off` *** — Two-tower vs. interaction-based models for candidate generation---when would you use each?

### [DL 13] Graph Neural Networks

- `L6` `Architecture Design` ** — Design a GNN-based system for real-time fraud detection on a financial transaction graph
- `L6` `Trade-off` ** — When should you NOT use a GNN? Give examples where simpler approaches win
- `L5` `Debugging` ** — Your GNN's performance degrades as you add more layers. What's happening and how do you fix it?
- `L5` `Trade-off` ** — GCN vs GraphSAGE vs GAT---when would you use each?
- `L6` `Estimation` * — Estimate the memory requirements for running a 3-layer GCN on a graph with 10M nodes
- `L6` `Trade-off` ** — GCN vs GraphSAGE vs GAT---which would you choose for a production recommendation system at scale?
- `L6` `First Principles` * — What are the theoretical expressivity limitations of message-passing GNNs, and when do they matter in practice?
- `L5` `Conceptual` ** — How do you handle heterogeneous graphs (multiple node types, multiple edge types) in a GNN?

### [DL 14] Efficient Architectures

- `L6` `Architecture Design` *** — Design an efficient architecture for on-device ML (phone, `<`100ms latency, `<`50MB model)
- `L6` `Trade-off` ** — When would you choose MoE over dense scaling, and vice versa?
- `L6` `Trade-off` ** — You're building a system that processes 100K-token documents. Compare transformer vs Mamba for this use case
- `L6` `Estimation` ** — Estimate the compute savings of a MoE model with 8 experts, top-2 routing, vs a dense model of the same quality
- `L7` `Architecture Design` *** — You have a serving budget of 80,GB of GPU memory and 20,ms/token decode latency, with contexts up to 128K. Design the model
- `L6` `Trade-off` *** — INT8 vs INT4 vs FP8 quantization---what is your decision framework?
- `L6` `Debugging` ** — Your quantized model performs well on benchmarks but fails on specific domains (code, math). Why?
- `L6` `Architecture Design` ** — Knowledge distillation: design a pipeline to compress a 70B teacher to a 7B student
- `L6` `Debugging` ** — MoE routing collapse: what is it, how do you detect it, and how do you fix it?
- `L6` `Trade-off` ** — Structured vs unstructured pruning---which actually gives real speedup?
- `L7` `Architecture Design` * — Design an efficient architecture for real-time video understanding on edge devices

### [DL 15] Training Optimization

- `L5` `Debugging` *** — Your model's loss plateaus after initial descent. Walk through your debugging process
- `L6` `Architecture Design` *** — How would you scale training of a 70B parameter model across 256 GPUs?
- `L5` `Trade-off` *** — Adam vs SGD with momentum---when would you pick each?
- `L5` `Conceptual` ** — Explain gradient accumulation and when you would use it
- `L6` `Debugging` ** — Your model's loss spikes every N steps then recovers. Diagnose
- `L5` `Debugging` *** — Training loss is NaN after 1000 steps. Walk through your debugging process
- `L7` `Debugging` *** — Your 70B pretraining run diverges at 200B tokens. Give a full differential diagnosis, ordered by prior probability, with the cheapest discriminating experiment for each
- `L6` `Debugging` ** — Your distributed training job is 60% as fast as expected with 8 GPUs. What is wrong?
- `L7` `Debugging` *** — A node dies every 3 hours on your 16K-GPU run. Design the recovery system
- `L6` `Estimation` *** — How much GPU memory is needed to fine-tune a 13B parameter model with LoRA vs. full fine-tuning?
- `L7` `Estimation` * — Calculate the training throughput (tokens/sec) for a 7B model on 8`times`A100s
- `L5` `Trade-off` ** — Cosine decay vs. constant LR with warmup---when would you use each?
- `L6` `Trade-off` *** — Data parallelism vs. tensor parallelism vs. pipeline parallelism---what is your decision framework?
- `L6` `Trade-off` ** — FP16 vs. BF16 vs. FP8 for training---when would you use each?

### [DL 16] Transfer Learning and Parameter-Efficient Fine-Tuning

- `L5` `Estimation` *** — Estimate the number of trainable parameters for LoRA with rank 16 on a 7B model
- `L6` `Debugging` *** — Your LoRA fine-tuned model performs well on your test set but catastrophically forgets general capabilities. How do you fix it?
- `L6` `Debugging` *** — Your fine-tuned model echoes the prompt before answering---why?
- `L6` `Trade-off` *** — LoRA vs QLoRA vs full fine-tuning vs prompt tuning---give a complete decision framework
- `L6` `Trade-off` *** — You have chosen LoRA to adapt a 7B instruction-tuned model to an internal support-and-policy domain: 20K curated examples, one 80,GB GPU, and the adapter will be served alongside the general assistant. Pick rank, `alpha`, and target modules, and defend the choices
- `L6` `Architecture Design` ** — Design a fine-tuning pipeline for adapting a foundation model to a new domain with 10K examples
- `L5` `Conceptual` ** — What is negative transfer and how do you detect and prevent it?
- `L6` `First Principles` ** — Why does LoRA work? What is the low-rank assumption about fine-tuning?
- `L5` `Trade-off` ** — Adapter layers vs LoRA vs prefix tuning---what are the architectural differences and when would you use each?
- `L5` `Architecture Design` *** — You need to fine-tune a 70B LLM for a customer's specific domain. You have access to a single A100 80GB GPU. What approach would you take?
- `L6` `Debugging` *** — You fine-tuned a model and it performs well on your test set but poorly in production. What happened?
- `L6` `Architecture Design` ** — You have one base model and 100 customers, each with their own LoRA adapter. Design the serving system

### [DL 17] Reinforcement Learning and RLHF

- `L5` `Conceptual` *** — Walk me through the RLHF pipeline for aligning an LLM
- `L6` `Trade-off` *** — PPO vs DPO vs RLHF---complete comparison for LLM alignment
- `L5` `Conceptual` *** — What is reward hacking? Give 3 concrete examples and how to mitigate each
- `L6` `Debugging` *** — Your RLHF-tuned model becomes sycophantic (agrees with everything). Diagnose and fix
- `L6` `Estimation` ** — Estimate the memory required to run RLHF (4 models: policy, reference, reward model, critic)
- `L5` `First Principles` *** — Why does the KL penalty matter in RLHF? What happens without it?
- `L6` `Architecture Design` ** — Design a reward model training pipeline. What data do you need?
- `L6` `Trade-off` ** — Constitutional AI vs RLHF vs DPO---when would you use each?
- `L7` `Debugging` * — Your RLHF training shows the reward score plateauing while KL divergence keeps climbing. What is happening?
- `L6` `Trade-off` ** — Process reward models vs outcome reward models---what are the tradeoffs and when does each win?
- `L6` `First Principles` *** — Explain GRPO. Why did it replace PPO's critic for reasoning RL, and what breaks when you use it naively?
- `L6` `Debugging` ** — Your GRPO run's mean reward climbs but responses grow unboundedly long and benchmark scores stall---diagnose
- `L7` `Trade-off` *** — When is test-time compute cheaper than a bigger model---and when does it stop working?

### [DL 18] Multimodal Learning

- `L6` `Trade-off` ** — Cascade vs native speech-to-speech for a voice assistant---decide and defend
- `L6` `Architecture Design` *** — Design a multimodal content understanding system for a social media platform
- `L6` `Trade-off` *** — Adapter-based (LLaVA) vs natively multimodal (Gemini)---tradeoffs
- `L5` `First Principles` ** — How does CLIP's contrastive loss create a shared embedding space?
- `L5` `Conceptual` ** — CLIP vs SigLIP---what changed and why?
- `L7` `Estimation` * — Estimate the compute cost of training a CLIP model on 400M image-text pairs
- `L6` `Debugging` *** — Your VLM hallucinates visual details that are not in the image. Why and how to fix?
- `L6` `Trade-off` ** — Visual tokens consume 576 tokens of context. How do you reduce this cost?
- `L5` `Conceptual` *** — Compare early, late, and cross-attention fusion for multimodal models. When would you use each?
- `L6` `Architecture Design` ** — Design a multimodal search system that lets users search a product catalog using text or image queries
- `L6` `Architecture Design` * — How would you add audio understanding to an existing vision-language model?

### [DL 19] Inference Optimization and LLM Serving

- `L7` `Architecture Design` *** — Design an LLM serving stack that handles 1,000 queries per second with a p99 latency of 2 seconds for a 70B parameter model
- `L7` `Architecture Design` *** — Your capacity plan assumed 500-token outputs. The reasoning model you are now deploying emits `sim`20K thinking tokens per query. Redo the plan
- `L5` `Conceptual` *** — Explain speculative decoding. When would you use it and when would you not?
- `L6` `Debugging` *** — KV cache memory is your bottleneck---you are running out of GPU memory and dropping requests. What do you do?
- `L7` `Estimation` ** — Calculate the throughput (tokens/sec) of a 70B model on 8`times`H100s with tensor parallelism
- `L6` `Estimation` *** — How much memory savings does INT4 quantization give for a 70B model? What is the quality trade-off?
- `L7` `Estimation` * — Your LLM service needs to handle 500 concurrent users with 2-second time-to-first-token. Size the infrastructure
- `L6` `Trade-off` ** — GPTQ vs AWQ vs GGUF quantization---when would you use each?
- `L6` `Trade-off` *** — Tensor parallelism vs pipeline parallelism for inference---what is your decision framework?
- `L7` `Trade-off` * — Prefill-optimized vs decode-optimized serving---when should you disaggregate them?
- `L6` `Debugging` ** — Your LLM serving system's p99 latency is 10`times` the p50. Diagnose
- `L6` `Debugging` ** — After quantizing your model to INT4, certain types of prompts give garbage output while others are fine. Why?
- `L6` `Debugging` * — Your speculative decoding setup is slower than standard decoding. What went wrong?

### [DL 20] Safety, Alignment, and Interpretability

- `L6` `Architecture Design` *** — Design a content filtering pipeline for an LLM chatbot with `<`50ms added latency
- `L5` `Conceptual` *** — Jailbreak attacks: how do they work and how do you defend?
- `L6` `Conceptual` ** — What is mechanistic interpretability? Give a concrete example of a discovered circuit
- `L7` `Trade-off` ** — Your LLM refuses too many benign queries. How do you reduce over-refusal without increasing safety risk?
- `L6` `Architecture Design` ** — Hallucination detection: design a system that flags when an LLM generates unsupported claims
- `L5` `Conceptual` *** — How do you detect and reduce hallucination in a production LLM system?
- `L6` `Architecture Design` * — How do you set up a red teaming program for an LLM product?
- `L7` `Conceptual` * — How can you use representation engineering (steering vectors) to control model behavior without retraining?
- `L7` `Architecture Design` *** — Design safety for an agent that can browse the web and execute code
- `L6` `Debugging` ** — A jailbreak goes viral against your production model---walk me through the first 24 hours

### [DL 21] Long Context and RAG Systems

- `L6` `Trade-off` *** — Your 8K-trained model must serve 64K contexts next month. What are your options, and what do they cost?
- `L6` `Debugging` ** — Your model's quality degrades beyond 32K context. Walk me through the diagnosis
- `L6` `Estimation` *** — Estimate the cost per query for a RAG system vs a long-context LLM (128K)
- `L5` `Trade-off` *** — Long context window vs RAG---when do you choose each?

### [DL 22] Production ML Systems

- `L6` `Architecture Design` *** — Design ML infrastructure for a search ranking system at 10K QPS
- `L6` `Debugging` *** — Your model's CTR dropped 5% overnight. Walk through your diagnosis
- `L6` `Architecture Design` *** — How would you set up an A/B test for a new recommendation model?
- `L5` `Conceptual` *** — How do you handle training-serving skew?
- `L6` `Debugging` *** — Your model's online metrics diverge from offline metrics after 2 weeks in production. What is happening?
- `L7` `Estimation` ** — Estimate the infrastructure cost for serving a recommendation model at 100K QPS
- `L6` `Architecture Design` ** — Design the monitoring system for an ML model in production
- `L5` `Trade-off` *** — Shadow deployment vs A/B test vs multi-armed bandit---when do you use each?
- `L5` `Conceptual` ** — How do you detect and handle concept drift vs data drift?
- `L6` `Architecture Design` ** — Feature store: batch vs streaming features---describe the architecture and tradeoffs

### [DL 23] Evaluation Metrics

- `L6` `Estimation` *** — Model B is 1.5 points better than model A on MMLU---do you ship it?
- `L7` `Architecture Design` *** — Design the eval stack that decides whether tomorrow's checkpoint ships
- `L5` `Conceptual` *** — Your model has 95% accuracy but stakeholders are unhappy. What is going on?
- `L5` `Conceptual` *** — How do you evaluate a search ranking system?
- `L5` `Trade-off` *** — Precision is 0.95 but recall is 0.30. What do you do?
- `L5` `Conceptual` *** — How do you know if a model improvement is statistically significant?
- `L6` `Debugging` *** — Your model's AUC is 0.95 but calibration is poor (ECE = 0.15). Why does this matter and how do you fix it?
- `L5` `Trade-off` *** — NDCG vs MAP vs MRR for ranking---when do you use each?
- `L6` `Estimation` ** — Estimate the sample size needed for an A/B test to detect a 1% improvement in CTR
- `L7` `Architecture Design` ** — Design an evaluation framework for an LLM-based chatbot
- `L6` `Debugging` *** — Offline metrics went up but online metrics went down. Give 5 possible reasons
- `L5` `Conceptual` *** — What is wrong with accuracy as a metric? When is it actually fine?

### [DL 24] Decision Frameworks: When to Use What

- `L6` `Architecture Design` *** — Design a content recommendation system for a news app
- `L6` `Architecture Design` *** — Design the ML architecture for a content moderation system
- `L6` `Trade-off` *** — Your model needs to run in `<`10ms. Current latency is 50ms. Walk through the optimization decision tree
- `L6` `Trade-off` *** — You have 3 months and a team of 3 ML engineers. Should you fine-tune an LLM or train a custom model?
- `L5` `Trade-off` ** — GPU vs TPU vs CPU for inference---give a decision framework for different workloads
- `L7` `Trade-off` ** — Build vs buy vs open-source---give a decision framework for ML infrastructure
- `L5` `Trade-off` *** — Would you use BERT or GPT for sentiment analysis?
- `L5` `Trade-off` *** — When should you use gradient boosting vs. neural networks for tabular data?

### [DL 25] Pretraining at Scale

- `L5` `Conceptual` *** — Walk me through what happens between ``we have a Common Crawl snapshot'' and the first training step
- `L7` `System Design` *** — You own data for the next frontier pretraining run. Design the pipeline and the ablation program
- `L6` `Trade-off` *** — How would you decide the code:web:math mixture for a pretraining run?
- `L6` `Debugging` ** — After a data-pipeline change, training loss improved---but downstream evals got worse. Diagnose
- `L5` `Conceptual` ** — Why does deduplication improve language models, and what goes wrong if you deduplicate too aggressively?
- `L6` `System Design` ** — Design the benchmark decontamination strategy for a pretraining run. What does n-gram overlap miss, and what do you do about it?
- `L6` `Conceptual` ** — What is mid-training (annealing), why does everyone do it now, and what is the catch?
- `L7` `Trade-off` *** — Your data lead proposes raising the synthetic fraction of the next run from 10% to 40%. How do you evaluate the proposal?
- `L6` `Trade-off` ** — You are choosing between a 32K, 128K, and 256K vocabulary for a new model family. What changes, and when would you retrain vs. reuse a tokenizer?
- `L6` `System Design` ** — Design the in-flight evaluation for a three-month pretraining run. When do in-loop evals mislead?
- `L6` `Debugging` * — At 40% through a flagship run, held-out code loss jumps and stays elevated. Walk me through the response

### [DL 26] GPU Performance Fundamentals

- `L5` `Conceptual` ** — Walk me through the GPU memory hierarchy. Why does it, rather than FLOPs, dominate ML performance thinking?
- `L5` `First Principles` ** — Why do tensor cores only accelerate matrix multiplication? What does that imply for how you design models and kernels?
- `L6` `Estimation` *** — I give you an op---say, a LayerNorm over a `[16{,}384 times 8{,}192]` BF16 tensor, or a `4096^3` GEMM. Is it compute-bound or bandwidth-bound on an H100? Show your method
- `L6` `Debugging` *** — Your training job shows the GPU at 20% utilization. Diagnose it
- `L6` `First Principles` ** — Kernel fusion doesn't reduce FLOPs. Why does it make models faster---and when does it stop helping? Use FlashAttention as your example
- `L6` `Estimation` ** — How long does it take to all-reduce the gradients of a 70B-parameter model across 8 GPUs? Derive it
- `L6` `Debugging` * — A teammate changed the hidden size from 4096 to 4100 ``to add a few features,'' and training throughput dropped 35%. What happened?
- `L6` `Trade-off` * — What problem do CUDA graphs solve, and what constraints do they impose? Where do they matter most in an LLM system?
- `L7` `Estimation` *** — Estimate the maximum tokens/second for a 70B dense model on 8`times`H100---for training, and for inference
- `L7` `Debugging` ** — Your 512-GPU pretraining run reports 25% MFU. Walk me through how you find the missing performance
- `L7` `Architecture Design` ** — You have 8 nodes of 8`times`H100. Map tensor, pipeline, data, and expert parallelism onto this cluster's fabric, justifying the mapping from link characteristics---then tell me what changes on a GB200 NVL72

### [DL 27] Agents and Tool Use

- `L5` `Conceptual` *** — What makes an LLM system an ``agent''? Walk me through the agent loop and where the engineering difficulty actually lives
- `L6` `Trade-off` *** — When should you NOT build an agent? Your PM wants ``an agent'' for a document-processing product---how do you decide?
- `L6` `System Design` *** — Design a code-review agent for your organization's monorepo
- `L7` `Debugging` *** — Your agent succeeds on 60% of an internal task suite. Leadership wants 95%. Take it there
- `L6` `System Design` ** — Design the evaluation for a customer-support agent that can read accounts, process refunds, and escalate to humans
- `L5` `Conceptual` ** — Explain pass@`k` vs. pass`k`. Your agent team reports 92% pass@5---what do you tell the VP who wants to ship?
- `L6` `Trade-off` ** — One agent with 30 tools, or an orchestrator with specialized sub-agents? Argue both sides with numbers
- `L6` `Conceptual` ** — Why do computer-use agents lag API-based agents so badly, and when would you deploy one anyway?
- `L7` `System Design` ** — Your agent's tasks run for hours and blow past the context window. Design its memory and state management
- `L6` `Trade-off` ** — Your agent product works but loses money: `4 of inference per task against `1.50 of revenue. Fix the economics without wrecking the success rate
- `L5` `Conceptual` ** — What does MCP actually standardize, and what does it change---and not change---about how you architect an agent?

### [DL 28] ML Coding Rounds

- `L5` `Coding` *** — Implement multi-head self-attention with a causal mask in numpy
- `L5` `Coding` *** — Implement one AdamW update step in numpy
- `L6` `Coding` ** — Implement LayerNorm forward and backward in numpy
- `L5` `Coding` *** — Write a minimal training loop with gradient clipping in PyTorch
- `L6` `Coding` * — Implement 2-D convolution as a matrix multiply (im2col)
- `L5` `Coding` *** — Implement BPE---train the merges on a corpus, then encode a word
- `L5` `Coding` *** — Implement sampling from logits with temperature, top-`k`, and top-`p`
- `L6` `Coding` *** — Implement greedy decoding with a KV cache
- `L6` `Coding` ** — Implement beam search with length normalization
- `L5` `Coding` ** — Build a toy inverted index and score queries with BM25
- `L5` `Coding` *** — Implement NDCG@k
- `L6` `Coding` ** — Implement the in-batch softmax loss for a two-tower retrieval model
- `L6` `Coding` ** — Implement greedy graph search over a fixed HNSW-style neighbor graph
- `L5` `Coding` *** — Implement logistic regression with SGD from scratch, deriving the gradient
- `L5` `Coding` *** — Implement k-means with k-means++ initialization
- `L6` `Coding` ** — Implement gradient boosting with decision stumps

### [DL 29] The Staff Loop

- `L6` `Behavioral` *** — Walk me through your most impactful project. (Then 40 minutes of drilling.)
- `L6` `Behavioral` *** — A peer team's launch is hurting your metric. Their leadership is celebrating it. What do you do?
- `L7` `Behavioral` ** — If we gave you a year and a small team, what would you work on, and why?
- `L6` `Behavioral` ** — Here's a paper claiming a new post-training method beats RLHF. Critique its evaluation
- `L6` `Behavioral` ** — You're two days from a launch your team has crunched for, and an eval shows a small but real regression in a harm category. Walk me through what you actually do
- `L6` `Behavioral` *** — Describe committing to a technical decision you argued against. Would you do it again?
- `L7` `Behavioral` ** — If you joined and we gave you no direction for your first month, what would you do---concretely, in week one?
- `L6` `System Design` *** — I'm going to give you a deliberately vague prompt---``design memory for our assistant.'' Run the first ten minutes

## Volume II — NLP Essentials

### [NLP 1] Classical NLP Foundations

- `L5` `Conceptual` *** — What is TF-IDF? Derive the formula. When would you use it over neural embeddings?
- `L5` `Conceptual` *** — Explain BM25. Why is it still used in production search systems?
- `L5` `Conceptual` ** — What is the difference between stemming and lemmatization? When does the choice matter?
- `L5` `Conceptual` ** — What is the BIO tagging scheme? Why not just classify each token independently?
- `L5` `Conceptual` *** — Explain the difference between rule-based, statistical, and neural NLP. When would you still use rule-based methods?
- `L5` `Conceptual` ** — What problems does the bag-of-words assumption create? How have subsequent methods addressed them?
- `L6` `System Design` ** — Design a text preprocessing pipeline for a multilingual sentiment analysis system covering 20+ languages
- `L5` `Conceptual` ** — Why is context modeling crucial in NLP? Give examples where lack of context leads to failures
- `L5` `Conceptual` ** — When would you choose Naive Bayes over a fine-tuned BERT for text classification? Justify your answer
- `L6` `System Design` ** — Design the classical-ML filtering stage for a 10T-token pretraining corpus: quality filtering, deduplication, and benchmark decontamination

### [NLP 2] Statistical NLP and Probabilistic Models

- `L5` `Mathematical` ** — Explain the Viterbi algorithm. What is its time complexity? Trace through a small example
- `L5` `Conceptual` *** — Why use a CRF layer on top of BERT for NER instead of just a softmax classifier?
- `L5` `Mathematical` *** — What is perplexity? How does it relate to cross-entropy? What are its limitations as an evaluation metric?
- `L5` `Conceptual` ** — Compare generative and discriminative sequence models. Why are CRFs generally preferred over HMMs for sequence labeling?
- `L6` `Mathematical` * — Explain Kneser-Ney smoothing. Why is it better than add-1 (Laplace) smoothing?
- `L5` `Conceptual` * — Explain LDA's generative process. What are the latent variables and how is inference performed?
- `L5` `Conceptual` ** — How do n-gram language models relate to modern neural language models? What did each generation preserve and what did it change?
- `L6` `Mathematical` ** — What is the partition function in a CRF? Why is it computationally challenging, and how is it computed efficiently?
- `L5` `Conceptual` * — Explain the noisy channel model for machine translation. Why was it eventually replaced by neural MT?
- `L5` `Applied` ** — You are building a named entity recognizer for a new domain (e.g., legal documents) with only 500 labeled sentences. Walk me through your approach

### [NLP 3] Word Representations and Embeddings

- `L5` `Mathematical` *** — Explain the Word2Vec Skip-gram model. What is the training objective? What is negative sampling and why is it needed?
- `L5` `Conceptual` *** — What are the differences between Word2Vec, GloVe, and FastText? When would you choose each one?
- `L5` `Conceptual` *** — Why are contextual embeddings better than static embeddings? Give a concrete example of when static embeddings fail
- `L5` `Conceptual` ** — How would you evaluate the quality of word or sentence embeddings?
- `L5` `Conceptual` ** — What is the distributional hypothesis and why is it foundational to all word embedding methods?
- `L5` `Conceptual` ** — How does FastText handle out-of-vocabulary words? Why is this an improvement over Word2Vec?
- `L6` `Conceptual` ** — Explain SimCSE. How does using dropout as data augmentation create meaningful positive pairs for contrastive learning?
- `L6` `System Design` ** — How would you choose an embedding model for a production semantic search system?
- `L6` `Mathematical` ** — Explain the connection between Word2Vec, GloVe, and matrix factorization on co-occurrence statistics. Why does this connection matter?
- `L6` `Debugging` *** — Your new embedding model wins offline---`+4` points Recall@100 on your evaluation set---but loses the online A/B test. Walk me through your investigation

### [NLP 4] Neural Sequence Models

- `L5` `Mathematical` *** — Explain the vanishing gradient problem in RNNs. Why does LSTM solve it? Walk through the math
- `L5` `Conceptual` *** — Walk me through the LSTM cell gate by gate. What is each gate's purpose?
- `L5` `Conceptual` *** — Why did transformers replace LSTMs? What specific limitations did they address?
- `L5` `Conceptual` *** — Explain the attention mechanism in seq2seq. Why was it needed?
- `L5` `Conceptual` ** — What is teacher forcing? What problems can it cause?
- `L6` `Conceptual` ** — Compare additive (Bahdanau) and multiplicative (Luong) attention. When would you prefer each?
- `L6` `Trade-off` ** — When would you still use an LSTM over a transformer?
- `L5` `Mathematical` ** — Explain Backpropagation Through Time (BPTT). How is it different from standard backpropagation?
- `L5` `Conceptual` ** — What are the key differences between LSTM and GRU? When would you choose one over the other?
- `L5` `Conceptual` ** — A candidate proposes using a bidirectional LSTM for a text generation task. What is wrong with this proposal?

### [NLP 5] Transformers and Modern Architectures

- `L5` `Mathematical` *** — Explain self-attention step by step, including the math
- `L5` `Mathematical` *** — Why do we scale by `d_k`? Derive the variance argument
- `L5` `Conceptual` *** — Why do we need multiple attention heads? What happens with just one head?
- `L5` `Conceptual` *** — Compare encoder-only, decoder-only, and encoder-decoder transformers. Why did decoder-only dominate the LLM era?
- `L5` `Mathematical` *** — What is the computational complexity of self-attention? Why is it a problem?
- `L6` `Conceptual` ** — Explain RoPE (Rotary Position Embedding). Why is it better than learned absolute positions for long sequences?
- `L6` `Conceptual` ** — What does the FFN layer do in a transformer? Is there evidence it stores factual knowledge?
- `L5` `Conceptual` *** — Explain BERT's pretraining objectives. Why was NSP removed in RoBERTa?
- `L6` `System Design` ** — What is Flash Attention? Why does it help if it computes the same result as standard attention?
- `L6` `Trade-off` ** — Compare Multi-Query Attention, Grouped-Query Attention, and standard Multi-Head Attention. When would you use each?
- `L5` `Conceptual` ** — Explain pre-norm vs. post-norm in transformers. Why did pre-norm become the standard?
- `L5` `Trade-off` ** — You need to build a production NLP system that classifies customer support tickets into 50 categories. Would you use BERT or GPT-4? Justify your choice

### [NLP 6] Pre-training, Fine-tuning, and Transfer Learning

- `L5` `Trade-off` *** — When would you use LoRA vs. full fine-tuning? When would LoRA fail?
- `L5` `System Design` *** — How would you fine-tune an LLM for a domain-specific task with only 1,000 labeled examples?
- `L5` `Conceptual` *** — What is catastrophic forgetting? How do you mitigate it during fine-tuning?
- `L5` `Conceptual` ** — Why is data quality more important than data quantity for SFT?
- `L6` `Conceptual` ** — Explain ELECTRA's pre-training approach. Why is it more efficient than MLM?
- `L5` `Conceptual` ** — What is the difference between feature extraction and fine-tuning? When would you use each?
- `L6` `Trade-off` ** — Compare the major pre-training objectives: MLM, CLM, span corruption, and ELECTRA. When would you choose each?

### [NLP 7] Large Language Models and In-Context Learning

- `L5` `Conceptual` *** — Explain scaling laws for LLMs. What does the Chinchilla paper tell us about compute-optimal training?
- `L5` `Conceptual` *** — Why does chain-of-thought prompting improve reasoning? What are its limitations?
- `L6` `Conceptual` *** — Explain in-context learning. Why can models learn from examples in the prompt without gradient updates?
- `L6` `Conceptual` ** — How do Mixture-of-Experts models work? What is the load balancing problem?
- `L6` `Trade-off` ** — When would you choose Mamba (or an SSM) over a Transformer?
- `L5` `System Design` *** — What is prompt injection? How would you defend against it in a production system?
- `L6` `Conceptual` ** — Explain test-time compute scaling (o1-style reasoning). How does it differ from simply making a bigger model?
- `L5` `Trade-off` *** — Compare few-shot prompting, fine-tuning, and RAG for adapting an LLM to a new task or domain. When would you use each?
- `L6` `Conceptual` ** — What are the key factors that affect in-context learning performance? Why do even random labels partially work?
- `L6` `System Design` ** — How would you handle the ``lost in the middle'' problem when building a system that processes long documents with an LLM?
- `L5` `Conceptual` *** — How does an LLM actually call a tool? Walk me through the full round trip from tool definition to final answer

### [NLP 8] RLHF and Alignment

- `L7` `System Design` *** — Design the RLHF/RLVR training system for a 70B model
- `L5` `Conceptual` *** — Walk me through the RLHF pipeline step by step. What are the three stages?
- `L5` `Conceptual` *** — What is reward hacking? How does the KL penalty help? When does it fail?
- `L6` `Mathematical` *** — Explain DPO. How does it relate to RLHF? What are its advantages and limitations?
- `L6` `Conceptual` ** — What is Constitutional AI? How does self-critique enable alignment without human feedback at every step?
- `L6` `Trade-off` ** — Compare RLHF, DPO, and KTO. When would you use each?
- `L5` `System Design` ** — What makes a good reward model? How do you evaluate it?
- `L7` `Mathematical` ** — Derive the DPO loss from the RLHF objective
- `L6` `Conceptual` ** — What is the difference between process and outcome reward models? When would you use each?

### [NLP 9] Retrieval-Augmented Generation

- `L5` `System Design` *** — Design a RAG system for a customer support chatbot that answers questions using your company's knowledge base. Walk through the key components and design decisions
- `L5` `Trade-off` *** — Compare BM25 vs. dense retrieval. When would you use each? When would you combine them?
- `L5` `System Design` *** — What chunking strategies exist for RAG? How do you choose chunk size?
- `L5` `Evaluation` *** — How would you evaluate a RAG system end-to-end? What metrics would you track?
- `L6` `Conceptual` ** — What is HyDE (Hypothetical Document Embeddings)? When would it help and when would it fail?
- `L6` `System Design` ** — How do you handle the ``lost in the middle'' problem in RAG?
- `L5` `Trade-off` *** — When should you use RAG vs. fine-tuning vs. long context?
- `L5` `Conceptual` *** — Explain the bi-encoder vs. cross-encoder tradeoff in retrieval. How would you use both in a production system?
- `L6` `Debugging` ** — Your RAG system is returning correct documents but the LLM's answers are still wrong. What would you investigate?
- `L6` `System Design` ** — How would you scale a RAG system to handle 10 million documents and 1000 queries per second?

### [NLP 10] Evaluation, Metrics, and Decoding

- `L5` `Mathematical` *** — What is BLEU? Derive it step by step. What are its limitations?
- `L5` `Mathematical` *** — What is perplexity? How does it relate to cross-entropy? What are its limitations?
- `L5` `System Design` *** — How would you evaluate a summarization system? Specifically, how would you measure faithfulness?
- `L5` `Conceptual` ** — What are the problems with LLM-as-judge evaluation? When would you use it and when would you not?
- `L5` `Conceptual` *** — Explain ROUGE. How is it different from BLEU?
- `L5` `Conceptual` ** — Compare greedy decoding, beam search, and nucleus sampling. When would you use each?
- `L6` `Conceptual` ** — How do you handle benchmark contamination? How would you design an evaluation pipeline that is robust to it?
- `L6` `System Design` ** — Design an evaluation pipeline for a production chatbot that handles customer support for an e-commerce company
- `L6` `Trade-off` ** — You are building a code generation system. BLEU is commonly reported for code generation---is it a good metric? What alternatives would you propose?
- `L6` `Trade-off` ** — A model achieves state-of-the-art on MMLU but users report it gives poor answers in practice. What could explain this gap?
- `L6` `Trade-off` *** — Your model gained 6 points on the internal benchmark suite, but users can't tell the difference---what happened?

### [NLP 11] Practical NLP Tasks and Applications

- `L5` `System Design` *** — How would you build a text classification system? Walk through the full pipeline from data to deployment
- `L5` `System Design` *** — How would you build a production NER system for a new domain (e.g., biomedical) with limited labeled data?
- `L5` `Trade-off` *** — Compare extractive vs. abstractive summarization. What are the tradeoffs and when would you use each?
- `L6` `System Design` ** — Design a question answering system for a company's internal documentation (Confluence, Notion, internal wikis)
- `L5` `Conceptual` ** — How would you handle aspect-based sentiment analysis at scale?
- `L5` `Trade-off` ** — Compare Naive Bayes vs. fine-tuned BERT for text classification. When might Naive Bayes win?
- `L6` `Conceptual` * — How would you detect sarcasm in text? Why is this hard?
- `L6` `System Design` ** — Design a task-oriented dialogue system for restaurant booking
- `L6` `System Design` ** — You need to build an information extraction system that populates a knowledge graph from news articles. Walk through your approach
- `L5` `Conceptual` *** — When is an LLM overkill for an NLP task? Give concrete examples

### [NLP 12] Production NLP and Deployment

- `L6` `System Design` *** — How would you serve an LLM at 10K QPS with less than 500ms time-to-first-token?
- `L5` `Conceptual` *** — Explain the KV-cache. Why is it necessary for autoregressive generation? How much memory does it use?
- `L6` `Conceptual` ** — What is speculative decoding? Why does it produce the same distribution as standard autoregressive decoding?
- `L6` `Trade-off` ** — Compare GPTQ, AWQ, and INT8 quantization. When would you use each?
- `L5` `System Design` *** — How would you reduce the latency of an NLP inference system by 5`times`?
- `L5` `Conceptual` *** — Explain BPE tokenization. How is it trained? How does it handle OOV words?
- `L5` `Trade-off` ** — When would you choose knowledge distillation over quantization or pruning to compress a model?
- `L6` `System Design` ** — How do you handle the ``tokenizer tax'' when deploying NLP systems for multilingual users?
- `L6` `System Design` ** — You have deployed an LLM-powered feature and users report quality has degraded over the past week, but your automated metrics show no change. How do you diagnose this?
- `L6` `System Design` * — Describe a model routing strategy that reduces serving cost by 5`times` while maintaining quality

### [NLP 13] Safety, Ethics, and Responsible AI

- `L5` `System Design` *** — How do you detect and mitigate hallucinations in an LLM-based system? Describe a practical defense strategy
- `L5` `System Design` *** — What is prompt injection? Design a defense system for a production LLM application
- `L5` `Conceptual` *** — How do you measure and reduce bias in an NLP system? Walk me through your approach for a resume screening application
- `L6` `System Design` ** — How would you build a content moderation pipeline for a social media platform serving 100M+ users?
- `L5` `Conceptual` ** — What ethical considerations arise when deploying LLMs at scale? How do you address them in practice?
- `L6` `Trade-off` ** — How do you handle the tradeoff between helpfulness and safety in an LLM system? Give concrete examples
- `L6` `System Design` ** — What is red teaming for LLMs? How would you design a red teaming process for a new model before deployment?
- `L6` `Conceptual` * — How does text watermarking for LLMs work? What are the tradeoffs?

## Volume III — Search & Recommendation Essentials

### [SR 1] Search Systems and Query Understanding

- `L5` `Conceptual` *** — Walk me through what happens, end to end, when a user types a query into a large-scale search engine
- `L6` `System Design` *** — Design the query understanding system for an e-commerce search engine
- `L6` `First Principles` ** — Design spell correction for a commerce search engine. What breaks a naive dictionary approach?
- `L6` `Trade-off` ** — How do you mine synonyms for query expansion without a hand-built ontology?
- `L7` `Trade-off` ** — When should a query understanding signal change retrieval, and when should it only be a ranking feature?
- `L7` `System Design` ** — You inherit a search system that is a single BM25 stage with a 15% zero-result rate. Sequence your first three quarters of investment
- `L6` `Debugging` *** — Your new L2 reranker improves offline NDCG by 4%, but online conversions are flat. Walk through your diagnosis
- `L5` `Conceptual` ** — Explain the head/torso/tail decomposition of query traffic. Why must strategy differ by segment?
- `L6` `Trade-off` ** — Where do LLMs fit in query understanding in 2026? Your PM wants ``LLM-powered search''---what do you actually build?
- `L5` `Debugging` * — Your search engine performs well in English but users report it is nearly useless in Japanese. Diagnose
- `L6` `Debugging` ** — Search relevance metrics degraded starting last Tuesday. No model was retrained. Where do you look?
- `L6` `First Principles` ** — Why is search a multi-stage funnel at all? Why not run your best model over the whole corpus for every query?

### [SR 2] Lexical Retrieval and Inverted Indexes

- `L5` `System Design` *** — Walk me through, mechanically, how BM25 over an inverted index returns the top-10 results from a billion documents in under 50,ms
- `L5` `Conceptual` *** — What do BM25's `k_1` and `b` actually control, and when would you change them from the defaults?
- `L6` `Debugging` ** — A search for ``red dress'' ranks a product titled ``Red'' with ``dress'' scattered through a long description above an exact ``Red Dress'' title match. Diagnose and fix
- `L6` `Trade-off` ** — Index-time vs. query-time synonyms---walk through the trade-offs and give your production policy
- `L6` `System Design` ** — Explain WAND and block-max WAND. Why are they safe, and when does the pruning stop helping?
- `L5` `Conceptual` * — Why do search engines compress postings so aggressively when disk is cheap? Walk through the encoding stack
- `L7` `System Design` ** — You must index 10B documents. Design the partitioning, tiering, and caching. Why did document partitioning beat term partitioning?
- `L6` `Conceptual` ** — Explain SPLADE. What do the log-saturation and the FLOPS regularizer each do, and how does it compare to BM25 and dense retrieval at serving time?
- `L6` `Trade-off` *** — Lexical vs. dense vs. hybrid---argue the retrieval split for a marketplace search engine
- `L5` `Conceptual` * — How do phrase queries actually execute, and what do positional postings cost? When would you use phrase boosts vs. phrase matching vs. shingles?

### [SR 3] Vector Retrieval Theory

- `L5` `Conceptual` *** — How does HNSW work, and why is it fast?
- `L6` `Architecture Design` *** — Index 500M `times` 768-d embeddings and serve top-100 under 15,ms p99 on a machine with 64,GB RAM---walk me through the design
- `L6` `Debugging` ** — Your ANN recall dropped after a reindex---same code, same corpus size. Debug it
- `L5` `Conceptual` ** — Why doesn't a `k`-d tree work for 768-dimensional embeddings, and what minimal changes rescue tree-based indexes?
- `L6` `Debugging` ** — A recsys team L2-normalizes item embeddings from a dot-product-trained two-tower model so they can reuse a cosine HNSW index. Ranking quality drops. Diagnose and fix
- `L6` `Debugging` ** — Your HNSW recall@10 drops from 0.95 to 0.6 when users apply a metadata filter matching 2% of the corpus. Why, and what are the fixes?
- `L5` `Conceptual` ** — Walk me through how a PQ index computes distances without decompressing anything, and where the time goes
- `L7` `Trade-off` * — When and why does ScaNN's anisotropic quantization beat plain PQ?
- `L6` `Estimation` * — You are offered an LSH family with `p_1 = 0.8`, `p_2 = 0.5` at your target radius. Size an index for 100M vectors and decide whether to use it
- `L7` `Conceptual` * — Why does greedy search on a proximity graph find the nearest neighbor at all? What breaks in high dimensions, and which practical graph has worst-case guarantees?
- `L6` `Architecture Design` ** — Tune an IVF index for 100M vectors: how many clusters? And what do you do when latency headroom remains but recall plateaus as you raise nprobe?
- `L6` `Debugging` * — Your ANN benchmark on synthetic Gaussian vectors shows terrible recall at any latency, but the same index on real embeddings is fine. Explain

### [SR 4] Neural Retrieval and Reranking

- `L5` `Conceptual` *** — Bi-encoder vs. cross-encoder: why not cross-encode everything, and what exactly does the cross-encoder buy?
- `L6` `First Principles` ** — Why do bi-encoders train with such large batches? Explain in-batch negatives and the logQ correction
- `L6` `Architecture Design` ** — Design the training recipe for a domain bi-encoder from your search logs. Walk through negatives, denoising, and distillation
- `L6` `Trade-off` ** — When does late interaction (ColBERT) earn its cost? Walk through the storage math
- `L7` `Trade-off` ** — A listwise LLM reranker beats your production cross-encoder by 3 NDCG points offline---at 200`times` the cost. What ships?
- `L6` `Debugging` *** — Your new reranker improved offline NDCG by 4%, but online CTR dropped. Walk through the diagnosis
- `L5` `Conceptual` *** — You run BM25 and dense retrieval in parallel. How do you combine them---and why does `0.5 cdot BM25 + 0.5 cdot cosine` fail?
- `L6` `System Design` *** — Design semantic search over 100 million documents, end to end
- `L6` `Debugging` ** — Dense retrieval tanks on queries containing part numbers. Fix it without abandoning dense retrieval
- `L6` `Conceptual` * — Your reranker's score gates behavior downstream---``no results'' messaging and RAG context filtering. What breaks, and how do you make the scores trustworthy?

### [SR 5] Learning to Rank

- `L5` `Conceptual` ** — Compare pointwise, pairwise, and listwise learning to rank. When is each the right choice?
- `L6` `Communication` *** — Explain LambdaRank to a strong engineer who knows GBDT but has never done ranking
- `L6` `Debugging` *** — Your click-trained ranker keeps favoring the items that have historically sat at position 1---better new items never rise. Diagnose and fix
- `L5` `Applied` ** — Walk me through constructing an LTR training set from a marketplace's search logs: labels, negatives, and splits
- `L5` `Trade-off` ** — When would you insist on a pointwise objective even though the product is a ranked list?
- `L6` `Trade-off` *** — GBDT or neural network for your L2 ranker? Decide for (a) a commerce search engine with rich engineered features, (b) a feed ranker over user history and item IDs
- `L6` `Applied` ** — How would you estimate position-bias propensities without degrading the user experience?
- `L6` `Debugging` ** — Your new LTR model improves overall NDCG but tail-query relevance regresses. Why, and what do you change?
- `L6` `Judgment Call` ** — Editorial judgments say document A beats B for this query; click data says B massively outperforms A. Which do you trust, and what do you do?
- `L7` `System Design` ** — Design the end-to-end unbiased-LTR loop for a search product---logging through training through evaluation---and tell me where it silently breaks
- `L7` `Depth` * — Your IPS-weighted training runs are unstable---a few examples dominate the gradient and validation NDCG oscillates. What is happening and what are your options?

### [SR 6] Recommendation Systems

- `L6` `Architecture Design` *** — Design a recommendation system for a video streaming platform with 100M users and 1M videos
- `L5` `Conceptual` *** — How do you handle the cold start problem for new users and new items?
- `L6` `Debugging` *** — Your recommendation model has great offline metrics but poor online performance. What went wrong?
- `L6` `Architecture Design` *** — Design the ranking model for an e-commerce product ads system
- `L7` `Architecture Design` ** — Design a real-time personalized notification system
- `L7` `Architecture Design` * — Design the embedding infrastructure for a recommendation system with 1B items
- `L6` `Estimation` *** — Estimate the memory required for DLRM embedding tables with 100M users, 10M items, and 1000 categorical features
- `L6` `Estimation` ** — Your ads ranking model needs to score 100 candidates in `<`50ms. What architecture constraints does this impose?
- `L7` `Estimation` * — Calculate the daily training data volume for an ads system serving 1B impressions/day
- `L6` `Debugging` *** — Your CTR model's offline AUC improved by 0.5% but online revenue dropped 3%. Diagnose
- `L7` `Debugging` ** — Your recommendation model shows high engagement but users are churning. What is happening?
- `L5` `Trade-off` *** — Two-Tower vs. interaction-based models for candidate generation---when would you use each?
- `L6` `Trade-off` ** — Shared-bottom vs. MMoE vs. PLE for multi-task ads modeling---give me a decision framework
- `L6` `Conceptual` ** — Explain the sample selection bias problem in CVR prediction. How does ESMM solve it?
- `L5` `Conceptual` *** — Why do recommendation models need calibration? What happens if pCTR is systematically over-confident?
- `L6` `Trade-off` ** — You run retrieval for a marketplace with 200M listings and heavy daily churn. Two-tower + ANN or generative retrieval with semantic IDs?
- `L6` `Conceptual` *** — Where would you use an LLM in a recommendation stack today---and why won't it replace your ranker?

### [SR 7] Search and RecSys Evaluation

- `L5` `Metric Derivation` *** — Derive NDCG from first principles, then compute NDCG@3 for a ranking with grades `(3, 0, 2)`, exponential gain, given the ideal available grades are `(3, 2, 0)`
- `L5` `Conceptual` ** — You have binary judgments only. When do Precision@`k`, Recall@`k`, MRR, and MAP each answer the right question---and construct a case where two of them disagree about which of two systems is better
- `L6` `Debugging` *** — Your reranker improved offline NDCG@10 by 3% on the judgment set, but the A/B test shows flat CTR. Walk me through your investigation
- `L6` `Trade-off` ** — Interleaving vs. A/B testing for a ranking change---how does team-draft interleaving work, why is it more sensitive, and when would it mislead you?
- `L6` `Debugging` ** — You replace a lexical retriever with a dense retriever. Offline NDCG on the existing judgment set drops. Is the new system worse?
- `L6` `Protocol Design` ** — A colleague evaluates a new sequential recommender with a random 80/20 interaction split and HR@10 against 100 sampled negatives, and reports a 12% win. What is wrong, and what protocol do you require before believing it?
- `L6` `Judgment Call` ** — Your new recommender lifts NDCG and short-term engagement, but intra-list diversity drops, catalog coverage falls, and exposure Gini rises 6 points. Do you ship it---and how should the evaluation have been set up so this is not a debate?
- `L7` `First Principles` ** — Without launching it, estimate what CTR a new ranking policy would achieve, using only logs from the current system. Derive the estimator, its requirements, and its failure modes
- `L6` `Program Design` ** — You want to replace most crowd relevance labeling with an LLM judge. Design the program so the labels are trustworthy---and tell me what stays human
- `L7` `Program Design` * — You own relevance for a search org of several teams. Design the evaluation program: what gets measured, at what layer, on what cadence---and how does a change get to ship?

### [SR 8] Production Retrieval and RAG Integration

- `L5` `System Design` *** — You own a search endpoint with a 250,ms server-side p99 SLO. Walk me through the latency budget---where does the time go, and what do you cut when you're over?
- `L6` `System Design` *** — Your search index must reflect catalog changes within 60 seconds---design it
- `L5` `Conceptual` ** — Why are deletes and updates hard in HNSW, and what does your vector database actually do when you call delete()?
- `L6` `Debugging` *** — Relevance regressed after last week's embedder upgrade---find it
- `L6` `System Design` ** — You have no labels in production. How do you know your retrieval quality hasn't regressed---and how do you find out fast?
- `L6` `Trade-off` ** — Traffic to your LLM-answer product is expensive. Design a semantic cache---and tell me how it goes wrong
- `L6` `Trade-off` ** — You're building the index behind a RAG product over 100M documents. Chunk-level or doc-level vectors---and what does the choice do to your index?
- `L5` `Trade-off` ** — We have 20M vectors, moderate QPS, and we already run Postgres and OpenSearch. Do we need a vector database?
- `L7` `System Design` ** — Cut retrieval serving cost 5`times` without visible quality loss
- `L7` `Estimation` ** — A new embedding model tops the leaderboard. Do you re-embed your 100M-document corpus? Walk through the decision and the cost

## Volume IV — Conventional ML Essentials

### [CML 2] Trees and Ensembles

- `L5` `Mathematical` *** — Derive the optimal leaf weight and the split-gain formula in XGBoost
- `L5` `Conceptual` *** — Random forest vs. gradient boosting---how do they differ, and when do you reach for each?
- `L5` `First Principles` *** — Explain gradient boosting to someone who already understands gradient descent
- `L6` `Trade-off` *** — XGBoost vs. LightGBM vs. CatBoost: 600K rows, 45 features of which 14 are categorical including a 40K-cardinality merchant_id. Choose and defend
- `L6` `Conceptual` *** — Why does a gradient-boosted tree beat your MLP on this tabular dataset---and when would it not?
- `L6` `Debugging` ** — Your LambdaMART ranker's training NDCG keeps improving, but validation NDCG peaked at tree 200 and is now degrading. Walk me through your response
- `L5` `Conceptual` ** — Why do trees split on Gini or entropy instead of misclassification error, since error is what we ultimately care about?
- `L5` `Mathematical` ** — What fraction of the training data does each tree in a random forest see? Derive it, and explain what the rest is used for
- `L6` `Conceptual` ** — How does CatBoost avoid target leakage with categorical features, and what is ``ordered boosting'' fixing?
- `L5` `Conceptual` ** — Why can't a tree-based model extrapolate, and where does that bite in practice?
- `L6` `Debugging` ** — Your random forest's impurity-based feature importance says session_id_hash is the top feature. The PM wants to build product strategy around the ranking. What do you tell them?
- `L7` `System Design` * — Design the serving story for a GBDT fraud model: p99 model latency under 2,ms at 20K QPS, with a compliance requirement that risk never decreases as chargeback count increases

### [CML 6] Experimentation and Causal Inference

- `L5` `Estimation` *** — We want to detect a 1% relative lift on a 2% click-through rate. How many users do we need, and how long should we run?
- `L6` `Trade-off` *** — Your test came back flat, but you are confident the feature helps. What do you do?
- `L6` `Debugging` *** — Your treatment shows +2% on the primary metric, but the SRM check fired. What now?
- `L6` `Conceptual` ** — Explain CUPED to a PM, and quantify what it buys us
- `L6` `Conceptual` *** — Your marketplace A/B test shows +2% GMV in treatment. Why might the true effect be smaller, or negative?
- `L6` `Conceptual` ** — Why not send retention offers to the users with the highest churn probability?
- `L6` `Trade-off` ** — Thompson sampling, UCB, or `epsilon`-greedy---mechanics, and when would you use a bandit instead of an A/B test?
- `L7` `Architecture Design` ** — We cannot randomize prices. How would you estimate price elasticity from historical data?
- `L5` `Conceptual` ** — What assumption does difference-in-differences require, and how would you probe it?
- `L7` `Architecture Design` ** — Design the experimentation platform for a 200-engineer organization

