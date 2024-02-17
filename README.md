## I'll share my LLM learning notes here!

# [Coursera](https://www.coursera.org/learn/generative-ai-with-llms)

[Papers](papers.md)
# Transformer
Bert: encoder only model  
Bard: encoder-decoder model  
GPT: decoder only model  

# Prompting
**in-context learning** = zero shot inference: include input data within the prompt. Large LLM is good at zero-shot but not trained on specific task.   
**one shot inference**: include the example that demonstrates the tasks to be carried out to the model  
**few shot inference**: include multiple examples  
**context window**: limit on the amount of in-context learning that you can pass in to the model  
If few shot doesn't work well, should try fine-tuning which trains on additional data to make it more capable of the task you want it to perform.  

# LLM Lifecycle
Choose an existing model.  
Adapt the model: prompt engineering, fine-tuning, align w/ human feedback (RLHF), and evaluation.  
LLM limitation: invent information when they don't know an answer, limited ability to come out complex reasoning and mathematics.  

# Fine-tuning
**Parameter efficient fine-tuning**: freeze the model parameter or add task-specific adaptive layer on top of that for the sake of a smaller memory footprint. Solves catastrophic forgetting problem.  
**LoRA**: use low rank matrices as opposed to full fine-tuning.  
Pretraining is selfsupervised learning on vast textual data, while fine tuning is supervised learning on set of labeled examples of prompt completion pairs.  
**Instruction fine-tuning**: train to respond to a specific instruction. The prompt completion examples allow the model to learn to generate responses that follow the given instructions. Prepare instruction dataset. Full fine-tuning.  
**Evaluation**: cross-entropy on the output token distribution and the completion.  
**Single-task fine-tuning**: 500-1000 examples would be enough. Cons: catastrophic forgetting.  
**Multi-task fine-tuning**: 50-100k examples.  

**PEFT** - 30%-50% parameters  
Method:  
**Selective**  
Reparameterizem model weights using a low-rank representation/LoRA  
Add trainable layers or parameters to model - Adapters (add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers) /Soft Prompts (keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance. This can be done by adding trainable parameters to the prompt embeddings or keeping the input fixed and retraining the embedding weights)  
  
**LoRA**  
original weight = d x k  
LoRA = r x k and d x r  
final weight = original weight + LoRA = d x k + d x r X r x k  
  
**Prompt tuning**  
Differnet from prompt engineering: one-shot or few-shot inference  
Add additional trainable "soft prompt" to inputs, prepend to embedding vectors, usually 20-100 tokens  


# Configuration  
Temperature: the higher the temperature, the higher the randomness is.   
It's a setting in softmax output layer that determines the shape of the output token distribution.  

# Evaluation  
Rouge  
BLEU  
Benchmark: MMLU  

# RLHF  
RLHF: fine tune LLM w/ human feedback data  
e.g., reduce harmful, aggressive, dangeours, dishonest and toxic content learnt from internet data. Responsible AI  
e.g., train the model to give caveats that acknowledge their limitations and to avoid toxic language and topics.  
e.g., personalization of LLM  

RL  
an agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward.  
Ageent RL policy = LLM, Environment = LLM context, Action = completion token, rewards = how the completions aligns w/ human knowledge. e.g., reward = 0/1.  
RLHF is expensive and time-consuming, alternatives is to have an additional model / reward model that classifies the outputs of LLM and evaluate the alignment w/ human preference. Train the reward model w/ human labels using supervised learning, then use the reward model to assess the output of LLM.  
Human: assess the LLM completion, mulitple labelers w/ consensus.  
Reward model: convert into pairwise training data? why not point-wise? also a language model w/ supervised learning  
RL aoglrithm to iteratively update the LLM model weights based on reward model. Algorithm: PPO.  

Reasoning engine

RAG to incoperate external information to LLM

# Applications
LLM Chanllenges: out-of-date knowledge, reasoning/math problem, tendency to generate response on things they don't know /hallucination  

Orchestration/Langchain: This layer can enable some powerful technologies that augment and enhance the performance of the LLM at runtime. By providing access to external data sources or connecting to existing APIs of other applications.  

RAG/framework: instead of retraining the model on new model, give model access to additional external data at inference time. Retriever: query dencoder + external information sources -> combines the new text (in vector store) w/ the original prompt -> pass the expanded prompt to LLM and generate the completion. Considerations: data must fit inside context window, external data must be available as embedding vectors (get from LLM) at inference time for LLM to consume  

reasoning and planning  
external application: LLM trigger API call. LLM generates completions including plan actions, format outputs, validate actions. Structuring the prompt in correct way make huge difference.  
multi-step math problems: CoT - prompt the model to think more like a person  

ReAct / prompt strategy (reasoning + action planning): LLM + websearch API: LLM formats the request in a ceratin way w/ a limited action space + identify the next actions, prompt w/ instruction + examples + question to answer.  

