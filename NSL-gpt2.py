import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    #翻译成函数 x³  →  ×0.044715  →  +x  →  ×√(2/π)  →  tanh  →  +1  →  ×x  →  ×0.5
    return x * 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    #输入一矩阵x（类型为torch.tensor），输出一矩阵y（类型为torch.tensor）
    #每行减去该行的最大值，再取指数，再除以该行所有元素指数之和
    x_max = x.max(dim=-1, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=-1, keepdim=True)



def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    #输入矩阵x（类型为torch.tensor），g_b['g']是x的权重，g_b['b']是x的偏置
    #进行归一化，先求均值，再求方差，再求标准差
    #稳定每层输入的分布，防止梯度爆炸或梯度消失
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * g + b
    

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    #输入一矩阵x（类型为torch.tensor），w_b['w']是x的权重，w_b['b']是x的偏置
    w, b = w_b['w'], w_b['b'] #读取权重和偏置 w是权重，b是偏置
    return x @ w + b
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    # 只带一个隐藏层的神经网络
    #输入矩阵x c_fc是输入到隐藏层权重，c_proj是隐藏层到输出层的权重
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    h = linear(x, w_b1) #h是隐藏层，3072维
    h = gelu(h) 
    return linear(h, w_b2)  
    


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    # q是query，k是key，v是value，mask是掩码
    # 本质是单头注意力，加上了因果mask
    d_k = k.shape[-1] # d_k是key的维度
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_k) # scores是注意力得分矩阵
    scores = scores + mask #加上因果mask
    weights = softmax(scores) #weights是注意力权重矩阵 softmax把得分转成比例
    return weights @ v #weights @ v是注意力输出矩阵
    

def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection 一次得到qkv三个矩阵的联合体
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    #2304维

    # Split into qkv 
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    qkv = x.chunk(3, dim=-1)  # [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]) 
    # need to modify 把联合在一起的矩阵拆分成三个矩阵

    # Split into heads 把768维度分成12个头，每头64维度
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    n_seq = x.shape[0] # n_seq是序列长度
    causal_mask = torch.triu(torch.full((n_seq, n_seq), float('-inf')), diagonal=1) # need to modify 构造因果mask
     #其实就是一个上三角矩阵，对角线及以下全为0，对角线以上全为-inf
    # Perform attention over each head 
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1) # need to modify 合并多头注意力
    
    # Out projection 再对多头注意力的结果进行一次融合
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    #在gpt2中循环被调用12次
    #block是gpt2的12个transformer block的参数
    #n_head是多头注意力的头数
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    #解析出本层的参数
    #ln_1, ln_2是两个layer norm的参数
    
    # multi-head causal self attention 残差连接
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network 残差链接
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    #这里的归一化都是在函数入口做的，所以产出没有进行归一化操作
    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab] 被generate调用
    #输入整理完毕的token ID序列、模型权重、多头注意力的头数
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    #进行embedding的线性变换（500267*768 =500267*12*64） + 位置编码（1024*768限定长度是1024）
    
    
    x = torch.Tensor(x) #numpy 转 torch
    # forward pass through n_layer transformer blocks
    for block in blocks: #十二层decoder layer
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd] 层间归一化
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab] 用来做输出打分，使用wte的转置矩阵作为权重，属于一个逆向过程


def generate(inputs, params, n_head, n_tokens_to_generate): # [n_seq] -> [n_tokens_to_generate]真实的自回归 decoding
    #输入inputs是经过处理后的token ID序列、全部权重、多头数、生成token数
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass 每个循环进行一次前向传播
        #这里每次都进行全部序列的前向传播，还不存在KV缓存
        
        next_id = np.argmax(logits[-1])  # greedy sampling 选择概率最大的token ID 贪心选择
        
        #由于是贪心选择，只看最大值，不用进行softmax，因为softmax只影响概率，不影响最大值
        
        inputs.append(int(next_id))  # append prediction to input 把预测的token ID添加到输入序列中

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids 只返回最新生成的token ID序列

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        pass

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    #1.装载函数
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    #2.对文本执行编码
    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    #3.真正的生成过程
    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    #4.解码输出文本
    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)