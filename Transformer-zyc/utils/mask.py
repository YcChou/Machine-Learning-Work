import torch

def create_masks(src, tgt):
    # src: [bs, n]
    # tgt: [bs, m-1]
    def get_pad_mask(src):
        return (src == 0).unsqueeze(-2).to('cuda') # [bs, 1, n] 在dim=1上广播
    src_mask = get_pad_mask(src) # [bs, n, n]

    def get_subsequent_mask(tgt): # 上三角矩阵
        bs, len_q = tgt.size()
        subsequent_mask = (torch.triu(torch.ones((1, len_q, len_q)), diagonal=1)).bool()
        # triu(, diagonal=1) 保留主对角线上面一行，及其往上的全部
        return subsequent_mask.to('cuda')
    tgt_mask = get_pad_mask(tgt) | get_subsequent_mask(tgt) # decoder自己本来对句末padding的mask和遮蔽当前时刻后的mask叠加

    return src_mask, tgt_mask