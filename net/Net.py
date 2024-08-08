from torch.nn import Module, Sequential, GELU, Linear, Conv2d, ModuleList, ReplicationPad2d
from net.Modules import *
from typing import Tuple, Optional

class UNet(Module):
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        emb_dim: int = 512,
        dims: Tuple[int] = (128, 256, 256, 512, 512, 512),
        dropouts: Tuple[float] = (0., 0., 0., 0., 0.1, 0.1),
        num_resblocks: Tuple[int] = (2, 2, 2, 2, 4, 4),
        use_attn: Tuple[bool] = (False, False, False, False, True, True),
        attn_heads: int = 4,
        in_conv_kernel_size: int = 5,
        t_dim_in: int = 1,
        fourier_features_scale: float = math.pi * 2
    ):
        super().__init__()
        self.t_dim_in = t_dim_in
        self.resolutions = len(dims)
        self.register_buffer('root2', torch.sqrt(torch.tensor(2)))
        self.time_mlp = Sequential(
            FourierFeatures(dims[0], std=1., num_features=t_dim_in, scale=fourier_features_scale),
            Linear(dims[0], emb_dim),
            GELU(),
            Linear(emb_dim, emb_dim)
        )
        self.in_conv = Conv2d(in_channels, dims[0], in_conv_kernel_size, padding = in_conv_kernel_size // 2)
        
        #self.out_conv = Conv2d(dims[0], out_channels, 1, bias=False)
        self.out_pad = ReplicationPad2d(1) # remove if this breaks things
        self.out_conv = Conv2d(dims[0], out_channels, 3, padding=0, bias=False) # set to padding=1 if this broke things

        def get_resblocks(lvl: int):
            return [ResBlock(dims[lvl], dims[lvl], emb_dim, dropouts[lvl]) for _ in range(num_resblocks[lvl])]

        def make_blocks(lvl: int):
            attn = [ImageSelfAttention2d(dims[lvl], attn_heads, dropouts[lvl])] if use_attn[lvl] else []
            return get_resblocks(lvl) + attn

        self.down_blocks = ModuleList([
            ModuleList(make_blocks(i) + [Downsample(dims[i], dims[i + 1])])
            for i in range(self.resolutions - 1)
        ])

        self.mid_resblocks1 = ModuleList(get_resblocks(-1))
        self.mid_attn = ImageSelfAttention2d(dims[-1], attn_heads, dropouts[-1])
        self.mid_resblocks2 = ModuleList(get_resblocks(-1))

        self.up_blocks = ModuleList([
            ModuleList([Upsample(dims[-i - 1], dims[-i - 2])] + make_blocks(-i - 2))
            for i in range(self.resolutions - 1)
        ])

        self.print_param_count()

    def print_param_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([p.numel() for p in model_parameters])
        print(f'Num params in network: {n_params}\n({int(n_params / 1000000)}M params)', )

    def get_device(self):
        return next(self.parameters()).device
    
    def forward(self, x, t, c: Optional[torch.tensor] = None):
        """
        x: (B,C,H,W) tensor
        t: (B,k) tensor
        """
        # If t is not a (B,k) tensor, ensure it becomes one
        if t.dim() == 0:
            assert self.t_dim_in == 1
            t = torch.ones(x.size(0), 1).to(self.get_device()) * t
        elif t.dim() == 1:
            t = t[:, None]
            assert self.t_dim_in == 1
        elif t.dim() == 2:
            assert t.size(1) == self.t_dim_in
        elif t.dim() == 4:
            assert t.size(2) == 1 and t.size(3) == 1
            t = t.view(-1, t.size(1))
            assert t.size(1) == self.t_dim_in
        else:
            raise ValueError("t expected to be a single float, a 1d tensor or a 4d tensor of shape (B, t_dim_in, 1, 1)")
        
        temb = self.time_mlp(t)
        hs = []

        def block_forward_down(h, block):
            h = block(h, temb) if isinstance(block, ResBlock) else block(h)
            hs.append(h)
            return h
        
        def block_forward_up(h, block):
            skip_h = hs.pop()
            h = (h + skip_h) / self.root2
            h = block(h, temb) if isinstance(block, ResBlock) else block(h)
            return h

        last_h = self.in_conv(x)

        for down_block in self.down_blocks:
            for block in down_block[:-1]:
                last_h = block_forward_down(last_h, block)
            downsample = down_block[-1]
            last_h = downsample(last_h)

        for block in self.mid_resblocks1:
            last_h = block_forward_down(last_h, block)
        last_h = self.mid_attn(last_h)
        for block in self.mid_resblocks2:
            last_h = block_forward_up(last_h, block)
        
        for up_block in self.up_blocks:
            upsample = up_block[0]
            last_h = upsample(last_h)
            for block in up_block[1:]:
                last_h = block_forward_up(last_h, block)

        return self.out_conv(self.out_pad(last_h)) # remove out_pad if needed