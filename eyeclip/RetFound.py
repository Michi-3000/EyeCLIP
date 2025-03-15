import torch
import torch.nn as nn
from clip_modules.modeling.models_vit import VisionTransformer
from functools import partial
from clip_modules.modeling.pos_embed import interpolate_pos_embed
from clip_modules.modeling.timm_models.models.layers import trunc_normal_
class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""
    def forward(self, x):
        return x
def vit_large_patch16(pretrained=True):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    if pretrained:
        # load RETFound weights
        checkpoint = torch.load(
            '/home/danli/workspace/pretrain/mae_zwy/RETFound_cfp_weights.pth',
            map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("msg.missing_keys === {}".format(msg.missing_keys))

        # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    # print("Model = %s" % str(model))
    model.head = Identity()
    return model


from clip_modules.modeling.LORA.lora_image_encoder import LoRA_ViT
def lora(pretrained=True,R=8):
    print("================ LORA R================{}".format(R))
    ViT= vit_large_patch16(pretrained=pretrained)
    model = LoRA_ViT(ViT, r=R)
    return model


if __name__ == "__main__":
    net = vit_large_patch16(pretrained=False)
    images = torch.rand(2, 3, 224, 224)
    output = net(images)
    print(output.shape)