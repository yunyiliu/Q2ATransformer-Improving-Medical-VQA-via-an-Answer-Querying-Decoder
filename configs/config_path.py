import argparse
import ast


parser = argparse.ArgumentParser(description="Benchmark training for VLP")

parser.add_argument('--test_mode', type=bool, default=False, help='pretrain or finetune')
parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or finetune')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

# ========================= Dataset Configs ==========================
# parser.add_argument('--annotation', type=str, default=r'data/annotation.json', help="mimic annotation")
parser.add_argument('--annotation', type=str, default=r'/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/path_annotation.json', help="mimic annotation")
parser.add_argument('--base_dir_path', type=str, default=r'/root/VQA_Main/Modified_MedVQA-main/data/path_VQA/images', help="/home/zhanyu_wang/data/mimic_cxr/images")
parser.add_argument('--base_dir', type=str, default=r'/root/VQA_Main/Modified_MedVQA-main/data/VQA_RAD_Images', help="/home/zhanyu_wang/data/mimic_cxr/images")
parser.add_argument('--batch_size', default=4, type=int, help="use for training duration per worker")
parser.add_argument('--val_batch_size', default=128, type=int, help="use for validation duration per worker")
parser.add_argument('--test_batch_size', default=128, type=int, help="use for testing duration per worker")
parser.add_argument('--prefetch', default=4, type=int, help="use for training duration per worker")
parser.add_argument('--cpu_num', default=4, type=int, help="Cpu num for dataloaders")

# ======================== SavedModel Configs =========================
parser.add_argument('--resume_training', default=0, type=int, help='resume training from checkpoints')
parser.add_argument('--savedmodel_path', type=str, default='save/vqa_rad/v1')
parser.add_argument('--ckpt_file', type=str, default=None)
# parser.add_argument('--ckpt_file', type=str, default='/home/zhanyu_wang/code/NIPS2022/save/candix/v2_swinfreeze_base_reduction4/checkpoints/epoch=6-step=1679-bleu=0.1298-cider=0.0822.ckpt')
parser.add_argument('--max_to_keep', default=60, type=int, help='the number of checkpoints to keep')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

# ========================= Learning Configs ==========================
parser.add_argument('--max_steps', default=100000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup_steps', default=4000, type=int, help="warm ups for parameters not in bert or vit")
parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--max_cache_batches', default=1, type=int, help="How many batch cache in momentum")
parser.add_argument('--softmax_smooth_value', default=0.05, type=float, help="Label smooth value")
parser.add_argument('--tau', default=0.05, type=float, help="Label smooth value")

# ========================== question BERT =============================
parser.add_argument('--bert_dir', type=str, default='bert-base-uncased')  # macbert-large macbert-base
parser.add_argument('--bert_cache_dir', type=str, default='./pretrained_models/cache')
parser.add_argument('--bert_question_max_length', type=int, default=20)
parser.add_argument('--bert_answer_max_length', type=int, default=10)
parser.add_argument('--bert_learning_rate', type=float, default=5e-6)
parser.add_argument('--bert_warmup_steps', type=int, default=4000)
parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
parser.add_argument("--training", type=ast.literal_eval, default=True, help="whether to parse tags or channels")
parser.add_argument("--shuffle_tag", type=ast.literal_eval, default=True, help="whether shuffle union_tags orders")
parser.add_argument('--bert_vocab_size', type=int, default=30522)  # 21128  30522

# ========================== Image VIT =============================
parser.add_argument('--vision_encoder', type=str, default='vit')  # vit\efficientnet
parser.add_argument('--vit_dir', type=str, default='google/vit-base-patch16-224-in21k')
parser.add_argument('--vit_cache_dir', type=str, default='./pretrained_models/cache')
parser.add_argument('--image_width', type=int, default=224, help="default for vit-16-base")
parser.add_argument('--image_height', type=int, default=224, help="default for vit-16-base")
parser.add_argument('--vision_learning_rate', type=float, default=5e-5)
parser.add_argument('--vision_warmup_steps', type=int, default=4000)
parser.add_argument("--vision_hidden_dropout_prob", type=float, default=0.1)
parser.add_argument("--img_arg", type=ast.literal_eval, default=False, help="whether to argument cover")

# ========================== Image Swin Transformer =============================
# parser.add_argument('--swin_transformer_yaml', type=str, default='./pretrained_models/Swin-S224/swin_small_patch4_window7_224_22k.yaml')  # swin_large_patch4_window7_224
# parser.add_argument('--swin_transformer_ckpt', type=str, default='./pretrained_models/Swin-S224/swin_small_patch4_window7_224_22k.pth')  # b5 2048 b7 2560
parser.add_argument('--swin_transformer_yaml', type=str, default='/root/VQA_Main/Modified_MedVQA-main/pretrained_models/Swin-S224/swin_base_patch4_window7_224_22k.yaml')  # swin_large_patch4_window7_224
parser.add_argument('--swin_transformer_ckpt', type=str, default='/root/VQA_Main/Modified_MedVQA-main/pretrained_models/Swin-S224/swin_base_patch4_window7_224_22k.pth')  # b5 2048 b7 2560

parser.add_argument('--swin_transformer_ckpt_unet', type=str, default='./pretrained_models/Swin-B224/swin_tiny_patch4_window7_224_22k.pth')  # b5 2048 b7 2560
parser.add_argument('--swin_hidden_size', type=int, default=49, help="Swin Transformers last layer hidden_size")  # 49

#=================================== Fusion cross Transformer ============================
parser.add_argument('--num_cross_layer', default=2, type=int,  help="Number of cross transformer")

#=================================== Transformer decoder ============================
parser.add_argument('--num_classes', default=3974, type=int, 
                        help="Number of answers")
# parser.add_argument('--num_classes', default=458, type=int, help="Number of answers")
parser.add_argument('--enc_layers', default=2, type=int, 
                        help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=2, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=8192, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=2048, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
# parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=4, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--backbone', default='resnet101', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                    help='keep the other self attention modules in transformer decoders, which will be removed default.')
parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                    help='keep the first self attention module in transformer decoders, which will be removed default.')
parser.add_argument('--keep_input_proj', action='store_true', 
                    help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

# ====================== Pytorch Lightning ===========================
parser.add_argument('--gpus', type=int, default=1, help='how many gpus to use')
parser.add_argument('--num_nodes', type=int, default=1, help='how many machines to use')
parser.add_argument('--accelerator', type=str, default="ddp", help='default ddp for multi-gpus')
parser.add_argument('--val_check_interval', type=float, default=1.0, help='how many training steps do validation once')
parser.add_argument('--amp_backend', type=str, default="native", help='apex for original pytorch amp auto cast')
parser.add_argument('--precision', type=int, default=16, help='16 or 32, using for original pytorch amp auto cast')
# parser.add_argument('--limit_val_batches', type=int, default=50, help='How many steps runs when validation')
# parser.add_argument('--limit_train_batches', type=int, default=2000, help='How many steps runs when training')
parser.add_argument('--max_epochs', type=int, default=50, help='How many epochs')
parser.add_argument("--enable_deepspeed", type=ast.literal_eval, default=False, help="whether to deepspeed")
parser.add_argument("--sync_batchnorm", type=ast.literal_eval, default=False, help="whether to sync_batchnorm across gpus")
parser.add_argument("--num_sanity_val_steps", type=int, default=0)
parser.add_argument("--auto_scale_batch_size", type=bool, default=False)
# parser.add_argument("--check_val_every_n_epoch", type=int, default=3)
parser.add_argument("--hparams_file", type=str, default='/home/zhanyu_wang/code/NIPS2022/save/candix/v2_swinfreeze_base_reduction4/logs/csvlog/version_0/hparams.yaml')
# parser.add_argument("--resume_from_checkpoint", type=str, default='/home/zhanyu_wang/code/MedVQA/save/vqa_rad/v6_cls/checkpoints/last.ckpt')
# ====================== SAN ===========================
parser.add_argument('--num_stacks', default=2, type=int,
                    help='num of stacks in Stack Attention Networks')
# Utilities - support testing, gpu training or sampling
parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                    help='print per certain number of steps')
parser.add_argument('--gpu', type=int, default=0,
                    help='specify index of GPU using for training, to use CPU: -1')
parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

# ====================== BAN - Bilinear Attention Networks ==========================
parser.add_argument('--gamma', type=int, default=2,
                    help='glimpse in Bilinear Attention Networks')
parser.add_argument('--use_counter', action='store_true', default=False,
                    help='use counter module')