# 注意：需要配置以下环境变量
unset NCCL_IB_GID_INDEX
unset NCCL_IB_HCA
unset NCCL_IB_SL
unset NCCL_IB_TC
unset NVIDIA_VISIBLE_DEVICES
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_QUADRUPLE_CHANNELS=0
export NCCL_DEBUG=INFO
export UNICM_LOG_LEVEL=ERROR
export NCCL_NET_GDR_LEVEL=PIX 
export NCCL_IB_DISABLE=1 
export NCCL_CROSS_NIC=0
export NCCL_PF_MTU=5000
export NCCL_DEBUG=info
export NCCL_PF_CQE_ORDER=1
export NCCL_PF_SEND_FLAG_RR=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="fic2_soe_bond0,fic2_soe_bond1,fic2_soe_bond2,fic2_soe_bond3"

/usr/local/bin/python /home/data/guojiawei/stella-master/finetune3/run.py \
--output_dir /home/data/guojiawei/stella-master/output/0705/ \
--model_name_or_path  /home/data/fanyue.zy/project/rag/models/stella_large_zh_v3_1792d \
--train_data /home/data/guojiawei/stella-master/data/train_data \
--learning_rate 1e-4 \
--fp16 \
--num_train_epochs 30 \
--per_device_train_batch_size 25 \
--gradient_accumulation_steps 20 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 256 \
--passage_max_len 500 \
--train_group_size 2 \
--logging_steps 10 \
--save_strategy "epoch" \
--query_instruction_for_retrieval ""