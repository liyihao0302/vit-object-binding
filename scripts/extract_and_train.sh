#  ----- DINOv2 family -----
# dinov2-large
python ../src/main.py mode=extract_and_save model.name='facebook/dinov2-large'
python ../src/main.py mode=train model.name='facebook/dinov2-large' trainer.layer=18

# dinov2-base
#python ../src/main.py mode=extract_and_save model.name='facebook/dinov2-base' model.embed_dim=768
#python ../src/main.py mode=train model.name='facebook/dinov2-base' model.embed_dim=768 probe.in_dim=768 trainer.layer=9

# dinov2-small
#python ../src/main.py mode=extract_and_save model.name='facebook/dinov2-small' model.embed_dim=384
#python ../src/main.py mode=train model.name='facebook/dinov2-small' model.embed_dim=384 probe.in_dim=384 trainer.layer=5

# dinov2-giant
#python ../src/main.py mode=extract_and_save model.name='facebook/dinov2-giant' model.embed_dim=1536
#python ../src/main.py mode=train model.name='facebook/dinov2-giant' model.embed_dim=1536 probe.in_dim=1536 trainer.layer=29

# ----- CLIP -----
#python ../src/main.py mode=extract_and_save model.name='openai/clip-vit-large-patch14' data_extractor.image_processor.crop_size=[224,224]
#python ../src/main.py mode=train model.name='openai/clip-vit-large-patch14' data_extractor.image_processor.crop_size=[224,224] trainer.layer=23

# ----- MAE -----
#python ../src/main.py mode=extract_and_save model.name='facebook/vit-mae-large' data_extractor.image_processor.crop_size=[224,224] data_extractor.image_processor.shortest_edge=592 model.patch_size=16
#python ../src/main.py mode=train model.name='facebook/vit-mae-large' data_extractor.image_processor.crop_size=[224,224] data_extractor.image_processor.shortest_edge=592 model.patch_size=16 trainer.layer=23

# ----- Supervised ViT on ImageNet -----
#python ../src/main.py mode=extract_and_save model.name='google/vit-large-patch16-224' data_extractor.image_processor.crop_size=[224,224] data_extractor.image_processor.shortest_edge=592 model.patch_size=16
#python ../src/main.py mode=train model.name='google/vit-large-patch16-224' data_extractor.image_processor.crop_size=[224,224] model.patch_size=16 trainer.layer=23
