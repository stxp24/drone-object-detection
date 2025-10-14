from ultralytics import YOLO

model = YOLO("../src/yolo11n.pt")

# train the model
results = model.train(
    data="config.yaml",        # path to dataset configuration
    epochs=3,                  # number of training epochs
    name="mclaren_tracker",      # name for this training run
    save=True,                   # save checkpoints
    project="runs/train",        # where to save results
    pretrained=True,             # use pretrained weights
    augment=True,                # use data augmentation
)

# after training completes, the best model is saved at:
# runs/train/mclaren_tracker/weights/best.pt
print("Training complete")
print(f"Best model saved to: {results.save_dir}/weights/best.pt")
