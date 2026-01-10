from ultralytics import YOLO
import configparser
import torch
import os

def model_training(model: YOLO, dataset: str, output: str, epochs: int = 10, batch: int = 5, workers: int = 8):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  results = model.train(
    data=dataset,
    epochs=epochs,
    batch=batch,
    workers=workers,
    device=device,
    project=output,
    cache=False,
    save=True
  )

  return results

if __name__ == "__main__":
  # Load a model
  model = YOLO("yolo11m.pt") 

  # Load configures from config file
  config = configparser.ConfigParser()
  config.read("settings.ini")

  # Train a model
  model_training(
    model=model,
    dataset=config["DEFAULT"]["DATASET"],
    output=f"{os.getcwd()}/datasets/result/",
    epochs=int(config["DEFAULT"]["EPOCHS"]),
    batch=int(config["DEFAULT"]["BATCH"]),
    workers=int(config["DEFAULT"]["WORKERS"])
  )