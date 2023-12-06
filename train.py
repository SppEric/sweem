import torch
from tqdm import tqdm
from collections import Counter 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

batch_size = 4

def train_loop(model, optimizer, num_epochs, train_dataloader, validation_dataloader, lr_scheduler, device):
  """
  :param torch.nn.Module model: the model to be trained
  :param torch.optim.Optimizer optimizer: the training optimizer
  :param int num_epochs: number of epochs to train for
  :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
  :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
  :param _ lr_scheduler: learning rate scheduler
  :param torch.device device: the device that we'll be training on
  
  :return int train_losses, int val_losses
  """
  model.to(device)
  for epoch in range(num_epochs):
    ## Training
    # put the model in training mode (important that this is done each epoch,
    # since we put the model into eval mode during validation)
    model.train()

    print(f"Epoch {epoch + 1} training:")
    progress_bar = tqdm(range(len(train_dataloader)))

    for i, batch in enumerate(train_dataloader):
      batch["input_ids"] = batch["input_ids"].to(device)
      batch["start_positions"] = batch["start_positions"].to(device)
      batch["end_positions"] = batch["end_positions"].to(device)
      batch["attention_mask"] = batch["attention_mask"].to(device)

      # Calculate the predictions
      predictions = model(**batch)

      # Back propagate
      loss = predictions.loss
      loss.backward()

      # Adjust learning rate
      optimizer.step()
      lr_scheduler.step()

      # Zero the optimizer 
      optimizer.zero_grad()
      
      progress_bar.update(1)

    ## Validation
    print("Running validation:")
    val_loss, val_metrics = eval_loop(model, validation_dataloader, device)
    print(f"Epoch {epoch+1} validation: {val_metrics}")
    print(f"Epoch {epoch+1} losses; Train: {loss}, Validation: {val_loss}")

  return loss, val_loss


def eval_loop(model, validation_dataloader, device):
  """
  :param torch.nn.Module model: the model to be trained
  :param torch.utils.data.DataLoader vaildation_data_loader: DataLoader containing the validation set
  :param torch.device device: the device that we'll be training on

  :return float precision, float recall, float f1_score
  """
  # Put model into evaluation mode
  model.eval()

  # we like progress bars :)
  progress_bar = tqdm(range(len(validation_dataloader)))

  model.to(device)

  # Store logit results to calculate metrics macro batch-wise
  start_logits = []
  end_logits = []
  for batch in validation_dataloader:
    ## Create predictions for this batch
    # Send the batch to the GPU
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["start_positions"] = batch["start_positions"].to(device)
    batch["end_positions"] = batch["end_positions"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)

    # Calculate the predictions
    with torch.no_grad():
      outputs = model(**batch)
    start_logits.append(outputs.start_logits)
    end_logits.append(outputs.end_logits)

    progress_bar.update(1)

  # Update the metrics
  metrics = compute_metrics(
      validation_data, start_logits, end_logits, validation, validation_offsets
  )

  print(metrics)
  return outputs.loss, metrics



def compute_metrics(validation_data, start_logits, end_logits, raw_data, offsets):
  # Initialize vars
  total_precision = 0
  total_recall = 0
  total_f1 = 0

  # Iterate for each logit guess
  for i, example in enumerate(validation_data):
    # Initialize vars for this batch's training
    context = example["contexts"]
    ids = example['input_ids']
    offset_mapping = offsets[i]
    data = raw_data[i]

    # Find guess beginning and ending indices in context
    start_guess = torch.argmax(start_logits[i // batch_size][i % batch_size]) # account for batch_size-ing
    end_guess = torch.argmax(end_logits[i // batch_size][i % batch_size])

    # Find matching tokens
    s = example['start_positions']
    e = example['end_positions'] 
    prediction = Counter(ids[start_guess : end_guess])
    ground_truth = Counter(ids[s : e])

    # Calculate metrics
    true_positives = sum((prediction & ground_truth).values())

    precision = (true_positives / (end_guess - start_guess)) if (end_guess - start_guess > 0) else 0  # TP / (TP + FP)
    recall = (true_positives / (e - s)) if (e - s > 0) else 0 # TP / (TP + FN)
    f1 = 2 / (1 / precision + 1 / recall) if precision and recall != 0 else 0

    # Sum to the total
    total_precision += precision
    total_recall += recall
    total_f1 += f1

    # print(f"Guessed answer: {tokenizer.decode(ids[start_guess : end_guess])}")
    # print(f"Guessed answer: {context[offset_mapping[start_guess][0] : offset_mapping[end_guess][1]]}")
    # print(f"Correct answer: {tokenizer.decode(ids[s : e])}")
    

  return {
      "precision": total_precision / len(validation_data),
      "recall": total_recall / len(validation_data),
      "f1": total_f1 / len(validation_data)
  }