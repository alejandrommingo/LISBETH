import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, LineByLineTextDataset
import os
import torch

def dapt(model_name, train_file, output_dir, epochs=1, batch_size=4):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    print(f"Loading dataset: {train_file}")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128, # Keep small for speed in demo, or 512 for production
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        dataloader_num_workers=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving adapted model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptive Pretraining (DAPT)")
    parser.add_argument("--model", type=str, default="PlanTL-GOB-ES/roberta-large-bne", help="Base model")
    parser.add_argument("--data", type=str, required=True, help="Path to corpus.txt")
    parser.add_argument("--output", type=str, default="models/roberta-large-bne-yape", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    
    args = parser.parse_args()
    dapt(args.model, args.data, args.output, args.epochs)
