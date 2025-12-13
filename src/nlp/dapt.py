import argparse
import os
import math
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dapt(model_name: str, train_file: str, output_dir: str, epochs: int = 1, batch_size: int = 4, fp16: bool = False):
    """
    Runs Domain Adaptive Pretraining (Masked Language Modeling) on a given corpus.
    """
    logger.info(f"Starting DAPT for model: {model_name}")
    logger.info(f"Training data: {train_file}")
    logger.info(f"Output directory: {output_dir}")

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise e

    # Load Model
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise e

    # Prepare Dataset
    # Using 'text' builder from datasets library is more robust than LineByLineTextDataset
    logger.info("Loading and tokenizing dataset...")
    raw_datasets = load_dataset("text", data_files={"train": train_file})
    
    context_length = 128 # Can be up to 512, but 128 is faster for DAPT and usually sufficient
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=context_length, padding="max_length")

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    
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
        weight_decay=0.01,
        fp16=fp16,
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )
    
    logger.info(f"Training on {len(tokenized_datasets['train'])} samples...")
    train_result = trainer.train()
    
    logger.info(f"Saving adapted model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info("DAPT complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptive Pretraining (DAPT)")
    parser.add_argument("--model", type=str, default="PlanTL-GOB-ES/roberta-large-bne", help="Base model HuggingFace ID")
    parser.add_argument("--data", type=str, required=True, help="Path to corpus.txt")
    parser.add_argument("--output", type=str, default="models/roberta-adapted", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    
    args = parser.parse_args()
    
    # Simple logic to handle model fallback name if user passed 'gov_roberta' or 'beto' manually
    # ideally CLI wires this, but safely handling here too:
    if args.model == "gov_roberta": args.model = "PlanTL-GOB-ES/roberta-large-bne"
    if args.model == "beto": args.model = "dccuchile/bert-base-spanish-wwm-uncased"
    
    dapt(args.model, args.data, args.output, args.epochs, fp16=args.fp16)
