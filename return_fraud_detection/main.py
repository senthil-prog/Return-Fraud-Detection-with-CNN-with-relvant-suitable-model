import argparse
import sys
import os

# Add src to path
sys.path.append('src')

from train import train_model
from evaluate import evaluate_model
from predict import example_prediction

def main():
    parser = argparse.ArgumentParser(description='Return Fraud Detection with CNN')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       required=True, help='Mode to run the application')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    if args.mode == 'train':
        print("Starting training mode...")
        train_model()
        
    elif args.mode == 'evaluate':
        print("Starting evaluation mode...")
        evaluate_model()
        
    elif args.mode == 'predict':
        print("Starting prediction mode...")
        example_prediction()

if __name__ == "__main__":
    main()