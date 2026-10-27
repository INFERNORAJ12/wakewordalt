import sys
import os
from model import WakeWordClassifier
from dataset import generate_wake_word_dataset
from train import train_network
from export import export_model_to_onnx, verify_onnx

def main():
    target_phrase = sys.argv[1] if len(sys.argv) > 1 else "autocon"
    output_filename = f"{target_phrase}.onnx"

    print(f"=== Starting Wake Word Generation: '{target_phrase}' ===")

    try:
        # Step 1: Initialize model
        print("\n[1/4] Building neural network graph...")
        classifier = WakeWordClassifier(input_dim=1536, hidden_dim=32)

        # Step 2: Prepare synthesized dataset
        print("\n[2/4] Synthesizing acoustic feature space...")
        loader = generate_wake_word_dataset(target_word=target_phrase, n_positives=3000, n_negatives=7000)

        # Step 3: Train
        print("\n[3/4] Optimizing binary cross-entropy loss...")
        trained_model = train_network(classifier, loader, epochs=8, lr=0.001)

        # Step 4: Export & verify ONNX
        print("\n[4/4] Exporting and validating ONNX runtime binary...")
        export_model_to_onnx(trained_model, output_filename, input_dim=1536)
        verify_onnx(output_filename, input_dim=1536)

        print(f"\n[+] Success! Your offline model is ready at: {os.path.abspath(output_filename)}")

    except Exception as err:
        print(f"\n[!] Pipeline execution failed: {err}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()