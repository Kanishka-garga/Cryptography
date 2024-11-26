import random 
import pandas as pd
from fractionated_morse import fractionated_morse_encrypt  
from grandpre_cipher import grandpre_encrypt  

# Generate random plaintexts
def generate_random_plaintext(min_length, max_length):
    """Generate random plaintext of given length range."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ ', k=length))

def create_combined_dataset(num_samples=400):
    """Create dataset with ciphertext and labels for both algorithms."""
    data = []
    plaintext_min_length = 110
    plaintext_max_length = 200

    # Generate for Fractionated Morse
    for _ in range(num_samples // 2):
        plaintext = generate_random_plaintext(plaintext_min_length, plaintext_max_length)
        ciphertext = fractionated_morse_encrypt(plaintext)
        data.append((ciphertext, "Fractionated Morse"))

    # Generate for Grandpré
    for _ in range(num_samples // 2):
        plaintext = generate_random_plaintext(plaintext_min_length, plaintext_max_length)
        ciphertext = grandpre_encrypt(plaintext)
        data.append((ciphertext, "Grandpré"))

    # Shuffle data
    random.shuffle(data)

    # Save to CSV
    df = pd.DataFrame(data, columns=["Ciphertext", "Algorithm"])
    df.to_csv("final_dataset.csv", index=False)

if __name__ == "__main__":
    create_combined_dataset()
